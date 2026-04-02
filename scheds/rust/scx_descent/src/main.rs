// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2024 Andrea Righi <arighi@nvidia.com>

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;
pub use bpf_intf::*;

mod classifier;
mod debug;
mod optimizer;
mod profiles;
mod safety;
mod stats;

use std::ffi::c_int;
use std::fmt::Write;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use classifier::TaskClassifier;
use crossbeam::channel::RecvTimeoutError;
use debug::GradientDebugger;
use libbpf_rs::OpenObject;
use libbpf_rs::ProgramInput;
use libbpf_rs::RingBufferBuilder;
use log::{debug, info, warn};
use optimizer::{DescentOptimizer, GradientEvent};
use profiles::ProfileConfig;
use safety::SafetyMonitor;
use scx_stats::prelude::*;
use scx_utils::autopower::{fetch_power_profile, PowerProfile};
use scx_utils::build_id;
use scx_utils::compat;
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::pm::{cpu_idle_resume_latency_supported, update_cpu_idle_resume_latency};
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::try_set_rlimit_infinity;
use scx_utils::uei_exited;
use scx_utils::uei_report;
use scx_utils::CoreType;
use scx_utils::Cpumask;
use scx_utils::Topology;
use scx_utils::UserExitInfo;
use scx_utils::NR_CPU_IDS;
use stats::Metrics;

const SCHEDULER_NAME: &str = "scx_descent";

#[derive(PartialEq)]
enum Powermode {
    Turbo,
    Performance,
    Powersave,
    Any,
}

fn get_primary_cpus(mode: Powermode) -> std::io::Result<Vec<usize>> {
    let topo = Topology::new().unwrap();

    let cpus: Vec<usize> = topo
        .all_cores
        .values()
        .flat_map(|core| &core.cpus)
        .filter_map(|(cpu_id, cpu)| match (&mode, &cpu.core_type) {
            // Turbo mode: prioritize CPUs with the highest max frequency
            (Powermode::Turbo, CoreType::Big { turbo: true }) |
            // Performance mode: add all the Big CPUs (either Turbo or non-Turbo)
            (Powermode::Performance, CoreType::Big { .. }) |
            // Powersave mode: add all the Little CPUs
            (Powermode::Powersave, CoreType::Little) => Some(*cpu_id),
            (Powermode::Any, ..) => Some(*cpu_id),
            _ => None,
        })
        .collect();

    Ok(cpus)
}

// Convert an array of CPUs to the corresponding cpumask of any arbitrary size.
fn cpus_to_cpumask(cpus: &Vec<usize>) -> String {
    if cpus.is_empty() {
        return String::from("none");
    }

    // Determine the maximum CPU ID to create a sufficiently large byte vector.
    let max_cpu_id = *cpus.iter().max().unwrap();

    // Create a byte vector with enough bytes to cover all CPU IDs.
    let mut bitmask = vec![0u8; (max_cpu_id + 1 + 7) / 8];

    // Set the appropriate bits for each CPU ID.
    for cpu_id in cpus {
        let byte_index = cpu_id / 8;
        let bit_index = cpu_id % 8;
        bitmask[byte_index] |= 1 << bit_index;
    }

    // Convert the byte vector to a hexadecimal string.
    let hex_str: String = bitmask.iter().rev().fold(String::new(), |mut f, byte| {
        let _ = write!(&mut f, "{:02x}", byte);
        f
    });

    format!("0x{}", hex_str)
}

#[derive(Debug, clap::Parser)]
#[command(
    name = "scx_descent",
    version,
    disable_version_flag = true,
    about = "A gradient descent-based scheduler that automatically optimizes scheduling parameters for different workload classes.",
    long_about = r#"
scx_descent is a scheduler that uses gradient descent to automatically optimize
scheduling parameters for different workload classes (interactive, audio, batch, kernel).

It operates using an earliest deadline first (EDF) policy with per-class tunable parameters:

1. latency_weight (θ₁): Virtual deadline offset
2. base_slice_ns (θ₂): Preferred time slice
3. vruntime_scale (θ₃): Vruntime multiplier
4. preemption_priority (θ₄): Urgency threshold
5. migration_cost (θ₅): Cross-CPU migration penalty

These parameters are optimized via gradient descent to minimize a composite loss function
that balances latency, throughput, fairness, and efficiency.

Key features:
- Automatic task classification into workload classes
- Per-class parameter optimization
- Three optimization profiles: gaming, productivity, server
- Safety mechanisms to prevent parameter oscillation
"#
)]
struct Opts {
    /// Exit debug dump buffer length. 0 indicates default.
    #[clap(long, default_value = "0")]
    exit_dump_len: u32,

    /// Maximum scheduling slice duration in microseconds.
    #[clap(short = 's', long, default_value = "700")]
    slice_us: u64,

    /// Maximum runtime budget that a task can accumulate while sleeping (in microseconds).
    #[clap(short = 'l', long, default_value = "20000")]
    slice_us_lag: u64,

    /// Throttle the running CPUs by periodically injecting idle cycles.
    #[clap(short = 't', long, default_value = "0")]
    throttle_us: u64,

    /// Set CPU idle QoS resume latency in microseconds (-1 = disabled).
    #[clap(short = 'I', long, allow_hyphen_values = true, default_value = "32")]
    idle_resume_us: i64,

    /// Enable tickless mode.
    #[clap(short = 'T', long, action = clap::ArgAction::SetTrue)]
    tickless: bool,

    /// Enable round-robin scheduling.
    #[clap(short = 'R', long, action = clap::ArgAction::SetTrue)]
    rr_sched: bool,

    /// Specifies the initial set of CPUs as a bitmask in hex (e.g., 0xff).
    #[clap(short = 'm', long, default_value = "auto")]
    primary_domain: String,

    /// Disable SMT awareness.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    disable_smt: bool,

    /// Disable NUMA rebalancing.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    disable_numa: bool,

    /// Enable CPU frequency control (only with schedutil governor).
    #[clap(short = 'f', long, action = clap::ArgAction::SetTrue)]
    cpufreq: bool,

    /// Profile selection (gaming, productivity, server)
    #[clap(short = 'p', long, default_value = "productivity")]
    profile: String,

    /// Enable gradient debug output every N ms
    #[clap(long, value_name = "N")]
    debug_gradients: Option<u64>,

    /// Update interval for parameter sync (ms)
    #[clap(long, default_value = "50")]
    update_interval_ms: u64,

    /// Audio cgroup path for automatic classification
    #[clap(long)]
    audio_cgroup: Option<String>,

    /// Enable stats monitoring with the specified interval.
    #[clap(long)]
    stats: Option<f64>,

    /// Run in stats monitoring mode with the specified interval. Scheduler
    /// is not launched.
    #[clap(long)]
    monitor: Option<f64>,

    /// Enable BPF debugging via /sys/kernel/tracing/trace_pipe.
    #[clap(short = 'd', long, action = clap::ArgAction::SetTrue)]
    debug: bool,

    /// Enable verbose output, including libbpf details.
    #[clap(short = 'v', long, action = clap::ArgAction::SetTrue)]
    verbose: bool,

    /// Print scheduler version and exit.
    #[clap(short = 'V', long, action = clap::ArgAction::SetTrue)]
    version: bool,

    /// Show descriptions for statistics.
    #[clap(long)]
    help_stats: bool,

    #[clap(flatten, next_help_heading = "Libbpf Options")]
    pub libbpf: LibbpfOpts,
}

// Shared counters for metrics
static GRADIENT_UPDATES: AtomicU64 = AtomicU64::new(0);
static OSCILLATIONS: AtomicU64 = AtomicU64::new(0);

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    struct_ops: Option<libbpf_rs::Link>,
    opts: &'a Opts,
    topo: Topology,
    power_profile: PowerProfile,
    stats_server: StatsServer<(), Metrics>,
    user_restart: bool,
    // Phase 2: optimizer and safety components
    optimizer: DescentOptimizer,
    _classifier: TaskClassifier,
    safety: SafetyMonitor,
    debugger: Option<GradientDebugger>,
}

impl<'a> Scheduler<'a> {
    fn init(opts: &'a Opts, open_object: &'a mut MaybeUninit<OpenObject>) -> Result<Self> {
        try_set_rlimit_infinity();

        // Initialize CPU topology.
        let topo = Topology::new().unwrap();

        // Check host topology to determine if we need to enable SMT capabilities.
        let smt_enabled = !opts.disable_smt && topo.smt_enabled;

        info!(
            "{} {} {}",
            SCHEDULER_NAME,
            build_id::full_version(env!("CARGO_PKG_VERSION")),
            if smt_enabled { "SMT on" } else { "SMT off" }
        );

        // Print command line.
        info!(
            "scheduler options: {}",
            std::env::args().collect::<Vec<_>>().join(" ")
        );

        // Load profile configuration
        let _profile_config = ProfileConfig::from_name(&opts.profile);
        info!("Using profile: {}", opts.profile);

        if opts.idle_resume_us >= 0 {
            if !cpu_idle_resume_latency_supported() {
                warn!("idle resume latency not supported");
            } else {
                info!("Setting idle QoS to {} us", opts.idle_resume_us);
                for cpu in topo.all_cpus.values() {
                    update_cpu_idle_resume_latency(
                        cpu.id,
                        opts.idle_resume_us.try_into().unwrap(),
                    )?;
                }
            }
        }

        // Determine the amount of non-empty NUMA nodes in the system.
        let nr_nodes = topo
            .nodes
            .values()
            .filter(|node| !node.all_cpus.is_empty())
            .count();
        info!("NUMA nodes: {}", nr_nodes);

        // Automatically disable NUMA optimizations when running on non-NUMA systems.
        let numa_disabled = opts.disable_numa || nr_nodes == 1;
        if numa_disabled {
            info!("Disabling NUMA optimizations");
        }

        // Determine the primary scheduling domain.
        let power_profile = Self::power_profile();
        let domain =
            Self::resolve_energy_domain(&opts.primary_domain, power_profile).map_err(|err| {
                anyhow!(
                    "failed to resolve primary domain '{}': {}",
                    &opts.primary_domain,
                    err
                )
            })?;

        // Initialize BPF connector.
        let mut skel_builder = BpfSkelBuilder::default();
        skel_builder.obj_builder.debug(opts.verbose);
        let open_opts = opts.libbpf.clone().into_bpf_open_opts();
        let mut skel = scx_ops_open!(skel_builder, open_object, descent_ops, open_opts)?;

        skel.struct_ops.descent_ops_mut().exit_dump_len = opts.exit_dump_len;

        // Override default BPF scheduling parameters.
        let rodata = skel.maps.rodata_data.as_mut().unwrap();
        rodata.debug = opts.debug;
        rodata.smt_enabled = smt_enabled;
        rodata.numa_disabled = numa_disabled;
        rodata.rr_sched = opts.rr_sched;
        rodata.tickless_sched = opts.tickless;
        rodata.slice_max = opts.slice_us * 1000;
        rodata.slice_lag = opts.slice_us_lag * 1000;
        rodata.throttle_ns = opts.throttle_us * 1000;
        rodata.primary_all = domain.weight() == *NR_CPU_IDS;

        // Set scheduler flags.
        skel.struct_ops.descent_ops_mut().flags = *compat::SCX_OPS_ENQ_EXITING
            | *compat::SCX_OPS_ENQ_LAST
            | *compat::SCX_OPS_ENQ_MIGRATION_DISABLED
            | *compat::SCX_OPS_ALLOW_QUEUED_WAKEUP
            | if numa_disabled {
                0
            } else {
                *compat::SCX_OPS_BUILTIN_IDLE_PER_NODE
            };
        info!(
            "scheduler flags: {:#x}",
            skel.struct_ops.descent_ops_mut().flags
        );

        // Load the BPF program for validation.
        let mut skel = scx_ops_load!(skel, descent_ops, uei)?;

        // Initialize the primary scheduling domain and the preferred domain.
        Self::init_energy_domain(&mut skel, &domain).map_err(|err| {
            anyhow!(
                "failed to initialize primary domain 0x{:x}: {}",
                domain,
                err
            )
        })?;

        if let Err(err) = Self::init_cpufreq_perf(&mut skel, &opts.primary_domain, opts.cpufreq) {
            bail!(
                "failed to initialize cpufreq performance level: error {}",
                err
            );
        }

        // Initialize SMT domains.
        if smt_enabled {
            Self::init_smt_domains(&mut skel, &topo)?;
        }

        // Attach the scheduler.
        let struct_ops = Some(scx_ops_attach!(skel, descent_ops)?);
        let stats_server = StatsServer::new(stats::server_data()).launch()?;

        // Initialize Phase 2 components
        let optimizer = DescentOptimizer::new();
        let classifier = TaskClassifier::new();
        let safety = SafetyMonitor::new();
        let debugger = opts
            .debug_gradients
            .map(|interval_ms| GradientDebugger::new(interval_ms));

        // Initialize optimizer states for all CPUs and classes with default params
        let mut scheduler = Self {
            skel,
            struct_ops,
            opts,
            topo,
            power_profile,
            stats_server,
            user_restart: false,
            optimizer,
            _classifier: classifier,
            safety,
            debugger,
        };

        // Initialize optimizer states with default parameters
        scheduler.init_optimizer_states();

        Ok(scheduler)
    }

    fn init_optimizer_states(&mut self) {
        // Default initial parameters for each class
        let default_params: [[u64; 5]; 4] = [
            // INTERACTIVE: [latency_weight, base_slice_ns, vruntime_scale, preemption_priority, migration_cost]
            [1_000_000, 600_000, 768, 500, 30_000],
            // AUDIO
            [100_000, 500_000, 512, 100, 10_000],
            // BATCH
            [10_000_000, 5_000_000, 1536, 5_000_000, 100_000],
            // KERNEL
            [2_000_000, 1_000_000, 1024, 1_000_000, 20_000],
        ];

        // Load profile configuration for weights
        let profile_config = ProfileConfig::from_name(&self.opts.profile);
        let weights = profile_config.profile.loss_weights();

        for cpu in self.topo.all_cpus.keys() {
            for class in 0..4u32 {
                let initial_params = default_params[class as usize];
                self.optimizer.init_state_with_weights(
                    *cpu as u32,
                    class,
                    initial_params,
                    weights.clone(),
                );
            }
        }

        info!(
            "Initialized optimizer states for {} CPUs with profile '{}'",
            self.topo.all_cpus.len(),
            self.opts.profile
        );
    }

    fn enable_primary_cpu(skel: &mut BpfSkel<'_>, cpu: i32) -> Result<(), u32> {
        let prog = &mut skel.progs.enable_primary_cpu;
        let mut args = cpu_arg {
            cpu_id: cpu as c_int,
        };
        let input = ProgramInput {
            context_in: Some(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut args as *mut _ as *mut u8,
                    std::mem::size_of_val(&args),
                )
            }),
            ..Default::default()
        };
        let out = prog.test_run(input).unwrap();
        if out.return_value != 0 {
            return Err(out.return_value);
        }

        Ok(())
    }

    fn epp_to_cpumask(profile: Powermode) -> Result<Cpumask> {
        let mut cpus = get_primary_cpus(profile).unwrap_or_default();
        if cpus.is_empty() {
            cpus = get_primary_cpus(Powermode::Any).unwrap_or_default();
        }
        Cpumask::from_str(&cpus_to_cpumask(&cpus))
    }

    fn resolve_energy_domain(primary_domain: &str, power_profile: PowerProfile) -> Result<Cpumask> {
        let domain = match primary_domain {
            "powersave" => Self::epp_to_cpumask(Powermode::Powersave)?,
            "performance" => Self::epp_to_cpumask(Powermode::Performance)?,
            "turbo" => Self::epp_to_cpumask(Powermode::Turbo)?,
            "auto" => match power_profile {
                PowerProfile::Powersave => Self::epp_to_cpumask(Powermode::Powersave)?,
                PowerProfile::Balanced { power: true } => {
                    Self::epp_to_cpumask(Powermode::Powersave)?
                }
                PowerProfile::Balanced { power: false }
                | PowerProfile::Performance
                | PowerProfile::Unknown => Self::epp_to_cpumask(Powermode::Any)?,
            },
            "all" => Self::epp_to_cpumask(Powermode::Any)?,
            &_ => Cpumask::from_str(primary_domain)?,
        };

        Ok(domain)
    }

    fn init_energy_domain(skel: &mut BpfSkel<'_>, domain: &Cpumask) -> Result<()> {
        info!("primary CPU domain = 0x{:x}", domain);

        // Clear the primary domain by passing a negative CPU id.
        if let Err(err) = Self::enable_primary_cpu(skel, -1) {
            bail!("failed to reset primary domain: error {}", err);
        }

        // Update primary scheduling domain.
        for cpu in 0..*NR_CPU_IDS {
            if domain.test_cpu(cpu) {
                if let Err(err) = Self::enable_primary_cpu(skel, cpu as i32) {
                    bail!("failed to add CPU {} to primary domain: error {}", cpu, err);
                }
            }
        }

        Ok(())
    }

    // Update hint for the cpufreq governor.
    fn init_cpufreq_perf(
        skel: &mut BpfSkel<'_>,
        primary_domain: &String,
        auto: bool,
    ) -> Result<()> {
        // If we are using the powersave profile always scale the CPU frequency to the minimum,
        // otherwise use the maximum, unless automatic frequency scaling is enabled.
        let perf_lvl: i64 = match primary_domain.as_str() {
            "powersave" => 0,
            _ if auto => -1,
            _ => 1024,
        };
        info!(
            "cpufreq performance level: {}",
            match perf_lvl {
                1024 => "max".into(),
                0 => "min".into(),
                n if n < 0 => "auto".into(),
                _ => perf_lvl.to_string(),
            }
        );
        skel.maps.bss_data.as_mut().unwrap().cpufreq_perf_lvl = perf_lvl;

        Ok(())
    }

    fn power_profile() -> PowerProfile {
        let profile = fetch_power_profile(true);
        if profile == PowerProfile::Unknown {
            fetch_power_profile(false)
        } else {
            profile
        }
    }

    fn refresh_sched_domain(&mut self) -> bool {
        if self.power_profile != PowerProfile::Unknown {
            let power_profile = Self::power_profile();
            if power_profile != self.power_profile {
                self.power_profile = power_profile;

                if self.opts.primary_domain == "auto" {
                    return true;
                }
                if let Err(err) = Self::init_cpufreq_perf(
                    &mut self.skel,
                    &self.opts.primary_domain,
                    self.opts.cpufreq,
                ) {
                    warn!("failed to refresh cpufreq performance level: error {}", err);
                }
            }
        }

        false
    }

    fn enable_sibling_cpu(
        skel: &mut BpfSkel<'_>,
        lvl: usize,
        cpu: usize,
        sibling_cpu: usize,
    ) -> Result<(), u32> {
        let prog = &mut skel.progs.enable_sibling_cpu;
        let mut args = domain_arg {
            lvl_id: lvl as c_int,
            cpu_id: cpu as c_int,
            sibling_cpu_id: sibling_cpu as c_int,
        };
        let input = ProgramInput {
            context_in: Some(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut args as *mut _ as *mut u8,
                    std::mem::size_of_val(&args),
                )
            }),
            ..Default::default()
        };
        let out = prog.test_run(input).unwrap();
        if out.return_value != 0 {
            return Err(out.return_value);
        }

        Ok(())
    }

    fn init_smt_domains(skel: &mut BpfSkel<'_>, topo: &Topology) -> Result<(), std::io::Error> {
        let smt_siblings = topo.sibling_cpus();

        info!("SMT sibling CPUs: {:?}", smt_siblings);
        for (cpu, sibling_cpu) in smt_siblings.iter().enumerate() {
            Self::enable_sibling_cpu(skel, 0, cpu, *sibling_cpu as usize).unwrap();
        }

        Ok(())
    }

    fn get_metrics(&self) -> Metrics {
        let bss_data = self.skel.maps.bss_data.as_ref().unwrap();
        Metrics {
            nr_running: bss_data.nr_running,
            nr_cpus: bss_data.nr_online_cpus,
            nr_kthread_dispatches: bss_data.nr_kthread_dispatches,
            nr_direct_dispatches: bss_data.nr_direct_dispatches,
            nr_shared_dispatches: bss_data.nr_shared_dispatches,
            nr_tasks_interactive: 0, // TODO: read from BPF
            nr_tasks_audio: 0,       // TODO: read from BPF
            nr_tasks_batch: 0,       // TODO: read from BPF
            nr_tasks_kernel: 0,      // TODO: read from BPF
            gradient_updates: GRADIENT_UPDATES.load(Ordering::Relaxed),
            oscillations: OSCILLATIONS.load(Ordering::Relaxed),
        }
    }

    pub fn exited(&mut self) -> bool {
        uei_exited!(&self.skel, uei)
    }

    fn update_bpf_params(&mut self, cpu: i32, class: u32, params: [u64; 5]) {
        let prog = &mut self.skel.progs.update_class_params;

        let mut args = descent_params_update {
            cpu_id: cpu,
            class_id: class,
            latency_weight: params[0],
            base_slice_ns: params[1],
            vruntime_scale: params[2],
            preemption_priority: params[3],
            migration_cost: params[4],
        };

        let input = ProgramInput {
            context_in: Some(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut args as *mut _ as *mut u8,
                    std::mem::size_of_val(&args),
                )
            }),
            ..Default::default()
        };

        match prog.test_run(input) {
            Ok(out) => {
                if out.return_value != 0 {
                    warn!(
                        "Failed to update BPF params for CPU {} class {}: {}",
                        cpu, class, out.return_value
                    );
                } else {
                    debug!(
                        "Updated BPF params for CPU {} class {:?}: {:?}",
                        cpu, class, params
                    );
                }
            }
            Err(e) => {
                warn!("Error updating BPF params: {}", e);
            }
        }
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
        let (res_ch, req_ch) = self.stats_server.channels();

        // Create a channel for gradient events to avoid borrow issues
        let (gradient_tx, gradient_rx) = crossbeam::channel::unbounded::<GradientEvent>();

        // Setup ring buffer for gradient events
        let mut ringbuf_builder = RingBufferBuilder::new();
        let tx = gradient_tx.clone();
        ringbuf_builder.add(&self.skel.maps.gradient_events, move |data| {
            if data.len() == std::mem::size_of::<GradientEvent>() {
                let event: GradientEvent =
                    unsafe { std::ptr::read_unaligned(data.as_ptr() as *const GradientEvent) };
                let _ = tx.send(event);
            }
            0
        })?;

        let ringbuf = ringbuf_builder.build()?;

        // NEW: Phase 3 - Counter for checkpoint creation
        let mut update_counter: u64 = 0;

        // Main loop
        while !shutdown.load(Ordering::Relaxed) && !self.exited() {
            if self.refresh_sched_domain() {
                self.user_restart = true;
                break;
            }

            // Poll ring buffer with timeout
            match ringbuf.poll(Duration::from_millis(10)) {
                Ok(_) => {}
                Err(e) => {
                    warn!("Ring buffer poll error: {}", e);
                }
            }

            // Process any gradient events
            while let Ok(event) = gradient_rx.try_recv() {
                debug!(
                    "Received gradient event: CPU {} class {} param {} loss_p {} loss_m {}",
                    event.cpu_id,
                    event.class_id,
                    event.param_idx,
                    event.loss_plus,
                    event.loss_minus
                );

                if let Some(new_params) = self.optimizer.process_gradient_event(&event) {
                    self.update_bpf_params(event.cpu_id, event.class_id, new_params);
                    GRADIENT_UPDATES.fetch_add(1, Ordering::Relaxed);
                    update_counter += 1;
                }
            }

            // NEW: Phase 3 - Checkpoint every 10 updates per CPU/class
            if update_counter > 0 && update_counter % 10 == 0 {
                let cpus: Vec<_> = self.topo.all_cpus.keys().cloned().collect();
                for cpu in &cpus {
                    for class in 0..4u32 {
                        self.optimizer.create_checkpoint(*cpu as u32, class);
                    }
                }
                debug!("Created checkpoints at update {}", update_counter);
            }

            // NEW: Phase 3 - Check for degradation and rollback if needed
            /*
            let cpus: Vec<_> = self.topo.all_cpus.keys().cloned().collect();
            for cpu in &cpus {
                for class in 0..4u32 {
                    if let Some(state) = self.optimizer.get_state(*cpu as u32, class) {
                        if let Some(checkpoint_loss) =
                            self.optimizer.get_checkpoint_loss(*cpu as u32, class)
                        {
                            if let Some(&current_loss) = state.loss_history.back() {
                                // Rollback if loss increased > 20%
                                if current_loss > checkpoint_loss * 1.2 && checkpoint_loss > 0.0 {
                                    warn!(
                                        "Degradation detected on CPU {} class {}: current_loss={:.2} > 1.2 * checkpoint_loss={:.2}, rolling back",
                                        cpu, class, current_loss, checkpoint_loss
                                    );
                                    OSCILLATIONS.fetch_add(1, Ordering::Relaxed);
                                    if let Some(new_params) =
                                        self.optimizer.rollback(*cpu as u32, class)
                                    {
                                        self.update_bpf_params(*cpu as i32, class, new_params);
                                    }
                                }
                            }
                        }
                    }
                }
            }*/

            // Handle stats
            match req_ch.recv_timeout(Duration::from_millis(10)) {
                Ok(()) => res_ch.send(self.get_metrics())?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(e) => Err(e)?,
            }

            // Optional gradient debugging output
            if let Some(ref mut debugger) = self.debugger {
                debugger.maybe_output(&self.optimizer);
            }
        }

        let _ = self.struct_ops.take();
        uei_report!(&self.skel, uei)
    }
}

impl Drop for Scheduler<'_> {
    fn drop(&mut self) {
        info!("Unregister {SCHEDULER_NAME} scheduler");

        // Restore default CPU idle QoS resume latency.
        if self.opts.idle_resume_us >= 0 {
            if cpu_idle_resume_latency_supported() {
                for cpu in self.topo.all_cpus.values() {
                    update_cpu_idle_resume_latency(cpu.id, cpu.pm_qos_resume_latency_us as i32)
                        .unwrap();
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    if opts.version {
        println!(
            "{} {}",
            SCHEDULER_NAME,
            build_id::full_version(env!("CARGO_PKG_VERSION"))
        );
        return Ok(());
    }

    if opts.help_stats {
        stats::server_data().describe_meta(&mut std::io::stdout(), None)?;
        return Ok(());
    }

    let loglevel = simplelog::LevelFilter::Info;

    let mut lcfg = simplelog::ConfigBuilder::new();
    lcfg.set_time_offset_to_local()
        .expect("Failed to set local time offset")
        .set_time_level(simplelog::LevelFilter::Error)
        .set_location_level(simplelog::LevelFilter::Off)
        .set_target_level(simplelog::LevelFilter::Off)
        .set_thread_level(simplelog::LevelFilter::Off);
    simplelog::TermLogger::init(
        loglevel,
        lcfg.build(),
        simplelog::TerminalMode::Stderr,
        simplelog::ColorChoice::Auto,
    )?;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    if let Some(intv) = opts.monitor.or(opts.stats) {
        let shutdown_copy = shutdown.clone();
        let jh = std::thread::spawn(move || {
            match stats::monitor(Duration::from_secs_f64(intv), shutdown_copy) {
                Ok(_) => {
                    debug!("stats monitor thread finished successfully")
                }
                Err(error_object) => {
                    warn!(
                        "stats monitor thread finished because of an error {}",
                        error_object
                    )
                }
            }
        });
        if opts.monitor.is_some() {
            let _ = jh.join();
            return Ok(());
        }
    }

    let mut open_object = MaybeUninit::uninit();
    loop {
        let mut sched = Scheduler::init(&opts, &mut open_object)?;
        if !sched.run(shutdown.clone())?.should_restart() {
            if sched.user_restart {
                continue;
            }
            break;
        }
    }

    Ok(())
}
