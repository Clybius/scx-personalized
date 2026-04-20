// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2024 scx_happy authors

use anyhow::{bail, Result};
use clap::Parser;
use libbpf_rs::MapCore as _;
use log::{debug, info};
use scx_utils::{
    libbpf_clap_opts::LibbpfOpts, scx_ops_attach, scx_ops_load, scx_ops_open,
    try_set_rlimit_infinity, uei_exited, uei_report, CoreType, Cpumask, Topology,
};
use std::collections::HashSet;
use std::fs;

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;
mod stats;
pub use stats::HappyMetrics;

#[derive(Debug, clap::Parser)]
#[command(
    name = "scx_happy",
    about = "Happy scheduler with virtual nice and multi-queue"
)]
struct Opts {
    // === Slice Configuration ===
    /// LC (latency-critical) queue max slice in microseconds.
    ///
    /// Tasks in this queue get the shortest time slices and highest priority, suitable for
    /// interactive and real-time workloads.
    #[clap(long, default_value = "500")]
    lc_slice_us: u64,

    /// NORMAL queue max slice in microseconds.
    ///
    /// Standard tasks run in this queue with moderate time slices.
    #[clap(long, default_value = "1000")]
    normal_slice_us: u64,

    /// HOG queue max slice in microseconds.
    ///
    /// CPU-intensive background tasks are demoted to this queue with longer time slices to
    /// prevent them from interfering with interactive workloads.
    #[clap(long, default_value = "3000")]
    hog_slice_us: u64,

    // === Domain Assignment ===
    /// LC domain - CPU set for latency-critical tasks.
    ///
    /// Special values:
    ///   - "turbo" = use turbo/performance CPUs (highest capacity)
    ///   - "performance" = use performance CPUs (big cores)
    ///   - "powersave" = use power-efficient CPUs (little cores)
    ///   - Custom cpumask in hex (e.g., "0xff") for specific CPUs
    #[clap(long, default_value = "turbo")]
    lc_domain: String,

    /// NORMAL domain - CPU set for standard tasks.
    ///
    /// Special values:
    ///   - "turbo" = use turbo/performance CPUs (highest capacity)
    ///   - "performance" = use performance CPUs (big cores)
    ///   - "powersave" = use power-efficient CPUs (little cores)
    ///   - Custom cpumask in hex (e.g., "0xff") for specific CPUs
    #[clap(long, default_value = "performance")]
    normal_domain: String,

    /// HOG domain - CPU set for CPU-intensive background tasks.
    ///
    /// Special values:
    ///   - "turbo" = use turbo/performance CPUs (highest capacity)
    ///   - "performance" = use performance CPUs (big cores)
    ///   - "powersave" = use power-efficient CPUs (little cores)
    ///   - Custom cpumask in hex (e.g., "0xff") for specific CPUs
    #[clap(long, default_value = "powersave")]
    hog_domain: String,

    // === Feature Disablers ===
    /// Disable SMT contention avoidance.
    ///
    /// When enabled, the scheduler avoids placing tasks on sibling threads of busy cores
    /// to reduce SMT contention. Disabling this may increase throughput for CPU-bound
    /// workloads but can hurt latency-sensitive tasks due to resource contention.
    #[clap(long)]
    disable_smt_avoid: bool,

    /// Disable cache affinity optimization.
    ///
    /// Cache affinity tries to keep tasks on CPUs where they have established cache
    /// residency. Disabling this may cause more task migrations, potentially hurting
    /// performance for cache-sensitive workloads.
    #[clap(long)]
    disable_cache_affinity: bool,

    /// Disable CPU frequency scaling.
    ///
    /// When enabled, the scheduler can request frequency changes based on workload
    /// characteristics. Disabling this prevents frequency adjustments, using the system's
    /// default governor behavior instead.
    #[clap(long)]
    disable_cpufreq: bool,

    /// Disable antistall mechanism.
    ///
    /// Antistall periodically boosts stuck tasks to prevent system hangs. Disabling
    /// this may improve performance in some cases but risks task starvation if the
    /// scheduler logic encounters edge cases.
    #[clap(long)]
    disable_antistall: bool,

    /// Disable dynamic virtual nice adjustment.
    ///
    /// Dynamic nice automatically adjusts task priorities based on their behavior.
    /// Disabling this keeps priorities static, which may reduce scheduler overhead
    /// but could hurt responsiveness for mixed workloads.
    #[clap(long)]
    disable_dynamic_nice: bool,

    /// Disable deadline-based preemption (EEVDF-style).
    ///
    /// Deadline preemption uses earliest-eligible-virtual-deadline-first scheduling
    /// for fair and responsive task ordering. Disabling this falls back to simpler
    /// priority-based scheduling, which may be less fair for bursty workloads.
    #[clap(long)]
    disable_deadline_preemption: bool,

    /// Disable HOG lag decay mechanism.
    ///
    /// Lag decay gradually reduces accumulated lag for sleeping HOG tasks, allowing
    /// them to eventually return to NORMAL queue. Disabling this means HOG tasks
    /// stay in HOG queue indefinitely, potentially improving batch throughput but
    /// hurting interactive response after long sleeps.
    #[clap(long)]
    disable_hog_lag_decay: bool,

    // === Thresholds ===
    /// CPU usage threshold (0-100) to demote NORMAL tasks to HOG queue.
    ///
    /// Tasks using more than this percentage of CPU are classified as HOG tasks
    /// and moved to the HOG queue for lower-priority scheduling.
    #[clap(long, value_name = "PERCENT", default_value = "50")]
    hog_cpu_threshold: u8,

    /// Interactive task threshold (0-1000, higher = more strict).
    ///
    /// Score threshold for classifying tasks as interactive. Higher values require
    /// tasks to show more interactive behavior (short bursts, frequent sleeps) to
    /// qualify for LC queue. Lower values are more permissive.
    #[clap(long, value_name = "SCORE", default_value = "700")]
    interactive_threshold: u32,

    /// Deadline preemption hysteresis threshold (0-100 percent of vslice).
    ///
    /// Hysteresis prevents excessive preemption by requiring a task's deadline
    /// advantage to exceed this percentage before preempting. Higher values reduce
    /// preemption frequency, improving throughput but potentially hurting latency.
    #[clap(long, default_value = "10")]
    preemption_hysteresis_pct: u8,

    // === Timing/Intervals ===
    /// Antistall timeout in seconds.
    ///
    /// Maximum time a task can wait before being forcefully dispatched to prevent
    /// system stalls.
    #[clap(long, default_value = "3")]
    antistall_sec: u64,

    /// TGID poll interval in milliseconds.
    ///
    /// How often to scan for and classify tasks based on their TGID and environment.
    #[clap(long, default_value = "500")]
    tgid_poll_ms: u64,

    /// Dynamic adjustment interval in microseconds.
    ///
    /// How often the scheduler re-evaluates task priorities and queue assignments
    /// for dynamic nice adjustments.
    #[clap(long, value_name = "MICROSECONDS", default_value = "10000")]
    adjust_interval_us: u64,

    /// HOG decay interval in microseconds (sleep time before decay).
    ///
    /// How long a HOG task must sleep before its accumulated lag begins to decay,
    /// making it eligible for promotion back to NORMAL queue.
    #[clap(long, default_value = "20000")]
    hog_decay_interval_us: u64,

    // === HOG Promotion ===
    /// Minimum total sleep duration in microseconds for HOG promotion.
    ///
    /// HOG tasks must accumulate at least this much sleep time across multiple
    /// sleep cycles before being considered for promotion to NORMAL queue.
    #[clap(long, default_value = "50000")]
    hog_min_sleep_duration_us: u64,

    /// Minimum sleep cycles before HOG promotion.
    ///
    /// Number of times a HOG task must go to sleep (at least) before being
    /// eligible for promotion back to NORMAL queue.
    #[clap(long, default_value = "3")]
    hog_min_sleep_count: u32,

    // === Latency Criticality Configuration ===
    /// Disable latency criticality calculation.
    ///
    /// By default, scx_happy calculates a normalized latency criticality score [0-1024]
    /// based on task behavior (wait frequency, wake frequency, runtime) and uses
    /// it to derive virtual nice values alongside the interactive score.
    /// Use this flag to disable the lat_cri mechanism entirely.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    disable_lat_cri: bool,

    /// Weight percentage for lat_cri vs interactive score (0-100).
    ///
    /// Higher values prioritize latency criticality over interactive score.
    /// For example, 60 means 60% lat_cri weight + 40% interactive score weight.
    #[clap(long, default_value = "60")]
    lat_cri_weight_pct: u32,

    /// Disable waker/wakee latency criticality inheritance.
    ///
    /// By default, tasks inherit latency criticality from their waker and
    /// pass it to tasks they wake. This helps propagate criticality through
    /// producer-consumer chains. Use this flag to disable inheritance.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    disable_lat_cri_inheritance: bool,

    // === Output/Monitoring ===
    /// Enable verbose output, including libbpf details.
    #[clap(short = 'v', long, action = clap::ArgAction::SetTrue)]
    verbose: bool,

    /// Print scheduler stats every N seconds.
    #[clap(long, value_name = "SECONDS")]
    stats: Option<u64>,

    // === Libbpf Options ===
    #[clap(flatten, next_help_heading = "Libbpf Options")]
    pub libbpf: LibbpfOpts,
}

struct TaskClassifier {
    scx_turbo_tgids: HashSet<u32>,
    steam_tgids: HashSet<u32>,
    de_tgids: HashSet<u32>,
    input_tgids: HashSet<u32>,
    audio_tgids: HashSet<u32>,
}

impl TaskClassifier {
    fn new() -> Self {
        Self {
            scx_turbo_tgids: HashSet::new(),
            steam_tgids: HashSet::new(),
            de_tgids: HashSet::new(),
            input_tgids: HashSet::new(),
            audio_tgids: HashSet::new(),
        }
    }

    fn detect_all(&mut self) {
        self.scx_turbo_tgids.clear();
        self.steam_tgids.clear();
        self.de_tgids.clear();
        self.input_tgids.clear();
        self.audio_tgids.clear();

        let Ok(entries) = fs::read_dir("/proc") else { return };

        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else { continue };
            let Ok(pid) = name.parse::<u32>() else { continue };

            // Fast path: read comm (small file)
            let Ok(comm) = fs::read_to_string(format!("/proc/{}/comm", pid)) else { continue };
            let comm = comm.trim().to_lowercase();

            // Classify by comm patterns - DE
            let de_patterns = ["kwin", "mutter", "compiz", "compositor", "wayfire",
                              "sway", "river", "dwl", "hyprland", "i3", "awesome"];
            for p in de_patterns {
                if comm.contains(p) {
                    self.de_tgids.insert(pid);
                }
            }

            // Classify by comm patterns - Input
            let input_patterns = [
                "fcitx", "fcitx5",
                "ibus", "ibus-daemon", "ibus-engine",
                "ksoftirqd",
                "input-",
                "evdev",
            ];
            for p in input_patterns {
                if comm.contains(p) {
                    self.input_tgids.insert(pid);
                }
            }

            // Classify by comm patterns - Audio
            let audio_patterns = [
                "pipewire", "pipewire-pulse", "wireplumber",
                "pulseaudio", "jackd", "pw-", "pw_",
                "speech-dispatcher", "canberra",
            ];
            for p in audio_patterns {
                if comm.contains(p) {
                    self.audio_tgids.insert(pid);
                }
            }

            // Always check environ for SCX_TURBO and Steam detection
            if let Ok(environ) = fs::read_to_string(format!("/proc/{}/environ", pid)) {
                // Check for SCX_TURBO with any truthy value (non-empty, non-"0")
                if let Some(start) = environ.find("SCX_TURBO=") {
                    let value_start = start + "SCX_TURBO=".len();
                    let remainder = &environ[value_start..];
                    let value_end = remainder.find('\0').unwrap_or(remainder.len());
                    let value = &remainder[..value_end];
                    if !value.is_empty() && value != "0" {
                        self.scx_turbo_tgids.insert(pid);
                    }
                }
                // Steam game detection
                if environ.contains("SteamGameId=") || environ.contains("STEAM_GAME=") {
                    self.steam_tgids.insert(pid);
                }
            }
        }
    }

    /// Detect all tasks and return true if any changes were detected
    fn detect_all_with_changes(&mut self) -> bool {
        // Store old state before clearing
        let old_scx_turbo = self.scx_turbo_tgids.clone();
        let old_steam = self.steam_tgids.clone();
        let old_de = self.de_tgids.clone();
        let old_input = self.input_tgids.clone();
        let old_audio = self.audio_tgids.clone();

        // Run detection (this clears and rebuilds all sets)
        self.detect_all();

        // Check if anything changed
        let changed = old_scx_turbo != self.scx_turbo_tgids
            || old_steam != self.steam_tgids
            || old_de != self.de_tgids
            || old_input != self.input_tgids
            || old_audio != self.audio_tgids;

        if changed {
            debug!("Task classification changes detected");
        }

        changed
    }

    fn print_stats(&self) {
        info!("Task classification stats:");
        info!("  SCX_TURBO tasks: {}", self.scx_turbo_tgids.len());
        info!("  Steam games: {}", self.steam_tgids.len());
        info!("  DE components: {}", self.de_tgids.len());
        info!("  Input tasks: {}", self.input_tgids.len());
        info!("  Audio tasks: {}", self.audio_tgids.len());
    }

    /// Write classified TGIDs to BPF maps
    fn update_bpf_maps(&self, skel: &mut BpfSkel) -> Result<()> {
        use libbpf_rs::MapFlags;

        // Update all classification maps
        for &tgid in &self.input_tgids {
            let val: u8 = 1;
            skel.maps
                .input_tgids
                .update(&tgid.to_ne_bytes(), &val.to_ne_bytes(), MapFlags::ANY)?;
        }

        for &tgid in &self.steam_tgids {
            let val: u8 = 1;
            skel.maps
                .steam_tgids
                .update(&tgid.to_ne_bytes(), &val.to_ne_bytes(), MapFlags::ANY)?;
        }

        for &tgid in &self.de_tgids {
            let val: u8 = 1;
            skel.maps
                .de_tgids
                .update(&tgid.to_ne_bytes(), &val.to_ne_bytes(), MapFlags::ANY)?;
        }

        for &tgid in &self.audio_tgids {
            let val: u8 = 1;
            skel.maps
                .audio_tgids
                .update(&tgid.to_ne_bytes(), &val.to_ne_bytes(), MapFlags::ANY)?;
        }

        for &tgid in &self.scx_turbo_tgids {
            let val: u8 = 1;
            skel.maps.scx_turbo_tgids.update(
                &tgid.to_ne_bytes(),
                &val.to_ne_bytes(),
                MapFlags::ANY,
            )?;
        }

        debug!(
            "Updated BPF maps: {} input, {} steam, {} de, {} audio, {} turbo",
            self.input_tgids.len(),
            self.steam_tgids.len(),
            self.de_tgids.len(),
            self.audio_tgids.len(),
            self.scx_turbo_tgids.len()
        );

        Ok(())
    }
}

fn setup_signal_handler() -> Arc<AtomicBool> {
    let should_exit = Arc::new(AtomicBool::new(false));
    let should_exit_clone = should_exit.clone();

    ctrlc::set_handler(move || {
        info!("Received Ctrl+C, shutting down...");
        should_exit_clone.store(true, Ordering::Relaxed);
    })
    .expect("Failed to set signal handler");

    should_exit
}

fn set_domain_cpu(skel: &mut BpfSkel, cpu: i32, queue: u32) -> Result<()> {
    use libbpf_rs::ProgramInput;
    use std::os::raw::c_int;

    // Domain cpu arg struct matching BPF
    #[repr(C)]
    struct DomainCpuArg {
        cpu_id: c_int,
        queue: u32,
    }

    let prog = &mut skel.progs.happy_set_domain_cpu;
    let mut args = DomainCpuArg {
        cpu_id: cpu as c_int,
        queue,
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

    let out = prog.test_run(input)?;
    if out.return_value != 0 {
        return Err(anyhow::anyhow!(
            "happy_set_domain_cpu failed with return value {}",
            out.return_value
        ));
    }

    Ok(())
}

/// Print scheduler statistics
fn print_scheduler_stats(skel: &BpfSkel, classifier: &TaskClassifier) {
    info!("=== Scheduler Stats ===");

    // Read BPF stats from BSS section
    let (
        lc,
        normal,
        hog,
        preemptions,
        migrations,
        antistall,
        smt,
        classified,
        deadline_preemptions,
        queue_priority_preemptions,
        same_queue_preemptions,
        hog_sleep_decayed,
        hog_promotion_checks,
        // Latency criticality stats
        lat_cri_calculations,
        high_lat_cri_tasks,
        lat_cri_inherited,
    ) = if let Some(bss) = skel.maps.bss_data.as_ref() {
        (
            bss.nr_lc_dispatches,
            bss.nr_normal_dispatches,
            bss.nr_hog_dispatches,
            bss.nr_preemptions,
            bss.nr_migrations,
            bss.nr_antistall_dispatches,
            bss.nr_smt_avoided,
            bss.nr_classified_tasks,
            bss.nr_deadline_preemptions,
            bss.nr_queue_priority_preemptions,
            bss.nr_same_queue_preemptions,
            bss.nr_hog_sleep_decayed,
            bss.nr_hog_promotion_checks,
            // Latency criticality
            bss.nr_lat_cri_calculations,
            bss.nr_high_lat_cri_tasks,
            bss.nr_lat_cri_inherited,
        )
    } else {
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    };

    let total_dispatches = lc + normal + hog;

    info!(
        "Dispatches: LC={}, NORMAL={}, HOG={} (total={})",
        lc, normal, hog, total_dispatches
    );
    info!(
        "Events: Preemptions={}, Migrations={}, SMT avoided={}, Antistall={}",
        preemptions, migrations, smt, antistall
    );
    info!("Classified tasks tracked: {}", classified);
    info!(
        "Active classified tasks: SCX_TURBO={}, Steam={}, DE={}, Input={}, Audio={}",
        classifier.scx_turbo_tgids.len(),
        classifier.steam_tgids.len(),
        classifier.de_tgids.len(),
        classifier.input_tgids.len(),
        classifier.audio_tgids.len()
    );
    info!(
        "Deadline Preemptions: Total={}, Queue Priority={}, Same Queue={}",
        deadline_preemptions, queue_priority_preemptions, same_queue_preemptions
    );
    info!(
        "HOG Lag Decay: sleep_decayed={}, promotion_checks={}",
        hog_sleep_decayed, hog_promotion_checks
    );
    info!(
        "Latency Criticality: calculations={}, high_lat_cri={}, inherited={}",
        lat_cri_calculations, high_lat_cri_tasks, lat_cri_inherited
    );
}

fn init_domain(skel: &mut BpfSkel, domain: &str, topo: &Topology, queue: u32) -> Result<()> {
    let mut cpumask = Cpumask::new();

    match domain {
        "turbo" => {
            for (_, cpu) in topo.all_cpus.iter() {
                if let CoreType::Big { turbo: true } = cpu.core_type {
                    let _ = cpumask.set_cpu(cpu.id);
                }
            }
        }
        "performance" => {
            for (_, cpu) in topo.all_cpus.iter() {
                if let CoreType::Big { turbo: false } = cpu.core_type {
                    let _ = cpumask.set_cpu(cpu.id);
                }
            }
        }
        "powersave" => {
            for (_, cpu) in topo.all_cpus.iter() {
                if cpu.core_type == CoreType::Little {
                    let _ = cpumask.set_cpu(cpu.id);
                }
            }
        }
        _ => {
            // Default: all CPUs
            for (_, cpu) in topo.all_cpus.iter() {
                let _ = cpumask.set_cpu(cpu.id);
            }
        }
    }

    info!(
        "Initializing domain {:?} with: {}",
        domain,
        cpumask.to_string()
    );

    // Clear the domain first (negative CPU means reset)
    if let Err(e) = set_domain_cpu(skel, -1, queue) {
        bail!("failed to reset domain: {}", e);
    }

    // Add each CPU to the domain
    for cpu in 0..*scx_utils::NR_CPU_IDS {
        if cpumask.test_cpu(cpu) {
            if let Err(e) = set_domain_cpu(skel, cpu as i32, queue) {
                bail!("failed to add CPU {} to domain: {}", cpu, e);
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let opts = Opts::parse();

    if opts.verbose {
        info!("scx_happy scheduler starting with verbose output");
        info!("Options: {:?}", opts);
    }

    // Increase rlimit for BPF
    try_set_rlimit_infinity();

    // Load topology
    let topo = Topology::new()?;
    info!("Loaded topology: {} CPUs", topo.all_cpus.len());

    // Setup signal handler
    let should_exit = setup_signal_handler();

    // Build and load BPF skeleton
    let mut open_object = MaybeUninit::uninit();
    let open_opts = opts.libbpf.clone().into_bpf_open_opts();

    let mut skel_builder = BpfSkelBuilder::default();
    skel_builder.obj_builder.debug(opts.verbose);

    let mut open_skel = scx_ops_open!(skel_builder, &mut open_object, happy_ops, open_opts)?;

    // Configure BPF parameters
    let rodata = open_skel.maps.rodata_data.as_mut().unwrap();
    rodata.lc_slice_ns = opts.lc_slice_us * 1000;
    rodata.normal_slice_ns = opts.normal_slice_us * 1000;
    rodata.hog_slice_ns = opts.hog_slice_us * 1000;
    rodata.avoid_smt = !opts.disable_smt_avoid;
    rodata.cache_affinity = !opts.disable_cache_affinity;
    rodata.cpufreq_enabled = !opts.disable_cpufreq;
    rodata.antistall_enabled = !opts.disable_antistall;
    rodata.antistall_sec = opts.antistall_sec;
    rodata.debug = opts.verbose as u32;
    rodata.hog_cpu_threshold = opts.hog_cpu_threshold;
    rodata.dynamic_nice_enabled = !opts.disable_dynamic_nice;
    rodata.adjust_interval_ns = opts.adjust_interval_us * 1000;
    rodata.interactive_threshold = opts.interactive_threshold;
    rodata.deadline_preemption_enabled = !opts.disable_deadline_preemption;
    rodata.preemption_hysteresis_pct = opts.preemption_hysteresis_pct;
    // NEW: HOG lag decay configuration
    rodata.hog_lag_decay_enabled = !opts.disable_hog_lag_decay;
    rodata.hog_decay_interval_ns = opts.hog_decay_interval_us * 1000;
    rodata.hog_min_sleep_duration_ns = opts.hog_min_sleep_duration_us * 1000;
    rodata.hog_min_sleep_count = opts.hog_min_sleep_count;
    // ========== Latency Criticality Configuration ==========
    rodata.lat_cri_enabled = !opts.disable_lat_cri;
    rodata.lat_cri_weight_pct = opts.lat_cri_weight_pct;
    rodata.lat_cri_inheritance = !opts.disable_lat_cri_inheritance;

    // Load the skeleton
    let mut skel = scx_ops_load!(open_skel, happy_ops, uei)?;

    // Initialize domain cpumasks AFTER loading
    // Queue IDs: LC=0, NORMAL=1, HOG=2
    init_domain(&mut skel, &opts.lc_domain, &topo, 0)?;
    init_domain(&mut skel, &opts.normal_domain, &topo, 1)?;
    init_domain(&mut skel, &opts.hog_domain, &topo, 2)?;

    // Attach the scheduler
    let _link = scx_ops_attach!(skel, happy_ops)?;

    info!("scx_happy scheduler attached successfully!");

    // Initialize task classifier
    let mut classifier = TaskClassifier::new();
    let poll_interval = Duration::from_millis(opts.tgid_poll_ms);
    let mut last_poll = Instant::now();

    // Stats interval setup
    let stats_interval = opts.stats.map(Duration::from_secs);
    let mut last_stats = Instant::now();

    // Run initial classification
    classifier.detect_all();
    classifier.print_stats();

    // Update BPF maps with detected TGIDs
    if let Err(e) = classifier.update_bpf_maps(&mut skel) {
        debug!("Failed to update BPF maps: {}", e);
    }

    // Print initial scheduler stats if --stats is enabled
    if stats_interval.is_some() {
        print_scheduler_stats(&skel, &classifier);
    }

    // Main loop
    while !should_exit.load(Ordering::Relaxed) {
        // Periodic task classification - only print on changes
        if last_poll.elapsed() >= poll_interval {
            debug!("Running task classification...");

            if classifier.detect_all_with_changes() {
                // Something changed - print stats and update BPF maps
                classifier.print_stats();
                if let Err(e) = classifier.update_bpf_maps(&mut skel) {
                    debug!("Failed to update BPF maps: {}", e);
                }
            }

            last_poll = Instant::now();
        }

        // Periodic scheduler stats printing
        if let Some(interval) = stats_interval {
            if last_stats.elapsed() >= interval {
                print_scheduler_stats(&skel, &classifier);
                last_stats = Instant::now();
            }
        }

        // Check for scheduler exit
        if uei_exited!(&skel, uei) {
            uei_report!(&skel, uei)?;
            break;
        }

        thread::sleep(Duration::from_millis(50));
    }

    info!("scx_happy scheduler shutting down");
    Ok(())
}
