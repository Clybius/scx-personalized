// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2024 scx_happy authors

use anyhow::{bail, Result};
use clap::Parser;
use log::{debug, info};
use scx_utils::{
    libbpf_clap_opts::LibbpfOpts, scx_ops_attach, scx_ops_load, scx_ops_open,
    try_set_rlimit_infinity, uei_exited, uei_report, CoreType, Cpumask, Topology,
};
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;

#[derive(Debug, clap::Parser)]
#[command(
    name = "scx_happy",
    about = "Happy scheduler with virtual nice and multi-queue"
)]
struct Opts {
    /// LC queue max slice (us)
    #[clap(long, default_value = "500")]
    lc_slice_us: u64,

    /// NORMAL queue max slice (us)
    #[clap(long, default_value = "1000")]
    normal_slice_us: u64,

    /// HOG queue max slice (us)
    #[clap(long, default_value = "3000")]
    hog_slice_us: u64,

    /// Disable SMT contention avoidance
    #[clap(long)]
    disable_smt_avoid: bool,

    /// Disable cache affinity
    #[clap(long)]
    disable_cache_affinity: bool,

    /// Disable cpufreq scaling
    #[clap(long)]
    disable_cpufreq: bool,

    /// LC domain (turbo/performance/powersave/CPUs)
    #[clap(long, default_value = "turbo")]
    lc_domain: String,

    /// NORMAL domain
    #[clap(long, default_value = "performance")]
    normal_domain: String,

    /// HOG domain
    #[clap(long, default_value = "powersave")]
    hog_domain: String,

    /// Antistall timeout (seconds)
    #[clap(long, default_value = "3")]
    antistall_sec: u64,

    /// TGID poll interval (ms)
    #[clap(long, default_value = "500")]
    tgid_poll_ms: u64,

    /// Disable antistall
    #[clap(long)]
    disable_antistall: bool,

    /// Verbose output
    #[clap(short, long)]
    verbose: bool,

    /// Print scheduler stats every N seconds
    #[clap(long, value_name = "SECONDS")]
    stats: Option<u64>,

    /// CPU usage threshold (%) to demote NORMAL tasks to HOG
    #[clap(long, value_name = "PERCENT", default_value = "50")]
    hog_cpu_threshold: u8,

    /// Disable dynamic virtual nice adjustment
    #[clap(long)]
    disable_dynamic_nice: bool,

    /// Dynamic adjustment interval (us)
    #[clap(long, value_name = "MICROSECONDS", default_value = "10000")]
    adjust_interval_us: u64,

    /// Interactive threshold (0-1000, higher = more strict)
    #[clap(long, value_name = "SCORE", default_value = "700")]
    interactive_threshold: u32,

    #[clap(flatten)]
    libbpf: LibbpfOpts,
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
        self.detect_scx_turbo();
        self.detect_steam_games();
        self.detect_de_tasks();
        self.detect_input_tasks();
        self.detect_audio_tasks();
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

    fn detect_scx_turbo(&mut self) {
        self.scx_turbo_tgids.clear();

        let entries = match fs::read_dir("/proc") {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let filename = entry.file_name();
            let name = match filename.to_str() {
                Some(n) => n,
                None => continue,
            };

            // Check if it's a PID directory
            let pid: u32 = match name.parse() {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Read environ file
            let environ_path = format!("/proc/{}/environ", pid);
            let mut content = String::new();

            if let Ok(mut file) = fs::File::open(&environ_path) {
                if file.read_to_string(&mut content).is_ok() {
                    // Check for SCX_TURBO=1
                    if content.contains("SCX_TURBO=1") {
                        self.scx_turbo_tgids.insert(pid);
                        debug!("Detected SCX_TURBO task: {}", pid);
                    }
                }
            }
        }
    }

    fn detect_steam_games(&mut self) {
        self.steam_tgids.clear();

        let entries = match fs::read_dir("/proc") {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let filename = entry.file_name();
            let name = match filename.to_str() {
                Some(n) => n,
                None => continue,
            };

            let pid: u32 = match name.parse() {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Read environ file for SteamGameId
            let environ_path = format!("/proc/{}/environ", pid);
            let mut content = String::new();

            if let Ok(mut file) = fs::File::open(&environ_path) {
                if file.read_to_string(&mut content).is_ok() {
                    if content.contains("SteamGameId=") || content.contains("STEAM_GAME=") {
                        self.steam_tgids.insert(pid);
                        debug!("Detected Steam game: {}", pid);
                    }
                }
            }

            // Also check comm for wine/game patterns
            let comm_path = format!("/proc/{}/comm", pid);
            if let Ok(comm) = fs::read_to_string(&comm_path) {
                let comm = comm.trim();
                if comm.contains("wine") || comm.contains("Game") || comm.contains("game") {
                    self.steam_tgids.insert(pid);
                    debug!("Detected game process by comm: {}", pid);
                }
            }
        }
    }

    fn detect_de_tasks(&mut self) {
        self.de_tgids.clear();

        let de_patterns = [
            "kwin",
            "mutter",
            "compiz",
            "compositor",
            "wayfire",
            "sway",
            "river",
            "dwl",
            "hyprland",
            "i3",
            "awesome",
        ];

        let entries = match fs::read_dir("/proc") {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let filename = entry.file_name();
            let name = match filename.to_str() {
                Some(n) => n,
                None => continue,
            };

            let pid: u32 = match name.parse() {
                Ok(p) => p,
                Err(_) => continue,
            };

            let comm_path = format!("/proc/{}/comm", pid);
            if let Ok(comm) = fs::read_to_string(&comm_path) {
                let comm = comm.trim().to_lowercase();
                for pattern in &de_patterns {
                    if comm.contains(pattern) {
                        self.de_tgids.insert(pid);
                        debug!("Detected DE component: {} ({})", pid, comm);
                        break;
                    }
                }
            }
        }
    }

    fn detect_input_tasks(&mut self) {
        self.input_tgids.clear();

        let input_patterns = ["input-", "evdev", "libinput", "keyboard", "mouse"];

        let entries = match fs::read_dir("/proc") {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let filename = entry.file_name();
            let name = match filename.to_str() {
                Some(n) => n,
                None => continue,
            };

            let pid: u32 = match name.parse() {
                Ok(p) => p,
                Err(_) => continue,
            };

            let comm_path = format!("/proc/{}/comm", pid);
            if let Ok(comm) = fs::read_to_string(&comm_path) {
                let comm = comm.trim().to_lowercase();
                for pattern in &input_patterns {
                    if comm.contains(pattern) {
                        self.input_tgids.insert(pid);
                        debug!("Detected input task: {} ({})", pid, comm);
                        break;
                    }
                }
            }
        }
    }

    fn detect_audio_tasks(&mut self) {
        self.audio_tgids.clear();

        let audio_patterns = ["pipewire", "pulseaudio", "jackd", "alsa", "snd"];

        let entries = match fs::read_dir("/proc") {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let filename = entry.file_name();
            let name = match filename.to_str() {
                Some(n) => n,
                None => continue,
            };

            let pid: u32 = match name.parse() {
                Ok(p) => p,
                Err(_) => continue,
            };

            let comm_path = format!("/proc/{}/comm", pid);
            if let Ok(comm) = fs::read_to_string(&comm_path) {
                let comm = comm.trim().to_lowercase();
                for pattern in &audio_patterns {
                    if comm.contains(pattern) {
                        self.audio_tgids.insert(pid);
                        debug!("Detected audio task: {} ({})", pid, comm);
                        break;
                    }
                }
            }
        }
    }

    fn print_stats(&self) {
        info!("Task classification stats:");
        info!("  SCX_TURBO tasks: {}", self.scx_turbo_tgids.len());
        info!("  Steam games: {}", self.steam_tgids.len());
        info!("  DE components: {}", self.de_tgids.len());
        info!("  Input tasks: {}", self.input_tgids.len());
        info!("  Audio tasks: {}", self.audio_tgids.len());
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
    let (lc, normal, hog, preemptions, migrations, antistall, smt, classified) =
        if let Some(bss) = skel.maps.bss_data.as_ref() {
            (
                bss.nr_lc_dispatches,
                bss.nr_normal_dispatches,
                bss.nr_hog_dispatches,
                bss.nr_preemptions,
                bss.nr_migrations,
                bss.nr_antistall_dispatches,
                bss.nr_smt_avoided,
                bss.nr_classified_tasks,
            )
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0)
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
    rodata.lc_slice_lag_ns = opts.lc_slice_us * 1000 * 20; // 20x slice
    rodata.normal_slice_lag_ns = opts.normal_slice_us * 1000 * 20;
    rodata.hog_slice_lag_ns = opts.hog_slice_us * 1000 * 20;
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
                // Something changed - print stats
                classifier.print_stats();
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
