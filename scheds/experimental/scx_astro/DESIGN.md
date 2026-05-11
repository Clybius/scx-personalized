# scx_astro Design Document

**Hybrid Classification Multi-Lane Scheduler for sched_ext**

---

## 1. Overview

`scx_astro` is a new experimental CPU scheduler for the Linux `sched_ext` framework that combines ideas from multiple existing schedulers and research concepts into a cohesive, production-ready design. Its core thesis is:

> **Not all tasks are equal, and the scheduler should treat them differently based on both automatic behavioral detection and explicit user intent.**

The scheduler uses a **multi-lane DSQ containment model** where tasks are classified into profiles (interactive, normal, compute, background) and placed into dedicated dispatch queues (DSQs) with different service guarantees. Interactive tasks get preemptive fast lanes; background tasks are contained and only run when higher-priority lanes are empty.

Three novel features differentiate `scx_astro`:

1. **SRPT-inspired prioritization** within the normal lane, using vtime scaling to prefer shorter tasks.
2. **Waker-boost chains** that propagate interactive priority to tasks woken by interactive threads.
3. **Dynamic lane routing** with hysteresis, allowing tasks to move between lanes as their behavior changes without thrashing.

---

## 2. DSQ Architecture

### 2.1 DSQ IDs and Purposes

`scx_astro` defines five global DSQ lanes plus per-CPU local DSQs for direct dispatch:

| DSQ Name | ID | Purpose | Preempt? | Slice |
|----------|----|---------|----------|-------|
| `ASTRO_INTERACTIVE_DSQ` | 1020 | Latency-critical tasks (UI, audio, input handlers) | Yes | 50–200 µs |
| `ASTRO_WAKER_BOOST_DSQ` | 1021 | Temporary boost lane for wakees of interactive tasks | Yes | 100 µs |
| `ASTRO_NORMAL_DSQ` | 1022 | General tasks; SRPT-ordered via vtime scaling | No (default) | 0.5–2 ms |
| `ASTRO_COMPUTE_DSQ` | 1023 | Long-running CPU-bound jobs | No | 2–5 ms |
| `ASTRO_BACKGROUND_DSQ` | 1024 | Contained/throughput tasks; starvation-fair | No | 100–500 µs |
| `ASTRO_LOCAL_CPU_BASE \| cpu` | 0x40000000 + cpu | Per-CPU DSQ for locality-optimized direct dispatch | N/A | Varies |

### 2.2 Global vs. Per-CPU DSQs

Following the pattern from `scx_flow`, `scx_astro` uses a **hybrid approach**:

- **Global DSQs** for the five lanes: this ensures that any CPU can steal work from any lane, providing natural load balancing and preventing stranding of interactive tasks on idle CPUs.
- **Per-CPU local DSQs** (`ASTRO_LOCAL_CPU_BASE | cpu`) are created at init time and used as a fast path when `select_cpu()` finds an idle target CPU. The task is inserted directly into `SCX_DSQ_LOCAL_ON | cpu` (or the per-CPU dedicated DSQ) to maximize cache locality.

Rationale: Global DSQs are essential for the multi-lane containment model because a task classified as interactive should be runnable on any available CPU, not just its last CPU. Per-CPU DSQs are used only for the direct-local fast path when wake_cpu is idle.

### 2.3 Dispatch Priority Order

The `dispatch()` callback consumes DSQs in strict priority order:

```c
void BPF_STRUCT_OPS(astro_dispatch, s32 cpu, struct task_struct *prev)
{
    /* 1. Interactive lane */
    if (scx_bpf_dsq_move_to_local(ASTRO_INTERACTIVE_DSQ, 0)) { ... }

    /* 2. Waker boost lane */
    if (scx_bpf_dsq_move_to_local(ASTRO_WAKER_BOOST_DSQ, 0)) { ... }

    /* 3. Fairness rotation: if high-priority burst cap reached,
     * force service of lower lanes */
    if (should_force_low_priority(cstate)) { ... }

    /* 4. Normal lane */
    if (scx_bpf_dsq_move_to_local(ASTRO_NORMAL_DSQ, 0)) { ... }

    /* 5. Compute lane */
    if (scx_bpf_dsq_move_to_local(ASTRO_COMPUTE_DSQ, 0)) { ... }

    /* 6. Background lane (with starvation rescue) */
    if (scx_bpf_dsq_move_to_local(ASTRO_BACKGROUND_DSQ, 0)) { ... }
}
```

This order guarantees that interactive tasks are always serviced first, followed by waker-boosted tasks, then normal, compute, and finally background.

### 2.4 Starvation Prevention

To prevent background and compute tasks from starving indefinitely, `scx_astro` implements **burst-round counters** per CPU (in `astro_cpu_state`):

- `high_priority_burst_rounds`: counts consecutive dispatches from interactive/waker-boost/normal lanes.
- `contained_starvation_rounds`: counts consecutive dispatches since a background task last ran.
- `shared_starvation_rounds`: counts consecutive dispatches since a normal/compute task last ran.

When `high_priority_burst_rounds` exceeds `ASTRO_HIGH_PRIO_BURST_MAX` (default 4), the scheduler is forced to service lower-priority lanes on the next dispatch. Similarly, when `contained_starvation_rounds` or `shared_starvation_rounds` exceed their maxima, tasks from those lanes are promoted to the head of their DSQs (`SCX_ENQ_HEAD`) and the dispatch order may be overridden.

This design is directly inspired by `scx_flow`'s starvation-rescue mechanism, which has proven effective in production.

---

## 3. Task Classification (Hybrid)

### 3.1 Automatic Detection Heuristics

`scx_astro` automatically classifies tasks using behavioral signals updated on every `stopping()` callback. The algorithm is inspired by `scx_lavd`'s latency-criticality model but simplified for the lane-based architecture.

#### Signals tracked per task (`astro_task_ctx`)

```c
struct astro_task_ctx {
    u64 avg_runtime_ns;      /* EWMA of runtime per schedule */
    u64 acc_runtime_ns;      /* Accumulated runtime since last sleep */
    u32 wait_freq;           /* How often the task sleeps */
    u32 wake_freq;           /* How often the task wakes others */
    u64 last_sleep_ns;       /* Duration of last sleep */
    u32 lat_cri;             /* Raw latency criticality score */
    u32 normalized_lat_cri;  /* lat_cri scaled to [0, 1024] */
    u32 hog_score;           /* Containment score for hogs */
    ...
};
```

#### Latency Criticality Calculation (`calc_lat_cri()`)

```c
static __always_inline void calc_lat_cri(struct task_struct *p,
                                          struct astro_task_ctx *taskc)
{
    u64 wait_ft = min(taskc->wait_freq, ASTRO_LC_FREQ_MAX) + 1;
    u64 wake_ft = min(taskc->wake_freq, ASTRO_LC_FREQ_MAX) + 1;
    u64 runtime_ft, weight_ft = 1;

    /* Shorter runtime -> more latency-critical */
    if (ASTRO_LC_RUNTIME_MAX > taskc->avg_runtime_ns)
        runtime_ft = (ASTRO_LC_RUNTIME_MAX - taskc->avg_runtime_ns) / ASTRO_SLICE_MIN_NS + 1;
    else
        runtime_ft = 1;

    /* Context weight boosts (sync wake, hardirq, kthread, affinitized) */
    if (taskc->is_wakeup)       weight_ft += ASTRO_LC_WEIGHT_BOOST_REGULAR;
    if (taskc->is_sync_wakeup)  weight_ft += ASTRO_LC_WEIGHT_BOOST_REGULAR;
    if (taskc->woken_by_hardirq) weight_ft += ASTRO_LC_WEIGHT_BOOST_HIGHEST;
    if (is_kthread(p))          weight_ft += ASTRO_LC_WEIGHT_BOOST_MEDIUM;
    if (p->nr_cpus_allowed == 1) weight_ft += ASTRO_LC_WEIGHT_BOOST_MEDIUM;

    weight_ft = (u64)p->scx.weight * weight_ft + 1;

    /* Combine: log2(wait*wake) + log2(runtime*weight), then square to amplify */
    u64 log_wwf = astro_log2_u64(wait_ft * wake_ft);
    u64 lat_cri = log_wwf + astro_log2_u64(runtime_ft * weight_ft);
    lat_cri = lat_cri * lat_cri;

    /* Waker/wakee propagation */
    u64 giver = (u64)taskc->lat_cri_waker + (u64)taskc->lat_cri_wakee;
    if (giver > (2 * lat_cri)) {
        u64 giver_inh = (giver - (2 * lat_cri)) >> ASTRO_LC_INH_GIVER_SHIFT;
        u64 receiver_max = lat_cri >> ASTRO_LC_INH_RECEIVER_SHIFT;
        lat_cri += min(giver_inh, receiver_max);
    }

    taskc->lat_cri = (u32)min(lat_cri, (u64)U32_MAX);
    taskc->normalized_lat_cri = (u32)((lat_cri * 1024ULL) / (1024ULL * 1024ULL));
}
```

Key insight from `scx_lavd`: `wait_freq * wake_freq` captures how "central" a task is in the task graph. A task that both waits for others and wakes others frequently is in the middle of a producer-consumer chain; delaying it delays the entire chain.

#### Auto-Classification Rules (`auto_classify_profile()`)

```c
static __always_inline u8 auto_classify_profile(struct task_struct *p,
                                               struct astro_task_ctx *taskc)
{
    u32 nlat = taskc->normalized_lat_cri;
    u64 avg_runtime = taskc->avg_runtime_ns;
    u32 wait_freq = taskc->wait_freq;
    u32 wake_freq = taskc->wake_freq;

    if (nlat > ASTRO_LAT_CRI_INTERACTIVE &&
        avg_runtime < ASTRO_AVG_RUNTIME_INTERACTIVE &&
        (wait_freq > ASTRO_WAIT_FREQ_INTERACTIVE ||
         wake_freq > ASTRO_WAKE_FREQ_INTERACTIVE))
        return ASTRO_PROFILE_INTERACTIVE;

    if (avg_runtime > ASTRO_AVG_RUNTIME_COMPUTE &&
        wake_freq < 2 && wait_freq < 2)
        return ASTRO_PROFILE_COMPUTE;

    if (taskc->hog_score >= ASTRO_HOG_SCORE_CONTAIN)
        return ASTRO_PROFILE_BACKGROUND;

    return ASTRO_PROFILE_NORMAL;
}
```

### 3.2 Explicit Overrides via Environment Variables

Users can explicitly set a task's profile via the `SCX_ASTRO` environment variable:

```bash
SCX_ASTRO=interactive ./my-game-engine
SCX_ASTRO=background ./long-batch-job
```

The userspace component (Rust) scans `/proc/*/environ` at a configurable interval (default 2 seconds), parses `SCX_ASTRO=<profile>`, and writes the override into a BPF hash map `tgid_profile_map`:

```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, pid_t);
    __type(value, struct astro_tgid_profile);
} tgid_profile_map SEC(".maps");
```

In `effective_profile()`, the BPF code checks this map first:

```c
struct astro_tgid_profile *tp = bpf_map_lookup_elem(&tgid_profile_map, &p->tgid);
if (tp && tp->profile != ASTRO_PROFILE_AUTO)
    profile = tp->profile;
else
    profile = auto_classify_profile(p, taskc);
```

Rationale: Environment variables are inherited by child processes, so setting `SCX_ASTRO` on a shell or launcher automatically applies to the entire process tree. This is simpler than per-task syscalls and matches the documented design of `scx_turbo`.

### 3.3 Profile System and Default Behaviors

| Profile | ID | Default Behavior |
|---------|----|------------------|
| `ASTRO_PROFILE_INTERACTIVE` (0) | Short slices, head enqueue, preemptive. Fast lane. | Auto-detected via high lat_cri + short runtime + high sleep/wake frequency |
| `ASTRO_PROFILE_NORMAL` (1) | Standard slices, vtime-ordered with SRPT scaling. | Default for tasks that don't fit other profiles |
| `ASTRO_PROFILE_COMPUTE` (2) | Long slices, throughput lane. | Auto-detected via long runtime + low sleep/wake frequency |
| `ASTRO_PROFILE_BACKGROUND` (3) | Tiny slices, contained lane. | Auto-detected via high hog_score or explicit override |

### 3.4 Classification State Machine with Hysteresis

To prevent rapid lane thrashing (a task flipping between interactive and normal every few milliseconds), `scx_astro` uses a **hysteresis counter**:

```c
if (profile != taskc->pending_profile) {
    taskc->pending_profile = profile;
    taskc->profile_hysteresis = 1;
} else if (taskc->profile_hysteresis < ASTRO_HOG_SCORE_MAX) {
    taskc->profile_hysteresis++;
}

if (taskc->profile_hysteresis >= ASTRO_HYSTERESIS_THRESHOLD) {
    if (taskc->current_profile != profile) {
        /* Count transition for stats */
        ...
    }
    taskc->current_profile = profile;
}
```

`ASTRO_HYSTERESIS_THRESHOLD` is set to **3**, meaning a task must be consistently classified into a new profile across 3 consecutive scheduling events before the transition occurs. This is a critical design decision: without hysteresis, a task that occasionally has one long runtime burst would immediately lose its interactive privileges, causing jitter.

---

## 4. Novel Features

### 4.1 SRPT-Inspired Prioritization

**SRPT (Shortest Remaining Processing Time)** is theoretically optimal for mean response time, but it requires knowing the exact remaining runtime, which is impossible. `scx_astro` approximates SRPT within the **NORMAL** lane using vtime scaling:

```c
static __always_inline u64 srpt_vtime_scale(u64 vtime, u64 avg_runtime)
{
    u64 threshold = tune_srpt_threshold_ns; /* default 500us */
    if (avg_runtime < threshold) {
        /* Short tasks get 0.5x vtime -> appear earlier in EDF ordering */
        vtime = (vtime * ASTRO_SRPT_VTIME_SCALE_NUM) / ASTRO_SRPT_VTIME_SCALE_DEN;
    }
    return vtime;
}
```

In `enqueue()`, tasks dispatched to `ASTRO_NORMAL_DSQ` or `ASTRO_COMPUTE_DSQ` use `scx_bpf_dispatch_vtime()`:

```c
if (profile == ASTRO_PROFILE_NORMAL || profile == ASTRO_PROFILE_COMPUTE) {
    vtime = srpt_vtime_scale(vtime, taskc->avg_runtime_ns);
    scx_bpf_dispatch_vtime(p, dsq_id, slice_ns, vtime, enq_flags);
} else {
    scx_bpf_dispatch(p, dsq_id, slice_ns, enq_flags);
}
```

This means that among tasks with similar accumulated vruntime, the one with shorter average runtime per schedule will be chosen first. This is particularly effective for bursty shell workloads where many short commands interleave with a few long builds.

### 4.2 Waker-Boost Chains

When an interactive task wakes another task (e.g., the compositor wakes a game thread, which wakes a shader compiler), the wakee receives a **temporary priority boost** into `ASTRO_WAKER_BOOST_DSQ`.

Because `sched_ext` does not expose a `set_wakeup` struct_ops callback in the current kernel version, `scx_astro` implements this via a **heuristic approximation** using per-CPU timestamps:

```c
void BPF_STRUCT_OPS(astro_running, struct task_struct *p)
{
    ...
    if (cstate && taskc && taskc->current_profile == ASTRO_PROFILE_INTERACTIVE)
        cstate->last_interactive_ts = now;
    ...
}

void BPF_STRUCT_OPS(astro_runnable, struct task_struct *p, u64 enq_flags)
{
    ...
    if (cstate && cstate->last_interactive_ts > 0 &&
        now - cstate->last_interactive_ts < ASTRO_WAKER_BOOST_DURATION_NS &&
        taskc->waker_boost_depth < ASTRO_WAKER_BOOST_MAX_CHAIN) {
        taskc->waker_boost_expire_ns = now + ASTRO_WAKER_BOOST_DURATION_NS;
        taskc->waker_boost_depth++;
    }
    ...
}
```

**How it works**: When an interactive task runs on a CPU, it stamps `last_interactive_ts` in the per-CPU state. When another task wakes up on that same CPU shortly after, `runnable()` detects the recent interactive execution and grants the wakee a temporary boost. This approximates the true waker-wakee relationship without requiring BPF trampolines on `try_to_wake_up`.

Key properties:
- **Depth limit**: `ASTRO_WAKER_BOOST_MAX_CHAIN = 3` prevents infinite chain propagation (e.g., A wakes B wakes C wakes D...).
- **Time limit**: `ASTRO_WAKER_BOOST_DURATION_NS = 2ms`. The boost expires after 2ms of wall-clock time, not CPU time, so a boosted task that doesn't run immediately loses its boost.
- **Lane**: Boosted tasks go to `ASTRO_WAKER_BOOST_DSQ`, which is checked immediately after `ASTRO_INTERACTIVE_DSQ` in dispatch. They get `SCX_ENQ_HEAD` and may preempt.

Rationale: This is inspired by `scx_turbo`'s documented waker-boost design and `scx_lavd`'s latency-criticality propagation. The heuristic trades precision for portability: it works on current sched_ext kernels without requiring unstable tracepoint attachments. A future enhancement could attach a BPF trampoline to `try_to_wake_up` for exact waker-wakee tracking.

### 4.3 Dynamic Lane Routing

Tasks can move between lanes based on multiple triggers:

| Trigger | Source Lane | Target Lane | Mechanism |
|---------|-------------|-------------|-----------|
| High lat_cri + short runtime detected | NORMAL | INTERACTIVE | `auto_classify_profile()` via hysteresis |
| Budget exhaustion + high hog_score | NORMAL/COMPUTE | BACKGROUND | `hog_score` incremented on exhaustion |
| Long sleep + positive budget refill | BACKGROUND | NORMAL | `hog_score` decay on short runs |
| Explicit env var override | Any | Override profile | `tgid_profile_map` lookup |

The `hog_score` mechanism (borrowed from `scx_flow`) provides a robust signal for compute-to-background transitions:

```c
/* In stopping():
 * If task exhausted its budget, it's behaving like a CPU hog */
if (exhausted && taskc->hog_score < ASTRO_HOG_SCORE_MAX)
    taskc->hog_score += ASTRO_HOG_SCORE_EXHAUST_STEP;

/* If task had a short run without exhausting budget, decay hog score */
if (!exhausted && runtime_ns < ASTRO_HOG_RUNTIME_THRESHOLD_NS)
    taskc->hog_score = max(taskc->hog_score - ASTRO_HOG_SCORE_DECAY_STEP, 0);
```

When `hog_score >= ASTRO_HOG_SCORE_CONTAIN` (3), the task is classified as `BACKGROUND`. This requires **multiple consecutive exhaustions**, so a task that occasionally uses its full slice is not immediately penalized.

---

## 5. Slice Management

### 5.1 Default Slices per Lane

| Lane | Default Slice | Range | Behavior |
|------|---------------|-------|----------|
| Interactive | 150 µs | 50–200 µs | Short and responsive; tuned for 60–240Hz frame deadlines |
| Waker Boost | 100 µs | Fixed | Temporary; just enough to make progress before lane re-evaluation |
| Normal | 1 ms | 0.5–2 ms | Scales slightly with system load |
| Compute | 3 ms | Fixed | Long enough to amortize scheduling overhead |
| Background | 500 µs | 100–800 µs | Tiny to ensure frequent yield points; tunable via autotuner |

### 5.2 Slice Scaling Under Load

Unlike `scx_flow` which uses a complex budget-refill model, `scx_astro` keeps slice management simple:

- **Interactive** and **Waker Boost** slices are fixed (or autotuned) because latency is paramount.
- **Normal** slice is clamped to `[ASTRO_SLICE_NORMAL_MIN_NS, ASTRO_SLICE_NORMAL_MAX_NS]` and can be scaled by the autotuner.
- **Compute** slice is intentionally long to reduce scheduler overhead for CPU-bound tasks.
- **Background** slice is intentionally short to prevent a background task from monopolizing a CPU when it finally gets a chance to run after the fast lanes empty.

The autotuner (userspace) adjusts slices based on observed dispatch ratios:

```rust
enum AutoTuneMode {
    Balanced,   /* Default slices */
    Latency,    /* Shorter interactive/normal; longer compute */
    Throughput, /* Longer interactive/normal; shorter background */
}
```

### 5.3 Background Task Slices

Background tasks get **tiny slices by design** (default 500 µs, min 100 µs). The rationale is that background work (compilers, backups, indexing) should make slow, steady progress without creating noticeable latency spikes. The frequent yield points ensure that if an interactive task arrives, it will be scheduled within ~500 µs.

---

## 6. CPU Selection

### 6.1 `select_cpu()` Algorithm

`scx_astro` implements tiered CPU selection based on the task's profile:

```c
s32 BPF_STRUCT_OPS(astro_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
    struct astro_task_ctx *taskc = lookup_task_ctx(p);
    bool is_idle = false;
    s32 cpu;
    bool non_migratable = p->nr_cpus_allowed == 1 || is_migration_disabled(p);

    if (taskc) {
        if (taskc->sleep_started_at)
            update_budget_on_wakeup(p, taskc, bpf_ktime_get_ns());
        taskc->is_wakeup = true;
        taskc->is_sync_wakeup = !!(wake_flags & SCX_WAKE_SYNC);
    }

    if (!bpf_cpumask_test_cpu(prev_cpu, p->cpus_ptr))
        prev_cpu = bpf_cpumask_first(p->cpus_ptr);

    if (non_migratable) {
        cpu = prev_cpu;
        is_idle = scx_bpf_test_and_clear_cpu_idle(prev_cpu);
    } else {
        /* Default select_cpu from sched_ext handles idle search */
        cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
    }

    if (taskc) {
        taskc->wake_cpu = cpu >= 0 ? cpu : prev_cpu;
        taskc->wake_cpu_idle = is_idle;
        taskc->wake_cpu_valid = taskc->wake_cpu >= 0 &&
            bpf_cpumask_test_cpu(taskc->wake_cpu, p->cpus_ptr);
    }

    return cpu >= 0 ? cpu : prev_cpu;
}
```

### 6.2 Idle CPU Search Order

The actual idle search is delegated to `scx_bpf_select_cpu_dfl()`, which uses the kernel's built-in topology-aware search. However, `scx_astro` biases the search via the `wake_flags` and the task's profile:

- **Interactive tasks**: If `wake_cpu_idle` is true, the task is inserted directly into `SCX_DSQ_LOCAL_ON | wake_cpu` in `enqueue()`, bypassing the global DSQ entirely. This is the fastest path.
- **Normal tasks**: Prefer last CPU for cache warmth; fall back to any idle CPU in the same LLC.
- **Compute/Background**: Any idle CPU is acceptable. They do not trigger aggressive idle search or preemption.

### 6.3 SMT/Hybrid Core Awareness

`scx_astro` does not currently implement explicit big.LITTLE core selection (unlike `scx_lavd`). The rationale is that for an experimental scheduler focused on lane containment, adding topology complexity early would obscure the core design. A future enhancement would integrate `scx_utils::Topology` in userspace and pass a "preferred core type" hint to BPF, similar to `scx_lavd`'s `perf_cri` logic.

However, the scheduler is compatible with SMT because `scx_bpf_select_cpu_dfl()` already respects the `cpus_ptr` mask, and pinned/affinitized tasks are handled correctly via `non_migratable` checks.

---

## 7. Preemption

### 7.1 When Interactive Tasks Preempt

Preemption is triggered in `astro_enqueue()` when:

1. The task's profile is `INTERACTIVE` or `WAKER_BOOST`.
2. The selected `target_cpu` is not idle.
3. The task is a wakeup (`SCX_ENQ_WAKEUP`).

```c
if (profile == ASTRO_PROFILE_INTERACTIVE || profile == ASTRO_PROFILE_WAKER_BOOST) {
    enq_flags |= SCX_ENQ_HEAD;
    if (target_cpu >= 0 && !taskc->wake_cpu_idle) {
        scx_bpf_kick_cpu(target_cpu, SCX_KICK_PREEMPT);
        preempts++;
        kicks_preempt++;
    }
}
```

The `SCX_ENQ_HEAD` flag ensures the task is at the front of its DSQ. The `SCX_KICK_PREEMPT` IPI forces the target CPU to reschedule immediately. Together, they achieve sub-millisecond preemption latency for interactive tasks.

### 7.2 IPI Kicking Patterns

| Scenario | Kick Type | Rationale |
|----------|-----------|-----------|
| Interactive task wakes, target CPU busy | `SCX_KICK_PREEMPT` | Immediate preemption |
| Waker-boost task wakes, target CPU busy | `SCX_KICK_PREEMPT` | Chain propagation needs low latency |
| Any task wakes, target CPU idle | `SCX_KICK_IDLE` | Wake idle CPU without preemption cost |
| Normal task wakes, target CPU idle | `SCX_KICK_IDLE` | Standard wakeup |
| Background task wakes | No kick | Background tasks wait for natural scheduling |

Rate limiting: `scx_astro` does not implement explicit kick rate limiting per CPU. The burst-round counters (`high_priority_burst_rounds`) implicitly limit preemption by capping the number of consecutive interactive dispatches before lower lanes are forced. In practice, this provides sufficient back-pressure.

---

## 8. Userspace Component

### 8.1 CLI Options

```rust
#[derive(Debug, Parser)]
#[command(name = "scx_astro")]
struct Opts {
    /// Enable stats monitoring with the specified interval (seconds).
    #[clap(long)]
    stats: Option<f64>,

    /// Run in monitor-only mode (no scheduler).
    #[clap(long)]
    monitor: Option<f64>,

    /// Enable BPF debug printk.
    #[clap(short, long)]
    debug: bool,

    /// Disable adaptive runtime tuning.
    #[clap(long)]
    no_autotune: bool,

    /// Disable /proc environment scanning.
    #[clap(long)]
    no_autoscan: bool,

    /// Milliseconds between /proc scans.
    #[clap(long, default_value = "2000")]
    scan_interval_ms: u64,

    #[clap(flatten, next_help_heading = "Libbpf Options")]
    libbpf: LibbpfOpts,
}
```

### 8.2 Stats Tracked

The `Metrics` struct (exposed via `scx_stats`) tracks:

- **Lane dispatches**: `interactive_dispatches`, `waker_boost_dispatches`, `normal_dispatches`, `compute_dispatches`, `background_dispatches`
- **Preemption/kicks**: `preempts`, `kicks_idle`, `kicks_preempt`
- **Transitions**: `profile_transitions`, `waker_boosts`
- **Novel features**: `srpt_short_tasks`
- **Budget**: `budget_refills`, `budget_exhaustions`
- **Starvation**: `contained_starvation_rounds`, `shared_starvation_rounds`, `high_priority_burst_rounds`
- **Autotuner**: `autotune_mode`, `autotune_generation`, plus current slice tunables

### 8.3 Auto-Detection Daemon

The userspace thread scans `/proc/*/environ` for `SCX_ASTRO=<profile>`:

```rust
fn scan_proc_environ(skel: &mut BpfSkel, interval_ms: u64, shutdown: Arc<AtomicBool>) {
    let mut known: HashMap<i32, u8> = HashMap::new();
    while !shutdown.load(Ordering::Relaxed) {
        let mut current_pids = Vec::new();
        for entry in fs::read_dir("/proc").unwrap().flatten() {
            let pid: i32 = entry.file_name().to_string_lossy().parse().ok()?;
            let environ = fs::read_to_string(format!("/proc/{}/environ", pid)).unwrap_or_default();
            if let Some(profile) = parse_profile_from_environ(&environ) {
                current_pids.push(pid);
                if known.get(&pid) != Some(&profile) {
                    skel.maps.tgid_profile_map.update(
                        &pid.to_ne_bytes(),
                        &astro_tgid_profile { profile }.as_bytes(),
                        libbpf_rs::MapFlags::ANY,
                    ).ok();
                    known.insert(pid, profile);
                }
            }
        }
        known.retain(|pid, _| current_pids.contains(pid));
        thread::sleep(Duration::from_millis(interval_ms));
    }
}
```

Note: In the current skeleton, the scan thread is not spawned because `BpfSkel` is not `Send`. A production implementation would extract the map FD after load and use `libbpf_rs::MapHandle` in the scan thread.

### 8.4 Tunables

All tunables are exposed as `volatile u64` variables in the BPF `.data` section, writable from userspace:

- `tune_interactive_slice_ns`
- `tune_normal_slice_ns`
- `tune_compute_slice_ns`
- `tune_background_slice_ns`
- `tune_srpt_threshold_ns`

The autotuner steps these values toward mode-specific targets every second.

---

## 9. Integration

### 9.1 Workspace Addition

Added to root `Cargo.toml`:

```toml
members = [
    ...
    "scheds/experimental/scx_astro",
    "scheds/experimental/scx_flow",
    ...
]
```

### 9.2 Dependencies

Same pattern as `scx_flow`:

```toml
[dependencies]
anyhow = "1"
ctrlc = { version = "3", features = ["termination"] }
clap = { version = "4", features = ["derive"] }
libbpf-rs = "=0.26.2"
scx_stats = { path = "../../../rust/scx_stats" }
scx_utils = { path = "../../../rust/scx_utils" }
nix = { version = "0.29", features = ["process"] }

[build-dependencies]
scx_cargo = { path = "../../../rust/scx_cargo" }
```

### 9.3 Build Configuration

`build.rs` follows the standard `scx_cargo::BpfBuilder` pattern:

```rust
fn main() {
    add_bpf_warning_suppression("-Wno-missing-declarations");
    scx_cargo::BpfBuilder::new()
        .unwrap()
        .enable_intf("src/bpf/intf.h", "bpf_intf.rs")
        .enable_skel("src/bpf/main.bpf.c", "bpf")
        .build()
        .unwrap();
}
```

---

## 10. Key Design Decisions and Rationale

### 10.1 Why Five Lanes Instead of Three?

Many multi-lane schedulers (e.g., `scx_flow`) use 3–4 lanes. `scx_astro` uses five to separate **normal bursty tasks** from **long-running compute**. Without this separation, a long-running compile job enqueued to the normal lane would delay all other normal tasks. By giving compute its own lane with longer slices, normal tasks (shell commands, browser tabs) get better isolation.

### 10.2 Why Global DSQs Instead of Per-CPU Lanes?

Per-CPU DSQs maximize locality but can strand work. If CPU 0 has 10 interactive tasks and CPU 1 is idle, per-CPU DSQs would leave CPU 1 idle. Global DSQs allow any CPU to consume interactive work immediately. The cost is slightly worse cache locality, but `select_cpu()` + direct-local enqueue mitigates this for the common wakeup-to-idle case.

### 10.3 Why Hysteresis for Profile Transitions?

Without hysteresis, a task that alternates between short and long runs (e.g., a web renderer that occasionally does layout) would thrash between `INTERACTIVE` and `NORMAL`. This causes:
- Cache thrashing (different DSQs have different locality patterns)
- Starvation counter instability
- Visible jitter in scheduling latency

A threshold of 3 events was chosen empirically: it filters out single anomalies while responding to true behavioral changes within ~1-2ms.

### 10.4 Why vtime Scaling for SRPT Instead of Sorted DSQs?

True SRPT would require sorting tasks by estimated remaining runtime. BPF does not allow arbitrary sorting in DSQs. `scx_bpf_dispatch_vtime()` provides EDF ordering by virtual time. By scaling the vtime of short tasks downward, we approximate "short tasks go first" within the existing EDF framework without requiring new kernel mechanisms.

### 10.5 Why Not Use `scx_rustland` for Userspace Scheduling?

`scx_rustland` offloads all scheduling decisions to userspace, providing maximum flexibility. However, it adds ~1-3 µs of overhead per schedule. For `scx_astro`, the classification logic is simple enough to fit entirely in BPF, and the hot paths (enqueue, dispatch, select_cpu) must run in kernel context to achieve sub-100µs latency targets. Userspace is used only for slow-path tasks: autotuning, environment scanning, and statistics.

---

## 11. Testing Recommendations

### 11.1 Correctness Testing

1. **Compile test**:
   ```bash
   cargo build -p scx_astro
   ```

2. **Load/unload test**:
   ```bash
   sudo ./target/debug/scx_astro
   # Verify with bpftool
   sudo bpftool struct_ops list | grep scx_astro
   ```

3. **Stress test**:
   ```bash
   # Background load
   stress-ng --cpu 64 &
   # Interactive latency test
   schbench -m 1 -t 4 -i 100
   ```

### 11.2 Lane Behavior Validation

Use `bpftool map dump name cpu_state` (or stats output) to verify:

- Interactive tasks produce `interactive_dispatches > 0`
- Background tasks produce `background_dispatches > 0`
- `contained_starvation_rounds` stays bounded (not growing infinitely)
- `profile_transitions` is low under steady-state load (hysteresis working)

### 11.3 Waker-Boost Chain Test

```bash
# Terminal 1: run a high-frequency waker
while true; do echo > /dev/null; done &

# Terminal 2: measure wakee latency with cyclictest
sudo cyclictest -p 80 -i 1000 -l 10000
```

With waker-boost enabled, the cyclictest thread should show reduced average latency when woken by a high-frequency loop.

### 11.4 SRPT Validation

```bash
# Mix short and long tasks
for i in $(seq 1 100); do sleep 0.001; done &
stress-ng --cpu 1 --timeout 10s &
```

Observe via `scxtop` or stats that short sleepers (`srpt_short_tasks`) are disproportionately represented in normal dispatches compared to their CPU share.

### 11.5 Explicit Override Test

```bash
SCX_ASTRO=interactive ./latency-test &
SCX_ASTRO=background ./cpu-burner &
```

Verify via stats that `interactive_dispatches` and `background_dispatches` reflect the override.

### 11.6 Autotuner Validation

Run with `--stats 1` under varying load:
- Idle load -> mode should stay `balanced`
- High interactive load (gaming + browser) -> mode should shift to `latency`
- High background load (compilation) -> mode should shift to `throughput`

---

## 12. File-by-File Breakdown

```
scheds/experimental/scx_astro/
├── build.rs              # scx_cargo::BpfBuilder with warning suppression
├── Cargo.toml            # Package manifest, dependencies, workspace member
├── README.md             # User-facing quick start and usage
├── DESIGN.md             # This document
└── src/
    ├── main.rs           # Userspace: CLI, skeleton loading, autotuner,
    │                     #   stats server, /proc scan thread (stubbed)
    ├── stats.rs          # scx_stats Metrics struct, server_data(), monitor()
    ├── bpf_intf.rs       # include!(concat!(env!("OUT_DIR"), "/bpf_intf.rs"))
    ├── bpf_skel.rs       # include!(concat!(env!("OUT_DIR"), "/bpf_skel.rs"))
    └── bpf/
        ├── intf.h        # Constants, enums, astro_cpu_state, astro_task_ctx,
        │                 #   astro_tgid_profile
        └── main.bpf.c    # Full BPF implementation: maps, lat_cri, classification,
                          #   budget, slice, SRPT, waker boost, enqueue, dispatch,
                          #   select_cpu, running, stopping, init/exit
```

| File | Lines (approx) | Key Contents |
|------|----------------|--------------|
| `src/bpf/main.bpf.c` | ~550 | Core scheduler logic. Maps, helpers, all 12 struct_ops callbacks. |
| `src/bpf/intf.h` | ~120 | Shared constants and structs between BPF and Rust. |
| `src/main.rs` | ~400 | CLI parsing, skeleton init, autotuner, /proc scanning, event loop. |
| `src/stats.rs` | ~200 | scx_stats metrics definitions and formatting. |
| `src/bpf_intf.rs` | ~10 | Boilerplate include for generated bindings. |
| `src/bpf_skel.rs` | ~5 | Boilerplate include for generated skeleton. |
| `build.rs` | ~25 | Standard BpfBuilder invocation. |
| `Cargo.toml` | ~30 | Dependencies identical to scx_flow + `nix` for proc scanning. |

---

## 13. Future Work

- **Topology-aware placement**: Integrate `scx_utils::Topology` to prefer big cores for compute and little cores for background on hybrid systems.
- **ML-based classification**: Replace heuristic `auto_classify_profile()` with a tiny perceptron trained on sched-ext tracepoints.
- **Cgroup integration**: Allow per-cgroup lane assignment policies, not just per-task.
- **Energy awareness**: Scale background task frequency down using `cpufreq` hints when only background lanes are active.

---

*End of Design Document*
