# scx_happy

This is a single user-defined scheduler used within [`sched_ext`](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about `sched_ext`](https://github.com/sched-ext/scx/tree/main).

## Overview

`scx_happy` is a latency-aware scheduler that implements a sophisticated hybrid scheduling algorithm combining **EEVDF (Earliest Eligible Virtual Deadline First)** with **WFQ (Weighted Fair Queueing)**. Its core innovation is a **virtual niceness system** that extends the kernel's standard nice values from the range -20 to +19 to an expanded range of -50 to +49, enabling finer-grained priority differentiation.

The scheduler organizes tasks into three distinct queues based on their virtual nice value:

- **LC (Latency-Critical) Queue**: virt_nice -50 to -20, 500μs time slices
- **NORMAL Queue**: virt_nice -19 to +10, 1000μs time slices
- **HOG Queue**: virt_nice +11 to +49, 3000μs time slices

Tasks are automatically classified into these queues using heuristics that detect Steam games, desktop environment components (kwin, mutter, gnome-shell), input handling threads (evdev, libinput), audio processing (pipewire, pulseaudio), and kernel threads. Tasks can also be manually marked via the `SCX_TURBO=1` environment variable.

The BPF component runs entirely in kernel context and makes all scheduling decisions, while the minimal Rust userspace component handles CLI argument parsing, task classification polling, and statistics reporting.

## How It Works

### Virtual Niceness and WFQ

The virtual niceness system maps the extended -50 to +49 range onto the kernel's existing nice-to-weight conversion table. The weight calculation follows the standard Linux weight table where nice 0 has weight 1024 and values range from 88761 (nice -20) down to 15 (nice +19).

Within each queue, tasks are scheduled using WFQ principles. The virtual slice (vslice) for a task is calculated as:

```
vslice = slice * NICE_0_WEIGHT / weight
```

This means higher-priority tasks (lower nice values, higher weights) receive proportionally smaller virtual time advances, giving them more frequent scheduling opportunities.

### EEVDF Implementation

Within each queue, tasks are ordered using EEVDF (Earliest Eligible Virtual Deadline First):

- **Eligibility**: A task becomes eligible when its virtual runtime (`vtime`) is less than or equal to the current average virtual time (`avg_vtime`)
- **Eligible tasks** are ordered by their deadline (vruntime + vslice)
- **Ineligible tasks** are ordered by their vruntime, allowing them to catch up in fairness
- The scheduler dispatches the earliest eligible deadline task first

Eligible tasks can preempt currently running tasks through the deadline-based preemption mechanism described below.

### Deadline-Based Preemption

Building on the EEVDF foundation, scx_happy implements deadline-based preemption that allows eligible tasks to interrupt running tasks when they have earlier virtual deadlines:

**Preemption Rules:**
1. **Queue Priority Preemption**: LC tasks can always preempt NORMAL and HOG tasks, and NORMAL tasks can preempt HOG tasks, regardless of deadlines
2. **Same-Queue Preemption**: Within the same queue, a task can preempt the running task if it has an earlier virtual deadline (`vruntime + vslice`)
3. **Eligibility Requirement**: Only eligible tasks (those owed CPU time, where `vtime <= avg_vtime`) can trigger preemption

**Hysteresis Mechanism:**
To prevent excessive context switching (ping-pong), preemption includes a hysteresis threshold (default 10% of the running task's vslice, configurable via `--preemption-hysteresis-pct`). A task will only preempt if its deadline is earlier by at least this threshold.

This mechanism ensures that latency-critical tasks get immediate access to the CPU when they have urgent deadlines, while preventing thrashing from tasks with nearly-identical deadlines. Use `--disable-deadline-preemption` to disable this feature.

### Dynamic Virtual Nice Adjustment

Every 10ms (configurable via `--adjust-interval-us`), the scheduler recalculates an interactive score (0-1000) for each task based on:

- **Wait frequency** (0-200 points): Higher sleep rate indicates more interactive behavior
- **Average runtime** (0-200 points): Shorter runtime bursts suggest interactive patterns
- **Sync wakeup flag** (+50 points): Indicates producer-consumer chains
- **IRQ-driven wakeup** (+100 points): Tasks woken by interrupts (input, timers) are typically interactive
- **Wake frequency** (0-100 points): High wake frequency suggests producer behavior

Tasks with scores above the interactive threshold (default 700) have their virtual nice value adjusted toward more favorable values, limited to maximum changes of 5 units per adjustment period for smooth transitions.

### HOG Demotion and Promotion

**Demotion:**
NORMAL tasks that consume more than 50% CPU over a 100ms measurement window (configurable via `--hog-cpu-threshold`) are automatically demoted to the HOG queue. This prevents background batch work from interfering with interactive tasks.

**Promotion via Lag Decay:**
HOG tasks can be promoted back to NORMAL through a sophisticated lag decay mechanism that tracks accumulated sleep time:

- **Lag Tracking**: Each task maintains a "lag" value representing CPU time owed to the task (similar to EEVDF eligibility). This accumulates while the task waits to run.
- **Sleep-Based Decay**: While a HOG task is sleeping (not runnable), its accumulated lag decays exponentially. The decay applies every 20ms of accumulated sleep time (configurable via `--hog-decay-interval-us`), reducing the lag by 75% each time (`lag = lag >> 2`, keeping 25%).
- **Promotion Criteria**: A HOG task is eligible for promotion when:
  1. It has accumulated at least 3 sleep cycles while in HOG (configurable via `--hog-min-sleep-count`)
  2. Total sleep time reaches at least 50ms (configurable via `--hog-min-sleep-duration-us`)
  3. The task is eligible (lag >= 0, meaning it's owed CPU time)
  4. The task shows interactive behavior patterns (wait frequency above threshold)

This decay mechanism allows batch tasks that periodically sleep (e.g., checking for work, I/O wait) to gradually become eligible for promotion back to NORMAL, while true CPU hogs remain in the HOG queue. Use `--disable-hog-lag-decay` to disable this promotion mechanism.

### Additional Features

- **Deadline-Based Preemption**: EEVDF-style preemption allows eligible tasks with earlier deadlines to interrupt running tasks (with hysteresis to prevent thrashing)
- **SMT Avoidance**: Prefers non-SMT siblings to reduce contention
- **Cache Affinity**: Attempts to keep tasks on the same LLC
- **CPU Frequency Scaling**: LC queue tasks run at max frequency, HOG at min
- **Antistall Protection**: Tasks stalled for more than 3 seconds are rescued
- **HOG Lag Decay**: Exponential decay of accumulated lag during sleep enables promotion of well-behaved batch tasks back to NORMAL queue

## Typical Use Case

`scx_happy` is designed for desktop and gaming workloads where interactive responsiveness is critical. It excels when:

- Gaming while background tasks (compilations, backups) run
- Running multimedia applications with real-time requirements
- Desktop usage with mixed interactive and batch workloads
- Systems with heterogeneous CPU topologies (P-cores/E-cores, big.LITTLE)

The scheduler's combination of fine-grained virtual niceness and dynamic adjustment ensures that latency-critical tasks maintain responsiveness even when CPU-intensive background work is present.

## Production Ready?

No. `scx_happy` is currently in the experimental schedulers directory and is under active development. While the underlying algorithms (EEVDF and WFQ) are well-established in scheduling theory, this specific implementation with virtual niceness extension and dynamic adjustment heuristics requires further testing and validation before production use.

The experimental status reflects:
- Novel dynamic adjustment heuristics that may need tuning
- Extended nice value system that hasn't been widely tested
- Task classification heuristics that may need refinement
- Limited real-world workload validation

Users are encouraged to test and provide feedback, but should be aware that behavior may change as the scheduler evolves.

## AI Notice

This scheduler was primarily written utilizing AI, in particular, Kimi 2.5. 
Research & implementation scheme was devised primarily by Clybius (Cole O.).
