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

### Dynamic Virtual Nice Adjustment

Every 10ms (configurable via `--adjust-interval-us`), the scheduler recalculates an interactive score (0-1000) for each task based on:

- **Wait frequency** (0-200 points): Higher sleep rate indicates more interactive behavior
- **Average runtime** (0-200 points): Shorter runtime bursts suggest interactive patterns
- **Sync wakeup flag** (+50 points): Indicates producer-consumer chains
- **IRQ-driven wakeup** (+100 points): Tasks woken by interrupts (input, timers) are typically interactive
- **Wake frequency** (0-100 points): High wake frequency suggests producer behavior

Tasks with scores above the interactive threshold (default 700) have their virtual nice value adjusted toward more favorable values, limited to maximum changes of 5 units per adjustment period for smooth transitions.

### HOG Demotion

NORMAL tasks that consume more than 50% CPU over a 100ms measurement window are automatically demoted to the HOG queue. This prevents background batch work from interfering with interactive tasks. HOG tasks can be promoted back to NORMAL if they exhibit highly interactive behavior patterns.

### Additional Features

- **SMT Avoidance**: Prefers non-SMT siblings to reduce contention
- **Cache Affinity**: Attempts to keep tasks on the same LLC
- **CPU Frequency Scaling**: LC queue tasks run at max frequency, HOG at min
- **Antistall Protection**: Tasks stalled for more than 3 seconds are rescued

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
