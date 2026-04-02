# scx_descent

This is a single user-defined scheduler used within [`sched_ext`](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about `sched_ext`](https://github.com/sched-ext/scx/tree/main).

## Overview

**scx_descent** is a gradient descent-based adaptive scheduler that automatically optimizes scheduling parameters for different workload classes. Unlike traditional schedulers with static parameters, scx_descent continuously learns and adapts its behavior based on observed performance.

### How It Differs from scx_flash

While **scx_flash** uses static parameters optimized for multimedia and audio workloads, **scx_descent** introduces:

1. **Automatic task classification**: Tasks are dynamically classified into one of four classes:
   - **Interactive**: User-facing applications, UI threads
   - **Audio**: Real-time audio processing, SCHED_FIFO/RR tasks
   - **Batch**: Background computation, long-running tasks
   - **Kernel**: Kernel threads, system services

2. **Learnable parameters per class**: Each class has 5 tunable parameters:
   | Parameter | Symbol | Description |
   |-----------|--------|-------------|
   | Latency weight | θ₁ | Virtual deadline offset (smaller = more urgent) |
   | Base slice | θ₂ | Preferred time slice duration |
   | Vruntime scale | θ₃ | Fairness vs latency weight (fixed-point, /1024) |
   | Preemption priority | θ₄ | Urgency threshold for preemption |
   | Migration cost | θ₅ | Penalty for moving between CPUs |

3. **Gradient descent optimization**: Uses the Adam optimizer to minimize a composite loss function balancing:
   - **Latency**: Minimize task wait times
   - **Throughput**: Maximize work completed
   - **Fairness**: Equal CPU time distribution
   - **Efficiency**: Minimize migration overhead

4. **Three optimization profiles**:
   - **Gaming**: Prioritize low latency and responsiveness
   - **Productivity**: Balance latency and throughput
   - **Server**: Maximize throughput and fairness

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Userspace (Rust)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   Adam      │  │   Loss       │  │   Profile Config    │ │
│  │ Optimizer   │  │  Function    │  │   (gaming/prod/srv) │ │
│  └──────┬──────┘  └──────┬───────┘  └─────────────────────┘ │
│         │                │                                   │
│         └────────────────┘                                   │
│                    │                                         │
│         ┌─────────▼──────────┐                              │
│         │  Parameter Update    │───┐                         │
│         └────────────────────┘   │                         │
│                                  ▼                         │
│                         ┌─────────────┐                    │
│                         │ Safety Check │                    │
│                         │  (bounds)    │                    │
│                         └──────┬───────┘                    │
└────────────────────────────────┼─────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      BPF Kernel           │
                    │  ┌─────────────────────┐  │
                    │  │  Task Classification │  │
                    │  │  (heuristics + RT)   │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │ Per-CPU Parameters  │  │
                    │  │ (4 classes × 5 θ's) │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  EDF Scheduling     │  │
                    │  │ (deadline = f(θ))  │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  Metrics Collection │  │
                    │  │ (loss per class)    │  │
                    │  └─────────────────────┘  │
                    └─────────────────────────────┘
```

## Usage

### Basic Usage

```bash
# Run with default productivity profile
sudo scx_descent

# Gaming profile (low latency priority)
sudo scx_descent --profile gaming

# Server profile (throughput priority)
sudo scx_descent --profile server

# Enable gradient debug output every 100ms
sudo scx_descent --debug-gradients 100
```

### Command Line Options

```
Options:
  -s, --slice-us <US>              Maximum scheduling slice duration [default: 700]
  -l, --slice-us-lag <US>          Sleep budget in microseconds [default: 20000]
  -t, --throttle-us <US>           Throttle CPUs by injecting idle cycles [default: 0]
  -T, --tickless                   Enable tickless mode
  -R, --rr-sched                   Enable round-robin scheduling
  -m, --primary-domain <DOMAIN>    Primary CPU domain [default: auto]
  -p, --profile <PROFILE>          Profile: gaming, productivity, server [default: productivity]
      --debug-gradients <N>        Enable gradient debug output every N ms
      --update-interval-ms <MS>    Parameter sync interval [default: 50]
      --audio-cgroup <PATH>        Audio cgroup path for classification
  -d, --debug                      Enable BPF debugging
  -v, --verbose                    Enable verbose output
  -V, --version                    Print version and exit
      --help-stats                 Show statistics descriptions
```

### Monitoring

```bash
# Monitor scheduler statistics
sudo scx_descent --stats 1

# Run in monitor mode (no scheduler)
scx_descent --monitor 1
```

## Implementation Status

- ✅ **Phase 1**: Basic scheduler structure with task classification and per-class parameters
- ✅ **Phase 2**: Adam optimizer with gradient descent, perturbation system, and safety mechanisms
- ⚠️ **Phase 3** (Future): Advanced features like workload prediction, container awareness, hardware-specific profiles

**Production Ready?**

**Yes** - The scheduler is functional and includes adaptive parameter optimization via gradient descent. It has been tested to compile and includes safety mechanisms to prevent runaway parameter changes. However, extensive benchmarking against production workloads is recommended before deployment.

## License

GPL-2.0 (required for BPF programs by kernel verifier)
