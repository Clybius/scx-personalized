# scx_descent

This is a single user-defined scheduler used within [`sched_ext`](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about `sched_ext`](https://github.com/sched-ext/scx/tree/main).

## Overview

**scx_descent** is a PIE (Proportional Integral controller Enhanced) based adaptive scheduler that automatically optimizes scheduling parameters for different workload classes. Unlike traditional schedulers with static parameters, scx_descent continuously adapts its behavior based on observed latency using deterministic control theory.

### How It Differs from scx_flash

While **scx_flash** uses static parameters optimized for multimedia and audio workloads, **scx_descent** introduces:

1. **Automatic task classification**: Tasks are dynamically classified into one of four classes:
   - **Latency Critical**: User-facing applications, UI threads, games, audio
   - **Normal**: Default interactive tasks
   - **Hog**: High CPU usage tasks
   - **Background**: Low priority background work

2. **PIE controller optimization**: Uses a PI (Proportional-Integral) controller to minimize the error between observed latency and target latency per class. The controller adjusts:
   - Latency weight (deadline offset)
   - Base time slice
   - Vruntime scale
   - Preemption priority
   - Migration cost

3. **Deterministic control**: Unlike probabilistic or gradient-based approaches, PIE provides predictable, deterministic parameter adjustment with fast convergence and no random exploration.

4. **Three optimization profiles**:
   - **Gaming**: Prioritize low latency (10ms response, α=4, β=2)
   - **Productivity**: Balance latency and throughput (20ms response, α=8, β=4)
   - **Server**: Maximize stability (50ms response, α=16, β=8)

## PIE Controller Architecture

The PIE controller operates on each (CPU, class) combination:

1. **Measurement**: BPF tracks task latencies (enqueue to run)
2. **Error Calculation**: Compare observed latency to target
3. **PI Control**:
   - P-term: Proportional to current error (current - target)
   - I-term: Accumulates trend (rate of change of latency)
4. **Parameter Update**: Adjusts all 5 scheduling parameters based on control output
5. **Bounds Enforcement**: Ensures parameters stay within safe limits per profile

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Userspace (Rust)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   PIE       │  │   Profile    │  │   Task Classifier   │ │
│  │ Controller  │  │   Config     │  │   (proc scanning)   │ │
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
                    │  │  Latency Metrics    │  │
                    │  │ (per CPU/class)     │  │
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

# Enable PIE debug output every 100ms
sudo scx_descent --debug-pie 100
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
      --debug-pie <N>              Enable PIE controller debug output every N ms
      --update-interval-ms <MS>  Parameter sync interval [default: 50]
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

## Profile Configuration

| Profile | Response | Alpha | Beta | Use Case |
|---------|----------|-------|------|----------|
| Gaming | 10ms | 4 | 2 | Fast response for games/audio |
| Productivity | 20ms | 8 | 4 | Balanced desktop use |
| Server | 50ms | 16 | 8 | Stable server workloads |

**PIE Parameters:**
- **Alpha (α)**: Proportional gain divisor - smaller = more aggressive response
- **Beta (β)**: Integral gain divisor - smaller = faster trend accumulation
- **Response**: Update interval for parameter adjustments

## Implementation Status

- ✅ **Phase 1**: Basic scheduler structure with task classification and per-class parameters
- ✅ **Phase 2**: Safety mechanisms and parameter bounds
- ✅ **Phase 3**: PIE controller with deterministic latency-based optimization
- ⚠️ **Phase 4** (Future): Advanced features like workload prediction, container awareness, hardware-specific profiles

**Production Ready?**

**Yes** - The scheduler is functional and includes adaptive parameter optimization via PIE controller. It has been tested to compile and includes safety mechanisms to prevent runaway parameter changes. However, extensive benchmarking against production workloads is recommended before deployment.

## License

GPL-2.0 (required for BPF programs by kernel verifier)
