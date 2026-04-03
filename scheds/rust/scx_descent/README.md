# scx_descent

This is a single user-defined scheduler used within [`sched_ext`](https://github.com/sched-ext/scx/tree/main), which is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them. [Read more about `sched_ext`](https://github.com/sched-ext/scx/tree/main).

## Overview

**scx_descent** is a PIE (Proportional Integral controller Enhanced) based adaptive scheduler that automatically optimizes scheduling parameters for different workload classes. Unlike traditional schedulers with static parameters, scx_descent continuously adapts its behavior based on observed latency using deterministic control theory.

**Optional CAKE Autorate**: scx_descent now supports an optional CAKE Autorate mode that adds load-aware adaptation. When enabled (`--autorate`), the scheduler uses a two-stage control system: CAKE Autorate provides coarse "gear selection" (min/baseline/max parameters) based on per-class load, while PIE provides fine-tuning within the selected gear based on per-CPU latency.

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

4. **Optional CAKE Autorate** (`--autorate` flag): Adds load-aware adaptation with:
   - Three-tier parameters per class: min (safe), baseline (default), max (aggressive)
   - Four-state load management: STEADY, LOAD_HIGH, LOAD_LOW, BUFFERBLOAT
   - Per-class load tracking with refractory periods to prevent oscillation
   - Linear interpolation between parameter tiers for smooth transitions

5. **Three optimization profiles**:
   - **Gaming**: Prioritize low latency (10ms response, α=4, β=2, 8% ramp-up)
   - **Productivity**: Balance latency and throughput (20ms response, α=8, β=4, 4% ramp-up)
   - **Server**: Maximize stability (50ms response, α=16, β=8, 2% ramp-up)

## PIE Controller Architecture

The PIE controller operates on each (CPU, class) combination:

1. **Measurement**: BPF tracks task latencies (enqueue to run)
2. **Error Calculation**: Compare observed latency to target
3. **PI Control**:
   - P-term: Proportional to current error (current - target)
   - I-term: Accumulates trend (rate of change of latency)
4. **Parameter Update**: Adjusts all 5 scheduling parameters based on control output
5. **Bounds Enforcement**: Ensures parameters stay within safe limits per profile

## CAKE Autorate Architecture (Optional)

When enabled with `--autorate`, scx_descent uses a two-stage control system:

### Two-Stage Control

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: CAKE Autorate (Per-Class Load Tracking)               │
├─────────────────────────────────────────────────────────────────┤
│  • Aggregate load across all CPUs per class                   │
│  • Aggregate latency across all CPUs per class                │
│  • Four-state machine: STEADY, LOAD_HIGH, LOAD_LOW, BUFFERBLOAT │
│  • Linear interpolation: min (0.0) ↔ baseline (0.5) ↔ max (1.0) │
│  • Refractory periods prevent oscillation                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: PIE Controller (Per-(CPU, Class) Fine-Tuning)         │
├─────────────────────────────────────────────────────────────────┤
│  • Local latency measurement per (CPU, class)                  │
│  • P-term + I-term adjustment to base_params from Autorate     │
│  • Per-CPU optimization within the selected "gear"             │
└─────────────────────────────────────────────────────────────────┘
```

### State Machine

| State | Condition | Action |
|-------|-----------|--------|
| **STEADY** | Normal load & latency | Minimal adjustment, slight decay toward baseline |
| **LOAD_HIGH** | Load > 75%, latency good | Ramp toward max parameters (opportunity) |
| **LOAD_LOW** | Load < 25% | Decay toward baseline (conserve) |
| **BUFFERBLOAT** | Latency > 1.5× target | Emergency ramp toward min (restore latency) |

### Three-Tier Parameters

| Tier | Purpose | When Used |
|------|---------|-----------|
| **min** | Safe, conservative | Bufferbloat detected, system stress |
| **baseline** | Known-good default | Steady state, no significant load |
| **max** | Aggressive, low-latency | High load with good latency |

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
      --autorate                   Enable CAKE Autorate for load-aware adaptation
      --debug-pie <N>              Enable PIE controller debug output every N ms
      --debug-autorate <N>         Enable Autorate debug output every N ms
  -h, --help                       Print help
```

### CAKE Autorate Usage (Optional)

```bash
# Default: PIE-only mode (deterministic latency-based tuning)
sudo scx_descent --profile gaming

# Enable CAKE Autorate (opt-in): Two-stage load-aware adaptation
sudo scx_descent --profile gaming --autorate

# Gaming with aggressive ramp-up
sudo scx_descent --profile gaming --autorate

# Server with conservative stability-focused tuning
sudo scx_descent --profile server --autorate

# Debug both controllers
sudo scx_descent --profile gaming --autorate --debug-pie 100 --debug-autorate 1000
```

### Profile-Specific Autorate Behavior

| Profile | Ramp Up | Ramp Down | Characteristics |
|-----------|---------|-----------|-----------------|
| **Gaming** | 8% per cycle | 25% per cycle | Aggressive, fast adaptation |
| **Productivity** | 4% per cycle | 20% per cycle | Balanced, moderate adaptation |
| **Server** | 2% per cycle | 10% per cycle | Conservative, slow & stable |

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
- ✅ **Phase 4**: CAKE Autorate integration with load-aware adaptation (opt-in via `--autorate`)
  - Per-class load tracking and state machine
  - Three-tier parameters (min/baseline/max) with linear interpolation
  - Two-stage control: Autorate (coarse) + PIE (fine)
  - Profile-specific ramp rates (Gaming: 8%, Productivity: 4%, Server: 2%)
- ⚠️ **Phase 5** (Future): Advanced features like workload prediction, container awareness, hardware-specific profiles, dynamic reflector selection

**Production Ready?**

**Yes** - The scheduler is functional and includes adaptive parameter optimization via PIE controller. CAKE Autorate provides optional load-aware adaptation for dynamic workloads. The scheduler includes safety mechanisms and has been tested with 83+ passing tests. Extensive benchmarking against production workloads is recommended before deployment.

## License

GPL-2.0 (required for BPF programs by kernel verifier)
