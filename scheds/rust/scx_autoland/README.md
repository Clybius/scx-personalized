# scx_autoland

An adaptive CPU scheduler for Linux using the `sched_ext` framework, evolved from scx_beerland with a **PID Tail-Latency-Target Based Controller**.

## Overview

scx_autoland automatically classifies tasks into three categories and applies adaptive scheduling policies based on **real-time P99 latency measurements**:

- **LATENCY_CRITICAL (LC)**: Gaming workloads, audio tasks, tasks with SCX_TURBO=1, desktop environment components
- **NORMAL**: Interactive desktop applications  
- **HOG**: CPU-intensive background tasks

## Key Improvements Over scx_beerland

### 🎯 PID Tail-Latency-Target Based Controller

Unlike beerland's **load-based controller** (which adjusts slices based on task counts), scx_autoland uses a sophisticated **PID controller** that:

- **Targets P99 latencies directly** (not just average latency)
  - LC target: 500μs P99
  - Normal target: 2000μs P99
  - Hog target: 50ms max P99
  
- **Uses control theory for smooth adaptation**
  - Proportional-Integral-Derivative (PID) control prevents oscillation
  - Per-class tuning: aggressive for LC, stable for Hog
  
- **EMA-based tracking** (Exponential Moving Average)
  - Natural decay of historical data (no periodic resets needed)
  - Automatically rebounds when load ends
  - Self-correcting without BPF modifications

### 📊 EMA Histogram for P99 Calculation

- **16 logarithmic latency buckets** (1μs to ~8.3ms range)
- **Exponential decay** of old samples (α=0.2: ~11% weight after 10 intervals)
- **No cumulative data problem** - old measurements fade naturally
- **Responsive to workload changes** - rebounds within 30 seconds after load ends

### 🎮 Enhanced Task Detection

- **One-time detection logging**: Logs when games/turbo tasks are detected and when they end
- **Automatic Steam game detection**: Checks `/proc/{pid}/environ` for `SteamGameId=`
- **SCX_TURBO environment variable**: Mark any process as latency-critical
- **Desktop environment and audio process recognition**

### ⚡ SMT-Aware Scheduling

- Migrates non-critical tasks away from SMT siblings of latency-critical tasks
- Full-idle-core preference for critical workloads
- Topology-aware CPU selection

### 🚀 Latency Optimizations

- Power-of-two random choice preemption (inspired by scx_lavd)
- Lock holder protection (never preempt lock holders)
- Cache stickiness for short-running tasks
- Wake-to-wake affinity
- **Priority-proportional slices** via `scale_by_task_weight()`

## Usage

```bash
# Run with defaults
sudo scx_autoland

# Adjust slice durations with ratios
sudo scx_autoland -s 700 --slice-ratios 0.5,1.0,2.0

# With statistics output (shows PID controller updates)
sudo scx_autoland --stats 1

# Disable adaptive controller (fixed parameters)
sudo scx_autoland --no-adaptive

# Debug output
sudo scx_autoland -d
```

## Command-Line Options

### Slice Configuration
- `-s, --slice-us <US>`: Base slice duration in microseconds (default: 2000)
- `--slice-ratios <RATIOS>`: Per-class ratios as "LC,Normal,Hog" (default: "0.25,1.0,2.0")
  - Example: `--slice-ratios 0.5,1.0,2.0` with base 700μs gives LC=350μs, Normal=700μs, Hog=1400μs

### Controller Options
- `--no-adaptive`: Disable PID controller (use fixed slices)
- `--stats <SECS>`: Enable statistics output every N seconds (shows controller updates)

### CPU Configuration
- `--cpu-busy-thresh <PCT>`: CPU busy threshold percentage (default: 75)
- `-p, --primary-cpus <CPUS>`: CPUs to use for primary scheduling domain

### Debug
- `-d, --debug`: Enable debug output

## Marking Tasks as Latency-Critical

```bash
# Using environment variable
SCX_TURBO=1 ./my_game

# Steam games are detected automatically
steam steam://rungameid/570  # Dota 2

# View detection logs
# Output: "Detected Steam game: PID=12345 comm=dota2"
# When game ends: "Steam game ended: PID=12345"
```

## How the PID Controller Works

### Control Loop

1. **Measure P99 latency** from EMA histogram (every control interval)
2. **Calculate error** = (measured_P99 - target_P99) / target_P99
3. **Apply LAVD-style criticality weighting** (wakeup frequency + runtime variance)
4. **Compute PID output** (proportional + integral + derivative terms)
5. **Adjust slices** to minimize latency error

### Example Behavior

```
Startup:            LC=500μs (at target)
FFMPEG starts:      LC=400μs (reduces due to +20% latency error)
During FFMPEG:      LC stabilizes at ~400-450μs (EMA tracks sustained load)
FFMPEG ends:        LC=500μs (rebounds within 30s via EMA decay)
```

### PID Tuning Per Class

| Class | Kp (Proportional) | Ki (Integral) | Kd (Derivative) | Strategy |
|-------|------------------|---------------|-----------------|----------|
| LC | 1.0 | 0.05 | 0.3 | Aggressive, fast response |
| Normal | 0.5 | 0.1 | 0.2 | Balanced |
| Hog | 0.3 | 0.15 | 0.1 | Slow, stable |

## Implementation Details

Forked from [scx_beerland](https://github.com/sched-ext/scx/tree/main/scheds/rust/scx_beerland) with major enhancements:

### New Components
- **PID Tail-Latency-Target Controller** (`pid_tail_latency_controller.rs`)
- **EMA Histogram Tracking** (pure userspace, no BPF resets)
- **LAVD-style Criticality Scoring** [0, 1024]

### Inspired By
- **scx_lavd**: Latency criticality calculation, preemption strategies
- **scx_cake**: Steam game detection, per-class scheduling concepts
- **scx_pandemonium**: Adaptive regime control ideas
- **scx_flash/bpfland**: SMT-aware idle CPU selection

## Architecture

```
BPF (Kernel)                    Userspace (Rust)
─────────────────────────────────────────────────────
task_slice()                    PidTailLatencyController
  ↓                               ↓
scale_by_task_weight()           EmaHistogram
  ↓                               ↓
Per-class slice_ns_*             EmaPidController
  ↓                               ↓
Histogram tracking               P99 calculation
  ↓                               ↓
class_latency_stats map          Slice adjustment
```

## Production Ready?

**Experimental** - in active development. Testing recommended before production use.

## Requirements

- Linux kernel with `sched_ext` support (6.12+ recommended)
- Rust toolchain
- BPF development libraries (libbpf)

## License

GPL-2.0-only
