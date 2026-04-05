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
   - **Gaming**: Prioritize low latency (20ms response, α=8, β=4, 8% ramp-up)
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

#### Core Scheduling Options
- `-s, --slice-us <US>` - Maximum scheduling slice duration in microseconds [default: 700]
- `-l, --slice-us-lag <US>` - Maximum runtime budget for sleeping tasks in microseconds [default: 20000]
- `-t, --throttle-us <US>` - Throttle CPUs by injecting idle cycles [default: 0]
- `-I, --idle-resume-us <US>` - Set CPU idle QoS resume latency in microseconds (-1 = disabled) [default: -1]
- `-T, --tickless` - Enable tickless mode
- `-R, --rr-sched` - Enable round-robin scheduling

#### CPU Domain Options
- `-m, --primary-domain <DOMAIN>` - Primary CPU domain (auto, powersave, performance, turbo, all, or hex mask) [default: auto]
- `--disable-smt` - Disable SMT awareness
- `--disable-numa` - Disable NUMA rebalancing
- `-f, --cpufreq` - Enable CPU frequency control (only with schedutil governor)

#### Profile and Tuning Options
- `-p, --profile <PROFILE>` - Profile selection: gaming, productivity, server [default: productivity]
- `--response-ms <MS>` - Override profile response interval (lower = faster updates)
- `--pie-alpha <N>` - Override PIE alpha (proportional gain divisor)
- `--pie-beta <N>` - Override PIE beta (integral gain divisor)
- `--target-latency-critical <US>` - Override target latency for LATENCY_CRITICAL class (microseconds)
- `--target-latency-normal <US>` - Override target latency for NORMAL class (microseconds)
- `--target-latency-hog <US>` - Override target latency for HOG class (microseconds)
- `--target-latency-background <US>` - Override target latency for BACKGROUND class (microseconds)
- `--audio-cgroup <PATH>` - Audio cgroup path for automatic classification
- `--update-interval-ms <MS>` - Update interval for parameter sync [default: 50]

#### Debug Options
- `--debug-pie <N>` - Enable PIE controller debug output every N milliseconds
- `--debug-bounds <N>` - Enable parameter bounds debugging every N milliseconds
- `-d, --debug` - Enable BPF debugging via /sys/kernel/tracing/trace_pipe
- `-v, --verbose` - Enable verbose output including libbpf details

#### Help and Stats Options
- `--help-profiles` - Show detailed profile parameter defaults and exit
- `--help-stats` - Show descriptions for statistics and exit
- `--stats <INTERVAL>` - Enable stats monitoring with specified interval in seconds
- `--monitor <INTERVAL>` - Run in stats monitoring mode (no scheduler)
- `-V, --version` - Print scheduler version and exit
- `-h, --help` - Print help

### Monitoring

```bash
# Monitor scheduler statistics
sudo scx_descent --stats 1

# Run in monitor mode (no scheduler)
scx_descent --monitor 1
```

### SCX_DESCENT_TURBO Environment Variable

scx_descent automatically detects processes with the `SCX_DESCENT_TURBO` environment variable set to a non-empty, non-zero value. These processes receive the highest scheduling priority:

- **Automatic classification**: Turbo processes are always classified as `LATENCY_CRITICAL`
- **Reduced time slices**: Turbo tasks receive half the normal slice for faster preemption
- **Deadline boost**: Earlier virtual deadlines (higher priority within their class)
- **SMT conflict avoidance**: Non-turbo tasks on SMT siblings of turbo tasks are migrated away or deprioritized with additional vruntime penalties

**Usage:**

```bash
# Run a single command with turbo priority
SCX_DESCENT_TURBO=1 ./benchmark

# Export for multiple commands
export SCX_DESCENT_TURBO=1
./app1 &
./app2 &
```

**How it works:**
- The scheduler scans `/proc/*/environ` every 5 seconds
- Processes with `SCX_DESCENT_TURBO=1` (or any non-zero value) are identified
- All threads in the process (matching TGID) receive turbo priority
- Changes are applied dynamically without restarting the scheduler

### Automatic Process Detection

scx_descent automatically detects and prioritizes several types of system processes by scanning `/proc`:

#### Audio Daemon Detection
Automatically detects and prioritizes audio daemons by scanning for known audio process names:
- **pipewire**, **wireplumber**, **pipewire-pulse** - Modern Linux audio stack
- **pulseaudio** - Legacy PulseAudio server
- **jackd**, **jackdbus** - JACK audio server

Detected audio processes are classified as `LATENCY_CRITICAL` to ensure uninterrupted audio playback.

#### Input Kworker Detection
Detects input-related kernel worker threads by identifying processes with parent PID 2 (kthreadd) and matching patterns:
- `ksoftirqd/*` - Deferred interrupt handlers (critical for input latency)
- `hid-*` - HID (Human Interface Device) workers
- `usbhid` - USB HID workers
- `input_*` - Input event handlers
- `irq/*` - IRQ workers for input devices

These threads are critical for input latency and receive priority scheduling.

#### ksoftirqd Detection
Dedicated detection for ksoftirqd threads which handle the bottom half of interrupt processing, including input device interrupts. All ksoftirqd threads (named `ksoftirqd/N` where N is the CPU number) are detected and prioritized.

#### Desktop Environment Detection
Detects running Desktop Environment components by process name to ensure UI responsiveness:

- **GNOME**: gnome-shell, gnome-panel, nautilus
- **KDE Plasma**: plasmashell, kwin_wayland, kwin_x11, plasma-desktop, dolphin
- **Sway**: sway, swaybar
- **Hyprland**: Hyprland
- **XFCE**: xfce4-panel, xfwm4, xfdesktop, thunar
- **i3/sway**: i3, i3bar
- **MATE**: marco, mate-panel, caja
- **Cinnamon**: cinnamon, muffin, nemo
- **LXQt**: lxqt-panel, pcmanfm-qt, pcmanfm
- **Budgie**: budgie-panel, budgie-wm
- **Wayfire**: wayfire
- **Weston**: weston
- **Gamescope**: gamescope (Steam Deck UI)
- **Pantheon**: gala, wingpanel

DE components are promoted to `LATENCY_CRITICAL` during non-GAMING states to ensure desktop responsiveness.

#### Game Detection
Detects game processes to automatically enter GAMING state:
- **Steam games**: Detected via `SteamGameId=` or `STEAM_GAME=` environment variables
- **Wine/Proton games**: Detected via `.exe` files in command line

When a game is detected, the scheduler enters GAMING state and prioritizes the game process.

This is ideal for benchmarks, real-time applications, or any workload that needs guaranteed low latency regardless of system load.

## Profile Configuration

| Profile | Response | Alpha | Beta | Use Case |
|---------|----------|-------|------|----------|
| Gaming | 20ms | 8 | 4 | Fast response for games/audio |
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
- ✅ **Phase 4**: PIE controller with deterministic latency-based optimization (fully operational)
  - Per-class parameter optimization
  - Safety mechanisms and parameter bounds
  - Automatic task classification with advanced detection (audio, input, DE, games)
- ⚠️ **Phase 5** (Future): Advanced features like workload prediction, container awareness, hardware-specific profiles, dynamic reflector selection

**Production Ready?**

**Yes** - The scheduler is functional and includes adaptive parameter optimization via PIE controller with automatic task classification. The scheduler includes safety mechanisms and has been tested with extensive unit tests. Extensive benchmarking against production workloads is recommended before deployment.

## License

GPL-2.0 (required for BPF programs by kernel verifier)
