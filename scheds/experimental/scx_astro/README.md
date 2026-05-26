# scx_astro — Hybrid Classification Multi-Lane Scheduler

`scx_astro` is an experimental sched_ext scheduler combining **automatic behavioral task classification** (inspired by scx_lavd) with **infrastructure for explicit user overrides via environment variables** (inspired by scx_turbo's documented design). It uses a **multi-lane DSQ architecture** where interactive tasks get fast lanes, background tasks are contained, and normal/compute tasks share throughput-oriented lanes.

> **Note:** The BPF-side map (`tgid_profile_map`) is ready to receive `SCX_ASTRO=<profile>` overrides, but the userspace `/proc/*/environ` scanning component is not yet implemented. Auto-classification is fully functional.

## Novel Features

- **SRPT-inspired prioritization**: Approximates Shortest Remaining Processing Time by applying a transient vtime bonus (negative offset) to short tasks in the normal and compute lanes, moving them earlier in EDF ordering without permanently distorting cumulative vruntime.
- **Waker-boost chains**: Tasks woken by interactive tasks receive temporary priority boosts into a dedicated fast lane.
- **Dynamic lane routing**: Tasks transition between lanes based on behavioral changes, with hysteresis to prevent thrashing.
- **Hybrid classification**: Auto-detects interactive/compute/background behavior using wait/wake frequencies and runtime heuristics, with BPF-side infrastructure prepared for explicit `SCX_ASTRO=<profile>` overrides (userspace scanning pending).

## Lane Architecture

| Lane | DSQ ID | Purpose | Slice |
|------|--------|---------|-------|
| Interactive | 1020 | Latency-sensitive tasks (UI, audio, input) | 50–200 µs |
| Waker Boost | 1021 | Temporary boost for wakees of interactive tasks | 100 µs |
| Normal | 1022 | Default tasks, SRPT-ordered via transient vtime bonus | 0.5–2 ms |
| Compute | 1023 | Long-running compute jobs | 3 ms default (tunable, min 50 µs) |
| Background | 1024 | Contained lane; runs only when fast lanes empty | 500 µs default (tunable, min 50 µs) |

## Usage

```bash
# Build
cargo build -p scx_astro

# Run with defaults
sudo ./target/debug/scx_astro

# Enable stats and debug
sudo ./target/debug/scx_astro --stats 1 --debug

# Explicit profile override (for a shell and its children)
SCX_ASTRO=interactive ./my-app
```

## CLI Options

```
--stats <f64>          Enable stats monitoring with interval (seconds)
--monitor <f64>        Run in monitor-only mode (no scheduler)
--debug, -d            Enable BPF debug printk
--version, -V          Print version and exit
--no-autotune          Disable adaptive runtime tuning
--completions <SHELL>  Generate shell completions
```

## Environment Variable Override (Planned)

When implemented, users will be able to set `SCX_ASTRO` to one of:
- `interactive` — Forces task into the interactive fast lane
- `normal` — Forces task into the normal lane
- `compute` — Forces task into the compute lane
- `background` — Forces task into the contained background lane

The BPF map `tgid_profile_map` is already defined and checked in `effective_profile()`, but the userspace component that scans `/proc/*/environ` and pushes overrides into BPF is **not yet implemented**. Auto-classification works independently of this feature.

## Architecture

See [DESIGN.md](DESIGN.md) for the complete design document with code snippets and rationale.

## License

GPL-2.0-only
