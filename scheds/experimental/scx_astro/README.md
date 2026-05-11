# scx_astro — Hybrid Classification Multi-Lane Scheduler

`scx_astro` is an experimental sched_ext scheduler combining **automatic behavioral task classification** (inspired by scx_lavd) with **explicit user overrides via environment variables** (inspired by scx_turbo's documented design). It uses a **multi-lane DSQ architecture** where interactive tasks get fast lanes, background tasks are contained, and normal/compute tasks share throughput-oriented lanes.

## Novel Features

- **SRPT-inspired prioritization**: Approximates Shortest Remaining Processing Time by scaling vruntime for short tasks within the normal lane.
- **Waker-boost chains**: Tasks woken by interactive tasks receive temporary priority boosts into a dedicated fast lane.
- **Dynamic lane routing**: Tasks transition between lanes based on behavioral changes, with hysteresis to prevent thrashing.
- **Hybrid classification**: Auto-detects interactive/compute/background behavior using wait/wake frequencies and runtime heuristics, while allowing explicit `SCX_ASTRO=<profile>` overrides.

## Lane Architecture

| Lane | DSQ ID | Purpose | Slice |
|------|--------|---------|-------|
| Interactive | 1020 | Latency-sensitive tasks (UI, audio, input) | 50–200 µs |
| Waker Boost | 1021 | Temporary boost for wakees of interactive tasks | 100 µs |
| Normal | 1022 | Default tasks, SRPT-ordered via vtime scaling | 0.5–2 ms |
| Compute | 1023 | Long-running compute jobs | 2–5 ms |
| Background | 1024 | Contained lane; runs only when fast lanes empty | 100–500 µs |

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

## Environment Variable Override

Set `SCX_ASTRO` to one of:
- `interactive` — Forces task into the interactive fast lane
- `normal` — Forces task into the normal lane
- `compute` — Forces task into the compute lane
- `background` — Forces task into the contained background lane

The userspace component scans `/proc/*/environ` periodically to discover these overrides and communicates them to BPF via a hash map.

## Architecture

See [DESIGN.md](DESIGN.md) for the complete design document with code snippets and rationale.

## License

GPL-2.0-only
