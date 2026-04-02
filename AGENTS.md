# SCX (sched_ext) - AI Agent Guide

This document provides essential context for AI agents working with the SCX (sched_ext) repository - a Linux kernel feature enabling kernel thread schedulers to be implemented in BPF and dynamically loaded.

## Project Overview

**What is sched_ext?**
- A Linux kernel feature (upstream since 6.12) that enables implementing kernel thread schedulers in BPF
- Allows safe, rapid iteration of scheduler implementations without kernel recompilation/reboot
- Provides BPF struct_ops callbacks (similar to `struct sched_class`) for implementing scheduling policies
- Used in production at Meta and Google

**Key Benefits:**
- Safe experimentation: BPF verifier prevents crashes and corruption
- Rapid deployment: Load/unload schedulers by running/terminating binaries
- Customization: Build application-specific schedulers
- Production-ready: Can be used for real workloads with acceptable performance

## Repository Structure

```
scx/
├── scheds/                    # Scheduler implementations
│   ├── include/              # Shared BPF headers, vmlinux.h, lib/
│   │   ├── lib/              # BPF library headers (cpumask, bitmap, etc.)
│   │   ├── bpf_*.h           # BPF helper headers
│   │   └── scx/              # Scheduler-specific includes
│   └── rust/                 # Rust-based schedulers
│       ├── scx_bpfland       # Production-ready vtime scheduler (BPF-only logic)
│       ├── scx_rustland      # Userspace scheduling framework (Rust-based policy)
│       ├── scx_rusty         # Production multi-domain work-stealing scheduler
│       ├── scx_layered       # Highly configurable multi-layer hybrid scheduler
│       ├── scx_lavd          # Gaming/media-focused low-latency scheduler
│       ├── scx_flash         # Fast scheduler focused on simplicity
│       ├── scx_p2dq          # Priority-weighted dual-queue scheduler
│       ├── scx_tickless      # Tickless scheduling for power saving
│       ├── scx_chaos         # Chaos engineering/testing scheduler
│       └── ... (more)
├── lib/                       # BPF library code (*.bpf.c files)
│   ├── scxtest/              # BPF unit testing framework
│   ├── selftests/            # Selftest BPF programs
│   ├── *.bpf.c               # Reusable BPF libraries (btree, rbtree, etc.)
│   └── alloc/                # Memory allocator tests
├── rust/                      # Rust support crates
│   ├── scx_utils/            # Core utilities (topology, cpumask, build helpers)
│   ├── scx_rustland_core/    # Framework for userspace schedulers
│   ├── scx_stats/            # Statistics transport library (UNIX socket)
│   ├── scx_stats_derive/     # Procedural macros for scx_stats
│   ├── scx_cargo/            # Build-time BPF compilation support
│   ├── scx_bpf_unittests/    # BPF unit test runner
│   └── ...
├── tools/                     # Development and debugging tools
│   ├── scxtop/               # Top-like observability tool with TUI, traces, MCP
│   ├── scxcash/              # BPF program caching utility
│   └── vmlinux_docify/       # Kernel docs generator
├── scripts/                   # Helper scripts (sched_ftrace.py, etc.)
├── services/                  # systemd service files
└── .nix/                      # Nix packaging
```

## Architecture

### BPF/Rust Split

**Schedulers typically have two components:**

1. **BPF component (`*.bpf.c`)**: Runs in kernel, handles hot paths
   - Implements `sched_ext_ops` callbacks (select_cpu, enqueue, dispatch, etc.)
   - Uses DSQs (dispatch queues) for task management
   - Must be GPL-licensed (enforced by BPF verifier)

2. **Rust component (`main.rs`)**: Userspace, handles complex/cold operations
   - CLI argument parsing, statistics display
   - Can implement full scheduling policy (for rustland schedulers)
   - Communicates with BPF via maps and ring buffers

### Core sched_ext Concepts

**Dispatch Queues (DSQs):**
- `SCX_DSQ_GLOBAL`: Built-in global FIFO queue
- `SCX_DSQ_LOCAL`: Built-in per-CPU queue (always consumed first)
- Custom DSQs: Created with `scx_bpf_create_dsq()`, can be FIFO or priority-based

**Key Callbacks in `sched_ext_ops`:**
- `select_cpu()`: Choose target CPU for waking task (can direct-dispatch)
- `enqueue()`: Handle newly runnable task (dispatch or queue in BPF)
- `dispatch()`: Called when CPU needs more work (consume from DSQs)
- `running()`: Task starts executing
- `stopping()`: Task stops executing (yields/timeslice expires)
- `init_task()`: Task first enters sched_ext
- `exit_task()`: Task exits

**Important BPF Helper Functions:**
- `scx_bpf_dispatch()`: Dispatch task to DSQ (FIFO)
- `scx_bpf_dispatch_vtime()`: Dispatch with vtime (priority queue)
- `scx_bpf_consume()`: Consume from non-local DSQ to local DSQ
- `scx_bpf_kick_cpu()`: Wake up idle CPU
- `scx_bpf_select_cpu_dfl()`: Get default CPU selection

## Build System

### Building Rust Schedulers

```bash
# Build everything
cargo build --release

# Build specific scheduler
cargo build --release -p scx_rusty

# Available profiles in Cargo.toml:
# - release (default, thin LTO)
# - release-tiny (stripped, small size)
# - release-fast (fast compile, no LTO)
cargo build --profile=release-tiny -p scx_flash
```

### Environment Variables for BPF Compilation

- `BPF_CLANG`: Clang command (default: `clang`, recommend >=17)
- `BPF_CFLAGS`: Override all compiler flags
- `BPF_BASE_CFLAGS`: Override base flags (non-include)
- `BPF_EXTRA_CFLAGS_PRE_INCL`: Extra flags before includes
- `BPF_EXTRA_CFLAGS_POST_INCL`: Extra flags after includes

Example:
```bash
BPF_CLANG=clang-17 cargo build --release -p scx_bpfland
```

### Installing from crates.io

```bash
cargo install scx_rusty
cargo install scxtop
```

## Key Libraries and Crates

### scx_utils (Core Utilities)

**Build Utilities:**
- `BpfBuilder` (now in `scx_cargo`): Automates BPF compilation in build.rs

**Runtime Utilities:**
- `Topology`: CPU topology detection (cores, LLCs, NUMA nodes)
- `Cpumask`: CPU mask operations
- `compat`: Kernel compatibility checking
- `ravg`: Running average calculations
- `infeasible`: Load balancing weight calculations
- `libbpf_logger`: BPF logging integration
- `user_exit_info`: Graceful exit handling

### scx_rustland_core

Framework for implementing schedulers in userspace Rust:
- `BpfScheduler`: Core interface to BPF component
- `dequeue_task()`: Get tasks needing scheduling
- `dispatch_task()`: Send tasks to CPUs
- `select_cpu()`: Find idle CPU for task

Example usage pattern in `scx_rlfifo` (simple round-robin).

### scx_stats

Statistics transport over UNIX domain socket:
- `#[derive(Stats)]` macro for defining stats structs
- `ScxStatsServer`: Serve stats to external tools
- `ScxStatsClient`: Query stats from scheduler
- Supports OpenMetrics integration via annotations (`_om_prefix`, `_om_label`)

### BPF Libraries (lib/*.bpf.c)

Reusable BPF data structures:
- `btree.bpf.c`: B-tree implementation
- `rbtree.bpf.c`: Red-black tree
- `minheap.bpf.c`: Min-heap for priority queues
- `bitmap.bpf.c`: Bitmap operations
- `topology.bpf.c`: Topology helpers
- `cpumask.bpf.c`: CPU mask operations in BPF
- `sdt_alloc.bpf.c`: Specialized task allocator
- `cgroup_bw.bpf.c`: Cgroup bandwidth management

## Development Patterns

### Typical Scheduler Structure

```
scheds/rust/scx_mysCHED/
├── src/
│   ├── main.rs           # Entry point, CLI, stats loop
│   ├── bpf/
│   │   ├── main.bpf.c    # BPF scheduler implementation
│   │   └── intf.h        # BPF/Rust interface definitions
│   ├── bpf_intf.rs       # Generated Rust bindings for intf.h
│   ├── bpf_skel.rs       # Generated skeleton (libbpf)
│   └── stats.rs          # Statistics definitions
├── Cargo.toml
├── build.rs              # BPF compilation setup
└── README.md
```

### build.rs Pattern

```rust
use scx_cargo::BpfBuilder;

fn main() {
    BpfBuilder::new()
        .unwrap()
        .compile_link_gen("main.bpf.c")  // Compile BPF, generate bindings
        .unwrap();
}
```

### BPF Code Structure

```c
#include <scx/common.bpf.h>

// Define scheduler operations
SEC("struct_ops")
struct sched_ext_ops my_sched_ops = {
    .select_cpu  = (void *)my_select_cpu,
    .enqueue     = (void *)my_enqueue,
    .dispatch    = (void *)my_dispatch,
    .running     = (void *)my_running,
    .stopping    = (void *)my_stopping,
    .init_task   = (void *)my_init_task,
    .exit_task   = (void *)my_exit_task,
    .name        = "my_scheduler",
};
```

## Testing

### BPF Unit Tests

1. Create `main.test.bpf.c` alongside `main.bpf.c`:
```c
#include <scx_test.h>
#include "main.bpf.c"

SCX_TEST(test_my_function)
{
    scx_test_assert(my_function(5) == 5);
}
```

2. Add to `rust/scx_bpf_unittests/build.rs`

3. Run: `cargo test -p scx_bpf_unittests`

### Selftests

Run `cargo test` for the entire workspace. Individual scheduler tests:
```bash
cargo test -p scx_flash
```

## Running Schedulers

```bash
# Load and run scheduler (requires root)
sudo scx_bpfland

# Run with monitoring
sudo scx_rusty --monitor 5

# Check scheduler status
cat /sys/kernel/sched_ext/state
cat /sys/kernel/sched_ext/*/ops

# Unload: Ctrl-C or kill the process
```

## Debugging and Observability

### scxtop (Primary Tool)

Three modes:
- **TUI Mode**: `sudo scxtop` - Interactive system monitoring
- **Trace Mode**: `sudo scxtop trace --duration 30` - Generate Perfetto traces
- **MCP Mode**: `sudo scxtop mcp --daemon` - AI assistant integration

### Other Tools

- `perf sched`: Timeline and latency analysis
- `bpftool`: List BPF programs, maps, struct_ops
- `bpftrace`: High-level BPF tracing (scripts in `scripts/`)
- `veristat`: BPF verifier statistics
- `retsnoop`: Kernel function flow tracing
- `systing`: Generate perfetto traces with stack traces

### Reading Scheduler Stats

```bash
# Most schedulers support --monitor or --stats
scx_bpfland --monitor 0.5
scx_rusty --stats 5
```

## Kernel Requirements

**Minimum kernel:** 6.12 (when sched_ext was upstreamed)

**Required kernel configs:**
```
CONFIG_BPF=y
CONFIG_BPF_SYSCALL=y
CONFIG_BPF_JIT=y
CONFIG_DEBUG_INFO_BTF=y
CONFIG_BPF_JIT_ALWAYS_ON=y
CONFIG_SCHED_CLASS_EXT=y
```

See `kernel.config` for complete recommended configuration.

## Common Pitfalls

### BPF Verifier Issues
- BPF programs must pass verifier - avoid unbounded loops, ensure null checks
- Use `bpf_loop()` for iteration with bounded complexity
- Watch for stack size limits (~512 bytes)

### Scheduling Deadlocks
- Never hold locks across `scx_bpf_dispatch()` or `scx_bpf_consume()`
- Always ensure tasks eventually get dispatched
- Watch for the watchdog timer (detects starved tasks)

### Memory Ordering
- BPF uses relaxed memory model - use atomics appropriately
- `__sync_fetch_and_add()` for counters, proper barriers when needed

### API Stability
- No ABI stability guarantee for BPF kfuncs (like other semi-internal interfaces)
- Check `BREAKING_CHANGES.md` for API updates

## Key Documentation

- `README.md`: Getting started, install by distro
- `OVERVIEW.md`: Detailed motivation and architecture
- `DEVELOPER_GUIDE.md`: Development tools and resources
- `CARGO_BUILD.md`: Build system details
- `UNIT_TESTING_GUIDE.md`: BPF unit testing
- `BREAKING_CHANGES.md`: API compatibility notes
- `DHQ_README.md`: Double Helix Queue data structure
- `scheds/rust/*/README.md`: Per-scheduler documentation
- `tools/scxtop/README.md`: Observability tool docs

## Important File Locations

| File | Purpose |
|------|---------|
| `scheds/include/scx/common.bpf.h` | Main BPF header |
| `scheds/vmlinux/vmlinux.h` | Kernel structure definitions |
| `lib/*.bpf.c` | Reusable BPF libraries |
| `rust/scx_utils/src/topology.rs` | Topology detection |
| `rust/scx_utils/src/cpumask.rs` | CPU mask operations |
| `rust/scx_rustland_core/src/` | Userspace scheduler framework |
| `scripts/sched_ftrace.py` | Generate Perfetto traces |

## Community

- GitHub: https://github.com/sched-ext/scx
- Discord: https://discord.gg/b2J8DrWa7t
- Weekly office hours: Tuesdays (see Discord #office-hours)
- Mailing list: sched-ext@lists.linux.dev (kernel development)

## License

All code is GPL-2.0 (required for BPF programs by kernel verifier).
