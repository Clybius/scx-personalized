/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2024 Andrea Righi <andrea.righi@linux.dev>
 * 
 * scx_descent: Gradient descent-based adaptive scheduler
 * Phase 1: Basic structure with classification and per-class parameters
 */
#include <scx/common.bpf.h>
#include "intf.h"
#include "descent.bpf.h"

#define MAX_VTIME (~0ULL)

#define CGROUP_WEIGHT_DFL 100

#define CONFIG_HZ 100

#define MAX_WAKEUP_FREQ 100

#define SLICE_MIN_NS (10ULL * NSEC_PER_USEC)

/*
 * Bit position for cached kthread flag in task_ctx->packed
 */
#define BIT_KTHREAD 23 /* Cached PF_KTHREAD from task flags */

/*
 * Bit position for turbo task flag in task_ctx->packed
 * Turbo tasks get highest priority scheduling
 */
#define BIT_TURBO 22 /* Task is a turbo-boosted process */

/*
 * Task classification thresholds
 */
#define WAKEUP_FREQ_INTERACTIVE_THRESH 1000
#define SLICE_NS_INTERACTIVE_THRESH (1ULL * NSEC_PER_MSEC)
#define WAKEUP_FREQ_BATCH_THRESH 10
#define SLICE_NS_BATCH_THRESH (10ULL * NSEC_PER_MSEC)

/*
 * Scheduling policy constants (from linux/sched.h)
 * These are needed for task classification
 */
#define SCHED_OTHER 0
#define SCHED_FIFO 1
#define SCHED_RR 2
#define SCHED_BATCH 3
#define SCHED_IDLE 5

/*
 * Return the time interval between two ticks in ns.
 */
static inline u64 tick_interval_ns(void)
{
	return NSEC_PER_SEC / CONFIG_HZ;
}

/*
 * Thresholds for applying hysteresis to CPU performance scaling:
 *  - CPUFREQ_LOW_THRESH: below this level, reduce performance to minimum
 *  - CPUFREQ_HIGH_THRESH: above this level, raise performance to maximum
 *
 * Values between the two thresholds retain the current smoothed performance level.
 */
#define CPUFREQ_LOW_THRESH (SCX_CPUPERF_ONE / 4)
#define CPUFREQ_HIGH_THRESH (SCX_CPUPERF_ONE - SCX_CPUPERF_ONE / 4)

char _license[] SEC("license") = "GPL";

/* Allow to use bpf_printk() only when @debug is set */
#define dbg_msg(_fmt, ...)                               \
	do {                                             \
		if (debug)                               \
			bpf_printk(_fmt, ##__VA_ARGS__); \
	} while (0)

/* Report additional debugging information */
const volatile bool debug;

/* Enable round-robin mode */
const volatile bool rr_sched;

/* Primary domain includes all CPU */
const volatile bool primary_all = true;

/*
 * Default task time slice (will be overridden by per-class params in descent).
 */
const volatile u64 slice_max = 700ULL * NSEC_PER_USEC;

/*
 * Maximum runtime budget that a task can accumulate while sleeping (used
 * to determine the task's minimum vruntime).
 */
const volatile u64 slice_lag = 20ULL * NSEC_PER_MSEC;

/*
 * Adjust the maximum sleep budget in function of the average CPU
 * utilization.
 */
const volatile bool slice_lag_scaling;

/*
 * Enable tickless mode.
 */
const volatile bool tickless_sched;

/*
 * The CPU frequency performance level: a negative value will not affect the
 * performance level and will be ignored.
 */
volatile s64 cpufreq_perf_lvl;

/*
 * Scheduling statistics.
 */
volatile u64 nr_kthread_dispatches, nr_direct_dispatches, nr_shared_dispatches;

/*
 * Amount of currently running tasks.
 */
volatile u64 nr_running;

/*
 * Amount of online CPUs.
 */
volatile u64 nr_online_cpus;

/*
 * Maximum possible CPU number.
 */
static u64 nr_cpu_ids;

/*
 * Runtime throttling.
 *
 * Throttle the CPUs by injecting @throttle_ns idle time every @slice_max.
 */
const volatile u64 throttle_ns;
static volatile u8 cpus_throttled;

/*
 * State machine variables - written by userspace, read by BPF
 * These need to be volatile since they're modified from userspace
 */
volatile u32 game_tgid; // Game process TGID
volatile u32 game_ppid; // Parent PID for Wine/Proton family
volatile u8  game_confidence; // 100=Steam, 90=Wine, 0=none
volatile u32 sched_state; // 0=IDLE, 1=COMPILATION, 2=GAMING
volatile u32 audio_tgids[16]; // Protected audio daemon TGIDs
volatile u32 nr_audio_tgids; // Number of valid audio TGIDs

/*
 * Turbo process tracking - processes with SCX_DESCENT_TURBO env var get
 * highest priority scheduling
 */
volatile u32 turbo_tgids[16]; // Turbo process TGIDs (array)
volatile u32 nr_turbo_tgids; // Number of valid turbo TGIDs

/*
 * Desktop Environment process tracking - compositors and shell processes get
 * elevated priority for responsive UI
 */
volatile u32	 de_tgids[16]; // Desktop Environment process TGIDs (array)
volatile u32	 nr_de_tgids; // Number of valid DE TGIDs
volatile u8	 de_detected; // Flag indicating if DE is currently active

static inline u8 is_throttled(void)
{
	if (!throttle_ns)
		return 0;

	return READ_ONCE(cpus_throttled) ? 1 : 0;
}

static inline void set_throttled(u8 state)
{
	WRITE_ONCE(cpus_throttled, state);
}

/*
 * Exit information.
 */
UEI_DEFINE(uei);

/*
 * Mask of CPUs that the scheduler can use until the system becomes saturated,
 * at which point tasks may overflow to other available CPUs.
 */
private(DESCENT) struct bpf_cpumask __kptr *primary_cpumask;

/*
 * CPUs in the system have SMT is enabled.
 */
const volatile bool smt_enabled = true;

/*
 * Disable NUMA optimizations.
 */
const volatile bool numa_disabled = false;

/*
 * Current global vruntime.
 */
static u64 vtime_now;

/*
 * Timer used to update NUMA statistics.
 */
struct numa_timer {
	struct bpf_timer timer;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct numa_timer);
} numa_timer SEC(".maps");

/*
 * Timer used to inject idle cycles when CPU throttling is enabled.
 */
struct throttle_timer {
	struct bpf_timer timer;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct throttle_timer);
} throttle_timer SEC(".maps");

/*
 * Timer used to preempt CPUs in tickless mode.
 */
struct tickless_timer {
	struct bpf_timer timer;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct tickless_timer);
} tickless_timer SEC(".maps");

/*
 * Per-CPU context.
 */
struct cpu_ctx {
	u64			   tot_runtime;
	u64			   prev_runtime;
	u64			   last_running;
	u64			   perf_lvl;
	struct bpf_cpumask __kptr *smt;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct cpu_ctx);
	__uint(max_entries, 1);
} cpu_ctx_stor SEC(".maps");

/*
 * Syscall-accessible class parameters map.
 * Uses a regular ARRAY map (not PERCPU) so it can be accessed from syscall programs.
 * Indexed by: cpu_id * DESCENT_CLASS_MAX + class_id
 */
#define MAX_CPUS 1024

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAX_CPUS *DESCENT_CLASS_MAX);
	__type(key, u32);
	__type(value, struct class_params);
} class_params_stor SEC(".maps");

/*
 * Per-CPU descent context map - kept for loss accumulators and other per-CPU state.
 * Note: syscall programs cannot use bpf_map_lookup_percpu_elem on this.
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct cpu_descent_ctx);
} cpu_descent_ctx_stor SEC(".maps");

/*
 * Return a CPU context.
 */
struct cpu_ctx *try_lookup_cpu_ctx(s32 cpu)
{
	const u32 idx = 0;
	return bpf_map_lookup_percpu_elem(&cpu_ctx_stor, &idx, cpu);
}

/*
 * Return a CPU descent context.
 */
struct cpu_descent_ctx *try_lookup_cpu_descent_ctx(void)
{
	const u32 idx = 0;
	return bpf_map_lookup_elem(&cpu_descent_ctx_stor, &idx);
}

/*
 * Parameter helper function: read parameters from syscall-accessible map
 * Key is computed as: cpu * DESCENT_CLASS_MAX + class_id
 */
static u32 params_key(u32 cpu, u32 class_id)
{
	return cpu * DESCENT_CLASS_MAX + class_id;
}

/*
 * Parameter helper function: read parameters from class_params_stor map
 * or fall back to default per-CPU params.
 * Also tracks last_param_sync timestamp when reading from syscall-accessible map.
 */
static struct class_params *get_class_params_for_scheduling(u32 class_id)
{
	struct class_params    *cp;
	struct cpu_descent_ctx *cdctx;
	s32			cpu = bpf_get_smp_processor_id();
	u32			key = params_key(cpu, class_id);

	/* Try syscall-accessible map first */
	cp = bpf_map_lookup_elem(&class_params_stor, &key);
	if (cp) {
		/* Track that we've read updated params - helps userspace correlate loss with params */
		cdctx = try_lookup_cpu_descent_ctx();
		if (cdctx)
			cdctx->last_param_sync = bpf_ktime_get_ns();
		return cp;
	}

	/* Fallback to default params from per-CPU context */
	cdctx = try_lookup_cpu_descent_ctx();
	if (cdctx && class_id < DESCENT_CLASS_MAX) {
		return &cdctx->class_params[class_id];
	}

	return NULL;
}

static u64 clamp_param_value(s32 idx, u64 value)
{
	switch (idx) {
	case 0:
		return CLAMP(value, PARAM_MIN_LATENCY_WEIGHT,
			     PARAM_MAX_LATENCY_WEIGHT);
	case 1:
		return CLAMP(value, PARAM_MIN_BASE_SLICE_NS,
			     PARAM_MAX_BASE_SLICE_NS);
	case 2:
		return CLAMP(value, PARAM_MIN_VRUNTIME_SCALE,
			     PARAM_MAX_VRUNTIME_SCALE);
	case 3:
		return CLAMP(value, PARAM_MIN_PREEMPT_PRIO,
			     PARAM_MAX_PREEMPT_PRIO);
	case 4:
		return CLAMP(value, PARAM_MIN_MIGRATION_COST,
			     PARAM_MAX_MIGRATION_COST);
	default:
		return value;
	}
}

/*
 * Initialize class parameters with defaults
 */
static void init_class_params(struct cpu_descent_ctx *cdctx)
{
	/* LATENCY_CRITICAL (Class 0): Games, audio, compositors, kthreads */
	cdctx->class_params[DESCENT_CLASS_LATENCY_CRITICAL].latency_weight =
		LATENCY_CRITICAL_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_LATENCY_CRITICAL].base_slice_ns =
		LATENCY_CRITICAL_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_LATENCY_CRITICAL].vruntime_scale =
		LATENCY_CRITICAL_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_LATENCY_CRITICAL].preemption_priority =
		LATENCY_CRITICAL_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_LATENCY_CRITICAL].migration_cost =
		LATENCY_CRITICAL_MIGRATION_COST_NS;

	/* NORMAL (Class 1): Default interactive */
	cdctx->class_params[DESCENT_CLASS_NORMAL].latency_weight =
		NORMAL_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_NORMAL].base_slice_ns =
		NORMAL_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_NORMAL].vruntime_scale =
		NORMAL_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_NORMAL].preemption_priority =
		NORMAL_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_NORMAL].migration_cost =
		NORMAL_MIGRATION_COST_NS;

	/* HOG (Class 2): High CPU usage */
	cdctx->class_params[DESCENT_CLASS_HOG].latency_weight =
		HOG_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_HOG].base_slice_ns =
		HOG_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_HOG].vruntime_scale =
		HOG_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_HOG].preemption_priority =
		HOG_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_HOG].migration_cost =
		HOG_MIGRATION_COST_NS;

	/* BACKGROUND (Class 3): Low priority */
	cdctx->class_params[DESCENT_CLASS_BACKGROUND].latency_weight =
		BACKGROUND_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_BACKGROUND].base_slice_ns =
		BACKGROUND_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_BACKGROUND].vruntime_scale =
		BACKGROUND_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_BACKGROUND].preemption_priority =
		BACKGROUND_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_BACKGROUND].migration_cost =
		BACKGROUND_MIGRATION_COST_NS;

	cdctx->last_param_sync = bpf_ktime_get_ns();

	/* NEW: Initialize latency accumulators for all classes (instead of loss) */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		cdctx->class_latency[i].total_latency_ns = 0;
		cdctx->class_latency[i].max_latency_ns	 = 0;
		cdctx->class_latency[i].sample_count	 = 0;
	}

	/* NEW: Initialize load accumulators for all classes */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		cdctx->class_load[i].cycles_spent   = 0;
		cdctx->class_load[i].sample_count   = 0;
		cdctx->class_load[i].last_update_ns = bpf_ktime_get_ns();
	}
}

/*
 * Reset latency accumulators for a class after userspace has read them.
 * Called from syscall program.
 */
static void reset_latency_accumulator(struct cpu_descent_ctx *cdctx,
				      u32		      class_id)
{
	if (class_id < DESCENT_CLASS_MAX) {
		cdctx->class_latency[class_id].total_latency_ns = 0;
		cdctx->class_latency[class_id].max_latency_ns	= 0;
		cdctx->class_latency[class_id].sample_count	= 0;
	}
}

/*
 * Reset load accumulators for a class after userspace has read them.
 */
static void reset_load_accumulator(struct cpu_descent_ctx *cdctx, u32 class_id)
{
	if (class_id < DESCENT_CLASS_MAX) {
		cdctx->class_load[class_id].cycles_spent   = 0;
		cdctx->class_load[class_id].sample_count   = 0;
		cdctx->class_load[class_id].last_update_ns = bpf_ktime_get_ns();
	}
}

/*
 * Helper to set default values in class_params_stor for a given CPU and class.
 */
static void init_class_params_stor(u32 cpu, u32 class_id,
				   struct class_params *defaults)
{
	struct class_params *cp;
	u32		     key = params_key(cpu, class_id);

	cp = bpf_map_lookup_elem(&class_params_stor, &key);
	if (cp) {
		cp->latency_weight	= defaults->latency_weight;
		cp->base_slice_ns	= defaults->base_slice_ns;
		cp->vruntime_scale	= defaults->vruntime_scale;
		cp->preemption_priority = defaults->preemption_priority;
		cp->migration_cost	= defaults->migration_cost;
	}
}

/*
 * Initialize syscall-accessible class parameters for a CPU.
 */
static void init_cpu_class_params(u32 cpu)
{
	struct class_params defaults;

	/* LATENCY_CRITICAL (Class 0) */
	defaults.latency_weight	     = LATENCY_CRITICAL_LATENCY_WEIGHT_NS;
	defaults.base_slice_ns	     = LATENCY_CRITICAL_BASE_SLICE_NS;
	defaults.vruntime_scale	     = LATENCY_CRITICAL_VRUNTIME_SCALE;
	defaults.preemption_priority = LATENCY_CRITICAL_PREEMPTION_PRIORITY;
	defaults.migration_cost	     = LATENCY_CRITICAL_MIGRATION_COST_NS;
	init_class_params_stor(cpu, DESCENT_CLASS_LATENCY_CRITICAL, &defaults);

	/* NORMAL (Class 1) */
	defaults.latency_weight	     = NORMAL_LATENCY_WEIGHT_NS;
	defaults.base_slice_ns	     = NORMAL_BASE_SLICE_NS;
	defaults.vruntime_scale	     = NORMAL_VRUNTIME_SCALE;
	defaults.preemption_priority = NORMAL_PREEMPTION_PRIORITY;
	defaults.migration_cost	     = NORMAL_MIGRATION_COST_NS;
	init_class_params_stor(cpu, DESCENT_CLASS_NORMAL, &defaults);

	/* HOG (Class 2) */
	defaults.latency_weight	     = HOG_LATENCY_WEIGHT_NS;
	defaults.base_slice_ns	     = HOG_BASE_SLICE_NS;
	defaults.vruntime_scale	     = HOG_VRUNTIME_SCALE;
	defaults.preemption_priority = HOG_PREEMPTION_PRIORITY;
	defaults.migration_cost	     = HOG_MIGRATION_COST_NS;
	init_class_params_stor(cpu, DESCENT_CLASS_HOG, &defaults);

	/* BACKGROUND (Class 3) */
	defaults.latency_weight	     = BACKGROUND_LATENCY_WEIGHT_NS;
	defaults.base_slice_ns	     = BACKGROUND_BASE_SLICE_NS;
	defaults.vruntime_scale	     = BACKGROUND_VRUNTIME_SCALE;
	defaults.preemption_priority = BACKGROUND_PREEMPTION_PRIORITY;
	defaults.migration_cost	     = BACKGROUND_MIGRATION_COST_NS;
	init_class_params_stor(cpu, DESCENT_CLASS_BACKGROUND, &defaults);
}

/*
 * Per-task local storage.
 *
 * This contain all the per-task information used internally by the BPF code.
 */
struct task_ctx {
	/*
	 * Timestamp when the task started to run on a CPU (used to
	 * evaluate the consumed time slice).
	 */
	u64 last_run_at;

	/*
	 * Task wakeup frequency.
	 */
	u64 wakeup_freq;
	u64 last_woke_at;

	/*
	 * EWMA of observed runtime slice.
	 */
	u64 slice_ns_ewma;

	/*
	 * Current runtime in this scheduling cycle (for HOG detection).
	 */
	u64 runtime_ns;

	/*
	 * cgroup weight (cpu.weight).
	 */
	u32 cgweight;

	/*
	 * NEW: Classification and descent fields
	 */
	u32 task_class; /* Current assigned class */
	u32 prev_class; /* Previous class (for hysteresis) */
	u64 class_entry_time; /* When entered current class */
	u32 reclassify_counter; /* Counts stops, classification every 64th */

	/*
	 * NEW: Latency tracking for PIE controller
	 */
	u64 enqueue_time_ns; /* Timestamp when task was enqueued */

	/*
	 * NEW: For load tracking - track when task started running
	 */
	u64 last_run_start_ns; /* Timestamp when task entered running state */

	/*
	 * Classification metrics.
	 */
	struct {
		u64 wakeup_latency_ewma;
		u64 runtime_per_sched_ewma;
		u64 wakeup_freq_ewma;
		u64 periodicity_score;
	} class_metrics;

	/*
	 * NEW: Momentum-based classification (Phase 3)
	 */
	struct {
		u64 vote_ewma[DESCENT_CLASS_MAX]; /* EWMA vote scores */
		u64 last_vote_update;
		u64 current_confidence; /* 0-100 */
	} class_momentum;

	/* Hysteresis threshold (profile-dependent) */
	u64 hysteresis_duration_ns;
	u32 hysteresis_threshold_pct;

	/* Profile ID for thresholds */
	u32 profile_id;

	/*
	 * Packed flags field:
	 * Bit 23 (BIT_KTHREAD): Cached PF_KTHREAD from task flags
	 */
	u32 packed;

	/*
	 * PPID for Wine/Proton family detection.
	 */
	u32 ppid;
};

/* Map that contains task-local storage. */
struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_ctx_stor SEC(".maps");

/*
 * Return a local task context from a generic task.
 */
struct task_ctx *try_lookup_task_ctx(const struct task_struct *p)
{
	return bpf_task_storage_get(&task_ctx_stor, (struct task_struct *)p, 0,
				    0);
}

/*
 * Per-cgroup context: tracks the cgroup's cpu.weight.
 */
struct cgrp_ctx {
	u32 weight;
};

struct {
	__uint(type, BPF_MAP_TYPE_CGRP_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct cgrp_ctx);
} cgrp_ctx_stor SEC(".maps");

/*
 * Return a local cgroup context from a generic task.
 */
struct cgrp_ctx *try_lookup_cgrp_ctx(struct cgroup *cgrp)
{
	return bpf_cgrp_storage_get(&cgrp_ctx_stor, cgrp, 0, 0);
}

/*
 * Return true if the target task @p is a kernel thread.
 */
static inline bool is_kthread(const struct task_struct *p)
{
	return p->flags & PF_KTHREAD;
}

/*
 * Return true if @p can only run on a single CPU, false otherwise.
 */
static u8 is_pcpu_task(const struct task_struct *p)
{
	return p->nr_cpus_allowed == 1 || is_migration_disabled(p) ? 1 : 0;
}

/*
 * Return true if @p still wants to run, false otherwise.
 */
static u8 is_queued(const struct task_struct *p)
{
	return p->scx.flags & SCX_TASK_QUEUED ? 1 : 0;
}

/*
 * Return the effective weight of a task, incorporating its cgroup weight.
 *
 * The effective weight is:
 *   task_nice_weight * cgroup_weight / CGROUP_WEIGHT_DFL
 *
 * This ensures tasks in a cgroup with weight 200 get twice the CPU time of
 * tasks in a cgroup with the default weight (100).
 */
static u64 task_weight(const struct task_struct *p)
{
	struct task_ctx *tctx;
	u32		 cgw;

	tctx = try_lookup_task_ctx(p);
	cgw  = tctx ? tctx->cgweight : CGROUP_WEIGHT_DFL;

	return (u64)p->scx.weight * cgw / CGROUP_WEIGHT_DFL;
}

static inline u64 scale_by_weight(const struct task_struct *p, u64 value)
{
	return value * task_weight(p) / CGROUP_WEIGHT_DFL;
}

static inline u64 scale_by_weight_inverse(const struct task_struct *p,
					  u64			    value)
{
	u64 w = task_weight(p);

	return w ? value * CGROUP_WEIGHT_DFL / w : value;
}

/*
 * Allocate/re-allocate a new cpumask.
 */
static int calloc_cpumask(struct bpf_cpumask **p_cpumask)
{
	struct bpf_cpumask *cpumask;

	cpumask = bpf_cpumask_create();
	if (!cpumask)
		return -ENOMEM;

	cpumask = bpf_kptr_xchg(p_cpumask, cpumask);
	if (cpumask)
		bpf_cpumask_release(cpumask);

	return 0;
}

/*
 * Return the time slice that can be assigned to a task.
 * Uses per-class base_slice_ns from descent parameters.
 */
static inline u64 task_slice(const struct task_struct *p, struct task_ctx *tctx)
{
	struct class_params *cp;
	u64		     base_slice;

	if (tickless_sched)
		return SCX_SLICE_INF;

	cp = get_class_params_for_scheduling(tctx->task_class);
	if (cp) {
		base_slice = cp->base_slice_ns;
		/* Scale by weight */
		return scale_by_weight(p, base_slice);
	}

	return scale_by_weight(p, slice_max);
}

/*
 * Forward declaration for SMT sibling lookup (defined later in file)
 */
static inline s32 smt_sibling(s32 cpu);

/*
 * Check if a task's TGID is in the turbo list.
 */
static inline u8 is_turbo_tgid(u32 task_tgid)
{
	if (nr_turbo_tgids == 0)
		return 0;

#pragma unroll
	for (u32 i = 0; i < 16; i++) {
		if (i >= nr_turbo_tgids)
			break;
		if (task_tgid == turbo_tgids[i])
			return 1;
	}
	return 0;
}

/*
 * Check if the SMT sibling of the given CPU is running a turbo task.
 */
static inline u8 is_sibling_turbo_task(s32 cpu)
{
	s32		    sibling_cpu;
	struct task_struct *sibling_task;

	if (!smt_enabled)
		return 0;

	sibling_cpu = smt_sibling(cpu);
	if (sibling_cpu == cpu)
		return 0;

	sibling_task = __COMPAT_scx_bpf_cpu_curr(sibling_cpu);
	if (!sibling_task || sibling_task->flags & PF_IDLE)
		return 0;

	return is_turbo_tgid(sibling_task->tgid);
}

/*
 * Classify a task into one of the descent classes using scx_cake methodology.
 * Classification runs every 64th stop for efficiency.
 *
 * Class 0: LATENCY_CRITICAL - Games, audio, compositors, kthreads (during GAMING)
 * Class 1: NORMAL          - Default interactive
 * Class 2: HOG             - High CPU usage (>=75% quantum)
 * Class 3: BACKGROUND      - Low priority, SCHED_IDLE, rare wakeups
 *
 * Default: NORMAL
 */
static u32 classify_task(struct task_struct *p, struct task_ctx *tctx)
{
	u32 class = DESCENT_CLASS_NORMAL; /* Default */

	/* Increment counter, skip expensive classification on 63/64 stops */
	tctx->reclassify_counter++;
	if (tctx->reclassify_counter & 63) { /* Check lower 6 bits */
		/* Fast path: return cached class if available */
		if (tctx->task_class < DESCENT_CLASS_MAX)
			return tctx->task_class;
		/* No cache: fall through to classification */
	}

	/* Check for real-time scheduling policies (always latency-critical) */
	if (p->policy == SCHED_FIFO || p->policy == SCHED_RR) {
		class = DESCENT_CLASS_LATENCY_CRITICAL;
		goto done;
	}

	/* Check for SCHED_IDLE (always background) */
	if (p->policy == SCHED_IDLE) {
		class = DESCENT_CLASS_BACKGROUND;
		goto done;
	}

	/*
	 * Check for Desktop Environment components (outside GAMING state only)
	 * DE components need responsiveness for UI interactions during desktop use
	 */
	if (sched_state != 2 && nr_de_tgids > 0) { // Not in GAMING state
		u32 task_tgid = p->tgid;

#pragma unroll
		for (u32 i = 0; i < 16; i++) {
			if (i >= nr_de_tgids)
				break;
			if (task_tgid == de_tgids[i]) {
				class = DESCENT_CLASS_LATENCY_CRITICAL;
				goto done;
			}
		}
	}

	/* Only during GAMING state: full classification */
	if (sched_state == 2) { /* GAMING */
		u8 is_kthread_cached = (tctx->packed >> BIT_KTHREAD) & 1;

		/* Class 0: LATENCY_CRITICAL */
		/* Game family matching */
		u8 is_game_family =
			(p->tgid == game_tgid) || (tctx->ppid == game_ppid) ||
			is_kthread_cached; /* Promote kthreads during gaming */

		if (is_game_family) {
			class = DESCENT_CLASS_LATENCY_CRITICAL;
			goto done;
		}

		/* Audio daemon matching */
		if (nr_audio_tgids > 0) {
			u32 task_tgid = p->tgid;
#pragma unroll
			for (u32 i = 0; i < 16; i++) {
				if (i >= nr_audio_tgids)
					break;
				if (task_tgid == audio_tgids[i]) {
					class = DESCENT_CLASS_LATENCY_CRITICAL;
					goto done;
				}
			}
		}

		/* Turbo process matching - highest priority */
		if (nr_turbo_tgids > 0 && is_turbo_tgid(p->tgid)) {
			class = DESCENT_CLASS_LATENCY_CRITICAL;
			tctx->packed |= (1 << BIT_TURBO);
			goto done;
		}

		/* Class 2: HOG (high CPU usage, non-critical) */
		/* HOG detection: runtime >= 75% of typical slice */
		u32 hog_thresh = (tctx->slice_ns_ewma >> 2) * 3; /* 75% */
		if (!is_game_family && tctx->runtime_ns >= hog_thresh &&
		    tctx->slice_ns_ewma > 0) {
			class = DESCENT_CLASS_HOG;
			goto done;
		}

		/* Class 3: BACKGROUND (rare wakeups, low activity) */
		if (tctx->wakeup_freq < 10 &&
		    tctx->slice_ns_ewma > 5 * NSEC_PER_MSEC) {
			class = DESCENT_CLASS_BACKGROUND;
			goto done;
		}
	}

	/* Non-GAMING or fallback: use heuristics for NORMAL vs BACKGROUND */
	if (tctx->slice_ns_ewma && tctx->wakeup_freq) {
		/* High CPU usage tasks go to HOG */
		u32 hog_thresh = (tctx->slice_ns_ewma >> 2) * 3;
		if (tctx->runtime_ns >= hog_thresh && tctx->slice_ns_ewma > 0) {
			class = DESCENT_CLASS_HOG;
			goto done;
		}

		/* Background: low wakeup freq + long slices */
		if (tctx->wakeup_freq < WAKEUP_FREQ_BATCH_THRESH &&
		    tctx->slice_ns_ewma > SLICE_NS_BATCH_THRESH) {
			class = DESCENT_CLASS_BACKGROUND;
			goto done;
		}
	}

done:
	tctx->task_class = class;
	return class;
}

/*
 * NEW: Phase 3 - Momentum-based classification helpers
 */

/* Profile IDs for threshold selection */
#define PROFILE_PRODUCTIVITY 0
#define PROFILE_GAMING 1
#define PROFILE_SERVER 2

/* Classification thresholds by profile */
static void set_profile_thresholds(struct task_ctx *tctx, u32 profile_id)
{
	switch (profile_id) {
	case PROFILE_GAMING: /* 60% / 300ms (more responsive) */
		tctx->hysteresis_threshold_pct = 60;
		tctx->hysteresis_duration_ns   = 300 * NSEC_PER_MSEC;
		break;
	case PROFILE_SERVER: /* 80% / 1000ms (more stable) */
		tctx->hysteresis_threshold_pct = 80;
		tctx->hysteresis_duration_ns   = 1000 * NSEC_PER_MSEC;
		break;
	case PROFILE_PRODUCTIVITY: /* 70% / 500ms (default) */
	default:
		tctx->hysteresis_threshold_pct = 70;
		tctx->hysteresis_duration_ns   = 500 * NSEC_PER_MSEC;
		break;
	}
}

/* Get raw classification votes based on current metrics */
static void get_raw_votes(struct task_struct *p, struct task_ctx *tctx,
			  u64 votes[DESCENT_CLASS_MAX])
{
	u8 is_kthread_cached;

	/* Initialize all votes to 0 */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		votes[i] = 0;
	}

	/* Real-time scheduling policy: always latency-critical */
	if (p->policy == SCHED_FIFO || p->policy == SCHED_RR) {
		votes[DESCENT_CLASS_LATENCY_CRITICAL] = 100;
		return;
	}

	/* SCHED_IDLE: always background */
	if (p->policy == SCHED_IDLE) {
		votes[DESCENT_CLASS_BACKGROUND] = 100;
		return;
	}

	/* Kernel threads: use cached flag */
	is_kthread_cached = (tctx->packed >> BIT_KTHREAD) & 1;
	if (is_kthread_cached) {
		/* During gaming, kthreads are latency-critical, otherwise normal */
		if (sched_state == 2) { /* GAMING */
			votes[DESCENT_CLASS_LATENCY_CRITICAL] = 100;
		} else {
			votes[DESCENT_CLASS_NORMAL] = 100;
		}
		return;
	}

	/* Use heuristics based on observed behavior */
	if (tctx->slice_ns_ewma && tctx->wakeup_freq) {
		/* HOG: high CPU usage */
		u32 hog_thresh = (tctx->slice_ns_ewma >> 2) * 3;
		if (tctx->runtime_ns >= hog_thresh && tctx->slice_ns_ewma > 0) {
			votes[DESCENT_CLASS_HOG]    = 80;
			votes[DESCENT_CLASS_NORMAL] = 20;
			return;
		}

		/* Background: low wakeup freq + long slices */
		if (tctx->wakeup_freq < WAKEUP_FREQ_BATCH_THRESH &&
		    tctx->slice_ns_ewma > SLICE_NS_BATCH_THRESH) {
			votes[DESCENT_CLASS_BACKGROUND] = 80;
			votes[DESCENT_CLASS_NORMAL]	= 20;
			return;
		}

		/* Normal: high wakeup freq + short slices */
		if (tctx->wakeup_freq > WAKEUP_FREQ_INTERACTIVE_THRESH &&
		    tctx->slice_ns_ewma < SLICE_NS_INTERACTIVE_THRESH) {
			votes[DESCENT_CLASS_NORMAL] = 100;
			return;
		}
	}

	/* Default to normal for unknown behavior */
	votes[DESCENT_CLASS_NORMAL] = 100;
}

/* Momentum alpha for EWMA (0.7 in percentage = 70) */
#define MOMENTUM_ALPHA 70

/* Update classification with momentum and hysteresis */
static void update_classification_momentum(struct task_struct *p,
					   struct task_ctx    *tctx)
{
	u64 now = bpf_ktime_get_ns();
	u64 raw_votes[DESCENT_CLASS_MAX];

	/* Get raw votes from current behavior */
	get_raw_votes(p, tctx, raw_votes);

	/* EWMA update: new = 0.7*old + 0.3*new */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		u64 old	     = tctx->class_momentum.vote_ewma[i];
		u64 new_vote = raw_votes[i];
		tctx->class_momentum.vote_ewma[i] =
			(old * MOMENTUM_ALPHA +
			 new_vote * (100 - MOMENTUM_ALPHA)) /
			100;
	}

	tctx->class_momentum.last_vote_update = now;

	/* Find highest voted class */
	u64 max_votes	   = 0;
	u32 proposed_class = tctx->task_class;
	u64 total_votes	   = 0;

	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		u64 v = tctx->class_momentum.vote_ewma[i];
		total_votes += v;
		if (v > max_votes) {
			max_votes      = v;
			proposed_class = i;
		}
	}

	/* Check confidence and hysteresis */
	if (proposed_class != tctx->task_class && total_votes > 0) {
		u64 confidence = (max_votes * 100) / total_votes;
		tctx->class_momentum.current_confidence = confidence;

		u64 time_in_class = now - tctx->class_entry_time;

		if (confidence >= tctx->hysteresis_threshold_pct &&
		    time_in_class >= tctx->hysteresis_duration_ns) {
			/* Actually change class */
			tctx->prev_class       = tctx->task_class;
			tctx->task_class       = proposed_class;
			tctx->class_entry_time = now;

			if (debug) {
				bpf_printk(
					"descent: task %d reclassified %d -> %d (conf %d%%)",
					p->pid, tctx->prev_class,
					proposed_class, confidence);
			}
		}
	}
}

/*
 * Return task deadline in function of the accumulated vruntime, using
 * class-specific parameters for gradient descent optimization.
 *
 * The deadline is calculated as:
 *   deadline = (vruntime * vruntime_scale / 1024) - latency_weight
 *
 * Earlier deadline = higher priority
 */
static u64 task_dl(struct task_struct *p, struct task_ctx *tctx, u64 enq_flags)
{
	struct class_params *cp;
	u64		     lag_scale, vsleep_max, vtime_min;
	u64		     vtime = p->scx.dsq_vtime;
	u64		     scaled_vtime;

	/* Get class parameters using new helper */
	cp = get_class_params_for_scheduling(tctx->task_class);
	if (!cp || tctx->task_class >= DESCENT_CLASS_MAX) {
		/* Fallback to flash behavior */
		lag_scale  = MAX(tctx->wakeup_freq, 1);
		vsleep_max = scale_by_weight(p, slice_lag * lag_scale);
		vtime_min  = vtime_now - vsleep_max;

		if (enq_flags & SCX_ENQ_REENQ)
			return vtime;

		if (time_before(vtime, vtime_min))
			vtime = vtime_min;

		if (tctx->slice_ns_ewma && tctx->slice_ns_ewma < SLICE_MIN_NS)
			vtime += slice_lag;

		return vtime;
	}

	/* Calculate scaled vruntime using class-specific scale */
	scaled_vtime = vtime * cp->vruntime_scale / 1024;

	/* Apply latency weight offset (negative = earlier deadline) */
	lag_scale  = MAX(tctx->wakeup_freq, 1);
	vsleep_max = cp->latency_weight * lag_scale;
	vsleep_max = scale_by_weight(p, vsleep_max);
	vtime_min  = vtime_now - vsleep_max;

	if (enq_flags & SCX_ENQ_REENQ)
		return scaled_vtime;

	if (time_before(scaled_vtime, vtime_min))
		scaled_vtime = vtime_min;

	/*
	 * Penalize tasks that are abusing the wakeup frequency
	 * prioritization by charging them additional latency.
	 */
	if (tctx->slice_ns_ewma &&
	    tctx->slice_ns_ewma < cp->preemption_priority)
		scaled_vtime += cp->latency_weight;

	/*
	 * Turbo task deadline boost: turbo tasks get earlier deadlines
	 * for faster scheduling within LATENCY_CRITICAL class
	 */
	if (is_turbo_tgid(p->tgid)) {
		/* Reduce deadline by latency_weight for turbo boost */
		if (scaled_vtime > cp->latency_weight)
			scaled_vtime -= cp->latency_weight;
	}

	/*
	 * Turbo conflict penalty: tasks on SMT siblings of turbo tasks
	 * get deprioritized with additional vruntime
	 */
	if (is_sibling_turbo_task(scx_bpf_task_cpu(p))) {
		scaled_vtime += cp->latency_weight;
	}

	return scaled_vtime;
}

/*
 * Find an idle CPU in the system.
 *
 * NOTE: the idle CPU selection doesn't need to be formally perfect, it is
 * totally fine to accept racy conditions and potentially make mistakes, by
 * picking CPUs that are not idle or even offline, the logic has been designed
 * to handle these mistakes in favor of a more efficient response and a reduced
 * scheduling overhead.
 */
static s32 pick_idle_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags,
			 u8 *is_idle)
{
	const struct cpumask *primary = cast_mask(primary_cpumask);
	s32		      cpu;

	/*
	 * Compatibility with older kernels (< v6.14).
	 */
	if (!__COMPAT_HAS_scx_bpf_select_cpu_and) {
		if (wake_flags) {
			_Bool local_is_idle;
			cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags,
						     &local_is_idle);
			*is_idle = local_is_idle ? 1 : 0;
			return cpu;
		}

		return prev_cpu;
	}

	/*
	 * Don't trust user-space about waker releasing the CPU: if it
	 * doesn't, we may have latency issues, so it's safer to just
	 * ignore the hint.
	 */
	wake_flags &= ~SCX_WAKE_SYNC;

	cpu = (primary_all || !primary) ?
		      -ENOENT :
		      scx_bpf_select_cpu_and(p, prev_cpu, wake_flags, primary,
					     0);
	if (cpu < 0) {
		cpu = scx_bpf_select_cpu_and(p, prev_cpu, wake_flags,
					     p->cpus_ptr, 0);
		if (cpu < 0)
			return prev_cpu;
	}
	*is_idle = 1;

	return cpu;
}

/*
 * Pick a target CPU for a task which is being woken up.
 *
 * If a task is dispatched here, ops.enqueue() will be skipped: task will be
 * dispatched directly to the CPU returned by this callback.
 */
s32 BPF_STRUCT_OPS(descent_select_cpu, struct task_struct *p, s32 prev_cpu,
		   u64 wake_flags)
{
	struct task_ctx *tctx;
	u8		 is_idle = 0;
	s32		 cpu;

	if (is_throttled())
		return prev_cpu;

	/* Update task classification using momentum-based method */
	tctx = try_lookup_task_ctx(p);
	if (tctx)
		update_classification_momentum(p, tctx);

	cpu = pick_idle_cpu(p, prev_cpu, wake_flags, &is_idle);

	/*
	 * If this task is NOT a turbo task but prev_cpu has a turbo task
	 * on its SMT sibling, try to migrate away to avoid contention.
	 */
	if (tctx && !is_turbo_tgid(p->tgid) &&
	    is_sibling_turbo_task(prev_cpu)) {
		/* Try to find a non-contended idle CPU */
		if (!is_pcpu_task(p)) {
			s32 new_cpu = pick_idle_cpu(p, prev_cpu, wake_flags,
						    &is_idle);
			if (is_idle && new_cpu != prev_cpu) {
				/* Migrate away from turbo task's SMT sibling */
				scx_bpf_dsq_insert(p,
						   SCX_DSQ_LOCAL_ON | new_cpu,
						   task_slice(p, tctx), 0);
				scx_bpf_kick_cpu(new_cpu, SCX_KICK_IDLE);
				return new_cpu;
			}
		}
	}

	if (rr_sched || is_idle) {
		if (tctx)
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL,
					   task_slice(p, tctx), 0);
		else
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SCX_SLICE_DFL, 0);
		__sync_fetch_and_add(&nr_direct_dispatches, 1);
	}

	return cpu;
}

/*
 * Return the cpumask of idle CPUs within the NUMA node that contains @cpu.
 *
 * If NUMA support is disabled, @cpu is ignored.
 */
static inline const struct cpumask *get_idle_cpumask(s32 cpu)
{
	if (numa_disabled)
		return scx_bpf_get_idle_cpumask();

	return __COMPAT_scx_bpf_get_idle_cpumask_node(
		__COMPAT_scx_bpf_cpu_node(cpu));
}

/*
 * Return the SMT sibling of @cpu, or @cpu if SMT is disabled.
 */
static inline s32 smt_sibling(s32 cpu)
{
	const struct cpumask *smt;
	struct cpu_ctx	     *cctx;

	if (!smt_enabled)
		return cpu;

	cctx = try_lookup_cpu_ctx(cpu);
	if (!cctx)
		return cpu;

	smt = cast_mask(cctx->smt);
	if (!smt)
		return cpu;

	return bpf_cpumask_first(smt);
}

/*
 * Return true if @cpu is in  a partially-idle SMT core, false otherwise.
 */
static u8 is_smt_contended(s32 cpu)
{
	const struct cpumask *idle_mask;
	u8		      is_contended;

	if (!smt_enabled)
		return 0;

	/*
	 * If the sibling SMT CPU is not idle and there are other full-idle
	 * SMT cores available, consider the current CPU as contended.
	 */
	idle_mask    = get_idle_cpumask(cpu);
	is_contended = !bpf_cpumask_test_cpu(smt_sibling(cpu), idle_mask) &&
		       !bpf_cpumask_empty(idle_mask);
	scx_bpf_put_cpumask(idle_mask);

	return is_contended;
}

/*
 * Return true if @p is running on a primary CPU (or can't run on a primary
 * CPU due to affinity constraints), false otherwise.
 */
static u8 is_primary_cpu(const struct task_struct *p, s32 cpu)
{
	if (!primary_all) {
		const struct cpumask *primary = cast_mask(primary_cpumask);

		if (primary && bpf_cpumask_intersects(primary, p->cpus_ptr) &&
		    !bpf_cpumask_test_cpu(cpu, primary))
			return 0;
	}

	return 1;
}

/*
 * Attempt to dispatch a task directly to its assigned CPU.
 *
 * Return true if the task is dispatched, false otherwise.
 */
static u8 try_direct_dispatch(struct task_struct *p, s32 prev_cpu,
			      u64 enq_flags, u8 is_running)
{
	u8		 is_idle = 0;
	s32		 cpu	 = prev_cpu;
	struct task_ctx *tctx;

	/*
	 * If throttling is enabled always dispatch critical kernel threads
	 * directly to prevent throttling the entire system.
	 */
	if (throttle_ns > 0 && is_kthread(p) && p->nr_cpus_allowed == 1) {
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SCX_SLICE_DFL, enq_flags);
		__sync_fetch_and_add(&nr_kthread_dispatches, 1);
		if (!is_running)
			scx_bpf_kick_cpu(prev_cpu, SCX_KICK_IDLE);

		return 1;
	}

	/*
	 * Don't attempt a migration if the task is running or if
	 * ops.select_cpu() was already called, as the task has already had
	 * an opportunity for direct dispatch there.
	 *
	 * Always attempt a migration if there's SMT contention on the
	 * current CPU or if the task has been re-enqueued.
	 */
	if (!is_running && __COMPAT_is_enq_cpu_selected(enq_flags) &&
	    (!is_smt_contended(prev_cpu) || is_pcpu_task(p)) &&
	    !(enq_flags & SCX_ENQ_REENQ))
		return 0;

	/*
	 * Try migrating to an idle CPU.
	 */
	if (!is_pcpu_task(p)) {
		cpu = pick_idle_cpu(p, prev_cpu, 0, &is_idle);
		if (!is_idle)
			return 0;
	} else {
		if (!scx_bpf_test_and_clear_cpu_idle(prev_cpu))
			return 0;
	}

	tctx = try_lookup_task_ctx(p);
	if (tctx)
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu,
				   task_slice(p, tctx), 0);
	else
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | cpu, SCX_SLICE_DFL, 0);
	__sync_fetch_and_add(&nr_direct_dispatches, 1);

	if (cpu != prev_cpu || !is_running)
		scx_bpf_kick_cpu(cpu, SCX_KICK_IDLE);

	return 1;
}

/*
 * Enqueue a task when running in round-robin mode.
 */
static void rr_enqueue(struct task_struct *p, s32 prev_cpu, u64 enq_flags)
{
	u8		 is_idle;
	s32		 cpu;
	struct task_ctx *tctx;

	/*
	 * Attempt to migrate on another CPU on wakeup or if the task has
	 * been re-enqueued due to a higher priority class stealing the
	 * CPU, otherwise always prefer running on the same CPU.
	 */
	if (!scx_bpf_task_running(p) || (enq_flags & SCX_ENQ_REENQ)) {
		if (is_pcpu_task(p)) {
			if (scx_bpf_test_and_clear_cpu_idle(prev_cpu))
				scx_bpf_kick_cpu(prev_cpu, SCX_KICK_IDLE);
		} else {
			cpu = pick_idle_cpu(p, prev_cpu, 0, &is_idle);
			if (is_idle) {
				tctx = try_lookup_task_ctx(p);
				if (tctx)
					scx_bpf_dsq_insert(
						p, SCX_DSQ_LOCAL_ON | cpu,
						task_slice(p, tctx), enq_flags);
				else
					scx_bpf_dsq_insert(
						p, SCX_DSQ_LOCAL_ON | cpu,
						SCX_SLICE_DFL, enq_flags);
				scx_bpf_kick_cpu(cpu, SCX_KICK_IDLE);
				return;
			}
		}
	}
	scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, SCX_SLICE_DFL, enq_flags);
}

/*
 * Dispatch all the other tasks that were not dispatched directly in
 * select_cpu().
 */
void BPF_STRUCT_OPS(descent_enqueue, struct task_struct *p, u64 enq_flags)
{
	s32		 prev_cpu   = scx_bpf_task_cpu(p);
	bool		 is_running = scx_bpf_task_running(p);
	struct task_ctx *tctx;

	/*
	 * NEW: Record enqueue time for latency tracking
	 */
	tctx = try_lookup_task_ctx(p);
	if (tctx) {
		tctx->enqueue_time_ns = bpf_ktime_get_ns();

		/*
		 * For turbo tasks: use reduced slice for faster preemption decisions
		 */
		if (is_turbo_tgid(p->tgid)) {
			struct class_params *cp =
				get_class_params_for_scheduling(
					tctx->task_class);
			if (cp) {
				/* Turbo tasks get half the normal slice for responsiveness */
				u64 turbo_slice = cp->base_slice_ns / 2;

				/* Update task slice for this enqueue */
				p->scx.slice = turbo_slice;
			}
		}
	}

	/*
	 * Keep reusing the same CPU in round-robin mode.
	 */
	if (rr_sched) {
		rr_enqueue(p, prev_cpu, enq_flags);
		return;
	}

	/*
	 * Try to dispatch the task directly, if possible.
	 */
	if (try_direct_dispatch(p, prev_cpu, enq_flags, is_running))
		return;

	/*
	 * Insert the task to the per-node DSQ using descent deadline.
	 */
	tctx = try_lookup_task_ctx(p);
	if (tctx) {
		int node = __COMPAT_scx_bpf_cpu_node(prev_cpu);

		scx_bpf_dsq_insert_vtime(p, node, task_slice(p, tctx),
					 task_dl(p, tctx, enq_flags),
					 enq_flags);
		__sync_fetch_and_add(&nr_shared_dispatches, 1);

		if (!is_running && !__COMPAT_is_enq_cpu_selected(enq_flags))
			scx_bpf_kick_cpu(prev_cpu, SCX_KICK_IDLE);
	} else {
		/* Fallback: use default insertion */
		int node = __COMPAT_scx_bpf_cpu_node(prev_cpu);
		scx_bpf_dsq_insert(p, node, SCX_SLICE_DFL, enq_flags);
		__sync_fetch_and_add(&nr_shared_dispatches, 1);
	}
}

/*
 * Return true if the task can keep running on its current CPU, false if
 * the task should migrate.
 */
static u8 keep_running(const struct task_struct *p, s32 cpu)
{
	/* Do not keep running if the task doesn't need to run */
	if (!is_queued(p))
		return 0;

	/*
	 * If the task can't migrate elsewhere, keep it running.
	 */
	if (p->nr_cpus_allowed == 1)
		return 1;

	/*
	 * Do not keep running if the CPU is not in the primary domain and
	 * the task can use the primary domain.
	 */
	if (!is_primary_cpu(p, cpu))
		return 0;

	/*
	 * If the task is running on a CPU with a busy SMT sibling, try to
	 * move it elsewhere.
	 */
	if (is_smt_contended(cpu))
		return 0;

	/*
	 * If a turbo task is running on SMT sibling, yield immediately.
	 * Non-turbo tasks should not contend with turbo tasks.
	 */
	if (is_sibling_turbo_task(cpu)) {
		return 0;
	}

	/*
	 * If this IS a turbo task and there's work waiting, keep running.
	 * Turbo tasks are sticky to maintain low latency.
	 */
	if (is_turbo_tgid(p->tgid) && is_queued(p)) {
		return 1;
	}

	return 1;
}

void BPF_STRUCT_OPS(descent_dispatch, s32 cpu, struct task_struct *prev)
{
	int node	 = __COMPAT_scx_bpf_cpu_node(cpu);
	u8  need_running = prev && keep_running(prev, cpu);

	/*
	 * Let the CPU go idle if the system is throttled.
	 */
	if (is_throttled())
		return;

	if (need_running) {
		struct task_ctx	   *tctx    = try_lookup_task_ctx(prev);
		struct task_struct *q	    = __COMPAT_scx_bpf_dsq_peek(node);
		u64		    q_vtime = q ? q->scx.dsq_vtime : ULLONG_MAX;

		if (tctx) {
			u64 slice      = bpf_ktime_get_ns() - tctx->last_run_at;
			u64 prev_vtime = prev->scx.dsq_vtime +
					 scale_by_weight_inverse(prev, slice);

			if (prev_vtime < q_vtime) {
				prev->scx.slice = task_slice(prev, tctx);
				return;
			}
		}
	}

	if (scx_bpf_dsq_move_to_local(node, 0))
		return;

	/*
	 * If the current task expired its time slice and no other task wants
	 * to run, simply replenish its time slice and let it run for another
	 * round on the same CPU.
	 */
	if (need_running) {
		struct task_ctx *tctx = try_lookup_task_ctx(prev);
		if (tctx)
			prev->scx.slice = task_slice(prev, tctx);
		else
			prev->scx.slice = SCX_SLICE_DFL;
	}
}

/*
 * Exponential weighted moving average (EWMA).
 *
 * Copied from scx_lavd. Returns the new average as:
 *
 *	new_avg := (old_avg * .75) + (new_val * .25);
 */
static u64 calc_avg(u64 old_val, u64 new_val)
{
	return (old_val - (old_val >> 2)) + (new_val >> 2);
}

/*
 * Update the average frequency of an event.
 *
 * The frequency is computed from the given interval since the last event
 * and combined with the previous frequency using an exponential weighted
 * moving average.
 */
static u64 update_freq(u64 freq, u64 interval)
{
	u64 new_freq;

	new_freq = (100 * NSEC_PER_MSEC) / interval;
	return calc_avg(freq, new_freq);
}

/*
 * Update CPU load and scale target performance level accordingly.
 */
static void update_cpu_load(struct task_struct *p, struct task_ctx *tctx)
{
	u64		now = bpf_ktime_get_ns();
	s32		cpu = scx_bpf_task_cpu(p);
	u64		perf_lvl, delta_runtime, delta_t;
	struct cpu_ctx *cctx;

	/*
	 * For non-interactive tasks determine their cpufreq scaling factor as
	 * a function of their CPU utilization.
	 */
	cctx = try_lookup_cpu_ctx(cpu);
	if (!cctx)
		return;

	/*
	 * Evaluate dynamic cpuperf scaling factor using the average CPU
	 * utilization, normalized in the range [0 .. SCX_CPUPERF_ONE].
	 */
	delta_t = now - cctx->last_running;
	if (!delta_t)
		return;

	/*
	 * Refresh target performance level.
	 */
	delta_runtime = cctx->tot_runtime - cctx->prev_runtime;
	perf_lvl =
		MIN(delta_runtime * SCX_CPUPERF_ONE / delta_t, SCX_CPUPERF_ONE);

	/*
	 * Use a moving average to evaluate the target performance level,
	 * giving more priority to the current average, so that we can
	 * react faster at CPU load variations and at the same time smooth
	 * the short spikes.
	 */
	cctx->perf_lvl = calc_avg(perf_lvl, cctx->perf_lvl);

	/*
	 * Refresh the dynamic cpuperf scaling factor if needed.
	 *
	 * Apply hysteresis to the scaling factor:
	 *  - if utilization is above the high threshold, bump to max;
	 *  - if it's below the low threshold, scale down to half capacity;
	 *  - otherwise, maintain the smoothed perf level.
	 */
	if (cpufreq_perf_lvl < 0) {
		if (cctx->perf_lvl >= CPUFREQ_HIGH_THRESH)
			perf_lvl = SCX_CPUPERF_ONE;
		else if (cctx->perf_lvl <= CPUFREQ_LOW_THRESH)
			perf_lvl = SCX_CPUPERF_ONE / 2;
		else
			perf_lvl = cctx->perf_lvl;
		scx_bpf_cpuperf_set(cpu, perf_lvl);
	}

	cctx->last_running = now;
	cctx->prev_runtime = cctx->tot_runtime;
}

void BPF_STRUCT_OPS(descent_running, struct task_struct *p)
{
	struct task_ctx *tctx;

	__sync_fetch_and_add(&nr_running, 1);

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	/*
	 * NEW: Calculate and accumulate enqueue-to-run latency
	 */
	if (tctx->enqueue_time_ns > 0) {
		u64 now	       = bpf_ktime_get_ns();
		u64 latency_ns = now - tctx->enqueue_time_ns;

		/* Accumulate to per-class metrics */
		struct cpu_descent_ctx *cdctx = try_lookup_cpu_descent_ctx();
		if (cdctx && tctx->task_class < DESCENT_CLASS_MAX) {
			struct class_latency_accumulator *accum =
				&cdctx->class_latency[tctx->task_class];

			/* Update atomically using __sync_fetch_and_add for 64-bit */
			__sync_fetch_and_add(&accum->total_latency_ns,
					     latency_ns);
			__sync_fetch_and_add(&accum->sample_count, 1);

			/* Track max (simple compare-and-set) */
			if (latency_ns > accum->max_latency_ns) {
				accum->max_latency_ns = latency_ns;
			}
		}

		/* Reset enqueue_time to prevent double counting */
		tctx->enqueue_time_ns = 0;
	}

	/*
	 * NEW: Record start time for load tracking
	 */
	tctx->last_run_start_ns = bpf_ktime_get_ns();

	tctx->last_run_at	= bpf_ktime_get_ns();

	/*
	 * Adjust target CPU frequency before the task starts to run.
	 */
	if (cpufreq_perf_lvl < 0)
		update_cpu_load(p, tctx);

	/*
	 * Update the global vruntime as a new task is starting to use a
	 * CPU.
	 */
	if (!rr_sched && time_before(vtime_now, p->scx.dsq_vtime))
		vtime_now = p->scx.dsq_vtime;
}

/*
 * Update task statistics when the task is releasing the CPU (either
 * voluntarily or because it expires its assigned time slice).
 */
void BPF_STRUCT_OPS(descent_stopping, struct task_struct *p, bool runnable)
{
	u64		 now = bpf_ktime_get_ns(), slice;
	s32		 cpu = scx_bpf_task_cpu(p);
	struct task_ctx *tctx;

	__sync_fetch_and_sub(&nr_running, 1);

	if (!rr_sched) {
		tctx = try_lookup_task_ctx(p);
		if (!tctx)
			return;

		/*
		 * Evaluate the time slice used by the task.
		 */
		slice = MAX(now - tctx->last_run_at, 1);

		/* Update runtime_ns for HOG detection */
		tctx->runtime_ns += slice;

		if (tctx->slice_ns_ewma)
			tctx->slice_ns_ewma =
				calc_avg(tctx->slice_ns_ewma, slice);
		else
			tctx->slice_ns_ewma = slice;

		/*
		 * Update task's vruntime and accumulated runtime.
		 */
		p->scx.dsq_vtime += scale_by_weight_inverse(p, slice);

		/*
		 * Update class metrics: runtime per schedule
		 */
		tctx->class_metrics.runtime_per_sched_ewma = calc_avg(
			tctx->class_metrics.runtime_per_sched_ewma, slice);

		/*
		 * Periodically reclassify task based on observed behavior
		 * using momentum-based classification with hysteresis
		 */
		if ((now - tctx->class_entry_time) > (100ULL * NSEC_PER_MSEC))
			update_classification_momentum(p, tctx);

		/*
		 * Also call classify_task for the scx_cake-style classification
		 * (runs the 64-counter based classification)
		 */
		classify_task(p, tctx);

		/*
		 * NEW: Load tracking - accumulate cycles spent running
		 */
		if (tctx->last_run_start_ns > 0) {
			u64 slice_ns = now - tctx->last_run_start_ns;

			/* Accumulate to per-class load metrics */
			struct cpu_descent_ctx *cdctx =
				try_lookup_cpu_descent_ctx();
			if (cdctx && tctx->task_class < DESCENT_CLASS_MAX) {
				struct class_load_accumulator *load =
					&cdctx->class_load[tctx->task_class];

				/* Per-CPU data doesn't need atomics - direct accumulation */
				load->cycles_spent += slice_ns;
				load->sample_count += 1;

				/* DEBUG: Trace load accumulation */
				bpf_printk(
					"LOAD: class=%u slice=%lu cycles=%lu samples=%lu",
					tctx->task_class, slice_ns,
					load->cycles_spent, load->sample_count);
			} else {
				/* DEBUG: Trace why accumulation failed */
				bpf_printk("LOAD SKIP: cdctx=%p class=%u",
					   cdctx, tctx->task_class);
			}

			/* Reset for next run */
			tctx->last_run_start_ns = 0;
		}
	}

	/*
	 * Update CPU runtime.
	 */
	if (cpufreq_perf_lvl < 0) {
		struct cpu_ctx *cctx;

		cctx = try_lookup_cpu_ctx(cpu);
		if (cctx)
			cctx->tot_runtime += now - cctx->last_running;
	}
}

void BPF_STRUCT_OPS(descent_runnable, struct task_struct *p, u64 enq_flags)
{
	u64		 now = bpf_ktime_get_ns(), delta_t;
	struct task_ctx *tctx;

	if (rr_sched)
		return;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	/*
	 * Update the task's wakeup frequency based on the time since
	 * the last wakeup, then cap the result at 1024 to avoid large
	 * spikes.
	 */
	delta_t		  = now - tctx->last_woke_at;
	tctx->wakeup_freq = update_freq(tctx->wakeup_freq, delta_t);
	tctx->wakeup_freq = MIN(tctx->wakeup_freq, MAX_WAKEUP_FREQ);

	/*
	 * Track wakeup latency EWMA for classification
	 */
	u64 wakeup_latency = now - tctx->last_woke_at;

	/* Update EWMA for classification */
	tctx->class_metrics.wakeup_latency_ewma = calc_avg(
		tctx->class_metrics.wakeup_latency_ewma, wakeup_latency);

	tctx->last_woke_at = now;

	/*
	 * Update class metrics: wakeup frequency EWMA
	 */
	tctx->class_metrics.wakeup_freq_ewma = calc_avg(
		tctx->class_metrics.wakeup_freq_ewma, tctx->wakeup_freq);
}

void BPF_STRUCT_OPS(descent_enable, struct task_struct *p)
{
	struct task_ctx *tctx;

	if (rr_sched)
		return;

	p->scx.dsq_vtime = vtime_now;

	/* Initialize task classification */
	tctx = try_lookup_task_ctx(p);
	if (tctx) {
		tctx->task_class       = classify_task(p, tctx);
		tctx->class_entry_time = bpf_ktime_get_ns();
	}
}

s32 BPF_STRUCT_OPS(descent_cgroup_init, struct cgroup *cgrp,
		   struct scx_cgroup_init_args *args)
{
	struct cgrp_ctx *cgc;

	cgc = bpf_cgrp_storage_get(&cgrp_ctx_stor, cgrp, 0,
				   BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!cgc)
		return -ENOMEM;

	cgc->weight = args->weight;

	return 0;
}

void BPF_STRUCT_OPS(descent_cgroup_set_weight, struct cgroup *cgrp, u32 weight)
{
	struct cgrp_ctx *cgc;

	cgc = try_lookup_cgrp_ctx(cgrp);
	if (cgc)
		cgc->weight = weight;
}

void BPF_STRUCT_OPS(descent_cgroup_move, struct task_struct *p,
		    struct cgroup *from, struct cgroup *to)
{
	struct task_ctx *tctx;
	struct cgrp_ctx *cgc;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	cgc	       = try_lookup_cgrp_ctx(to);
	tctx->cgweight = cgc ? cgc->weight : CGROUP_WEIGHT_DFL;
}

static int init_cpumask(struct bpf_cpumask **cpumask)
{
	struct bpf_cpumask *mask;
	int		    err = 0;

	/*
	 * Do nothing if the mask is already initialized.
	 */
	mask = *cpumask;
	if (mask)
		return 0;
	/*
	 * Create the CPU mask.
	 */
	err = calloc_cpumask(cpumask);
	if (!err)
		mask = *cpumask;
	if (!mask)
		err = -ENOMEM;

	return err;
}

s32 BPF_STRUCT_OPS(descent_init_task, struct task_struct *p,
		   struct scx_init_task_args *args)
{
	struct task_ctx *tctx;

	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0,
				    BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx)
		return -ENOMEM;

	if (args->cgroup) {
		struct cgrp_ctx *cgc = try_lookup_cgrp_ctx(args->cgroup);
		tctx->cgweight	     = cgc ? cgc->weight : CGROUP_WEIGHT_DFL;
	} else {
		tctx->cgweight = CGROUP_WEIGHT_DFL;
	}

	/* Initialize classification fields */
	tctx->task_class	 = DESCENT_CLASS_NORMAL;
	tctx->prev_class	 = DESCENT_CLASS_NORMAL;
	tctx->class_entry_time	 = bpf_ktime_get_ns();
	tctx->reclassify_counter = 0;
	tctx->runtime_ns	 = 0;

	/* Cache kthread flag from task flags (PF_KTHREAD is bit 21) */
	u8 is_kthread = ((u32)(p->flags >> 21) & 1u);
	tctx->packed  = (is_kthread << BIT_KTHREAD);

	/* Initialize PPID from parent */
	tctx->ppid = 0;

	/* NEW: Phase 3 - Initialize momentum-based classification */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		tctx->class_momentum.vote_ewma[i] = 0;
	}
	tctx->class_momentum.last_vote_update	= 0;
	tctx->class_momentum.current_confidence = 0;

	/* Set default profile thresholds (productivity) */
	set_profile_thresholds(tctx, PROFILE_PRODUCTIVITY);
	tctx->profile_id = PROFILE_PRODUCTIVITY;

	return 0;
}

/*
 * Evaluate the amount of online CPUs.
 */
s32 get_nr_online_cpus(void)
{
	const struct cpumask *online_cpumask;
	int		      cpus;

	online_cpumask = scx_bpf_get_online_cpumask();
	cpus	       = bpf_cpumask_weight(online_cpumask);
	scx_bpf_put_cpumask(online_cpumask);

	return cpus;
}

SEC("syscall")
int enable_sibling_cpu(struct domain_arg *input)
{
	struct cpu_ctx	   *cctx;
	struct bpf_cpumask *mask, **pmask;
	int		    err = 0;

	cctx			= try_lookup_cpu_ctx(input->cpu_id);
	if (!cctx)
		return -ENOENT;

	/* Make sure the target CPU mask is initialized */
	switch (input->lvl_id) {
	case 0:
		pmask = &cctx->smt;
		break;
	default:
		return -EINVAL;
	}
	err = init_cpumask(pmask);
	if (err)
		return err;

	bpf_rcu_read_lock();
	mask = *pmask;
	if (mask)
		bpf_cpumask_set_cpu(input->sibling_cpu_id, mask);
	bpf_rcu_read_unlock();

	return err;
}

SEC("syscall")
int enable_primary_cpu(struct cpu_arg *input)
{
	struct bpf_cpumask *mask;
	int		    err = 0;

	/* Make sure the primary CPU mask is initialized */
	err = init_cpumask(&primary_cpumask);
	if (err)
		return err;
	/*
	 * Enable the target CPU in the primary scheduling domain. If the
	 * target CPU is a negative value, clear the whole mask (this can be
	 * used to reset the primary domain).
	 */
	bpf_rcu_read_lock();
	mask = primary_cpumask;
	if (mask) {
		s32 cpu = input->cpu_id;

		if (cpu < 0)
			bpf_cpumask_clear(mask);
		else
			bpf_cpumask_set_cpu(cpu, mask);
	}
	bpf_rcu_read_unlock();

	return err;
}

/*
 * Initialize cpufreq performance level on all the online CPUs.
 */
static void init_cpuperf_target(void)
{
	const struct cpumask *online_cpumask;
	u64		      perf_lvl;
	s32		      cpu;

	online_cpumask = scx_bpf_get_online_cpumask();
	bpf_for(cpu, 0, nr_cpu_ids)
	{
		if (!bpf_cpumask_test_cpu(cpu, online_cpumask))
			continue;

		/* Set the initial cpufreq performance level  */
		if (cpufreq_perf_lvl < 0)
			perf_lvl = SCX_CPUPERF_ONE;
		else
			perf_lvl = MIN(cpufreq_perf_lvl, SCX_CPUPERF_ONE);
		scx_bpf_cpuperf_set(cpu, perf_lvl);
	}
	scx_bpf_put_cpumask(online_cpumask);
}

/*
 * Tickless timer used to preempt CPUs.
 */
static int tickless_timerfn(void *map, int *key, struct bpf_timer *timer)
{
	int		 node, err;
	s32		 cpu;
	struct task_ctx *tctx;

	/*
	 * Check if we need to preempt the running tasks.
	 */
	bpf_for(cpu, 0, nr_cpu_ids)
	{
		struct task_struct *p = __COMPAT_scx_bpf_cpu_curr(cpu);

		/*
		 * Ignore CPU if idle task is running.
		 */
		if (!p || p->flags & PF_IDLE)
			continue;

		/*
		 * Ignore CPUs without any task waiting.
		 */
		node = __COMPAT_scx_bpf_cpu_node(cpu);
		if (!scx_bpf_dsq_nr_queued(node) &&
		    !scx_bpf_dsq_nr_queued(SCX_DSQ_LOCAL_ON | cpu))
			continue;

		if (p->scx.slice != SCX_SLICE_INF)
			continue;

		p = bpf_task_from_pid(p->pid);
		if (!p)
			continue;

		/*
		 * Preempt the running task if it has an infinite time
		 * slice and has been running for more than base_slice.
		 */
		tctx = try_lookup_task_ctx(p);
		if (tctx) {
			u64 base_slice = DEFAULT_BASE_SLICE_NS;
			struct cpu_descent_ctx *cdctx =
				try_lookup_cpu_descent_ctx();
			if (cdctx && tctx->task_class < DESCENT_CLASS_MAX)
				base_slice =
					cdctx->class_params[tctx->task_class]
						.base_slice_ns;

			u64 slice = bpf_ktime_get_ns() - tctx->last_run_at;

			if (slice > base_slice)
				p->scx.slice = 0;
		}
		bpf_task_release(p);
	}

	err = bpf_timer_start(timer, tick_interval_ns(), 0);
	if (err)
		scx_bpf_error("Failed to re-arm tickless timer");

	return 0;
}

/*
 * Throttle timer used to inject idle time across all the CPUs.
 */
static int throttle_timerfn(void *map, int *key, struct bpf_timer *timer)
{
	bool throttled = is_throttled();
	u64  flags, duration;
	s32  cpu;
	int  err;

	/*
	 * Stop the CPUs sending a preemption IPI (SCX_KICK_PREEMPT) if we
	 * need to interrupt the running tasks and inject the idle sleep.
	 *
	 * Otherwise, send a wakeup IPI to resume from the injected idle
	 * sleep.
	 */
	if (throttled) {
		flags	 = SCX_KICK_IDLE;
		duration = slice_max;
	} else {
		flags	 = SCX_KICK_PREEMPT;
		duration = throttle_ns;
	}

	/*
	 * Flip the throttled state.
	 */
	set_throttled(!throttled);

	bpf_for(cpu, 0, nr_cpu_ids) scx_bpf_kick_cpu(cpu, flags);

	/*
	 * Re-arm the duty-cycle timer setting the runtime or the idle time
	 * duration.
	 */
	err = bpf_timer_start(timer, duration, 0);
	if (err)
		scx_bpf_error("Failed to re-arm duty cycle timer");

	return 0;
}

s32 BPF_STRUCT_OPS_SLEEPABLE(descent_init)
{
	struct bpf_timer       *timer;
	struct cpu_descent_ctx *cdctx;
	int			err, node, cpu;
	u32			key = 0;

	/* Initialize amount of online and possible CPUs */
	nr_online_cpus = get_nr_online_cpus();
	nr_cpu_ids     = scx_bpf_nr_cpu_ids();

	/* Initialize CPUs and NUMA properties */
	init_cpuperf_target();

	/* Create per-node DSQs */
	bpf_for(node, 0, __COMPAT_scx_bpf_nr_node_ids())
	{
		err = scx_bpf_create_dsq(node, node);
		if (err) {
			scx_bpf_error("failed to create DSQ %d: %d", node, err);
			return err;
		}
	}

	/* Initialize the primary scheduling domain */
	err = init_cpumask(&primary_cpumask);
	if (err)
		return err;

	/* Initialize per-CPU descent parameters */
	bpf_for(cpu, 0, nr_cpu_ids)
	{
		cdctx = bpf_map_lookup_percpu_elem(&cpu_descent_ctx_stor, &key,
						   cpu);
		if (cdctx)
			init_class_params(cdctx);

		/* Also initialize syscall-accessible class_params_stor map */
		init_cpu_class_params(cpu);
	}

	timer = bpf_map_lookup_elem(&tickless_timer, &key);
	if (!timer) {
		scx_bpf_error("Failed to lookup tickless timer");
		return -ESRCH;
	}

	/*
	 * Fire the tickless timer if tickless mode is enabled.
	 */
	if (tickless_sched) {
		bpf_timer_init(timer, &tickless_timer, CLOCK_MONOTONIC);
		bpf_timer_set_callback(timer, tickless_timerfn);
		err = bpf_timer_start(timer, tick_interval_ns(), 0);
		if (err) {
			scx_bpf_error("Failed to arm tickless timer");
			return err;
		}
	}

	timer = bpf_map_lookup_elem(&throttle_timer, &key);
	if (!timer) {
		scx_bpf_error("Failed to lookup throttle timer");
		return -ESRCH;
	}

	/*
	 * Fire the throttle timer if CPU throttling is enabled.
	 */
	if (throttle_ns) {
		bpf_timer_init(timer, &throttle_timer, CLOCK_MONOTONIC);
		bpf_timer_set_callback(timer, throttle_timerfn);
		err = bpf_timer_start(timer, tick_interval_ns(), 0);
		if (err) {
			scx_bpf_error("Failed to arm throttle timer");
			return err;
		}
	}

	return 0;
}

/*
 * Syscall program to update class parameters from userspace.
 * Uses syscall-accessible class_params_stor map (BPF_MAP_TYPE_ARRAY)
 * instead of percpu map to avoid kfunc dependency.
 */
SEC("syscall")
int update_class_params(struct descent_params_update *input)
{
	struct class_params    *cp;
	struct cpu_descent_ctx *cdctx;
	u32			key;

	/* Validate inputs */
	if (input->cpu_id < 0 || (u32)input->cpu_id >= MAX_CPUS)
		return -EINVAL;
	if (input->class_id >= DESCENT_CLASS_MAX)
		return -EINVAL;

	key = params_key(input->cpu_id, input->class_id);

	/* Update class_params_stor using regular bpf_map_lookup_elem (syscall-safe) */
	cp = bpf_map_lookup_elem(&class_params_stor, &key);
	if (!cp)
		return -ENOENT;

	/* Update parameters with bounds checking */
	cp->latency_weight = clamp_param_value(0, input->latency_weight);
	cp->base_slice_ns  = clamp_param_value(1, input->base_slice_ns);
	cp->vruntime_scale = clamp_param_value(2, input->vruntime_scale);
	cp->preemption_priority =
		clamp_param_value(3, input->preemption_priority);
	cp->migration_cost = clamp_param_value(4, input->migration_cost);

	/* NEW: Reset latency accumulators for this class in the per-CPU context */
	cdctx = try_lookup_cpu_descent_ctx();
	if (cdctx && input->class_id < DESCENT_CLASS_MAX) {
		reset_latency_accumulator(cdctx, input->class_id);
	}

	return 0;
}

/*
 * Syscall program to reset load accumulators after userspace has read them.
 * This is called after update_autorate_and_pie() processes load metrics.
 */
SEC("syscall")
int reset_load_accumulators(struct reset_load_args *args)
{
	struct cpu_descent_ctx *cdctx;
	u32			key = 0;
	int			cpu;

	/* Validate class_id */
	if (args->class_id >= DESCENT_CLASS_MAX)
		return -EINVAL;

	/* Reset for specific CPU or all CPUs */
	if (args->cpu_id >= 0) {
		/* Single CPU mode */
		if ((u32)args->cpu_id >= MAX_CPUS)
			return -EINVAL;

		cdctx = bpf_map_lookup_percpu_elem(&cpu_descent_ctx_stor, &key,
						   args->cpu_id);
		if (!cdctx)
			return -ENOENT;

		reset_load_accumulator(cdctx, args->class_id);
	} else {
		/* All CPUs mode (-1) */
		for (cpu = 0; cpu < MAX_CPUS; cpu++) {
			cdctx = bpf_map_lookup_percpu_elem(
				&cpu_descent_ctx_stor, &key, cpu);
			if (cdctx)
				reset_load_accumulator(cdctx, args->class_id);
		}
	}

	return 0;
}

void BPF_STRUCT_OPS(descent_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(descent_ops, .select_cpu = (void *)descent_select_cpu,
	       .enqueue		  = (void *)descent_enqueue,
	       .dispatch	  = (void *)descent_dispatch,
	       .running		  = (void *)descent_running,
	       .stopping	  = (void *)descent_stopping,
	       .runnable	  = (void *)descent_runnable,
	       .enable		  = (void *)descent_enable,
	       .cgroup_init	  = (void *)descent_cgroup_init,
	       .cgroup_set_weight = (void *)descent_cgroup_set_weight,
	       .cgroup_move	  = (void *)descent_cgroup_move,
	       .init_task	  = (void *)descent_init_task,
	       .init = (void *)descent_init, .exit = (void *)descent_exit,
	       .timeout_ms = 5000, .name = "descent");
