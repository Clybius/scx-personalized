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
const volatile u64   throttle_ns;
static volatile bool cpus_throttled;

static inline bool   is_throttled(void)
{
	if (!throttle_ns)
		return false;

	return READ_ONCE(cpus_throttled);
}

static inline void set_throttled(bool state)
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
 * Per-CPU descent context map
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct cpu_descent_ctx);
} cpu_descent_ctx_stor SEC(".maps");

/*
 * Timer used to inject perturbations for gradient estimation.
 */
struct perturb_timer {
	struct bpf_timer timer;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct perturb_timer);
} perturb_timer SEC(".maps");

/*
 * Ring buffer for gradient events to userspace.
 */
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024); /* 256KB buffer */
} gradient_events SEC(".maps");

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
 * Parameter helper functions for perturbation
 */
static u64 get_param_value(struct class_params *cp, s32 idx)
{
	switch (idx) {
	case 0:
		return cp->latency_weight;
	case 1:
		return cp->base_slice_ns;
	case 2:
		return cp->vruntime_scale;
	case 3:
		return cp->preemption_priority;
	case 4:
		return cp->migration_cost;
	default:
		return 0;
	}
}

static void set_param_value(struct class_params *cp, s32 idx, u64 value)
{
	switch (idx) {
	case 0:
		cp->latency_weight = value;
		break;
	case 1:
		cp->base_slice_ns = value;
		break;
	case 2:
		cp->vruntime_scale = value;
		break;
	case 3:
		cp->preemption_priority = value;
		break;
	case 4:
		cp->migration_cost = value;
		break;
	}
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

static void apply_perturbation(struct class_params *cp, s32 idx, s64 delta)
{
	u64 current = get_param_value(cp, idx);
	u64 new_val;

	if (delta < 0 && current < (u64)(-delta))
		new_val = 0;
	else
		new_val = (s64)current + delta;

	/* Apply bounds */
	new_val = clamp_param_value(idx, new_val);
	set_param_value(cp, idx, new_val);
}

static u64 calculate_epsilon(struct class_params *cp, s32 idx)
{
	/* Adaptive epsilon: 5% perturbation */
	u64 base    = get_param_value(cp, idx);
	u64 epsilon = base / 20;

	/* Minimum epsilon to ensure measurable effect */
	u64 min_epsilon = 1;
	if (idx == 0 || idx == 1 || idx == 4) {
		/* Time-based parameters: 100us minimum */
		min_epsilon = 100 * NSEC_PER_USEC;
	} else if (idx == 2) {
		/* Vruntime scale: 10 minimum */
		min_epsilon = 10;
	} else if (idx == 3) {
		/* Preemption priority: 1 minimum */
		min_epsilon = 1;
	}

	if (epsilon < min_epsilon)
		epsilon = min_epsilon;

	return epsilon;
}

static u64 read_accumulated_loss(struct cpu_descent_ctx *cdctx, u32 class_id)
{
	if (class_id >= DESCENT_CLASS_MAX)
		return 0;

	struct class_loss_accumulator *accum = &cdctx->class_loss[class_id];

	/* Compute composite loss: latency + deadline misses */
	u64 loss = accum->latency_loss_sum;
	if (accum->sample_count > 0)
		loss = loss / accum->sample_count;

	/* Add penalty for deadline misses */
	loss += accum->deadline_misses * NSEC_PER_MSEC;

	return loss;
}

static void reset_loss_accumulator(struct cpu_descent_ctx *cdctx, u32 class_id)
{
	if (class_id >= DESCENT_CLASS_MAX)
		return;

	struct class_loss_accumulator *accum = &cdctx->class_loss[class_id];
	accum->latency_loss_sum		     = 0;
	accum->deadline_misses		     = 0;
	accum->cpu_time_ns		     = 0;
	accum->sample_count		     = 0;
}

static void send_gradient_ready_event(s32 cpu, u32 class_id, s32 param_idx,
				      u64 loss_plus, u64 loss_minus, u64 eps)
{
	struct gradient_ready_event *e =
		bpf_ringbuf_reserve(&gradient_events, sizeof(*e), 0);
	if (!e)
		return;

	e->cpu_id     = cpu;
	e->class_id   = class_id;
	e->param_idx  = param_idx;
	e->loss_plus  = loss_plus;
	e->loss_minus = loss_minus;
	e->epsilon    = eps;
	e->timestamp  = bpf_ktime_get_ns();

	bpf_ringbuf_submit(e, 0);
}

/*
 * Initialize class parameters with defaults
 */
static void init_class_params(struct cpu_descent_ctx *cdctx)
{
	/* Interactive */
	cdctx->class_params[DESCENT_CLASS_INTERACTIVE].latency_weight =
		INTERACTIVE_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_INTERACTIVE].base_slice_ns =
		INTERACTIVE_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_INTERACTIVE].vruntime_scale =
		INTERACTIVE_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_INTERACTIVE].preemption_priority =
		INTERACTIVE_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_INTERACTIVE].migration_cost =
		INTERACTIVE_MIGRATION_COST_NS;

	/* Audio */
	cdctx->class_params[DESCENT_CLASS_AUDIO].latency_weight =
		AUDIO_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_AUDIO].base_slice_ns =
		AUDIO_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_AUDIO].vruntime_scale =
		AUDIO_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_AUDIO].preemption_priority =
		AUDIO_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_AUDIO].migration_cost =
		AUDIO_MIGRATION_COST_NS;

	/* Batch */
	cdctx->class_params[DESCENT_CLASS_BATCH].latency_weight =
		BATCH_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_BATCH].base_slice_ns =
		BATCH_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_BATCH].vruntime_scale =
		BATCH_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_BATCH].preemption_priority =
		BATCH_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_BATCH].migration_cost =
		BATCH_MIGRATION_COST_NS;

	/* Kernel */
	cdctx->class_params[DESCENT_CLASS_KERNEL].latency_weight =
		KERNEL_LATENCY_WEIGHT_NS;
	cdctx->class_params[DESCENT_CLASS_KERNEL].base_slice_ns =
		KERNEL_BASE_SLICE_NS;
	cdctx->class_params[DESCENT_CLASS_KERNEL].vruntime_scale =
		KERNEL_VRUNTIME_SCALE;
	cdctx->class_params[DESCENT_CLASS_KERNEL].preemption_priority =
		KERNEL_PREEMPTION_PRIORITY;
	cdctx->class_params[DESCENT_CLASS_KERNEL].migration_cost =
		KERNEL_MIGRATION_COST_NS;

	cdctx->last_param_sync		  = bpf_ktime_get_ns();
	cdctx->current_perturbation_start = 0;

	/* Initialize perturbation state for all classes */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		/* Initialize loss accumulators */
		cdctx->class_loss[i].latency_loss_sum = 0;
		cdctx->class_loss[i].deadline_misses  = 0;
		cdctx->class_loss[i].cpu_time_ns      = 0;
		cdctx->class_loss[i].target_share_ns  = 0;
		cdctx->class_loss[i].sample_count     = 0;
		cdctx->class_loss[i].active	      = false;

		/* Initialize baseline loss storage */
		cdctx->loss_baseline[i] = 0;
	}

	/* Initialize perturbation state machine */
	cdctx->perturb.param_idx      = 0;
	cdctx->perturb.phase	      = PERTURB_BASELINE;
	cdctx->perturb.phase_start_ns = 0;
	cdctx->perturb.current_class  = 0;
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
	 * cgroup weight (cpu.weight).
	 */
	u32 cgweight;

	/*
	 * NEW: Classification and descent fields
	 */
	u32 task_class; /* Current assigned class */
	u32 prev_class; /* Previous class (for hysteresis) */
	u64 class_entry_time; /* When entered current class */
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
static bool is_pcpu_task(const struct task_struct *p)
{
	return p->nr_cpus_allowed == 1 || is_migration_disabled(p);
}

/*
 * Return true if @p still wants to run, false otherwise.
 */
static bool is_queued(const struct task_struct *p)
{
	return p->scx.flags & SCX_TASK_QUEUED;
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
	struct cpu_descent_ctx *cdctx;
	u64			base_slice;

	if (tickless_sched)
		return SCX_SLICE_INF;

	cdctx = try_lookup_cpu_descent_ctx();
	if (cdctx && tctx->task_class < DESCENT_CLASS_MAX) {
		base_slice =
			cdctx->class_params[tctx->task_class].base_slice_ns;
		/* Scale by weight */
		return scale_by_weight(p, base_slice);
	}

	return scale_by_weight(p, slice_max);
}

/*
 * Classify a task into one of the descent classes.
 *
 * AUDIO: Real-time scheduling policy (SCHED_FIFO/SCHED_RR)
 * INTERACTIVE: High wakeup frequency AND short runtime slices
 * BATCH: Long slices AND low wakeup frequency
 * KERNEL: Kernel threads
 * Default: INTERACTIVE
 */
static u32 classify_task(struct task_struct *p, struct task_ctx *tctx)
{
	/* Audio: Real-time policy */
	if (p->policy == SCHED_FIFO || p->policy == SCHED_RR)
		return DESCENT_CLASS_AUDIO;

	/* Kernel: Kernel threads */
	if (is_kthread(p))
		return DESCENT_CLASS_KERNEL;

	/* Use heuristics based on observed behavior */
	if (tctx->slice_ns_ewma && tctx->wakeup_freq) {
		/* Interactive: high wakeup freq + short slices */
		if (tctx->wakeup_freq > WAKEUP_FREQ_INTERACTIVE_THRESH &&
		    tctx->slice_ns_ewma < SLICE_NS_INTERACTIVE_THRESH)
			return DESCENT_CLASS_INTERACTIVE;

		/* Batch: low wakeup freq + long slices */
		if (tctx->wakeup_freq < WAKEUP_FREQ_BATCH_THRESH &&
		    tctx->slice_ns_ewma > SLICE_NS_BATCH_THRESH)
			return DESCENT_CLASS_BATCH;
	}

	/* Default to interactive for unknown behavior */
	return DESCENT_CLASS_INTERACTIVE;
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
	/* Initialize all votes to 0 */
	for (int i = 0; i < DESCENT_CLASS_MAX; i++) {
		votes[i] = 0;
	}

	/* Audio: Real-time scheduling policy */
	if (p->policy == SCHED_FIFO || p->policy == SCHED_RR) {
		votes[DESCENT_CLASS_AUDIO] = 100;
		return;
	}

	/* Kernel: Kernel threads */
	if (is_kthread(p)) {
		votes[DESCENT_CLASS_KERNEL] = 100;
		return;
	}

	/* Use heuristics based on observed behavior */
	if (tctx->slice_ns_ewma && tctx->wakeup_freq) {
		/* Interactive: high wakeup freq + short slices */
		if (tctx->wakeup_freq > WAKEUP_FREQ_INTERACTIVE_THRESH &&
		    tctx->slice_ns_ewma < SLICE_NS_INTERACTIVE_THRESH) {
			votes[DESCENT_CLASS_INTERACTIVE] = 80;
			votes[DESCENT_CLASS_BATCH]	 = 20;
		}
		/* Batch: low wakeup freq + long slices */
		else if (tctx->wakeup_freq < WAKEUP_FREQ_BATCH_THRESH &&
			 tctx->slice_ns_ewma > SLICE_NS_BATCH_THRESH) {
			votes[DESCENT_CLASS_BATCH]	 = 80;
			votes[DESCENT_CLASS_INTERACTIVE] = 20;
		}
		/* Mixed - prefer interactive for unknown */
		else {
			votes[DESCENT_CLASS_INTERACTIVE] = 60;
			votes[DESCENT_CLASS_BATCH]	 = 40;
		}
	} else {
		/* Default to interactive for unknown behavior */
		votes[DESCENT_CLASS_INTERACTIVE] = 100;
	}
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
	struct cpu_descent_ctx *cdctx;
	struct class_params    *cp;
	u64			lag_scale, vsleep_max, vtime_min;
	u64			vtime = p->scx.dsq_vtime;
	u64			scaled_vtime;

	/* Get per-CPU descent context */
	cdctx = try_lookup_cpu_descent_ctx();
	if (!cdctx || tctx->task_class >= DESCENT_CLASS_MAX) {
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

	cp = &cdctx->class_params[tctx->task_class];

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
			 bool *is_idle)
{
	const struct cpumask *primary = cast_mask(primary_cpumask);
	s32		      cpu;

	/*
	 * Compatibility with older kernels (< v6.14).
	 */
	if (!__COMPAT_HAS_scx_bpf_select_cpu_and) {
		if (wake_flags)
			return scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags,
						      is_idle);

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
	*is_idle = true;

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
	bool		 is_idle = false;
	s32		 cpu;

	if (is_throttled())
		return prev_cpu;

	/* Update task classification using momentum-based method */
	tctx = try_lookup_task_ctx(p);
	if (tctx)
		update_classification_momentum(p, tctx);

	cpu = pick_idle_cpu(p, prev_cpu, wake_flags, &is_idle);
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
static bool is_smt_contended(s32 cpu)
{
	const struct cpumask *idle_mask;
	bool		      is_contended;

	if (!smt_enabled)
		return false;

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
static bool is_primary_cpu(const struct task_struct *p, s32 cpu)
{
	if (!primary_all) {
		const struct cpumask *primary = cast_mask(primary_cpumask);

		if (primary && bpf_cpumask_intersects(primary, p->cpus_ptr) &&
		    !bpf_cpumask_test_cpu(cpu, primary))
			return false;
	}

	return true;
}

/*
 * Attempt to dispatch a task directly to its assigned CPU.
 *
 * Return true if the task is dispatched, false otherwise.
 */
static bool try_direct_dispatch(struct task_struct *p, s32 prev_cpu,
				u64 enq_flags, bool is_running)
{
	bool		 is_idle = false;
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

		return true;
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
		return false;

	/*
	 * Try migrating to an idle CPU.
	 */
	if (!is_pcpu_task(p)) {
		cpu = pick_idle_cpu(p, prev_cpu, 0, &is_idle);
		if (!is_idle)
			return false;
	} else {
		if (!scx_bpf_test_and_clear_cpu_idle(prev_cpu))
			return false;
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

	return true;
}

/*
 * Enqueue a task when running in round-robin mode.
 */
static void rr_enqueue(struct task_struct *p, s32 prev_cpu, u64 enq_flags)
{
	bool		 is_idle;
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
static bool keep_running(const struct task_struct *p, s32 cpu)
{
	/* Do not keep running if the task doesn't need to run */
	if (!is_queued(p))
		return false;

	/*
	 * If the task can't migrate elsewhere, keep it running.
	 */
	if (p->nr_cpus_allowed == 1)
		return true;

	/*
	 * Do not keep running if the CPU is not in the primary domain and
	 * the task can use the primary domain.
	 */
	if (!is_primary_cpu(p, cpu))
		return false;

	/*
	 * If the task is running on a CPU with a busy SMT sibling, try to
	 * move it elsewhere.
	 */
	if (is_smt_contended(cpu))
		return false;

	return true;
}

void BPF_STRUCT_OPS(descent_dispatch, s32 cpu, struct task_struct *prev)
{
	int  node	  = __COMPAT_scx_bpf_cpu_node(cpu);
	bool need_running = prev && keep_running(prev, cpu);

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
	tctx->last_run_at = bpf_ktime_get_ns();

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
		 * NEW: Phase 3 - Populate loss accumulator for gradient descent
		 */
		struct cpu_descent_ctx *cdctx = try_lookup_cpu_descent_ctx();
		if (cdctx) {
			u32 class_id = tctx->task_class;
			if (class_id < DESCENT_CLASS_MAX) {
				struct class_loss_accumulator *accum =
					&cdctx->class_loss[class_id];

				if (accum->active) {
					/* Track CPU time for throughput/fairness calculation */
					accum->cpu_time_ns += slice;

					/* Track actual runtime vs expected (for throughput_loss) */
					struct class_params *cp =
						&cdctx->class_params[class_id];
					if (slice > cp->base_slice_ns) {
						accum->deadline_misses++;
					}

					accum->sample_count++;
				}
			}
		}

		/*
		 * Periodically reclassify task based on observed behavior
		 * using momentum-based classification with hysteresis
		 */
		if ((now - tctx->class_entry_time) > (100ULL * NSEC_PER_MSEC))
			update_classification_momentum(p, tctx);
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
	 * NEW: Phase 3 - Track wakeup latency for loss computation
	 */
	u64 wakeup_latency = now - tctx->last_woke_at;

	/* Update EWMA for classification */
	tctx->class_metrics.wakeup_latency_ewma = calc_avg(
		tctx->class_metrics.wakeup_latency_ewma, wakeup_latency);

	/* Accumulate for loss if active */
	struct cpu_descent_ctx *cdctx = try_lookup_cpu_descent_ctx();
	if (cdctx) {
		u32 class_id = tctx->task_class;
		if (class_id < DESCENT_CLASS_MAX) {
			struct class_loss_accumulator *accum =
				&cdctx->class_loss[class_id];

			if (accum->active) {
				/* Square latency to penalize outliers (scale to μs to avoid overflow) */
				u64 latency_us = wakeup_latency / 1000;
				u64 latency_sq = latency_us * latency_us;
				accum->latency_loss_sum += latency_sq;
			}
		}
	}

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
	tctx->task_class       = DESCENT_CLASS_INTERACTIVE;
	tctx->prev_class       = DESCENT_CLASS_INTERACTIVE;
	tctx->class_entry_time = bpf_ktime_get_ns();

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

/*
 * Perturbation timer used for gradient estimation.
 * Called every 10ms to advance perturbations through state machine.
 * 
 * Phase 3: Implements proper sequential perturbation with state tracking:
 * - State tracked per-CPU, not per-class
 * - Sequential: parameter 0 → 1 → 2 → 3 → 4 for class 0, then same for class 1, etc.
 * - Adaptive timing: extend window if not enough samples (< 3)
 * - Proper phase cycle: BASELINE → PLUS → MINUS → send event → next
 */
static int perturb_timerfn(void *map, int *key, struct bpf_timer *timer)
{
	s32			cpu   = bpf_get_smp_processor_id();
	struct cpu_descent_ctx *cdctx = try_lookup_cpu_descent_ctx();
	u64			now   = bpf_ktime_get_ns();

	if (!cdctx)
		goto rearm;

	struct perturb_state_machine *ps       = &cdctx->perturb;
	u32			      class_id = ps->current_class;

	if (class_id >= DESCENT_CLASS_MAX)
		goto rearm;

	struct class_params	      *cp    = &cdctx->class_params[class_id];
	struct class_loss_accumulator *accum = &cdctx->class_loss[class_id];

	/* Check if phase should advance (use 10-20ms adaptive window) */
	u64 elapsed = now - ps->phase_start_ns;

	/* Extend window if not enough samples collected */
	if (accum->sample_count < 3 && elapsed < 20 * NSEC_PER_MSEC) {
		/* Wait for more samples */
		bpf_timer_start(timer, 5 * NSEC_PER_MSEC, 0);
		return 0;
	}

	/* Advance state machine */
	switch (ps->phase) {
	case PERTURB_BASELINE:
		/* Start +ε perturbation */
		cp->original_value  = get_param_value(cp, ps->param_idx);
		cp->perturb_epsilon = calculate_epsilon(cp, ps->param_idx);
		apply_perturbation(cp, ps->param_idx, (s64)cp->perturb_epsilon);
		cdctx->loss_baseline[class_id] =
			read_accumulated_loss(cdctx, class_id);
		reset_loss_accumulator(cdctx, class_id);
		accum->active = true;
		ps->phase     = PERTURB_PLUS;
		break;

	case PERTURB_PLUS:
		/* Switch to -ε (apply -2ε from current to get to θ-ε) */
		apply_perturbation(cp, ps->param_idx,
				   -2 * (s64)cp->perturb_epsilon);
		cp->loss_plus = read_accumulated_loss(cdctx, class_id);
		reset_loss_accumulator(cdctx, class_id);
		accum->active = true;
		ps->phase     = PERTURB_MINUS;
		break;

	case PERTURB_MINUS:
		/* Restore and compute gradient */
		apply_perturbation(cp, ps->param_idx, (s64)cp->perturb_epsilon);
		cp->loss_minus = read_accumulated_loss(cdctx, class_id);
		accum->active  = false;

		/* Send event to userspace */
		send_gradient_ready_event(cpu, class_id, ps->param_idx,
					  cp->loss_plus, cp->loss_minus,
					  cp->perturb_epsilon);

		/* Move to next parameter or class */
		ps->phase = PERTURB_BASELINE;
		ps->param_idx++;
		if (ps->param_idx >= 5) {
			ps->param_idx = 0;
			ps->current_class++;
			if (ps->current_class >= DESCENT_CLASS_MAX) {
				ps->current_class = 0;
			}
		}
		break;

	default:
		/* Reset to baseline if in unknown state */
		ps->phase = PERTURB_BASELINE;
		break;
	}

	ps->phase_start_ns = now;

rearm:
	/* Re-arm timer for 10ms */
	bpf_timer_start(timer, 10 * NSEC_PER_MSEC, 0);
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

	/*
	 * Initialize and start perturbation timer for gradient estimation.
	 */
	timer = bpf_map_lookup_elem(&perturb_timer, &key);
	if (timer) {
		bpf_timer_init(timer, &perturb_timer, CLOCK_MONOTONIC);
		bpf_timer_set_callback(timer, perturb_timerfn);
		err = bpf_timer_start(timer, 10 * NSEC_PER_MSEC, 0);
		if (err)
			scx_bpf_error("Failed to arm perturbation timer");
	}

	return 0;
}

/*
 * Syscall program to update class parameters from userspace.
 */
SEC("syscall")
int update_class_params(struct descent_params_update *input)
{
	struct cpu_descent_ctx *cdctx;
	u32			key = 0;

	/* Get the per-CPU context for the target CPU */
	cdctx = bpf_map_lookup_percpu_elem(&cpu_descent_ctx_stor, &key,
					   input->cpu_id);
	if (!cdctx)
		return -ENOENT;

	if (input->class_id >= DESCENT_CLASS_MAX)
		return -EINVAL;

	/* Update parameters with bounds checking */
	struct class_params *cp = &cdctx->class_params[input->class_id];
	cp->latency_weight	= clamp_param_value(0, input->latency_weight);
	cp->base_slice_ns	= clamp_param_value(1, input->base_slice_ns);
	cp->vruntime_scale	= clamp_param_value(2, input->vruntime_scale);
	cp->preemption_priority =
		clamp_param_value(3, input->preemption_priority);
	cp->migration_cost     = clamp_param_value(4, input->migration_cost);

	cdctx->last_param_sync = bpf_ktime_get_ns();

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
