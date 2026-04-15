/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2024 scx_happy authors
 *
 * scx_happy: A latency-aware scheduler with virtual nice values
 */

#ifdef LSP
#define __bpf__
#include "../../../../scheds/include/scx/common.bpf.h"
#include "../../../../scheds/include/scx/percpu.bpf.h"
#else
#include <scx/common.bpf.h>
#include <scx/percpu.bpf.h>
#endif

#include <errno.h>
#include <stdbool.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "intf.h"

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

/* Maximum values */
#define MAX_CPUS 1024
#define MAX_LLCS 64
#define MAX_NODES 64
#define MAX_TGIDS 256

/* DSQ ID layout: [queue_id:8][llc_id:56] */
#define DSQ_ID_QUEUE_SHIFT 56
#define LC_DSQ_ID 0
#define NORMAL_DSQ_ID 1
#define HOG_DSQ_ID 2
#define SHARED_DSQ 3

/* Queue configuration - can be overridden from userspace */
const volatile u64 lc_slice_ns	   = 500000; /* 500us */
const volatile u64 normal_slice_ns = 1000000; /* 1000us */
const volatile u64 hog_slice_ns	   = 3000000; /* 3000us */

/* Feature toggles */
const volatile bool avoid_smt	      = true;
const volatile bool cache_affinity    = true;
const volatile bool cpufreq_enabled   = true;
const volatile bool antistall_enabled = true;
const volatile u64  antistall_sec     = 3;

/* HOG demotion parameters */
const volatile u8  hog_cpu_threshold = 50; /* Default 50% CPU threshold */
const volatile u64 hog_cpu_window_ns = 100000000; /* 100ms measurement window */

/* Dynamic virtual nice adjustment parameters */
const volatile bool dynamic_nice_enabled   = true;
const volatile u64  adjust_interval_ns	   = 10000000; /* 10ms default */
const volatile s16  lc_virt_nice_boost	   = 15; /* Boost within LC range */
const volatile s16  normal_virt_nice_boost = 15; /* Boost within NORMAL range */
const volatile s16  hog_virt_nice_penalty  = 10; /* Penalty within HOG range */
const volatile u32  interactive_threshold  = 700; /* Score >700 = interactive */

/* Dynamic adjustment flags */
#define HAPPY_FLAG_IS_SYNC_WAKEUP 0x00000001 /* Woken via sync wakeup */
#define HAPPY_FLAG_IS_WAKEUP 0x00000002 /* Recently woken */
#define HAPPY_FLAG_WOKEN_BY_IRQ 0x00000004 /* Woken by IRQ handler */
#define HAPPY_FLAG_IS_GREEDY 0x00000008 /* Using more than fair share */
#define HAPPY_FLAG_IS_INTERACTIVE 0x00000010 /* Detected interactive pattern */
#define HAPPY_FLAG_DYNAMIC_ADJUST 0x00000020 /* Subject to dynamic adjustment */

#define HAPPY_ADJUST_INTERVAL_NS 10000000 /* Recalculate every 10ms */
#define HAPPY_FREQ_MAX 100000 /* Max frequency cap (100K/sec) */
#define HAPPY_RUNTIME_LC_THRESH_NS 1000000 /* <1ms runtime = very interactive */
#define HAPPY_RUNTIME_NORMAL_THRESH_NS 5000000 /* <5ms = interactive */

/* Domain cpumask storage - global variables with __kptr */
private(HAPPY) struct bpf_cpumask __kptr *lc_cpumask;
private(HAPPY) struct bpf_cpumask __kptr *normal_cpumask;
private(HAPPY) struct bpf_cpumask __kptr *hog_cpumask;

/* Scheduling statistics - in BSS section, exposed to userspace */
volatile u64 nr_lc_dispatches;
volatile u64 nr_normal_dispatches;
volatile u64 nr_hog_dispatches;
volatile u64 nr_preemptions;
volatile u64 nr_migrations;
volatile u64 nr_antistall_dispatches;
volatile u64 nr_smt_avoided;
volatile u64 nr_classified_tasks;
/* NEW: Dynamic adjustment statistics */
volatile u64 nr_dynamic_adjustments;
volatile u64 nr_interactive_detected;
volatile u64 nr_promotions;
volatile u64 nr_demotions;

/* NEW: EEVDF statistics counters */
volatile u64 nr_eligible_dispatches;
volatile u64 nr_ineligible_dispatches;
volatile u64 nr_deadline_expired;

/* NEW: Deadline preemption statistics */
volatile u64	   nr_deadline_preemptions;
volatile u64	   nr_queue_priority_preemptions;
volatile u64	   nr_same_queue_preemptions;
volatile u64	   nr_preemptions_skipped;
volatile u64	   nr_preemptions_ineligible;
volatile u64	   nr_preemptions_later_deadline;

const volatile u32 debug    = 0;
const u32	   zero_u32 = 0;

/* Deadline preemption configuration */
const volatile bool deadline_preemption_enabled = true;
const volatile u64  preemption_min_interval_ns =
	500000; /* 500us minimum between preemptions */
const volatile u8 preemption_hysteresis_pct =
	10; /* Deadline diff threshold % */

/* Task context stored in task storage map */
struct task_ctx {
	s16 virt_nice; /* Virtual nice: -50 to 49 */
	s16 base_virt_nice; /* Original/static virt_nice from classification */
	enum happy_queue queue; /* Assigned queue */
	enum happy_queue base_queue; /* Original queue from classification */
	u64		 last_run_at;
	u64		 exec_runtime;
	u64		 vtime; /* Virtual runtime */
	s32 last_cpu; /* Last CPU task ran on, for migration tracking */

	/* CPU tracking for HOG demotion */
	u64  cpu_window_start; /* Start of current measurement window */
	u64  cpu_window_runtime; /* Cumulative runtime in current window */
	bool demote_to_hog; /* Flag: should be demoted on next enqueue */

	/* NEW: Behavioral tracking for dynamic adjustment */
	u64 wait_freq; /* How often task sleeps (waits) */
	u64 wake_freq; /* How often task wakes others */
	u64 avg_runtime_ns; /* Average runtime per schedule */
	u64 last_runnable_ns; /* Timestamp when became runnable */
	u64 last_quiescent_ns; /* Timestamp when went to sleep */
	u32 flags; /* HAPPY_FLAG_* indicators */

	/* NEW: Dynamic adjustment state */
	s16 target_virt_nice; /* Calculated target (smoothed toward this) */
	u32 dynamic_score; /* Raw 0-1000 score before normalization */
	u64 last_recalc_ns; /* When we last recalculated */

	/* NEW: EEVDF/WFQ fields */
	u64 eligible_vtime; /* When task becomes eligible */
	u64 deadline_vtime; /* Virtual deadline */
	u64 weight; /* WFQ weight based on virt_nice */
	u64 vslice; /* Virtual slice = slice * NICE_0_WEIGHT / weight */
};

/* Per-task storage map */
struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_ctx_stor SEC(".maps");

/* Classification TGID maps from userspace */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_TGIDS);
	__type(key, u32); /* TGID */
	__type(value, u8); /* Classification type */
} scx_turbo_tgids SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_TGIDS);
	__type(key, u32);
	__type(value, u8);
} input_tgids SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_TGIDS);
	__type(key, u32);
	__type(value, u8);
} steam_tgids SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_TGIDS);
	__type(key, u32);
	__type(value, u8);
} de_tgids SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_TGIDS);
	__type(key, u32);
	__type(value, u8);
} audio_tgids SEC(".maps");

/* Per-CPU context */
struct cpu_ctx {
	u64 current_vtime;
	u32 current_queue;
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct cpu_ctx);
	__uint(max_entries, 1);
} cpu_ctx_stor SEC(".maps");

/* Per-CPU running task deadline tracking for preemption */
struct cpu_running_task {
	u64		 deadline_vtime; /* Running task's virtual deadline */
	u64		 vtime; /* Running task's virtual runtime */
	s32		 pid; /* Running task's PID (for debugging) */
	enum happy_queue queue; /* Running task's queue */
	u64 preemption_count; /* Count of preemptions on this CPU */
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct cpu_running_task);
	__uint(max_entries, 1);
} cpu_running_task SEC(".maps");

/* Antistall tracking */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, u64); /* Delayed DSQ ID */
	__uint(max_entries, 1);
} antistall_dsq SEC(".maps");

/* SMT sibling map */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAX_CPUS);
	__type(key, u32);
	__type(value, s32); /* Sibling CPU or -1 */
} smt_sibling_map SEC(".maps");

/* Per-queue EEVDF state - stored in BPF map for mutable global state */
struct queue_eevdf_state {
	u64 min_vtime; /* Minimum vtime in queue */
	u64 avg_vtime; /* Weighted sum for avg calculation */
	u64 total_weight; /* Sum of weights of running tasks */
	u32 nr_tasks; /* Number of active tasks */
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, HAPPY_QUEUE_MAX); /* 3 entries: LC, NORMAL, HOG */
	__type(key, u32); /* queue index */
	__type(value, struct queue_eevdf_state);
} queue_eevdf_states SEC(".maps");

/* Global vtime clock */
static u64 vtime_now;

/* Helper: get domain cpumask for a queue */
static inline const struct cpumask *get_domain_cpumask(enum happy_queue queue)
{
	struct bpf_cpumask *cpumask = NULL;

	switch (queue) {
	case HAPPY_QUEUE_LC:
		cpumask = lc_cpumask;
		break;
	case HAPPY_QUEUE_NORMAL:
		cpumask = normal_cpumask;
		break;
	case HAPPY_QUEUE_HOG:
		cpumask = hog_cpumask;
		break;
	default:
		return NULL;
	}

	if (!cpumask)
		return NULL;

	return (const struct cpumask *)cpumask;
}

/* BPF program to enable/disable a CPU in a domain */
SEC("syscall")
int happy_set_domain_cpu(struct domain_cpu_arg *input)
{
	struct bpf_cpumask **cpumask_ptr = NULL;
	struct bpf_cpumask  *cpumask;

	if (!input)
		return -EINVAL;

	/* Validate queue first */
	if (input->queue > HAPPY_QUEUE_HOG)
		return -EINVAL;

	/* Get pointer to the correct cpumask variable */
	switch (input->queue) {
	case HAPPY_QUEUE_LC:
		cpumask_ptr = &lc_cpumask;
		break;
	case HAPPY_QUEUE_NORMAL:
		cpumask_ptr = &normal_cpumask;
		break;
	case HAPPY_QUEUE_HOG:
		cpumask_ptr = &hog_cpumask;
		break;
	default:
		return -EINVAL;
	}

	/* Initialize cpumask if needed (lazy initialization) */
	if (!*cpumask_ptr) {
		/* Use single variable pattern - cpumask holds new mask initially */
		cpumask = bpf_cpumask_create();
		if (!cpumask)
			return -ENOMEM;

		/* Reassign to xchg result - if non-NULL, release (we lost the race) */
		cpumask = bpf_kptr_xchg(cpumask_ptr, cpumask);
		if (cpumask)
			bpf_cpumask_release(cpumask);
	}

	/* Read and use cpumask INSIDE RCU critical section */
	bpf_rcu_read_lock();

	/* Read from global INSIDE RCU - now verifier sees it as rcu pointer */
	cpumask = *cpumask_ptr;
	if (!cpumask) {
		bpf_rcu_read_unlock();
		return -ENOENT;
	}

	if (input->cpu_id < 0) {
		/* Clear all CPUs (negative cpu_id means reset) */
		bpf_cpumask_clear(cpumask);
	} else {
		/* Set specific CPU */
		bpf_cpumask_set_cpu(input->cpu_id, cpumask);
	}

	bpf_rcu_read_unlock();

	return 0;
}

/* Helper: lookup or create task context */
static __always_inline struct task_ctx *lookup_task_ctx(struct task_struct *p)
{
	struct task_ctx *tctx;

	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0, 0);
	if (!tctx) {
		tctx = bpf_task_storage_get(&task_ctx_stor, p, 0,
					    BPF_LOCAL_STORAGE_GET_F_CREATE);
	}
	return tctx;
}

/* Helper: convert prio to nice (same as prio_to_nice macro) */
static inline s32 prio_to_nice(s32 static_prio)
{
	return static_prio - 120;
}

/* Linux kernel nice-to-weight table for nice -20 to 19 */
/* NICE_0_WEIGHT = 1024 */
static const u32 nice_to_weight[40] = {
	/* nice -20 to -16 */ 88761,
	71755,
	56483,
	46273,
	36291,
	/* nice -15 to -11 */ 29154,
	23254,
	18705,
	14949,
	11916,
	/* nice -10 to -6  */ 9548,
	7620,
	6100,
	4904,
	3906,
	/* nice -5 to -1   */ 3121,
	2501,
	1991,
	1586,
	1277,
	/* nice 0 to 4     */ 1024,
	820,
	655,
	526,
	423,
	/* nice 5 to 9     */ 335,
	272,
	215,
	172,
	137,
	/* nice 10 to 14   */ 110,
	87,
	70,
	56,
	45,
	/* nice 15 to 19   */ 36,
	29,
	23,
	18,
	15
};

#define NICE_0_WEIGHT 1024

/* Convert virtual nice (-50..49) to WFQ weight */
static inline u64 calc_weight_from_virt_nice(s16 virt_nice)
{
	/* Map virt_nice to kernel nice (-20..19) then to weight */
	s32 kernel_nice = ((virt_nice + 50) * 39) / 99 - 20;

	if (kernel_nice < -20)
		kernel_nice = -20;
	if (kernel_nice > 19)
		kernel_nice = 19;

	return (u64)nice_to_weight[kernel_nice + 20];
}

/* Calculate virtual slice: vslice = slice * NICE_0_WEIGHT / weight */
static inline u64 calc_vslice(u64 slice_ns, u64 weight)
{
	if (weight == 0)
		return slice_ns; /* Fallback: no weight scaling */
	return slice_ns * NICE_0_WEIGHT / weight;
}

/* Calculate virtual deadline: vd = vruntime + vslice */
static inline u64 calc_deadline(u64 vtime, u64 vslice)
{
	return vtime + vslice;
}

/* Check if task is eligible to run (lag >= 0) */
/* A task is eligible when its vruntime <= weighted average vtime */
static inline bool is_eligible(struct task_ctx *tctx, enum happy_queue queue)
{
	struct queue_eevdf_state *state;
	u32			  key = (u32)queue;
	u64			  avg_vtime;

	state = bpf_map_lookup_elem(&queue_eevdf_states, &key);
	if (!state)
		return true; /* Default to eligible if lookup fails */

	/* Calculate weighted average vtime */
	avg_vtime = state->min_vtime;
	if (state->total_weight > 0) {
		avg_vtime += state->avg_vtime / state->total_weight;
	}

	/* Eligible if vruntime <= avg_vtime (task is owed CPU time) */
	return tctx->vtime <= avg_vtime;
}

/*
 * Check if enqueuing task should preempt running task.
 *
 * Preemption rules (following Linux EEVDF):
 * 1. Task must be eligible (owed CPU time)
 * 2. Task must have earlier deadline than running task
 * 3. Queue priority is respected (LC > NORMAL > HOG)
 */
static inline bool should_preempt_running(struct task_struct *p,
					  struct task_ctx *tctx, s32 cpu)
{
	struct cpu_running_task *running;

	if (!deadline_preemption_enabled)
		return false;

	/* Get current running task on target CPU */
	running = bpf_map_lookup_percpu_elem(&cpu_running_task, &zero_u32, cpu);
	if (!running || running->pid == 0)
		return false;

	/* 1. Task must be eligible */
	if (!is_eligible(tctx, tctx->queue)) {
		__sync_fetch_and_add(&nr_preemptions_ineligible, 1);
		return false;
	}

	/* 2. Check queue priority first */
	if (tctx->queue < running->queue) {
		/* Higher priority queue - can preempt regardless of deadline */
		__sync_fetch_and_add(&nr_queue_priority_preemptions, 1);
		return true;
	} else if (tctx->queue > running->queue) {
		/* Lower priority queue - cannot preempt */
		return false;
	}

	/* 3. Same queue: compare deadlines (EEVDF rule) */
	if (tctx->deadline_vtime < running->deadline_vtime) {
		/* Add hysteresis to avoid ping-pong with similar deadlines */
		u64 deadline_diff =
			running->deadline_vtime - tctx->deadline_vtime;
		u64 vslice_threshold = tctx->vslice / 10; /* 10% of slice */

		if (deadline_diff < vslice_threshold) {
			__sync_fetch_and_add(&nr_preemptions_skipped, 1);
			return false;
		}

		__sync_fetch_and_add(&nr_same_queue_preemptions, 1);
		return true;
	}

	__sync_fetch_and_add(&nr_preemptions_later_deadline, 1);
	return false;
}

/* Calculate EWMA frequency: freq = alpha * (1/interval) + (1-alpha) * freq */
static inline u64 calc_avg_freq(u64 curr_freq, u64 interval_ns)
{
	/* Using fixed-point arithmetic: EWMA with 1/4 decay */
	u64 new_freq;

	if (interval_ns == 0)
		return curr_freq;

	/* Prevent division by zero and extreme values */
	if (interval_ns < 1000) /* <1us, assume 1us */
		interval_ns = 1000;

	new_freq = 1000000000ULL / interval_ns; /* Frequency in Hz */

	/* EWMA: new = (old * 3 + new) / 4 */
	if (curr_freq == 0)
		return new_freq;

	return ((curr_freq * 3) + new_freq) / 4;
}

/* Calculate absolute value for s16 */
static inline s16 abs_s16(s16 x)
{
	return x < 0 ? -x : x;
}

/* Helper: map existing nice to virtual nice range */
static inline s16 map_nice_to_virt(s32 nice)
{
	/*
	 * Map kernel nice (-20 to 19) to virtual nice (-50 to 49).
	 * Linear mapping: spread 39 nice values across 99 virt values.
	 * Formula: virt = (nice + 20) * 99 / 39 - 50
	 *
	 * nice=-20 → virt=-50 (LC min)
	 * nice=0   → virt=0   (NORMAL center)
	 * nice=19  → virt=49  (HOG max)
	 */
	s16 virt = (s16)(((nice + 20) * 99) / 39 - 50);
	if (virt < HAPPY_VIRT_NICE_MIN)
		virt = HAPPY_VIRT_NICE_MIN;
	if (virt > HAPPY_VIRT_NICE_MAX)
		virt = HAPPY_VIRT_NICE_MAX;
	return virt;
}

/* Task classification: determine queue and virtual nice */
static enum happy_queue classify_task(struct task_struct *p,
				      struct task_ctx	 *tctx)
{
	u32 tgid = p->tgid;
	u8 *val;

	/* Check SCX_TURBO TGIDs */
	val = bpf_map_lookup_elem(&scx_turbo_tgids, &tgid);
	if (val) {
		tctx->virt_nice = HAPPY_VIRT_NICE_TURBO;
		return HAPPY_QUEUE_LC;
	}

	/* Check Steam game TGIDs */
	val = bpf_map_lookup_elem(&steam_tgids, &tgid);
	if (val) {
		tctx->virt_nice = HAPPY_VIRT_NICE_STEAM;
		return HAPPY_QUEUE_LC;
	}

	/* Check DE component TGIDs */
	val = bpf_map_lookup_elem(&de_tgids, &tgid);
	if (val) {
		tctx->virt_nice = HAPPY_VIRT_NICE_DE;
		return HAPPY_QUEUE_LC;
	}

	/* Check input threads */
	val = bpf_map_lookup_elem(&input_tgids, &tgid);
	if (val) {
		tctx->virt_nice = HAPPY_VIRT_NICE_INPUT;
		return HAPPY_QUEUE_LC;
	}

	/* Check audio threads */
	val = bpf_map_lookup_elem(&audio_tgids, &tgid);
	if (val) {
		tctx->virt_nice = HAPPY_VIRT_NICE_AUDIO;
		return HAPPY_QUEUE_LC;
	}

	/* Check kernel threads - slight boost */
	if (p->flags & PF_KTHREAD) {
		tctx->virt_nice = HAPPY_VIRT_NICE_KTHREAD;
		return HAPPY_QUEUE_NORMAL;
	}

	/* Default: map existing nice to virtual nice */
	s32 nice	= prio_to_nice((s32)p->static_prio);
	tctx->virt_nice = map_nice_to_virt(nice);

	/* Determine queue based on virtual nice */
	if (tctx->virt_nice <= HAPPY_LC_MAX_VIRT_NICE)
		return HAPPY_QUEUE_LC;
	else if (tctx->virt_nice <= HAPPY_NORMAL_MAX_VIRT_NICE)
		return HAPPY_QUEUE_NORMAL;
	else
		return HAPPY_QUEUE_HOG;
}

/* Calculate raw interactive score (0-1000) based on behavioral metrics */
static u32 calc_interactive_score(struct task_struct *p, struct task_ctx *tctx)
{
	u32 score = 500; /* Start at neutral */
	u32 freq_factor, runtime_factor;

	/* Factor 1: Wait frequency (higher = more interactive) */
	/* wait_freq measures sleeps/sec. High = waiting for I/O, input, etc. */
	freq_factor = (u32)(tctx->wait_freq < HAPPY_FREQ_MAX ? tctx->wait_freq :
							       HAPPY_FREQ_MAX);
	score += (freq_factor * 200) / HAPPY_FREQ_MAX; /* +0 to +200 */

	/* Factor 2: Runtime inverse (shorter = more interactive) */
	if (tctx->avg_runtime_ns < HAPPY_RUNTIME_LC_THRESH_NS)
		runtime_factor = 200; /* Very short = max bonus */
	else if (tctx->avg_runtime_ns < HAPPY_RUNTIME_NORMAL_THRESH_NS)
		runtime_factor = 100; /* Short = moderate bonus */
	else if (tctx->avg_runtime_ns < 20000000) /* 20ms */
		runtime_factor = 0; /* Normal */
	else
		runtime_factor = -100; /* Long runtime = penalty */
	score += runtime_factor;

	/* Factor 3: Context flags */
	if (tctx->flags & HAPPY_FLAG_IS_SYNC_WAKEUP)
		score +=
			50; /* Sync wakeups indicate producer-consumer chains */
	if (tctx->flags & HAPPY_FLAG_WOKEN_BY_IRQ)
		score += 100; /* IRQ-driven tasks are usually interactive */
	if (p->flags & PF_KTHREAD)
		score += 25; /* Kernel threads slightly boosted */

	/* Factor 4: Wake frequency (producer indicator) */
	freq_factor = (u32)(tctx->wake_freq < HAPPY_FREQ_MAX ? tctx->wake_freq :
							       HAPPY_FREQ_MAX);
	score += (freq_factor * 100) / HAPPY_FREQ_MAX; /* +0 to +100 */

	/* Clamp to valid range */
	if (score > 1000)
		score = 1000;

	return score;
}

/* Calculate target virt_nice within the task's base queue boundaries */
static s16 calc_dynamic_virt_nice(struct task_struct *p, struct task_ctx *tctx)
{
	u32 score;
	s16 base = tctx->base_virt_nice;
	s16 min_nice, max_nice, target;
	s16 range;

	/* Get queue boundaries */
	switch (tctx->base_queue) {
	case HAPPY_QUEUE_LC:
		min_nice = HAPPY_LC_MIN_VIRT_NICE; /* -50 */
		max_nice = HAPPY_LC_MAX_VIRT_NICE; /* -20 */
		break;
	case HAPPY_QUEUE_NORMAL:
		min_nice = HAPPY_NORMAL_MIN_VIRT_NICE; /* -19 */
		max_nice = HAPPY_NORMAL_MAX_VIRT_NICE; /* 10 */
		break;
	case HAPPY_QUEUE_HOG:
		/* HOG tasks: high score can promote to NORMAL */
		min_nice = HAPPY_HOG_MIN_VIRT_NICE; /* 11 */
		max_nice = HAPPY_HOG_MAX_VIRT_NICE; /* 49 */
		break;
	default:
		return base;
	}

	/* Calculate score */
	score		    = calc_interactive_score(p, tctx);
	tctx->dynamic_score = score;

	/* Map score (0-1000) to virt_nice range */
	if (tctx->base_queue == HAPPY_QUEUE_HOG) {
		/* HOG tasks: high score can promote to NORMAL */
		if (score > interactive_threshold) {
			/* Promote toward NORMAL range */
			s16 promotion = ((score - interactive_threshold) * 15) /
					(1000 - interactive_threshold);
			target	      = HAPPY_HOG_MIN_VIRT_NICE - promotion;
			if (target < HAPPY_NORMAL_MIN_VIRT_NICE)
				target = HAPPY_NORMAL_MIN_VIRT_NICE;
		} else {
			target = base;
		}
	} else {
		/* LC and NORMAL: interactive score adjusts within range */
		/* Score 0 = max_nice (least priority in queue) */
		/* Score 1000 = min_nice (highest priority in queue) */
		range	   = max_nice - min_nice;
		s16 offset = (s16)((score * range) / 1000);
		target = max_nice -
			 offset; /* Higher score = lower (better) virt_nice */

		/* Smooth toward base if not very different */
		if (abs_s16(target - base) < 5)
			target = base;
	}

	return target;
}

/* Apply smoothing to avoid virt_nice thrashing */
static s16 smooth_virt_nice_transition(struct task_ctx *tctx, s16 target)
{
	s16 current = tctx->virt_nice;
	s16 diff    = target - current;
	s16 step;

	/* Don't adjust too quickly - max 5 units per adjustment */
	if (diff > 5)
		step = 5;
	else if (diff < -5)
		step = -5;
	else
		step = diff;

	return current + step;
}

/* Get DSQ ID for queue and LLC */
static inline u64 get_dsq_for_queue(enum happy_queue queue, u32 llc_id)
{
	return ((u64)queue << DSQ_ID_QUEUE_SHIFT) | llc_id;
}

/* Get slice for queue */
static inline u64 get_slice_for_queue(enum happy_queue queue)
{
	switch (queue) {
	case HAPPY_QUEUE_LC:
		return lc_slice_ns;
	case HAPPY_QUEUE_NORMAL:
		return normal_slice_ns;
	case HAPPY_QUEUE_HOG:
		return hog_slice_ns;
	default:
		return normal_slice_ns;
	}
}

/* Update task's EEVDF state on enqueue */
static inline void update_eevdf_state(struct task_ctx *tctx,
				      enum happy_queue queue)
{
	u64 slice = get_slice_for_queue(queue);

	/* Get or calculate weight */
	if (tctx->weight == 0) {
		tctx->weight = calc_weight_from_virt_nice(tctx->virt_nice);
	}

	/* Calculate virtual slice: time charged per unit of real time */
	tctx->vslice = calc_vslice(slice, tctx->weight);

	/* Calculate deadline: when this slice should complete */
	tctx->deadline_vtime = calc_deadline(tctx->vtime, tctx->vslice);

	/* Eligibility starts at current vtime (eligible immediately after sleep) */
	tctx->eligible_vtime = tctx->vtime;
}

/* Init task - called when task is first seen by scheduler */
static void init_task(struct task_struct *p, struct task_ctx *tctx)
{
	enum happy_queue queue;
	u64		 now;

	queue		     = classify_task(p, tctx);
	tctx->queue	     = queue;
	tctx->base_queue     = queue; /* Store original queue */
	tctx->base_virt_nice = tctx->virt_nice; /* Store original virt_nice */
	tctx->vtime	     = vtime_now;
	tctx->exec_runtime   = 0;
	tctx->last_run_at    = 0;
	tctx->last_cpu	     = -1; /* Initialize to -1 to indicate never ran */

	/* CPU tracking init for HOG demotion */
	now			 = bpf_ktime_get_ns();
	tctx->cpu_window_start	 = now;
	tctx->cpu_window_runtime = 0;
	tctx->demote_to_hog	 = false;

	/* NEW: Initialize behavioral tracking */
	tctx->wait_freq		= 0;
	tctx->wake_freq		= 0;
	tctx->avg_runtime_ns	= 0;
	tctx->last_runnable_ns	= 0;
	tctx->last_quiescent_ns = now;
	tctx->flags		= HAPPY_FLAG_DYNAMIC_ADJUST;
	tctx->target_virt_nice	= tctx->virt_nice;
	tctx->dynamic_score	= 500; /* Start neutral */
	tctx->last_recalc_ns	= now;

	/* NEW: Initialize EEVDF/WFQ fields */
	tctx->eligible_vtime = vtime_now;
	tctx->deadline_vtime = vtime_now;
	tctx->weight	     = calc_weight_from_virt_nice(tctx->virt_nice);
	tctx->vslice	     = 0;

	/* Count classified tasks (non-default queue) */
	if (tctx->queue != HAPPY_QUEUE_NORMAL)
		__sync_fetch_and_add(&nr_classified_tasks, 1);
}

/* Helper: initialize a cpumask using bpf_kptr_xchg */
static int init_cpumask(struct bpf_cpumask **p_cpumask)
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

/* SCX operations */

s32 BPF_STRUCT_OPS_SLEEPABLE(happy_init)
{
	u32 i;
	s32 err;

	/* Initialize global state */
	vtime_now = 0;

	/* Initialize domain cpumasks using kptr_xchg for proper registration */
	if (init_cpumask(&lc_cpumask) < 0)
		return -ENOMEM;
	if (init_cpumask(&normal_cpumask) < 0)
		return -ENOMEM;
	if (init_cpumask(&hog_cpumask) < 0)
		return -ENOMEM;

	bpf_for(i, 0, MAX_LLCS)
	{
		u64 dsq_id;

		/* Create DSQ for each queue × LLC combination */
		dsq_id = get_dsq_for_queue(HAPPY_QUEUE_LC, i);
		err    = scx_bpf_create_dsq(dsq_id, -1);
		if (err && err != -EEXIST) {
			scx_bpf_error("failed to create LC DSQ %llu: %d",
				      dsq_id, err);
			return err;
		}

		dsq_id = get_dsq_for_queue(HAPPY_QUEUE_NORMAL, i);
		err    = scx_bpf_create_dsq(dsq_id, -1);
		if (err && err != -EEXIST) {
			scx_bpf_error("failed to create NORMAL DSQ %llu: %d",
				      dsq_id, err);
			return err;
		}

		dsq_id = get_dsq_for_queue(HAPPY_QUEUE_HOG, i);
		err    = scx_bpf_create_dsq(dsq_id, -1);
		if (err && err != -EEXIST) {
			scx_bpf_error("failed to create HOG DSQ %llu: %d",
				      dsq_id, err);
			return err;
		}
	}

	/* Initialize antistall tracking */
	for (i = 0; i < MAX_CPUS; i++) {
		u64 *antistall = bpf_map_lookup_percpu_elem(&antistall_dsq,
							    &zero_u32, i);
		if (antistall)
			*antistall = SCX_DSQ_INVALID;
	}

	return 0;
}

void BPF_STRUCT_OPS(happy_exit, struct scx_exit_info *ei)
{
	struct bpf_cpumask *cpumask;

	/* Release domain cpumasks using kptr_xchg for proper cleanup */
	cpumask = bpf_kptr_xchg(&lc_cpumask, NULL);
	if (cpumask)
		bpf_cpumask_release(cpumask);

	cpumask = bpf_kptr_xchg(&normal_cpumask, NULL);
	if (cpumask)
		bpf_cpumask_release(cpumask);

	cpumask = bpf_kptr_xchg(&hog_cpumask, NULL);
	if (cpumask)
		bpf_cpumask_release(cpumask);

	UEI_RECORD(uei, ei);
}

void BPF_STRUCT_OPS(happy_enable, struct task_struct *p)
{
	struct task_ctx *tctx;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	init_task(p, tctx);
}

void BPF_STRUCT_OPS(happy_disable, struct task_struct *p)
{
	/* Task is being disabled - cleanup if needed */
}

s32 BPF_STRUCT_OPS(happy_select_cpu, struct task_struct *p, s32 prev_cpu,
		   u64 wake_flags)
{
	struct task_ctx	     *tctx;
	const struct cpumask *domain_mask;
	s32		      cpu;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return prev_cpu;

	/* Get domain cpumask based on queue */
	domain_mask = get_domain_cpumask(tctx->queue);

	/* Try preferred domain first */
	if (domain_mask) {
		cpu = scx_bpf_select_cpu_and(
			p, prev_cpu, wake_flags, domain_mask,
			avoid_smt ? SCX_PICK_IDLE_CORE : 0);
		if (cpu >= 0) {
			if (avoid_smt)
				__sync_fetch_and_add(&nr_smt_avoided, 1);
			return cpu;
		}
	}

	/* Fallback: any allowed CPU */
	cpu = scx_bpf_select_cpu_and(p, prev_cpu, wake_flags, p->cpus_ptr,
				     avoid_smt ? SCX_PICK_IDLE_CORE : 0);
	if (cpu >= 0) {
		if (avoid_smt)
			__sync_fetch_and_add(&nr_smt_avoided, 1);
		return cpu;
	}

	return prev_cpu; /* Never return error - use prev_cpu as final fallback */
}

void BPF_STRUCT_OPS(happy_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tctx;
	u64		 dsq_id, slice, vtime_for_dsq;
	u32		 llc_id = 0; /* Simplified: always use LLC 0 for now */
	bool		 eligible;
	s32		 target_cpu;
	bool		 do_preempt = false;

	tctx			    = lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Handle demotion if flagged */
	if (tctx->demote_to_hog && tctx->queue == HAPPY_QUEUE_NORMAL) {
		tctx->queue	    = HAPPY_QUEUE_HOG;
		tctx->virt_nice	    = HAPPY_VIRT_NICE_HOG;
		tctx->demote_to_hog = false;
		/* Reset weight for new queue */
		tctx->weight = calc_weight_from_virt_nice(tctx->virt_nice);
		if (debug)
			bpf_printk("Task %d demoted to HOG queue", p->pid);
	}

	/* Update EEVDF state (weight, vslice, deadline) */
	update_eevdf_state(tctx, tctx->queue);

	/* Check eligibility - is task owed CPU time? */
	eligible = is_eligible(tctx, tctx->queue);

	/* Check for deadline-based preemption */
	target_cpu = scx_bpf_task_cpu(p);
	if (eligible && target_cpu >= 0) {
		do_preempt = should_preempt_running(p, tctx, target_cpu);
	}

	/*
	 * EEVDF dispatch ordering:
	 * - Eligible tasks: order by deadline (earliest deadline first)
	 * - Ineligible tasks: order by vruntime (fairness catch-up)
	 */
	if (eligible) {
		/* Use deadline for ordering among eligible tasks */
		vtime_for_dsq = tctx->deadline_vtime;
		__sync_fetch_and_add(&nr_eligible_dispatches, 1);
		if (debug)
			bpf_printk("Task %d eligible, deadline=%llu", p->pid,
				   vtime_for_dsq);
	} else {
		/* Not eligible - use vruntime for fairness ordering */
		vtime_for_dsq = tctx->vtime;
		__sync_fetch_and_add(&nr_ineligible_dispatches, 1);
		if (debug)
			bpf_printk("Task %d NOT eligible, vtime=%llu", p->pid,
				   vtime_for_dsq);
	}

	/* Get queue configuration */
	dsq_id = get_dsq_for_queue(tctx->queue, llc_id);
	slice  = get_slice_for_queue(tctx->queue);

	/* Insert with appropriate vtime (deadline or vruntime) */
	scx_bpf_dsq_insert_vtime(p, dsq_id, slice, vtime_for_dsq, enq_flags);

	/* Update antistall tracking if enabled */
	if (antistall_enabled) {
		s32  cpu       = scx_bpf_task_cpu(p);
		u64 *antistall = bpf_map_lookup_percpu_elem(&antistall_dsq,
							    &zero_u32, cpu);
		if (antistall && *antistall == SCX_DSQ_INVALID)
			*antistall = dsq_id;
	}

	/* Perform preemption if needed */
	if (do_preempt && target_cpu >= 0) {
		scx_bpf_kick_cpu(target_cpu, SCX_KICK_PREEMPT);
		__sync_fetch_and_add(&nr_deadline_preemptions, 1);

		/* Update running task's preemption count for throttling */
		struct cpu_running_task *running = bpf_map_lookup_percpu_elem(
			&cpu_running_task, &zero_u32, target_cpu);
		if (running) {
			running->preemption_count++;
		}

		if (debug)
			bpf_printk("Preempting CPU %d for task %d", target_cpu,
				   p->pid);
	}
}

void BPF_STRUCT_OPS(happy_dispatch, s32 cpu, struct task_struct *prev)
{
	u32 llc_id = 0; /* Simplified */
	u64 dsq_id;

	/* Try antistall first */
	if (antistall_enabled) {
		u64 *antistall = bpf_map_lookup_elem(&antistall_dsq, &zero_u32);
		if (antistall && *antistall != SCX_DSQ_INVALID) {
			if (scx_bpf_dsq_move_to_local(*antistall, 0)) {
				__sync_fetch_and_add(&nr_antistall_dispatches,
						     1);
				*antistall = SCX_DSQ_INVALID;
				return;
			}
		}
	}

	/* Consume in priority order: LC -> NORMAL -> HOG */
	dsq_id = get_dsq_for_queue(HAPPY_QUEUE_LC, llc_id);
	if (scx_bpf_dsq_move_to_local(dsq_id, 0)) {
		__sync_fetch_and_add(&nr_lc_dispatches, 1);
		return;
	}

	dsq_id = get_dsq_for_queue(HAPPY_QUEUE_NORMAL, llc_id);
	if (scx_bpf_dsq_move_to_local(dsq_id, 0)) {
		__sync_fetch_and_add(&nr_normal_dispatches, 1);
		return;
	}

	dsq_id = get_dsq_for_queue(HAPPY_QUEUE_HOG, llc_id);
	if (scx_bpf_dsq_move_to_local(dsq_id, 0)) {
		__sync_fetch_and_add(&nr_hog_dispatches, 1);
	}
}

void BPF_STRUCT_OPS(happy_tick, struct task_struct *p)
{
	struct task_ctx *tctx;
	u64		 now;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Yield-based preemption per queue thresholds */
	switch (tctx->queue) {
	case HAPPY_QUEUE_HOG:
		/* HOG tasks yield to any LC or NORMAL task */
		/* Check if higher priority queues have waiting tasks */
		break;
	case HAPPY_QUEUE_NORMAL:
		/* NORMAL tasks yield only to LC tasks */
		break;
	case HAPPY_QUEUE_LC:
		/* LC tasks don't yield based on queue priority */
		break;
	default:
		break;
	}

	/* CPU tracking for HOG demotion - only for NORMAL tasks */
	if (tctx->queue == HAPPY_QUEUE_NORMAL) {
		now = bpf_ktime_get_ns();

		/* Accumulate runtime in current window */
		if (tctx->last_run_at > 0) {
			u64 delta = now - tctx->last_run_at;
			tctx->cpu_window_runtime += delta;
		}

		/* Check if window has elapsed */
		u64 window_elapsed = now - tctx->cpu_window_start;
		if (window_elapsed >= hog_cpu_window_ns) {
			/* Calculate CPU percentage for completed window */
			u64 cpu_percent = (tctx->cpu_window_runtime * 100) /
					  window_elapsed;

			/* Check if exceeds threshold */
			if (cpu_percent > hog_cpu_threshold) {
				tctx->demote_to_hog = true;
				if (debug)
					bpf_printk(
						"Task %d CPU %llu%% exceeds %d%%, demoting to HOG",
						p->pid, cpu_percent,
						hog_cpu_threshold);
			}

			/* Reset window */
			tctx->cpu_window_start	 = now;
			tctx->cpu_window_runtime = 0;
		}
	}
}

void BPF_STRUCT_OPS(happy_running, struct task_struct *p)
{
	struct task_ctx *tctx;
	struct cpu_ctx	*cpuc;
	u64		 now, delta;
	s32		 cpu;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	now		  = bpf_ktime_get_ns();
	tctx->last_run_at = now;

	/* Track migrations */
	cpu = scx_bpf_task_cpu(p);
	if (tctx->last_cpu >= 0 && tctx->last_cpu != cpu)
		__sync_fetch_and_add(&nr_migrations, 1);
	tctx->last_cpu = cpu;

	/* Update CPU context */
	cpuc = bpf_map_lookup_elem(&cpu_ctx_stor, &zero_u32);
	if (cpuc) {
		cpuc->current_queue = tctx->queue;
	}

	/* CPU frequency scaling if enabled */
	if (cpufreq_enabled) {
		u32 perf = SCX_CPUPERF_ONE;

		switch (tctx->queue) {
		case HAPPY_QUEUE_LC:
			perf = SCX_CPUPERF_ONE; /* Max frequency */
			break;
		case HAPPY_QUEUE_NORMAL:
			perf = SCX_CPUPERF_ONE / 2; /* Medium */
			break;
		case HAPPY_QUEUE_HOG:
			perf = SCX_CPUPERF_ONE / 4; /* Low */
			break;
		default:
			break;
		}

		scx_bpf_cpuperf_set(cpu, perf);
	}

	/* NEW: Track running task's deadline for preemption decisions */
	struct cpu_running_task *running;

	running = bpf_map_lookup_elem(&cpu_running_task, &zero_u32);
	if (running) {
		running->deadline_vtime = tctx->deadline_vtime;
		running->vtime		= tctx->vtime;
		running->pid		= p->pid;
		running->queue		= tctx->queue;
	}

	/* NEW: Update queue EEVDF state - task is now running */
	u32			  key = (u32)tctx->queue;
	struct queue_eevdf_state *state =
		bpf_map_lookup_elem(&queue_eevdf_states, &key);
	if (state) {
		if (tctx->weight == 0)
			tctx->weight =
				calc_weight_from_virt_nice(tctx->virt_nice);

		state->total_weight += tctx->weight;
		state->nr_tasks++;
	}

	/* NEW: Update average runtime from previous execution */
	if (tctx->exec_runtime > 0) {
		/* EWMA of runtime: new = (old * 3 + new) / 4 */
		if (tctx->avg_runtime_ns == 0)
			tctx->avg_runtime_ns = tctx->exec_runtime;
		else
			tctx->avg_runtime_ns = (tctx->avg_runtime_ns * 3 +
						tctx->exec_runtime) /
					       4;
	}

	/* NEW: Recalculate dynamic virt_nice periodically */
	if (dynamic_nice_enabled && (tctx->flags & HAPPY_FLAG_DYNAMIC_ADJUST)) {
		delta = now - tctx->last_recalc_ns;

		if (delta >= adjust_interval_ns) {
			s16 target = calc_dynamic_virt_nice(p, tctx);
			s16 new_nice =
				smooth_virt_nice_transition(tctx, target);

			/* Apply change if different */
			if (new_nice != tctx->virt_nice) {
				enum happy_queue new_queue;
				u64		 old_weight = tctx->weight;

				/* Determine new queue */
				if (new_nice <= HAPPY_LC_MAX_VIRT_NICE)
					new_queue = HAPPY_QUEUE_LC;
				else if (new_nice <= HAPPY_NORMAL_MAX_VIRT_NICE)
					new_queue = HAPPY_QUEUE_NORMAL;
				else
					new_queue = HAPPY_QUEUE_HOG;

				/* Remove from old queue state */
				if (tctx->queue < HAPPY_QUEUE_MAX &&
				    old_weight > 0) {
					u32 old_key = (u32)tctx->queue;
					struct queue_eevdf_state *old_state =
						bpf_map_lookup_elem(
							&queue_eevdf_states,
							&old_key);
					if (old_state) {
						if (old_state->total_weight >=
						    old_weight)
							old_state->total_weight -=
								old_weight;
						if (old_state->nr_tasks > 0)
							old_state->nr_tasks--;
					}
				}

				/* Apply new nice and recalculate weight */
				enum happy_queue old_queue = tctx->queue;
				tctx->virt_nice		   = new_nice;
				tctx->queue		   = new_queue;
				tctx->weight =
					calc_weight_from_virt_nice(new_nice);
				tctx->last_recalc_ns = now;

				/* Add to new queue state */
				u32 new_key = (u32)new_queue;
				struct queue_eevdf_state *new_state =
					bpf_map_lookup_elem(&queue_eevdf_states,
							    &new_key);
				if (new_state) {
					new_state->total_weight += tctx->weight;
					new_state->nr_tasks++;
				}

				/* Count statistics */
				__sync_fetch_and_add(&nr_dynamic_adjustments,
						     1);
				if (tctx->dynamic_score > interactive_threshold)
					__sync_fetch_and_add(
						&nr_interactive_detected, 1);
				if (new_queue < old_queue)
					__sync_fetch_and_add(&nr_promotions, 1);
				else if (new_queue > old_queue)
					__sync_fetch_and_add(&nr_demotions, 1);

				if (debug)
					bpf_printk(
						"Task %d: virt_nice -> %d (score %u, weight %llu)",
						p->pid, new_nice,
						tctx->dynamic_score,
						tctx->weight);
			}
		}
	}
}

void BPF_STRUCT_OPS(happy_stopping, struct task_struct *p, bool runnable)
{
	struct task_ctx *tctx;
	u64		 now = bpf_ktime_get_ns();
	u64		 delta;
	u64		 vdelta;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Clear running task tracking */
	struct cpu_running_task *running;

	running = bpf_map_lookup_elem(&cpu_running_task, &zero_u32);
	if (running && running->pid == p->pid) {
		running->deadline_vtime = 0;
		running->vtime		= 0;
		running->pid		= 0;
		running->queue		= HAPPY_QUEUE_MAX;

		/* Reset preemption count periodically */
		if (running->preemption_count > 10)
			running->preemption_count =
				running->preemption_count / 2;
	}

	/* Count preemptions (task still runnable but being stopped) */
	if (runnable)
		__sync_fetch_and_add(&nr_preemptions, 1);

	/* Update runtime */
	delta		   = now - tctx->last_run_at;
	tctx->exec_runtime = delta;

	/* NEW: WFQ-style vtime update - vruntime += delta * NICE_0_WEIGHT / weight */
	if (tctx->weight == 0)
		tctx->weight = calc_weight_from_virt_nice(tctx->virt_nice);

	vdelta = delta * NICE_0_WEIGHT / tctx->weight;
	tctx->vtime += vdelta;

	/* Update queue EEVDF state - task is no longer running */
	u32			  key = (u32)tctx->queue;
	struct queue_eevdf_state *state =
		bpf_map_lookup_elem(&queue_eevdf_states, &key);
	if (state) {
		/* Update min_vtime if this is the new minimum */
		if (tctx->vtime < state->min_vtime || state->min_vtime == 0) {
			state->min_vtime = tctx->vtime;
		}

		/* Update weighted average accumulator */
		state->avg_vtime += vdelta * tctx->weight;

		/* Remove task from running count */
		if (state->total_weight >= tctx->weight)
			state->total_weight -= tctx->weight;
		if (state->nr_tasks > 0)
			state->nr_tasks--;
	}
}

void BPF_STRUCT_OPS(happy_set_weight, struct task_struct *p, u32 weight)
{
	struct task_ctx *tctx;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Re-classify when weight changes */
	init_task(p, tctx);
}

void BPF_STRUCT_OPS(happy_set_cpumask, struct task_struct *p,
		    const struct cpumask *cpumask)
{
	/* Task cpumask changed - may need reclassification */
}

s32 BPF_STRUCT_OPS(happy_init_task, struct task_struct *p,
		   struct scx_init_task_args *args)
{
	struct task_ctx *tctx;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return -ENOMEM;

	init_task(p, tctx);
	return 0;
}

void BPF_STRUCT_OPS(happy_exit_task, struct task_struct *p,
		    struct scx_exit_task_args *args)
{
	/* Task exiting - cleanup */
}

void BPF_STRUCT_OPS(happy_update_idle, s32 cpu, bool idle)
{
	/* CPU idle state changed */
}

/* NEW: Track when task becomes runnable for wait frequency calculation */
void BPF_STRUCT_OPS(happy_runnable, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx	   *tctx, *waker_tctx;
	struct task_struct *waker;
	u64		    now, interval;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	now = bpf_ktime_get_ns();

	/* Track how often this task becomes runnable (wait frequency inverse) */
	if (tctx->last_quiescent_ns > 0) {
		interval	= now - tctx->last_quiescent_ns;
		tctx->wait_freq = calc_avg_freq(tctx->wait_freq, interval);
	}
	tctx->last_runnable_ns = now;

	/* Set wakeup flags based on how we were woken */
	if (enq_flags & SCX_ENQ_WAKEUP) {
		tctx->flags |= HAPPY_FLAG_IS_WAKEUP;

		/* Check for sync wakeup */
		if ((enq_flags & SCX_WAKE_SYNC))
			tctx->flags |= HAPPY_FLAG_IS_SYNC_WAKEUP;

		/* Check if woken by IRQ context (x86/arm64 only) */
#if defined(__x86_64__) || defined(__aarch64__)
		if (bpf_in_hardirq() || bpf_in_nmi() ||
		    bpf_in_serving_softirq())
			tctx->flags |= HAPPY_FLAG_WOKEN_BY_IRQ;
#endif
	}

	/* Only track waker relationships for actual wakeups */
	if (!(enq_flags & SCX_ENQ_WAKEUP))
		return;

	/* Filter out preempt/reenqueue cases */
	if (enq_flags & (SCX_ENQ_PREEMPT | SCX_ENQ_REENQ | SCX_ENQ_LAST))
		return;

	/* Architecture-specific: IRQ detection only on x86/arm64 */
#if defined(__x86_64__) || defined(__aarch64__)
	if (bpf_in_hardirq() || bpf_in_nmi() || bpf_in_serving_softirq())
		return;
#endif

	/* Get waker */
	waker = bpf_get_current_task_btf();
	if (!waker || waker == p)
		return;

	/* Confine to related tasks (same thread group) to reduce noise */
	if (p->tgid != waker->tgid)
		return;

	/* Track waker's wake frequency (producer behavior) */
	waker_tctx = lookup_task_ctx(waker);
	if (waker_tctx && waker_tctx->last_runnable_ns > 0) {
		interval = now - waker_tctx->last_runnable_ns;
		if (interval >= 500000) { /* Min 500us between wake updates */
			waker_tctx->wake_freq =
				calc_avg_freq(waker_tctx->wake_freq, interval);
		}
	}
}

/* NEW: Track when task goes to sleep for wait frequency calculation */
void BPF_STRUCT_OPS(happy_quiescent, struct task_struct *p, u64 deq_flags)
{
	struct task_ctx *tctx;
	u64		 now;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Only care about tasks going to sleep */
	if (!(deq_flags & SCX_DEQ_SLEEP))
		return;

	now			= bpf_ktime_get_ns();
	tctx->last_quiescent_ns = now;

	/* Clear one-shot flags */
	tctx->flags &= ~(HAPPY_FLAG_IS_WAKEUP | HAPPY_FLAG_IS_SYNC_WAKEUP |
			 HAPPY_FLAG_WOKEN_BY_IRQ);
}

SCX_OPS_DEFINE(happy_ops, .init = (void *)happy_init,
	       .exit = (void *)happy_exit, .enable = (void *)happy_enable,
	       .disable	   = (void *)happy_disable,
	       .select_cpu = (void *)happy_select_cpu,
	       .enqueue	   = (void *)happy_enqueue,
	       .dispatch = (void *)happy_dispatch, .tick = (void *)happy_tick,
	       .running	    = (void *)happy_running,
	       .stopping    = (void *)happy_stopping,
	       .runnable    = (void *)happy_runnable, /* NEW */
	       .quiescent   = (void *)happy_quiescent, /* NEW */
	       .set_weight  = (void *)happy_set_weight,
	       .set_cpumask = (void *)happy_set_cpumask,
	       .init_task   = (void *)happy_init_task,
	       .exit_task   = (void *)happy_exit_task,
	       .update_idle = (void *)happy_update_idle,
	       .flags = SCX_OPS_KEEP_BUILTIN_IDLE, .timeout_ms = 5000,
	       .name = "scx_happy");
