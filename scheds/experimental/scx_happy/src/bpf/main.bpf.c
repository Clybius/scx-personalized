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
const volatile u64 lc_slice_ns	       = 500000; /* 500us */
const volatile u64 lc_slice_lag_ns     = 10000000; /* 10000us */
const volatile u64 normal_slice_ns     = 1000000; /* 1000us */
const volatile u64 normal_slice_lag_ns = 20000000; /* 20000us */
const volatile u64 hog_slice_ns	       = 3000000; /* 3000us */
const volatile u64 hog_slice_lag_ns    = 60000000; /* 60000us */

/* Feature toggles */
const volatile bool avoid_smt	      = true;
const volatile bool cache_affinity    = true;
const volatile bool cpufreq_enabled   = true;
const volatile bool antistall_enabled = true;
const volatile u64  antistall_sec     = 3;

/* HOG demotion parameters */
const volatile u8  hog_cpu_threshold = 50; /* Default 50% CPU threshold */
const volatile u64 hog_cpu_window_ns = 100000000; /* 100ms measurement window */

/* Domain cpumask storage - global variables with __kptr */
private(HAPPY) struct bpf_cpumask __kptr *lc_cpumask;
private(HAPPY) struct bpf_cpumask __kptr *normal_cpumask;
private(HAPPY) struct bpf_cpumask __kptr *hog_cpumask;

/* Scheduling statistics - in BSS section, exposed to userspace */
volatile u64	   nr_lc_dispatches;
volatile u64	   nr_normal_dispatches;
volatile u64	   nr_hog_dispatches;
volatile u64	   nr_preemptions;
volatile u64	   nr_migrations;
volatile u64	   nr_antistall_dispatches;
volatile u64	   nr_smt_avoided;
volatile u64	   nr_classified_tasks;

const volatile u32 debug    = 0;
const u32	   zero_u32 = 0;

/* Task context stored in task storage map */
struct task_ctx {
	s16		 virt_nice; /* Virtual nice: -50 to 49 */
	enum happy_queue queue; /* Assigned queue */
	u64		 last_run_at;
	u64		 exec_runtime;
	u64		 vtime; /* Virtual runtime */
	s32 last_cpu; /* Last CPU task ran on, for migration tracking */

	/* CPU tracking for HOG demotion */
	u64  cpu_window_start; /* Start of current measurement window */
	u64  cpu_window_runtime; /* Cumulative runtime in current window */
	bool demote_to_hog; /* Flag: should be demoted on next enqueue */
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

/* Helper: calculate vtime scaling factor from virt_nice */
static inline u64 calc_vtime_scale(s16 virt_nice)
{
	/* virt_nice: -50 to 49, scale = 100 + virt_nice = 50 to 149 */
	s64 scale = 100 + virt_nice;
	if (scale < 50)
		scale = 50;
	if (scale > 149)
		scale = 149;
	return (u64)scale;
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

/* Get slice lag for queue */
static inline u64 get_lag_for_queue(enum happy_queue queue)
{
	switch (queue) {
	case HAPPY_QUEUE_LC:
		return lc_slice_lag_ns;
	case HAPPY_QUEUE_NORMAL:
		return normal_slice_lag_ns;
	case HAPPY_QUEUE_HOG:
		return hog_slice_lag_ns;
	default:
		return normal_slice_lag_ns;
	}
}

/* Init task - called when task is first seen by scheduler */
static void init_task(struct task_struct *p, struct task_ctx *tctx)
{
	tctx->queue	   = classify_task(p, tctx);
	tctx->vtime	   = vtime_now;
	tctx->exec_runtime = 0;
	tctx->last_run_at  = 0;
	tctx->last_cpu	   = -1; /* Initialize to -1 to indicate never ran */

	/* CPU tracking init for HOG demotion */
	tctx->cpu_window_start	 = bpf_ktime_get_ns();
	tctx->cpu_window_runtime = 0;
	tctx->demote_to_hog	 = false;

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
	cpu = scx_bpf_select_cpu_and(p, prev_cpu, wake_flags, p->cpus_ptr, 0);
	if (cpu < 0)
		return prev_cpu; /* Never return error - use prev_cpu as final fallback */

	return cpu;
}

void BPF_STRUCT_OPS(happy_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tctx;
	u64		 dsq_id, slice, vtime;
	u32		 llc_id = 0; /* Simplified: always use LLC 0 for now */

	tctx			= lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Handle demotion if flagged */
	if (tctx->demote_to_hog && tctx->queue == HAPPY_QUEUE_NORMAL) {
		tctx->queue	    = HAPPY_QUEUE_HOG;
		tctx->virt_nice	    = HAPPY_VIRT_NICE_HOG;
		tctx->demote_to_hog = false;
		if (debug)
			bpf_printk("Task %d demoted to HOG queue", p->pid);
	}

	/* Update vtime */
	u64 vtime_min = vtime_now - get_lag_for_queue(tctx->queue);
	if (tctx->vtime < vtime_min)
		tctx->vtime = vtime_min;

	/* Calculate scaled vtime */
	u64 scale = calc_vtime_scale(tctx->virt_nice);
	vtime	  = tctx->vtime + (tctx->exec_runtime * scale / 100);

	/* Get queue configuration */
	dsq_id = get_dsq_for_queue(tctx->queue, llc_id);
	slice  = get_slice_for_queue(tctx->queue);

	/* Insert into queue with vtime */
	scx_bpf_dsq_insert_vtime(p, dsq_id, slice, vtime, enq_flags);

	/* Update antistall tracking if enabled */
	if (antistall_enabled) {
		s32  cpu       = scx_bpf_task_cpu(p);
		u64 *antistall = bpf_map_lookup_percpu_elem(&antistall_dsq,
							    &zero_u32, cpu);
		if (antistall && *antistall == SCX_DSQ_INVALID)
			*antistall = dsq_id;
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

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	tctx->last_run_at = bpf_ktime_get_ns();

	/* Track migrations */
	s32 cpu = scx_bpf_task_cpu(p);
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
}

void BPF_STRUCT_OPS(happy_stopping, struct task_struct *p, bool runnable)
{
	struct task_ctx *tctx;
	u64		 now = bpf_ktime_get_ns();
	u64		 delta;

	tctx = lookup_task_ctx(p);
	if (!tctx)
		return;

	/* Count preemptions (task still runnable but being stopped) */
	if (runnable)
		__sync_fetch_and_add(&nr_preemptions, 1);

	/* Update runtime */
	delta		   = now - tctx->last_run_at;
	tctx->exec_runtime = delta;

	/* Update vtime with scaling */
	u64 scale = calc_vtime_scale(tctx->virt_nice);
	tctx->vtime += (delta * scale / 100);

	/* Advance global vtime if needed */
	if (tctx->vtime > vtime_now)
		vtime_now = tctx->vtime;
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

SCX_OPS_DEFINE(happy_ops, .init = (void *)happy_init,
	       .exit = (void *)happy_exit, .enable = (void *)happy_enable,
	       .disable	   = (void *)happy_disable,
	       .select_cpu = (void *)happy_select_cpu,
	       .enqueue	   = (void *)happy_enqueue,
	       .dispatch = (void *)happy_dispatch, .tick = (void *)happy_tick,
	       .running	    = (void *)happy_running,
	       .stopping    = (void *)happy_stopping,
	       .set_weight  = (void *)happy_set_weight,
	       .set_cpumask = (void *)happy_set_cpumask,
	       .init_task   = (void *)happy_init_task,
	       .exit_task   = (void *)happy_exit_task,
	       .update_idle = (void *)happy_update_idle,
	       .flags = SCX_OPS_KEEP_BUILTIN_IDLE, .timeout_ms = 5000,
	       .name = "scx_happy");
