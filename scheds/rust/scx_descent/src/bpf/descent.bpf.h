/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2024 Andrea Righi <andrea.righi@linux.dev>
 */
#ifndef __DESCENT_BPF_H
#define __DESCENT_BPF_H

#include <scx/common.bpf.h>
#include "intf.h"

/*
 * New scx_cake compatible class structure:
 * Class 0: LATENCY_CRITICAL - Games, audio, compositors, kthreads
 * Class 1: NORMAL          - Default interactive
 * Class 2: HOG             - High CPU usage (≥75% quantum)
 * Class 3: BACKGROUND      - Low priority, SCHED_IDLE, rare wakeups
 *
 * 5 learnable parameters per class per CPU:
 * - latency_weight:     Virtual deadline offset (θ₁)
 * - base_slice_ns:      Preferred time slice (θ₂)
 * - vruntime_scale:     Vruntime multiplier (θ₃), fixed-point divide by 1024
 * - preemption_priority: Urgency threshold (θ₄)
 * - migration_cost:     Cross-CPU migration penalty (θ₅)
 */
struct class_params {
	u64 latency_weight;
	u64 base_slice_ns;
	u64 vruntime_scale;
	u64 preemption_priority;
	u64 migration_cost;
};

/* Default parameter values (will be optimized by gradient descent) */
#define DEFAULT_LATENCY_WEIGHT_NS (5ULL * NSEC_PER_MSEC)
#define DEFAULT_BASE_SLICE_NS (700ULL * NSEC_PER_USEC)
#define DEFAULT_VRUNTIME_SCALE 1024ULL /* 1.0 in fixed-point */
#define DEFAULT_PREEMPTION_PRIORITY (1ULL * NSEC_PER_MSEC)
#define DEFAULT_MIGRATION_COST_NS (50ULL * NSEC_PER_USEC)

/*
 * LATENCY_CRITICAL (Class 0): Games, audio, compositors, kthreads - lowest latency
 */
#define LATENCY_CRITICAL_LATENCY_WEIGHT_NS (100ULL * NSEC_PER_USEC)
#define LATENCY_CRITICAL_BASE_SLICE_NS (500ULL * NSEC_PER_USEC)
#define LATENCY_CRITICAL_VRUNTIME_SCALE 512ULL /* 0.5 in fixed-point */
#define LATENCY_CRITICAL_PREEMPTION_PRIORITY (100ULL * NSEC_PER_USEC)
#define LATENCY_CRITICAL_MIGRATION_COST_NS (10ULL * NSEC_PER_USEC)

/*
 * NORMAL (Class 1): Default interactive tasks
 */
#define NORMAL_LATENCY_WEIGHT_NS (1ULL * NSEC_PER_MSEC)
#define NORMAL_BASE_SLICE_NS (600ULL * NSEC_PER_USEC)
#define NORMAL_VRUNTIME_SCALE 768ULL /* 0.75 in fixed-point */
#define NORMAL_PREEMPTION_PRIORITY (500ULL * NSEC_PER_USEC)
#define NORMAL_MIGRATION_COST_NS (30ULL * NSEC_PER_USEC)

/*
 * HOG (Class 2): High CPU usage (≥75% quantum), non-critical
 */
#define HOG_LATENCY_WEIGHT_NS (10ULL * NSEC_PER_MSEC)
#define HOG_BASE_SLICE_NS (5ULL * NSEC_PER_MSEC)
#define HOG_VRUNTIME_SCALE 1536ULL /* 1.5 in fixed-point */
#define HOG_PREEMPTION_PRIORITY (5ULL * NSEC_PER_MSEC)
#define HOG_MIGRATION_COST_NS (100ULL * NSEC_PER_USEC)

/*
 * BACKGROUND (Class 3): Low priority, SCHED_IDLE, rare wakeups
 */
#define BACKGROUND_LATENCY_WEIGHT_NS (2ULL * NSEC_PER_MSEC)
#define BACKGROUND_BASE_SLICE_NS (1ULL * NSEC_PER_MSEC)
#define BACKGROUND_VRUNTIME_SCALE 1024ULL /* 1.0 in fixed-point */
#define BACKGROUND_PREEMPTION_PRIORITY (1ULL * NSEC_PER_MSEC)
#define BACKGROUND_MIGRATION_COST_NS (20ULL * NSEC_PER_USEC)

/*
 * Parameter bounds for safety clamping
 */
#define PARAM_MIN_LATENCY_WEIGHT 0ULL
#define PARAM_MAX_LATENCY_WEIGHT (10ULL * NSEC_PER_MSEC)
#define PARAM_MIN_BASE_SLICE_NS (100ULL * NSEC_PER_USEC)
#define PARAM_MAX_BASE_SLICE_NS (50ULL * NSEC_PER_MSEC)
#define PARAM_MIN_VRUNTIME_SCALE 512ULL /* 0.5 * 1024 */
#define PARAM_MAX_VRUNTIME_SCALE 2048ULL /* 2.0 * 1024 */
#define PARAM_MIN_PREEMPT_PRIO 0ULL
#define PARAM_MAX_PREEMPT_PRIO 100ULL
#define PARAM_MIN_MIGRATION_COST 0ULL
#define PARAM_MAX_MIGRATION_COST (10ULL * NSEC_PER_MSEC)

/* Per-class latency accumulation structure for PIE controller */
struct class_latency_accumulator {
	u64 total_latency_ns; /* Sum of all enqueue-to-run latencies */
	u64 max_latency_ns; /* Maximum observed latency */
	u64 sample_count; /* Number of samples */
};

/*
 * Per-class load accumulator for CAKE Autorate
 * Tracks CPU cycles consumed by each task class
 */
struct class_load_accumulator {
	u64 cycles_spent; /* Total CPU cycles spent by this class */
	u64 sample_count; /* Number of load samples */
	u64 last_update_ns; /* Last time accumulator was reset */
};

/* Per-CPU descent context */
struct cpu_descent_ctx {
	struct class_params class_params[DESCENT_CLASS_MAX];
	struct class_latency_accumulator
		class_latency[DESCENT_CLASS_MAX]; /* CHANGED */
	struct class_load_accumulator
		class_load[DESCENT_CLASS_MAX]; /* NEW: Load tracking */
	u64	last_param_sync;
};

#endif /* __DESCENT_BPF_H */
