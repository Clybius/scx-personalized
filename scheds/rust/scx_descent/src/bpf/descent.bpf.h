/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2024 Andrea Righi <andrea.righi@linux.dev>
 */
#ifndef __DESCENT_BPF_H
#define __DESCENT_BPF_H

#include <scx/common.bpf.h>
#include "intf.h"

/*
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

	/* Phase 3: Perturbation state for gradient estimation */
	u64 original_value; /* Saved original parameter value */
	u64 perturb_epsilon; /* Current epsilon for perturbation */
	u64 loss_plus; /* Loss after +ε perturbation */
	u64 loss_minus; /* Loss after -ε perturbation */
};

/* Default parameter values (will be optimized by gradient descent) */
#define DEFAULT_LATENCY_WEIGHT_NS (5ULL * NSEC_PER_MSEC)
#define DEFAULT_BASE_SLICE_NS (700ULL * NSEC_PER_USEC)
#define DEFAULT_VRUNTIME_SCALE 1024ULL /* 1.0 in fixed-point */
#define DEFAULT_PREEMPTION_PRIORITY (1ULL * NSEC_PER_MSEC)
#define DEFAULT_MIGRATION_COST_NS (50ULL * NSEC_PER_USEC)

/*
 * Audio tasks: prioritize very low latency
 */
#define AUDIO_LATENCY_WEIGHT_NS (100ULL * NSEC_PER_USEC)
#define AUDIO_BASE_SLICE_NS (500ULL * NSEC_PER_USEC)
#define AUDIO_VRUNTIME_SCALE 512ULL /* 0.5 in fixed-point */
#define AUDIO_PREEMPTION_PRIORITY (100ULL * NSEC_PER_USEC)
#define AUDIO_MIGRATION_COST_NS (10ULL * NSEC_PER_USEC)

/*
 * Interactive tasks: moderate latency preference
 */
#define INTERACTIVE_LATENCY_WEIGHT_NS (1ULL * NSEC_PER_MSEC)
#define INTERACTIVE_BASE_SLICE_NS (600ULL * NSEC_PER_USEC)
#define INTERACTIVE_VRUNTIME_SCALE 768ULL /* 0.75 in fixed-point */
#define INTERACTIVE_PREEMPTION_PRIORITY (500ULL * NSEC_PER_USEC)
#define INTERACTIVE_MIGRATION_COST_NS (30ULL * NSEC_PER_USEC)

/*
 * Batch tasks: prefer throughput over latency
 */
#define BATCH_LATENCY_WEIGHT_NS (10ULL * NSEC_PER_MSEC)
#define BATCH_BASE_SLICE_NS (5ULL * NSEC_PER_MSEC)
#define BATCH_VRUNTIME_SCALE 1536ULL /* 1.5 in fixed-point */
#define BATCH_PREEMPTION_PRIORITY (5ULL * NSEC_PER_MSEC)
#define BATCH_MIGRATION_COST_NS (100ULL * NSEC_PER_USEC)

/*
 * Kernel tasks: balanced but conservative
 */
#define KERNEL_LATENCY_WEIGHT_NS (2ULL * NSEC_PER_MSEC)
#define KERNEL_BASE_SLICE_NS (1ULL * NSEC_PER_MSEC)
#define KERNEL_VRUNTIME_SCALE 1024ULL /* 1.0 in fixed-point */
#define KERNEL_PREEMPTION_PRIORITY (1ULL * NSEC_PER_MSEC)
#define KERNEL_MIGRATION_COST_NS (20ULL * NSEC_PER_USEC)

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

/*
 * Perturbation states for gradient estimation
 */
enum perturb_state {
	PERTURB_NONE	= 0, /* No perturbation, measure baseline */
	PERTURB_PLUS	= 1, /* Apply +ε */
	PERTURB_MINUS	= 2, /* Apply -ε (actually -2ε from +ε point) */
	PERTURB_RESTORE = 3, /* Restore original value */
};

/* Per-class loss accumulation structure */
struct class_loss_accumulator {
	u64  latency_loss_sum; /* Sum of squared wakeup latencies */
	u64  deadline_misses; /* Count of scheduling deadline misses */
	u64  cpu_time_ns; /* Total CPU time consumed */
	u64  target_share_ns; /* Expected fair share */
	u32  sample_count; /* Number of samples in this window */
	bool active; /* Whether currently collecting */
};

/* Per-CPU perturbation state (not per-class!) */
struct perturb_state_machine {
	u32 param_idx; /* Which parameter (0-4) */
	u32 phase; /* 0=BASELINE, 1=PLUS, 2=MINUS, 3=RESTORE */
	u64 phase_start_ns; /* When current phase started */
	u32 current_class; /* Which class is being perturbed (cycles 0-3) */
};

/* State constants */
#define PERTURB_BASELINE 0
#define PERTURB_PLUS 1
#define PERTURB_MINUS 2
#define PERTURB_RESTORE 3

/* Per-CPU descent context */
struct cpu_descent_ctx {
	struct class_params	      class_params[DESCENT_CLASS_MAX];
	struct class_loss_accumulator class_loss[DESCENT_CLASS_MAX];
	u64			      last_param_sync;
	u64 current_perturbation_start; /* When current perturbation began */
	struct perturb_state_machine perturb; /* Per-CPU perturbation state */
	u64 loss_baseline[DESCENT_CLASS_MAX]; /* Baseline loss before perturbation */
};

#endif /* __DESCENT_BPF_H */
