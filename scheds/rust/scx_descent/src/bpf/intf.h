/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2024 Andrea Righi <andrea.righi@linux.dev>
 *
 * This software may be used and distributed according to the terms of the GNU
 * General Public License version 2.
 */
#ifndef __INTF_H
#define __INTF_H

#include <limits.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define CLAMP(val, lo, hi) MIN(MAX(val, lo), hi)
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

enum consts {
	NSEC_PER_USEC = 1000ULL,
	NSEC_PER_MSEC = (1000ULL * NSEC_PER_USEC),
	NSEC_PER_SEC  = (1000ULL * NSEC_PER_MSEC),
};

#ifndef __VMLINUX_H__
typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;

typedef signed char    s8;
typedef signed short   s16;
typedef signed int     s32;
typedef signed long    s64;

typedef int	       pid_t;
#endif /* __VMLINUX_H__ */

struct cpu_arg {
	s32 cpu_id;
};

struct domain_arg {
	s32 lvl_id;
	s32 cpu_id;
	s32 sibling_cpu_id;
};

/* Task classes for descent scheduler - scx_cake compatible */
enum descent_class {
	DESCENT_CLASS_LATENCY_CRITICAL =
		0, // Games, audio, compositors, kthreads
	DESCENT_CLASS_NORMAL	 = 1, // Default interactive
	DESCENT_CLASS_HOG	 = 2, // High CPU usage
	DESCENT_CLASS_BACKGROUND = 3, // Low priority
	DESCENT_CLASS_MAX	 = 4,
};

/* Loss metrics passed to userspace */
struct descent_metrics {
	u64 loss_by_class[DESCENT_CLASS_MAX];
	u64 sample_count[DESCENT_CLASS_MAX];
};

/* Parameter update from userspace to BPF */
struct descent_params_update {
	s32 cpu_id;
	u32 class_id;
	u64 latency_weight;
	u64 base_slice_ns;
	u64 vruntime_scale;
	u64 preemption_priority;
	u64 migration_cost;
};

/* Gradient event from BPF to userspace */
struct gradient_ready_event {
	s32 cpu_id;
	u32 class_id;
	s32 param_idx;
	u64 loss_plus;
	u64 loss_minus;
	u64 epsilon;
	u64 timestamp;
};

/* Load metrics for CAKE Autorate - request/response struct */
struct load_metrics_request {
	s32 cpu_id; /* -1 for aggregate across all CPUs */
	u32 class_id; /* Class to query */
};

struct load_metrics_response {
	u64 cycles_spent; /* Total cycles consumed */
	u64 sample_count; /* Number of samples */
	u64 elapsed_ns; /* Time period for calculation */
};

/* Arguments for reset_load_accumulators syscall */
struct reset_load_args {
	s32 cpu_id; /* -1 for all CPUs, >=0 for specific CPU */
	u32 class_id; /* Class to reset */
};

#endif /* __INTF_H */
