/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2024 scx_happy authors
 *
 * Interface definitions for scx_happy scheduler
 */

#ifndef __HAPPY_INTF_H
#define __HAPPY_INTF_H

/* Type defs for BPF/userspace compat - defined when vmlinux.h is not included */
#ifndef __VMLINUX_H__
typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;

typedef signed char    s8;
typedef signed short   s16;
typedef signed int     s32;
typedef signed long    s64;
#endif

/* Queue IDs */
enum happy_queue {
	HAPPY_QUEUE_LC = 0,
	HAPPY_QUEUE_NORMAL,
	HAPPY_QUEUE_HOG,
	HAPPY_QUEUE_MAX,
};

/* Virtual nice range: -50 to 49 */
#define HAPPY_VIRT_NICE_MIN (-50)
#define HAPPY_VIRT_NICE_MAX 49
#define HAPPY_VIRT_NICE_DEFAULT 0

/* Queue boundaries in virtual nice space */
#define HAPPY_LC_MAX_VIRT_NICE (-20)
#define HAPPY_LC_MIN_VIRT_NICE (-50)
#define HAPPY_NORMAL_MAX_VIRT_NICE 10
#define HAPPY_NORMAL_MIN_VIRT_NICE (-19)
#define HAPPY_HOG_MAX_VIRT_NICE 49
#define HAPPY_HOG_MIN_VIRT_NICE 11

/* Default virtual nice values for classified tasks */
#define HAPPY_VIRT_NICE_INPUT (-45) /* Highest LC priority */
#define HAPPY_VIRT_NICE_TURBO (-35) /* SCX_TURBO tasks */
#define HAPPY_VIRT_NICE_STEAM (-30) /* Steam games */
#define HAPPY_VIRT_NICE_AUDIO (-28) /* Audio threads */
#define HAPPY_VIRT_NICE_DE (-25) /* DE components */
#define HAPPY_VIRT_NICE_KTHREAD (-10) /* Kernel threads (high-normal) */

/* Task classification types for TGID maps */
enum happy_task_type {
	HAPPY_TASK_TYPE_SCX_TURBO = 1,
	HAPPY_TASK_TYPE_STEAM,
	HAPPY_TASK_TYPE_DE,
	HAPPY_TASK_TYPE_INPUT,
	HAPPY_TASK_TYPE_AUDIO,
};

/* Statistics structure - 64 byte aligned */
struct happy_stats {
	u64 nr_lc_dispatches;
	u64 nr_normal_dispatches;
	u64 nr_hog_dispatches;
	u64 nr_preemptions;
	u64 nr_migrations;
	u64 nr_antistall_dispatches;
	u64 nr_smt_avoided;
	u64 nr_classified_tasks;
};

/* Per-queue configuration */
struct happy_queue_config {
	u64 slice_ns;
	u64 slice_lag_ns;
	u32 preempts_queues; /* Bitmask of queues this queue preempts */
};

/* Domain configuration flags */
#define HAPPY_DOMAIN_TURBO 0
#define HAPPY_DOMAIN_PERFORMANCE 1
#define HAPPY_DOMAIN_POWERSAVE 2

/* Argument structure for happy_set_domain_cpu syscall program */
struct domain_cpu_arg {
	s32 cpu_id;
	u32 queue; /* enum happy_queue */
};

#endif /* __HAPPY_INTF_H */
