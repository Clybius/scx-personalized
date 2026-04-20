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

/* Virtual nice range: -20 to 19 (standard Linux nice) */
#define HAPPY_VIRT_NICE_MIN (-20)
#define HAPPY_VIRT_NICE_MAX 19
#define HAPPY_VIRT_NICE_DEFAULT 0

/* Queue boundaries in virtual nice space */
#define HAPPY_LC_MAX_VIRT_NICE (-10)       /* LC: -20 to -10 */
#define HAPPY_LC_MIN_VIRT_NICE (-20)
#define HAPPY_NORMAL_MAX_VIRT_NICE 5       /* NORMAL: -9 to 5 */
#define HAPPY_NORMAL_MIN_VIRT_NICE (-9)
#define HAPPY_HOG_MAX_VIRT_NICE 19         /* HOG: 6 to 19 */
#define HAPPY_HOG_MIN_VIRT_NICE 6

/* Default virtual nice values for classified tasks (within LC range) */
#define HAPPY_VIRT_NICE_INPUT (-20)  /* Highest LC priority */
#define HAPPY_VIRT_NICE_TURBO (-18)  /* SCX_TURBO tasks */
#define HAPPY_VIRT_NICE_STEAM (-16)  /* Steam games */
#define HAPPY_VIRT_NICE_AUDIO (-14)  /* Audio threads */
#define HAPPY_VIRT_NICE_DE (-12)     /* DE components */
#define HAPPY_VIRT_NICE_KTHREAD (-8) /* Kernel threads (high-normal, NORMAL queue) */
#define HAPPY_VIRT_NICE_HOG 15       /* Demoted to HOG queue */

/* ========== Latency Criticality Constants ========== */

/* Scale for normalized latency criticality */
#define HAPPY_LAT_CRI_SHIFT     10     /* 2^10 = 1024 */
#define HAPPY_LAT_CRI_SCALE     1024   /* Normalized range [0, 1024] */

/* Frequency caps for lat_cri calculation */
#define HAPPY_LAT_CRI_FREQ_MAX  100000 /* Max frequency (100K/sec) */

/* Runtime thresholds for lat_cri */
#define HAPPY_LAT_CRI_RUNTIME_MAX_NS 1000000000ULL /* 1 second max */

/* Weight boost constants for context-aware lat_cri */
#define HAPPY_LC_WEIGHT_BOOST_WAKEUP    128   /* Regular wakeup boost */
#define HAPPY_LC_WEIGHT_BOOST_SYNC      128   /* Additional for sync wakeup */
#define HAPPY_LC_WEIGHT_BOOST_IRQ       512   /* IRQ-driven boost (highest) */
#define HAPPY_LC_WEIGHT_BOOST_KTHREAD   64    /* Kernel thread boost */

/* Inheritance shift for waker/wakee propagation */
#define HAPPY_LC_INH_GIVER_SHIFT        3     /* 12.5% of giver's surplus */
#define HAPPY_LC_INH_RECEIVER_SHIFT     2     /* 25% of receiver's lat_cri max */

/* System stats update interval */
#define HAPPY_SYS_STAT_INTERVAL_NS      10000000ULL /* 10ms */

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
	/* NEW: Dynamic adjustment statistics */
	u64 nr_dynamic_adjustments; /* How many times virt_nice was adjusted */
	u64 nr_interactive_detected; /* Tasks detected as interactive */
	u64 nr_promotions; /* Tasks promoted to better queue */
	u64 nr_demotions; /* Tasks demoted to worse queue */
	/* NEW: EEVDF/WFQ statistics */
	u64 nr_eligible_dispatches; /* Dispatched as eligible */
	u64 nr_ineligible_dispatches; /* Dispatched as ineligible */
	u64 nr_deadline_expired; /* Tasks that exceeded deadline */
	/* NEW: Deadline preemption statistics */
	u64 nr_deadline_preemptions; /* Total preemptions triggered */
	u64 nr_queue_priority_preemptions; /* Preemptions due to queue priority */
	u64 nr_same_queue_preemptions; /* Preemptions within same queue */
	u64 nr_preemptions_skipped; /* Skipped due to hysteresis */
	u64 nr_preemptions_ineligible; /* Skipped because task not eligible */
	u64 nr_preemptions_later_deadline; /* Skipped because later deadline */
	/* Per-queue vtime tracking */
	u64 lc_min_vtime; /* LC queue min vtime */
	u64 lc_avg_vtime; /* LC queue weighted avg vtime */
	u64 normal_min_vtime; /* NORMAL queue min vtime */
	u64 normal_avg_vtime; /* NORMAL queue weighted avg vtime */
	u64 hog_min_vtime; /* HOG queue min vtime */
	u64 hog_avg_vtime; /* HOG queue weighted avg vtime */
	/* NEW: Lag decay statistics */
	u64 nr_hog_sleep_decayed; /* Times HOG sleep was decayed */
	u64 nr_hog_promotion_checks; /* Promotion eligibility checks */
	/* ========== Latency Criticality Statistics ========== */
	u64 nr_lat_cri_calculations;  /* Total lat_cri calculations performed */
	u64 nr_high_lat_cri_tasks;    /* Tasks with normalized_lat_cri > 768 */
	u64 nr_lat_cri_inherited;     /* Times lat_cri was inherited from waker/wakee */
	u32 sys_max_lat_cri;          /* System-wide max lat_cri for normalization */
	u32 sys_avg_lat_cri;          /* System-wide avg lat_cri */
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
