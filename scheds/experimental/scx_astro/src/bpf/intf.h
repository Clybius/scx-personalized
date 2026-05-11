/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2025 scx_astro contributors
 *
 * Shared BPF constants and structs for scx_astro.
 */
#ifndef __INTF_H
#define __INTF_H

enum consts {
	NSEC_PER_USEC		= 1000ULL,
	NSEC_PER_MSEC		= (1000ULL * NSEC_PER_USEC),

	/* DSQ IDs */
	ASTRO_INTERACTIVE_DSQ	= 1020ULL,
	ASTRO_WAKER_BOOST_DSQ	= 1021ULL,
	ASTRO_NORMAL_DSQ	= 1022ULL,
	ASTRO_COMPUTE_DSQ	= 1023ULL,
	ASTRO_BACKGROUND_DSQ	= 1024ULL,

	/* Slice constants (ns) */
	ASTRO_SLICE_INTERACTIVE_MIN_NS	= (50ULL * NSEC_PER_USEC),
	ASTRO_SLICE_INTERACTIVE_MAX_NS	= (200ULL * NSEC_PER_USEC),
	ASTRO_SLICE_BOOST_NS		= (100ULL * NSEC_PER_USEC),
	ASTRO_SLICE_NORMAL_BASE_NS	= (1ULL * NSEC_PER_MSEC),
	ASTRO_SLICE_NORMAL_MIN_NS	= (500ULL * NSEC_PER_USEC),
	ASTRO_SLICE_NORMAL_MAX_NS	= (2ULL * NSEC_PER_MSEC),
	ASTRO_SLICE_COMPUTE_NS		= (3ULL * NSEC_PER_MSEC),
	ASTRO_SLICE_BACKGROUND_NS	= (500ULL * NSEC_PER_USEC),
	ASTRO_SLICE_MIN_NS		= (50ULL * NSEC_PER_USEC),

	/* Classification thresholds */
	ASTRO_LAT_CRI_INTERACTIVE	= 700ULL,
	ASTRO_LAT_CRI_NORMAL		= 400ULL,
	ASTRO_AVG_RUNTIME_INTERACTIVE	= (1ULL * NSEC_PER_MSEC),
	ASTRO_AVG_RUNTIME_COMPUTE	= (10ULL * NSEC_PER_MSEC),
	ASTRO_WAIT_FREQ_INTERACTIVE	= 5ULL,
	ASTRO_WAKE_FREQ_INTERACTIVE	= 5ULL,
	ASTRO_HYSTERESIS_THRESHOLD	= 3ULL,

	/* SRPT */
	ASTRO_SRPT_THRESHOLD_NS		= (500ULL * NSEC_PER_USEC),
	ASTRO_SRPT_VTIME_SCALE_NUM	= 1ULL,
	ASTRO_SRPT_VTIME_SCALE_DEN	= 2ULL, /* 0.5x vtime for short tasks */

	/* Waker boost */
	ASTRO_WAKER_BOOST_DURATION_NS	= (2ULL * NSEC_PER_MSEC),
	ASTRO_WAKER_BOOST_MAX_CHAIN	= 3ULL,

	/* Hog / background detection */
	ASTRO_HOG_RUNTIME_THRESHOLD_NS	= (50ULL * NSEC_PER_MSEC),
	ASTRO_HOG_SCORE_CONTAIN		= 3ULL,
	ASTRO_HOG_SCORE_MAX		= 8ULL,
	ASTRO_HOG_SCORE_EXHAUST_STEP	= 2ULL,
	ASTRO_HOG_SCORE_DECAY_STEP	= 1ULL,
	ASTRO_HOG_SCORE_DECAY_SHIFT	= 1ULL,

	/* Starvation / fairness */
	ASTRO_CONTAINED_STARVATION_MAX	= 6ULL,
	ASTRO_SHARED_STARVATION_MAX	= 12ULL,
	ASTRO_HIGH_PRIO_BURST_MAX	= 4ULL,

	/* Latency criticality scaling */
	ASTRO_LC_FREQ_MAX		= 1024ULL,
	ASTRO_LC_RUNTIME_MAX		= (10ULL * NSEC_PER_MSEC),
	ASTRO_LC_WEIGHT_BOOST_HIGHEST	= 512ULL,
	ASTRO_LC_WEIGHT_BOOST_HIGH	= 128ULL,
	ASTRO_LC_WEIGHT_BOOST_MEDIUM	= 64ULL,
	ASTRO_LC_WEIGHT_BOOST_REGULAR	= 32ULL,
	ASTRO_LC_INH_GIVER_SHIFT	= 3ULL,
	ASTRO_LC_INH_RECEIVER_SHIFT	= 2ULL,

	/* Budget */
	ASTRO_BUDGET_MAX_NS		= (2ULL * NSEC_PER_MSEC),
	ASTRO_BUDGET_MIN_NS		= (500ULL * NSEC_PER_USEC),
	ASTRO_INTERACTIVE_FLOOR_NS	= (100ULL * NSEC_PER_USEC),
	ASTRO_INTERACTIVE_SLEEP_MIN_NS	= (750ULL * NSEC_PER_USEC),
	ASTRO_REFILL_DIV		= 100ULL,
	ASTRO_SLEEP_MAX_NS		= (250ULL * NSEC_PER_MSEC),

	/* Profiles */
	ASTRO_PROFILE_INTERACTIVE	= 0,
	ASTRO_PROFILE_NORMAL		= 1,
	ASTRO_PROFILE_COMPUTE		= 2,
	ASTRO_PROFILE_BACKGROUND	= 3,
	ASTRO_PROFILE_WAKER_BOOST	= 4,
	ASTRO_PROFILE_AUTO		= 255,
};

#ifndef __VMLINUX_H__
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long s64;

typedef int pid_t;
#endif /* __VMLINUX_H__ */

struct astro_cpu_state {
	u64 interactive_dispatches;
	u64 waker_boost_dispatches;
	u64 normal_dispatches;
	u64 compute_dispatches;
	u64 background_dispatches;
	u64 preempts;
	u64 kicks_idle;
	u64 kicks_preempt;
	u64 contained_starvation_rounds;
	u64 shared_starvation_rounds;
	u64 high_priority_burst_rounds;
	u64 profile_transitions;
	u64 waker_boosts;
	u64 srpt_short_tasks;
	u64 budget_refills;
	u64 budget_exhaustions;
	u64 hog_scores;
	u64 last_interactive_ts;
};

struct astro_task_ctx {
	/* Behavioral tracking */
	u64 avg_runtime_ns;
	u64 acc_runtime_ns;
	u32 wait_freq;
	u32 wake_freq;
	u64 last_sleep_ns;
	u64 sleep_started_at;
	u64 last_run_at;

	/* Latency criticality */
	u32 lat_cri;
	u32 lat_cri_waker;
	u32 lat_cri_wakee;
	u32 normalized_lat_cri;

	/* Profile / lane */
	u8 current_profile;
	u8 pending_profile;
	u8 profile_hysteresis;
	u8 explicit_profile; /* 255 = none */

	/* Budget (for interactive containment / recovery) */
	s64 budget_ns;
	u64 last_refill_ns;

	/* Waker boost */
	u64 waker_boost_expire_ns;
	u32 waker_boost_depth;

	/* Hog detection */
	u32 hog_score;

	/* CPU / locality */
	s32 last_cpu;
	s32 wake_cpu;
	u8 wake_cpu_idle;
	u8 wake_cpu_valid;

	/* Flags */
	u8 is_wakeup;
	u8 is_sync_wakeup;
};

/* Userspace -> BPF explicit profile override map value */
struct astro_tgid_profile {
	u8 profile; /* one of ASTRO_PROFILE_* */
};

#endif /* __INTF_H */
