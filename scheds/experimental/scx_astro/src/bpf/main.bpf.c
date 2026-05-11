/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2025 scx_astro contributors
 *
 * scx_astro: Hybrid Classification Multi-Lane Scheduler
 *
 * Combines automatic behavioral detection (LAVD-inspired latency criticality)
 * with explicit SCX_ASTRO env-var overrides (profile-based). Uses a multi-lane
 * DSQ architecture with SRPT-inspired vtime scaling and waker-boost chains.
 */

#include <scx/common.bpf.h>
#include <scx/compat.bpf.h>
#include <scx/user_exit_info.bpf.h>
#include "intf.h"

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

/*
 * Per-task context storage
 */
struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct astro_task_ctx);
} task_ctx_stor SEC(".maps");

/*
 * Per-CPU scheduler state
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct astro_cpu_state);
} cpu_state SEC(".maps");

/*
 * Userspace-supplied explicit profile overrides per tgid/pid
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, pid_t);
	__type(value, struct astro_tgid_profile);
} tgid_profile_map SEC(".maps");

/* Global statistics (BSS) */
volatile u64 nr_running;
volatile u64 interactive_dispatches;
volatile u64 waker_boost_dispatches;
volatile u64 normal_dispatches;
volatile u64 compute_dispatches;
volatile u64 background_dispatches;
volatile u64 preempts;
volatile u64 kicks_idle;
volatile u64 kicks_preempt;
volatile u64 profile_transitions;
volatile u64 waker_boosts;
volatile u64 srpt_short_tasks;
volatile u64 budget_refills;
volatile u64 budget_exhaustions;
volatile u64 autotune_mode;
volatile u64 autotune_generation;

/* Tunables (read-write from userspace via .data) */
volatile u64 tune_interactive_slice_ns = ASTRO_SLICE_INTERACTIVE_MAX_NS;
volatile u64 tune_normal_slice_ns = ASTRO_SLICE_NORMAL_BASE_NS;
volatile u64 tune_compute_slice_ns = ASTRO_SLICE_COMPUTE_NS;
volatile u64 tune_background_slice_ns = ASTRO_SLICE_BACKGROUND_NS;
volatile u64 tune_srpt_threshold_ns = ASTRO_SRPT_THRESHOLD_NS;

#ifndef min
#define min(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef U32_MAX
#define U32_MAX ((u32)~0U)
#endif

/*
 * Helper macros
 */
#define ASTRO_CPUSTAT_INC(_cstate, _field) do { \
	typeof(_cstate) __cstate = (_cstate); \
	if (__cstate) __cstate->_field++; \
	else __sync_fetch_and_add(&_field, 1); \
} while (0)

static inline struct astro_task_ctx *lookup_task_ctx(const struct task_struct *p)
{
	return bpf_task_storage_get(&task_ctx_stor, (struct task_struct *)p, 0, 0);
}

static inline struct astro_task_ctx *alloc_task_ctx(struct task_struct *p)
{
	return bpf_task_storage_get(&task_ctx_stor, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
}

static __always_inline struct astro_cpu_state *lookup_cpu_state(void)
{
	u32 key = 0;
	return bpf_map_lookup_elem(&cpu_state, &key);
}

static __always_inline bool is_kthread(const struct task_struct *p)
{
	return p->flags & PF_KTHREAD;
}

static __always_inline bool is_pinned_kthread(const struct task_struct *p)
{
	return is_kthread(p) && p->nr_cpus_allowed == 1;
}

static __always_inline s64 clamp_budget(s64 budget_ns)
{
	if (budget_ns > (s64)ASTRO_BUDGET_MAX_NS)
		return ASTRO_BUDGET_MAX_NS;
	if (budget_ns < -(s64)ASTRO_BUDGET_MIN_NS)
		return -(s64)ASTRO_BUDGET_MIN_NS;
	return budget_ns;
}

/*
 * Simple fixed-point log2 approximation for u64.
 */
static __always_inline u32 astro_log2_u64(u64 v)
{
	u32 l = 0;
	if (v == 0)
		return 0;
	#pragma unroll
	for (int i = 32; i >= 1; i >>= 1) {
		u64 shift = 1ULL << i;
		if (v >= shift) {
			v >>= i;
			l += i;
		}
	}
	return l;
}

/*
 * Latency criticality calculation (LAVD-inspired).
 *
 * A task is more latency-critical if:
 *   - wait_freq and wake_freq are higher (middle of task graph)
 *   - avg_runtime_ns is shorter (short jobs hurt more from delay)
 */
static __always_inline void calc_lat_cri(struct task_struct *p,
					  struct astro_task_ctx *taskc)
{
	u64 wait_ft, wake_ft, runtime_ft, weight_ft;
	u64 log_wwf, lat_cri;
	u32 weight_boost = 1;

	if (!taskc)
		return;

	wait_ft = min(taskc->wait_freq, (u32)ASTRO_LC_FREQ_MAX) + 1;
	wake_ft = min(taskc->wake_freq, (u32)ASTRO_LC_FREQ_MAX) + 1;

	if (ASTRO_LC_RUNTIME_MAX > taskc->avg_runtime_ns) {
		u64 delta = ASTRO_LC_RUNTIME_MAX - taskc->avg_runtime_ns;
		runtime_ft = (delta / ASTRO_SLICE_MIN_NS) + 1;
	} else {
		runtime_ft = 1;
	}

	/* Context weight boosts */
	if (taskc->is_wakeup)
		weight_boost += ASTRO_LC_WEIGHT_BOOST_REGULAR;
	if (taskc->is_sync_wakeup)
		weight_boost += ASTRO_LC_WEIGHT_BOOST_REGULAR;
	if (is_kthread(p))
		weight_boost += ASTRO_LC_WEIGHT_BOOST_MEDIUM;
	if (p->nr_cpus_allowed == 1 || is_migration_disabled(p))
		weight_boost += ASTRO_LC_WEIGHT_BOOST_MEDIUM;

	weight_ft = (u64)p->scx.weight * weight_boost + 1;

	log_wwf = astro_log2_u64((u64)wait_ft * (u64)wake_ft);
	lat_cri = log_wwf + astro_log2_u64(runtime_ft * weight_ft);
	lat_cri = lat_cri * lat_cri;

	/* Waker/wakee propagation (boost chains) */
	u64 giver = (u64)taskc->lat_cri_waker + (u64)taskc->lat_cri_wakee;
	if (giver > (2 * lat_cri)) {
		u64 giver_inh = (giver - (2 * lat_cri)) >> ASTRO_LC_INH_GIVER_SHIFT;
		u64 receiver_max = lat_cri >> ASTRO_LC_INH_RECEIVER_SHIFT;
		lat_cri += min(giver_inh, receiver_max);
	}

	taskc->lat_cri = (u32)min(lat_cri, (u64)U32_MAX);
	taskc->lat_cri_waker = 0;
	taskc->lat_cri_wakee = 0;

	/* Normalize to [0, 1024] */
	u64 max_cri = 1024ULL * 1024ULL; /* lat_cri max before sqrt scaling approx */
	if (lat_cri > max_cri)
		lat_cri = max_cri;
	taskc->normalized_lat_cri = (u32)((lat_cri * 1024ULL) / max_cri);
}

/*
 * Automatic profile classification based on behavioral signals.
 */
static __always_inline u8 auto_classify_profile(struct task_struct *p,
						struct astro_task_ctx *taskc)
{
	u32 nlat = taskc->normalized_lat_cri;
	u64 avg_runtime = taskc->avg_runtime_ns;
	u32 wait_freq = taskc->wait_freq;
	u32 wake_freq = taskc->wake_freq;

	if (nlat > ASTRO_LAT_CRI_INTERACTIVE &&
	    avg_runtime < ASTRO_AVG_RUNTIME_INTERACTIVE &&
	    (wait_freq > ASTRO_WAIT_FREQ_INTERACTIVE || wake_freq > ASTRO_WAKE_FREQ_INTERACTIVE)) {
		return ASTRO_PROFILE_INTERACTIVE;
	}

	if (avg_runtime > ASTRO_AVG_RUNTIME_COMPUTE &&
	    wake_freq < 2 && wait_freq < 2) {
		return ASTRO_PROFILE_COMPUTE;
	}

	if (taskc->hog_score >= ASTRO_HOG_SCORE_CONTAIN)
		return ASTRO_PROFILE_BACKGROUND;

	return ASTRO_PROFILE_NORMAL;
}

/*
 * Determine the effective profile for a task.
 * Explicit overrides take precedence but auto-detection still runs underneath.
 */
static __always_inline u8 effective_profile(struct task_struct *p,
					      struct astro_task_ctx *taskc)
{
	u8 profile;
	struct astro_tgid_profile *tp;

	if (!taskc)
		return ASTRO_PROFILE_NORMAL;

	/* Check explicit override */
	pid_t tgid = p->tgid;
	tp = bpf_map_lookup_elem(&tgid_profile_map, &tgid);
	if (tp && tp->profile != ASTRO_PROFILE_AUTO) {
		profile = tp->profile;
	} else {
		profile = auto_classify_profile(p, taskc);
	}

	/* Hysteresis to prevent rapid profile thrashing */
	if (profile != taskc->pending_profile) {
		taskc->pending_profile = profile;
		taskc->profile_hysteresis = 1;
	} else if (taskc->profile_hysteresis < ASTRO_HOG_SCORE_MAX) {
		taskc->profile_hysteresis++;
	}

	if (taskc->profile_hysteresis >= ASTRO_HYSTERESIS_THRESHOLD) {
		if (taskc->current_profile != profile) {
			struct astro_cpu_state *cstate = lookup_cpu_state();
			ASTRO_CPUSTAT_INC(cstate, profile_transitions);
		}
		taskc->current_profile = profile;
	}

	return taskc->current_profile;
}

/*
 * Budget refill on wakeup (Flow-inspired).
 */
static __always_inline void update_budget_on_wakeup(struct task_struct *p,
					    struct astro_task_ctx *taskc,
					    u64 now)
{
	u64 sleep_ns;
	s64 refill_ns;

	if (!taskc)
		return;

	taskc->last_refill_ns = 0;
	if (!taskc->sleep_started_at || now <= taskc->sleep_started_at) {
		taskc->last_sleep_ns = 0;
		return;
	}

	sleep_ns = now - taskc->sleep_started_at;
	if (sleep_ns > ASTRO_SLEEP_MAX_NS)
		sleep_ns = ASTRO_SLEEP_MAX_NS;

	refill_ns = (s64)(sleep_ns / ASTRO_REFILL_DIV);
	if (refill_ns > 0) {
		refill_ns = (s64)scale_by_task_weight((struct task_struct *)p, (u64)refill_ns);
		if (refill_ns < (s64)ASTRO_INTERACTIVE_FLOOR_NS &&
		    sleep_ns >= ASTRO_INTERACTIVE_SLEEP_MIN_NS)
			refill_ns = (s64)ASTRO_INTERACTIVE_FLOOR_NS;
	}

	taskc->budget_ns = clamp_budget(taskc->budget_ns + refill_ns);
	taskc->last_refill_ns = refill_ns > 0 ? (u64)refill_ns : 0;
	taskc->last_sleep_ns = sleep_ns;
	taskc->sleep_started_at = 0;

	if (refill_ns > 0) {
		struct astro_cpu_state *cstate = lookup_cpu_state();
		ASTRO_CPUSTAT_INC(cstate, budget_refills);
	}
}

/*
 * Slice selection per profile.
 */
static __always_inline u64 profile_slice_ns(u8 profile)
{
	switch (profile) {
	case ASTRO_PROFILE_INTERACTIVE: {
		u64 slice = tune_interactive_slice_ns;
		if (slice < ASTRO_SLICE_INTERACTIVE_MIN_NS)
			slice = ASTRO_SLICE_INTERACTIVE_MIN_NS;
		if (slice > ASTRO_SLICE_INTERACTIVE_MAX_NS)
			slice = ASTRO_SLICE_INTERACTIVE_MAX_NS;
		return slice;
	}
	case ASTRO_PROFILE_WAKER_BOOST: /* not a stored profile, but used for boost */
		return ASTRO_SLICE_BOOST_NS;
	case ASTRO_PROFILE_NORMAL: {
		u64 slice = tune_normal_slice_ns;
		if (slice < ASTRO_SLICE_NORMAL_MIN_NS)
			slice = ASTRO_SLICE_NORMAL_MIN_NS;
		if (slice > ASTRO_SLICE_NORMAL_MAX_NS)
			slice = ASTRO_SLICE_NORMAL_MAX_NS;
		return slice;
	}
	case ASTRO_PROFILE_COMPUTE:
		return tune_compute_slice_ns;
	case ASTRO_PROFILE_BACKGROUND:
		return tune_background_slice_ns;
	default:
		return tune_normal_slice_ns;
	}
}

/*
 * DSQ selection per profile, with waker-boost override.
 */
static __always_inline u64 profile_dsq_id(u8 profile)
{
	switch (profile) {
	case ASTRO_PROFILE_INTERACTIVE:
		return ASTRO_INTERACTIVE_DSQ;
	case ASTRO_PROFILE_NORMAL:
		return ASTRO_NORMAL_DSQ;
	case ASTRO_PROFILE_COMPUTE:
		return ASTRO_COMPUTE_DSQ;
	case ASTRO_PROFILE_BACKGROUND:
		return ASTRO_BACKGROUND_DSQ;
	default:
		return ASTRO_NORMAL_DSQ;
	}
}

/*
 * Check if waker boost is active; if so, route to boost DSQ.
 */
static __always_inline u8 maybe_boost_profile(struct astro_task_ctx *taskc, u64 now)
{
	if (!taskc)
		return ASTRO_PROFILE_NORMAL;
	if (taskc->waker_boost_expire_ns > now) {
		struct astro_cpu_state *cstate = lookup_cpu_state();
		ASTRO_CPUSTAT_INC(cstate, waker_boosts);
		return ASTRO_PROFILE_WAKER_BOOST;
	}
	return taskc->current_profile;
}

/*
 * SRPT-inspired vtime scaling.
 * Short tasks (avg_runtime < threshold) get scaled-down vtime so they
 * are scheduled earlier within the normal/compute lanes.
 */
static __always_inline u64 srpt_vtime_scale(u64 vtime, u64 avg_runtime)
{
	u64 threshold = tune_srpt_threshold_ns;
	if (threshold == 0)
		threshold = ASTRO_SRPT_THRESHOLD_NS;

	if (avg_runtime < threshold) {
		/* Scale vtime by 0.5 for short tasks */
		vtime = (vtime * ASTRO_SRPT_VTIME_SCALE_NUM) / ASTRO_SRPT_VTIME_SCALE_DEN;
	}
	return vtime;
}

/*
 * Starvation tracking helpers
 */
static __always_inline void bump_starvation(u64 *counter, u64 max)
{
	if (counter && *counter < max)
		(*counter)++;
}

static __always_inline void reset_starvation(u64 *counter)
{
	if (counter)
		*counter = 0;
}

/*
 * Dispatch accounting helpers
 */
static __always_inline void note_high_priority_dispatch(struct astro_cpu_state *cstate)
{
	u64 max = ASTRO_HIGH_PRIO_BURST_MAX;
	bump_starvation(cstate ? &cstate->contained_starvation_rounds : NULL,
			ASTRO_CONTAINED_STARVATION_MAX);
	bump_starvation(cstate ? &cstate->shared_starvation_rounds : NULL,
			ASTRO_SHARED_STARVATION_MAX);
	if (cstate && cstate->high_priority_burst_rounds < max)
		cstate->high_priority_burst_rounds++;
}

static __always_inline void note_low_priority_dispatch(struct astro_cpu_state *cstate,
						bool contained)
{
	if (cstate)
		cstate->high_priority_burst_rounds = 0;
	if (contained) {
		reset_starvation(cstate ? &cstate->contained_starvation_rounds : NULL);
		bump_starvation(cstate ? &cstate->shared_starvation_rounds : NULL,
				ASTRO_SHARED_STARVATION_MAX);
	} else {
		reset_starvation(cstate ? &cstate->shared_starvation_rounds : NULL);
		bump_starvation(cstate ? &cstate->contained_starvation_rounds : NULL,
				ASTRO_CONTAINED_STARVATION_MAX);
	}
}

static __always_inline bool should_force_low_priority(struct astro_cpu_state *cstate)
{
	u64 hp = cstate ? cstate->high_priority_burst_rounds : 0;
	return hp >= ASTRO_HIGH_PRIO_BURST_MAX;
}

/*
 * scx callbacks
 */

s32 BPF_STRUCT_OPS_SLEEPABLE(astro_init)
{
	s32 ret;

	ret = scx_lib_init();
	if (ret)
		return ret;

	ret = scx_bpf_create_dsq(ASTRO_INTERACTIVE_DSQ, -1);
	if (ret < 0 && ret != -EEXIST) {
		scx_bpf_error("failed to create interactive DSQ: %d", ret);
		return ret;
	}
	ret = scx_bpf_create_dsq(ASTRO_WAKER_BOOST_DSQ, -1);
	if (ret < 0 && ret != -EEXIST) {
		scx_bpf_error("failed to create waker boost DSQ: %d", ret);
		return ret;
	}
	ret = scx_bpf_create_dsq(ASTRO_NORMAL_DSQ, -1);
	if (ret < 0 && ret != -EEXIST) {
		scx_bpf_error("failed to create normal DSQ: %d", ret);
		return ret;
	}
	ret = scx_bpf_create_dsq(ASTRO_COMPUTE_DSQ, -1);
	if (ret < 0 && ret != -EEXIST) {
		scx_bpf_error("failed to create compute DSQ: %d", ret);
		return ret;
	}
	ret = scx_bpf_create_dsq(ASTRO_BACKGROUND_DSQ, -1);
	if (ret < 0 && ret != -EEXIST) {
		scx_bpf_error("failed to create background DSQ: %d", ret);
		return ret;
	}

	return 0;
}

s32 BPF_STRUCT_OPS_SLEEPABLE(astro_init_task, struct task_struct *p,
			     struct scx_init_task_args *args)
{
	struct astro_task_ctx *taskc;
	u64 now;

	taskc = alloc_task_ctx(p);
	if (!taskc)
		return -ENOMEM;

	now = bpf_ktime_get_ns();
	__builtin_memset(taskc, 0, sizeof(*taskc));
	taskc->current_profile = ASTRO_PROFILE_NORMAL;
	taskc->pending_profile = ASTRO_PROFILE_NORMAL;
	taskc->last_cpu = -1;
	taskc->wake_cpu = -1;
	taskc->sleep_started_at = now;
	taskc->budget_ns = 0;

	return 0;
}

void BPF_STRUCT_OPS(astro_enable, struct task_struct *p)
{
	struct astro_task_ctx *taskc;
	u64 now;

	taskc = lookup_task_ctx(p);
	if (!taskc)
		return;

	now = bpf_ktime_get_ns();
	__builtin_memset(taskc, 0, sizeof(*taskc));
	taskc->current_profile = ASTRO_PROFILE_NORMAL;
	taskc->pending_profile = ASTRO_PROFILE_NORMAL;
	taskc->last_cpu = -1;
	taskc->wake_cpu = -1;
	taskc->sleep_started_at = now;
	taskc->budget_ns = 0;
}

s32 BPF_STRUCT_OPS(astro_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	struct astro_task_ctx *taskc;
	bool is_idle = false;
	s32 cpu;
	bool non_migratable = p->nr_cpus_allowed == 1 || is_migration_disabled(p);

	taskc = lookup_task_ctx(p);
	if (taskc) {
		if (taskc->sleep_started_at)
			update_budget_on_wakeup(p, taskc, bpf_ktime_get_ns());

		/* Mark wakeup flags for classification */
		taskc->is_wakeup = true;
		taskc->is_sync_wakeup = !!(wake_flags & SCX_WAKE_SYNC);
		taskc->wake_cpu = -1;
		taskc->wake_cpu_idle = false;
		taskc->wake_cpu_valid = false;


	}

	if (!bpf_cpumask_test_cpu(prev_cpu, p->cpus_ptr))
		prev_cpu = bpf_cpumask_first(p->cpus_ptr);

	if (non_migratable) {
		cpu = prev_cpu;
		is_idle = scx_bpf_test_and_clear_cpu_idle(prev_cpu);
	} else {
		cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
	}

	if (taskc) {
		taskc->wake_cpu = cpu >= 0 ? cpu : prev_cpu;
		taskc->wake_cpu_idle = is_idle;
		taskc->wake_cpu_valid = taskc->wake_cpu >= 0 &&
			bpf_cpumask_test_cpu(taskc->wake_cpu, p->cpus_ptr);
	}

	return cpu >= 0 ? cpu : prev_cpu;
}

void BPF_STRUCT_OPS(astro_runnable, struct task_struct *p, u64 enq_flags)
{
	struct astro_task_ctx *taskc;
	struct astro_cpu_state *cstate;
	u64 now;

	taskc = lookup_task_ctx(p);
	cstate = lookup_cpu_state();
	if (!taskc)
		return;

	now = bpf_ktime_get_ns();
	taskc->is_wakeup = true;
	if (taskc->sleep_started_at && now > taskc->sleep_started_at)
		update_budget_on_wakeup(p, taskc, now);

	/* Approximate waker-boost chain: if this CPU recently ran an interactive
	 * task, grant the wakee a temporary boost. This is a heuristic in lieu
	 * of a true waker-wakee tracking mechanism (e.g., BPF trampoline on
	 * try_to_wake_up), which is not available via standard struct_ops.
	 */
	if (cstate && cstate->last_interactive_ts > 0 &&
	    now - cstate->last_interactive_ts < ASTRO_WAKER_BOOST_DURATION_NS &&
	    taskc->waker_boost_depth < ASTRO_WAKER_BOOST_MAX_CHAIN) {
		taskc->waker_boost_expire_ns = now + ASTRO_WAKER_BOOST_DURATION_NS;
		taskc->waker_boost_depth++;
	}
}

void BPF_STRUCT_OPS(astro_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct astro_task_ctx *taskc;
	struct astro_cpu_state *cstate;
	u64 now = bpf_ktime_get_ns();
	u8 profile;
	u64 dsq_id, slice_ns, vtime = p->scx.dsq_vtime;
	s32 target_cpu = -1;
	bool is_wakeup = enq_flags & SCX_ENQ_WAKEUP;

	taskc = lookup_task_ctx(p);
	cstate = lookup_cpu_state();

	if (is_pinned_kthread(p)) {
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, ASTRO_SLICE_MIN_NS, enq_flags);
		return;
	}

	/* Classification */
	if (taskc) {
		calc_lat_cri(p, taskc);
		profile = effective_profile(p, taskc);

		/* Waker-boost override */
		profile = maybe_boost_profile(taskc, now);

		/* Budget exhaustion -> hog score for background containment */
		if (taskc->budget_ns <= 0 && is_wakeup) {
			if (taskc->hog_score < ASTRO_HOG_SCORE_MAX)
				taskc->hog_score += ASTRO_HOG_SCORE_EXHAUST_STEP;
		}

		if (taskc->wake_cpu_valid)
			target_cpu = taskc->wake_cpu;
	} else {
		profile = ASTRO_PROFILE_NORMAL;
	}

	slice_ns = profile_slice_ns(profile);
	dsq_id = profile_dsq_id(profile);

	/* SRPT scaling for normal/compute lanes */
	if ((profile == ASTRO_PROFILE_NORMAL || profile == ASTRO_PROFILE_COMPUTE) && taskc) {
		u64 old_vtime = vtime;
		vtime = srpt_vtime_scale(vtime, taskc->avg_runtime_ns);
		if (vtime != old_vtime) {
			ASTRO_CPUSTAT_INC(cstate, srpt_short_tasks);
		}
	}

	/* Interactive / waker-boost tasks get head insertion and may preempt */
	if (profile == ASTRO_PROFILE_INTERACTIVE || profile == ASTRO_PROFILE_WAKER_BOOST) {
		enq_flags |= SCX_ENQ_HEAD;
		if (target_cpu >= 0 && !taskc->wake_cpu_idle) {
			scx_bpf_kick_cpu(target_cpu, SCX_KICK_PREEMPT);
			ASTRO_CPUSTAT_INC(cstate, preempts);
			ASTRO_CPUSTAT_INC(cstate, kicks_preempt);
		} else if (target_cpu >= 0 && taskc->wake_cpu_idle) {
			scx_bpf_kick_cpu(target_cpu, SCX_KICK_IDLE);
			ASTRO_CPUSTAT_INC(cstate, kicks_idle);
		}
	} else {
		if (target_cpu >= 0 && taskc && taskc->wake_cpu_idle) {
			scx_bpf_kick_cpu(target_cpu, SCX_KICK_IDLE);
			ASTRO_CPUSTAT_INC(cstate, kicks_idle);
		}
	}

	/*
	 * Direct local enqueue for locality if CPU is idle and task is
	 * interactive/normal.  Re-check cpus_ptr because affinity may have
	 * changed between select_cpu() and enqueue().
	 */
	if (taskc && target_cpu >= 0 && taskc->wake_cpu_idle &&
	    bpf_cpumask_test_cpu(target_cpu, p->cpus_ptr) &&
	    (profile == ASTRO_PROFILE_INTERACTIVE || profile == ASTRO_PROFILE_WAKER_BOOST ||
	     profile == ASTRO_PROFILE_NORMAL)) {
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | target_cpu, slice_ns, enq_flags);
		return;
	}

	/* Use vtime dispatch for normal/compute to enable SRPT ordering within lane */
	if (profile == ASTRO_PROFILE_NORMAL || profile == ASTRO_PROFILE_COMPUTE) {
		scx_bpf_dsq_insert_vtime(p, dsq_id, slice_ns, vtime, enq_flags);
	} else {
		scx_bpf_dsq_insert(p, dsq_id, slice_ns, enq_flags);
	}

	if (taskc) {
		taskc->wake_cpu_valid = false;
		taskc->wake_cpu = -1;
	}
}

void BPF_STRUCT_OPS(astro_dispatch, s32 cpu, struct task_struct *prev)
{
	struct astro_cpu_state *cstate = lookup_cpu_state();
	bool force_bg, force_shared;
	u64 contained_rounds, shared_rounds;

	contained_rounds = cstate ? cstate->contained_starvation_rounds : 0;
	shared_rounds = cstate ? cstate->shared_starvation_rounds : 0;
	force_bg = contained_rounds >= ASTRO_CONTAINED_STARVATION_MAX;
	force_shared = shared_rounds >= ASTRO_SHARED_STARVATION_MAX;

	/* 1. Interactive lane (highest priority, bounded burst) */
	if (scx_bpf_dsq_move_to_local(ASTRO_INTERACTIVE_DSQ, 0)) {
		ASTRO_CPUSTAT_INC(cstate, interactive_dispatches);
		note_high_priority_dispatch(cstate);
		return;
	}

	/* 2. Waker boost lane */
	if (scx_bpf_dsq_move_to_local(ASTRO_WAKER_BOOST_DSQ, 0)) {
		ASTRO_CPUSTAT_INC(cstate, waker_boost_dispatches);
		note_high_priority_dispatch(cstate);
		return;
	}

	/* Fairness: if we've been consuming high-priority tasks for too long,
	 * force service of lower lanes to prevent starvation.
	 */
	if (should_force_low_priority(cstate)) {
		if (force_shared && scx_bpf_dsq_move_to_local(ASTRO_NORMAL_DSQ, 0)) {
			ASTRO_CPUSTAT_INC(cstate, normal_dispatches);
			note_low_priority_dispatch(cstate, false);
			return;
		}
		if (force_bg && scx_bpf_dsq_move_to_local(ASTRO_BACKGROUND_DSQ, 0)) {
			ASTRO_CPUSTAT_INC(cstate, background_dispatches);
			note_low_priority_dispatch(cstate, true);
			return;
		}
	}

	/* 3. Normal lane */
	if (scx_bpf_dsq_move_to_local(ASTRO_NORMAL_DSQ, 0)) {
		ASTRO_CPUSTAT_INC(cstate, normal_dispatches);
		note_high_priority_dispatch(cstate);
		return;
	}

	/* 4. Compute lane */
	if (scx_bpf_dsq_move_to_local(ASTRO_COMPUTE_DSQ, 0)) {
		ASTRO_CPUSTAT_INC(cstate, compute_dispatches);
		note_high_priority_dispatch(cstate);
		return;
	}

	/* 5. Background lane (starvation rescue via head promotion) */
	if (force_bg) {
		if (scx_bpf_dsq_move_to_local(ASTRO_BACKGROUND_DSQ, 0)) {
			ASTRO_CPUSTAT_INC(cstate, background_dispatches);
			note_low_priority_dispatch(cstate, true);
			return;
		}
	}
	if (scx_bpf_dsq_move_to_local(ASTRO_BACKGROUND_DSQ, 0)) {
		ASTRO_CPUSTAT_INC(cstate, background_dispatches);
		note_low_priority_dispatch(cstate, true);
		return;
	}

	/* If prev is still runnable and queued, refresh its slice */
	if (prev && (prev->scx.flags & SCX_TASK_QUEUED)) {
		struct astro_task_ctx *tctx = lookup_task_ctx(prev);
		prev->scx.slice = profile_slice_ns(tctx ? tctx->current_profile : ASTRO_PROFILE_NORMAL);
	}
}

void BPF_STRUCT_OPS(astro_running, struct task_struct *p)
{
	struct astro_task_ctx *taskc;
	struct astro_cpu_state *cstate;
	s32 cpu;
	u64 now;

	taskc = lookup_task_ctx(p);
	cstate = lookup_cpu_state();
	cpu = bpf_get_smp_processor_id();
	now = bpf_ktime_get_ns();

	if (taskc) {
		taskc->last_cpu = cpu;
		taskc->last_run_at = now;
		taskc->is_wakeup = false;
		taskc->is_sync_wakeup = false;
	}

	/* Track interactive task execution for waker-boost chain approximation */
	if (cstate && taskc && taskc->current_profile == ASTRO_PROFILE_INTERACTIVE)
		cstate->last_interactive_ts = now;

	__sync_fetch_and_add(&nr_running, 1);
}

void BPF_STRUCT_OPS(astro_stopping, struct task_struct *p, bool runnable)
{
	struct astro_task_ctx *taskc;
	u64 now;
	u64 runtime_ns = 0;
	bool exhausted = false;

	taskc = lookup_task_ctx(p);
	now = bpf_ktime_get_ns();

	if (taskc) {
		if (taskc->last_run_at && now > taskc->last_run_at)
			runtime_ns = now - taskc->last_run_at;

		/* Update EWMA of runtime */
		if (taskc->avg_runtime_ns == 0) {
			taskc->avg_runtime_ns = runtime_ns;
		} else {
			/* EWMA: avg = (3*avg + runtime) / 4 */
			taskc->avg_runtime_ns = (3 * taskc->avg_runtime_ns + runtime_ns) / 4;
		}
		taskc->acc_runtime_ns += runtime_ns;

		/* Budget exhaustion tracking */
		exhausted = taskc->budget_ns > 0 &&
		    taskc->budget_ns - (s64)runtime_ns <= 0;
		if (exhausted) {
			struct astro_cpu_state *cstate = lookup_cpu_state();
			ASTRO_CPUSTAT_INC(cstate, budget_exhaustions);
			if (taskc->hog_score < ASTRO_HOG_SCORE_MAX)
				taskc->hog_score += ASTRO_HOG_SCORE_EXHAUST_STEP;
		}

		taskc->budget_ns = clamp_budget(taskc->budget_ns - (s64)runtime_ns);

		/* Update frequencies heuristically */
		if (!runnable) {
			taskc->sleep_started_at = now;
			taskc->wait_freq = min(taskc->wait_freq + 1, (u32)ASTRO_LC_FREQ_MAX);
		} else {
			taskc->wake_freq = min(taskc->wake_freq + 1, (u32)ASTRO_LC_FREQ_MAX);
		}

		/* Decay hog score on short runs that don't exhaust budget */
		if (!exhausted && runtime_ns > 0 && runtime_ns < ASTRO_HOG_RUNTIME_THRESHOLD_NS) {
			if (taskc->hog_score > ASTRO_HOG_SCORE_DECAY_STEP)
				taskc->hog_score -= ASTRO_HOG_SCORE_DECAY_STEP;
			else
				taskc->hog_score = 0;
		}

		/* Waker boost expiration */
		if (taskc->waker_boost_expire_ns && now >= taskc->waker_boost_expire_ns) {
			taskc->waker_boost_expire_ns = 0;
			taskc->waker_boost_depth = 0;
		}

		/* Latency criticality propagation to wakee/waker would happen here
		 * in a full implementation with task_graph tracking.
		 */
	}

	__sync_fetch_and_sub(&nr_running, 1);
}

void BPF_STRUCT_OPS(astro_cpu_release, s32 cpu, struct scx_cpu_release_args *args)
{
	scx_bpf_reenqueue_local();
}

void BPF_STRUCT_OPS(astro_exit_task, struct task_struct *p,
		    struct scx_exit_task_args *args)
{
	struct astro_task_ctx *taskc = lookup_task_ctx(p);
	if (!taskc)
		return;
	__builtin_memset(taskc, 0, sizeof(*taskc));
}

void BPF_STRUCT_OPS(astro_set_cpumask, struct task_struct *p,
		    const struct cpumask *cpumask)
{
	struct astro_task_ctx *taskc = lookup_task_ctx(p);

	if (!taskc)
		return;

	/* Affinity changed — invalidate any stale wake_cpu so that
	 * astro_enqueue() does not dispatch to a now-forbidden CPU.
	 */
	taskc->wake_cpu_valid = false;
	taskc->wake_cpu = -1;
}

void BPF_STRUCT_OPS(astro_exit, struct scx_exit_info *info)
{
	UEI_RECORD(uei, info);
}

SCX_OPS_DEFINE(astro_ops,
	       .select_cpu		= (void *)astro_select_cpu,
	       .enqueue			= (void *)astro_enqueue,
	       .dispatch		= (void *)astro_dispatch,
	       .cpu_release		= (void *)astro_cpu_release,
	       .runnable		= (void *)astro_runnable,
	       .enable			= (void *)astro_enable,
	       .running			= (void *)astro_running,
	       .stopping		= (void *)astro_stopping,
       .init_task		= (void *)astro_init_task,
       .exit_task		= (void *)astro_exit_task,
       .set_cpumask		= (void *)astro_set_cpumask,
       .init			= (void *)astro_init,
	       .exit			= (void *)astro_exit,
	       .timeout_ms		= 5000,
	       .name			= "scx_astro");
