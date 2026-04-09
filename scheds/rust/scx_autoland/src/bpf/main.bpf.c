/* SPDX-License-Identifier: GPL-2.0 */

#include <scx/common.bpf.h>
#include <scx/percpu.bpf.h>
#include "intf.h"

/*
 * Maximum amount of CPUs supported by the scheduler when flat or preferred
 * idle CPU scan is enabled.
 */
#define MAX_CPUS 4096

/*
 * Maximum rate of task wakeups/sec (tasks with a higher rate are capped to
 * this value).
 */
#define MAX_WAKEUP_FREQ 64ULL

/*
 * Return true if @cpu is valid, false otherwise.
 */
#define IS_CPU_VALID(__cpu) ((__cpu) >= 0 && (__cpu) < MAX_CPUS)

/*
 * Return the LLC id associated to a CPU, or -1 if the CPU is invalid.
 */
#define CPU_LLC_ID(__cpu) (IS_CPU_VALID(__cpu) ? cpu_llc_id(__cpu) : -1)

/*
 * Return the capacity of a CPU, or -1 if the CPU is invalid.
 */
#define CPU_CAPACITY(__cpu) (IS_CPU_VALID(__cpu) ? cpu_capacity[__cpu] : -1)

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

/*
 * Default time slice.
 */
const volatile u64 slice_ns = NSEC_PER_MSEC;

/*
 * Maximum time slice lag.
 *
 * Increasing this value can help to increase the responsiveness of interactive
 * tasks at the cost of making regular and newly created tasks less responsive
 * (0 = disabled).
 */
const volatile u64 slice_lag = 40ULL * NSEC_PER_MSEC;

/*
 * Per-class time slice durations (in nanoseconds).
 * Set at initialization; runtime changes require restart.
 */
const volatile u64 slice_ns_lc =
	500 * NSEC_PER_USEC; /* 500us for latency-critical */
const volatile u64 slice_ns_normal = 2 * NSEC_PER_MSEC; /* 2ms for normal */
const volatile u64 slice_ns_hog	   = 4 * NSEC_PER_MSEC; /* 4ms for hog */

/*
 * Preemption threshold (latency criticality >= this value can preempt).
 * Set at initialization; runtime changes require restart.
 */
const volatile u32 preempt_threshold = 128;

/*
 * Minimum slice duration (floor for load-adjusted slice calculation).
 */
const volatile u64 slice_min_ns = 100 * NSEC_PER_USEC; /* 100us minimum */

/*
 * Maximum amount of CPUs supported by the system.
 */
static u64 nr_cpu_ids;

/*
 * CPUs in the system have SMT is enabled.
 */
const volatile bool smt_enabled = true;

/*
 * User CPU utilization threshold to determine when the system is busy.
 */
const volatile u64 busy_threshold;

/*
 * Current global CPU utilization percentage in the range [0 .. 1024].
 */
volatile u64 cpu_util;

/*
 * Subset of CPUs to prioritize.
 */
private(AUTOLAND) struct bpf_cpumask __kptr *primary_cpumask;

/*
 * Set to true when @primary_cpumask is empty (primary domain includes all
 * the CPU).
 */
const volatile bool primary_all = false;

/*
 * Enable preferred cores prioritization.
 */
const volatile bool preferred_idle_scan;

/*
 * CPUs sorted by their capacity in descendent order.
 */
const volatile u64 preferred_cpus[MAX_CPUS];

/*
 * Cache CPU capacity values.
 */
const volatile u64 cpu_capacity[MAX_CPUS];

/*
 * Scheduling statistics.
 */
volatile u64 nr_local_dispatch, nr_remote_dispatch, nr_keep_running;
volatile u64 nr_lc_tasks, nr_normal_tasks, nr_hog_tasks;

/*
 * Current system vruntime.
 */
static u64 vtime_now;

/*
 * Per-CPU context.
 */
struct cpu_ctx {
	struct bpf_cpumask __kptr *smt;
	u32 current_class; /* enum autoland_task_class of current task */
	u64 current_task_start; /* When current task started running */
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, u32);
	__type(value, struct cpu_ctx);
	__uint(max_entries, 1);
} cpu_ctx_stor SEC(".maps");

/*
 * Return a CPU context.
 */
struct cpu_ctx *try_lookup_cpu_ctx(s32 cpu)
{
	const u32 idx = 0;
	return bpf_map_lookup_percpu_elem(&cpu_ctx_stor, &idx, cpu);
}

/*
 * Per-task context.
 */
struct task_ctx {
	u64  last_run_at;
	u64  last_woke_at;
	u64  wakeup_freq;
	u64  awake_vtime;
	u64  avg_runtime;
	u32  task_class; /* enum autoland_task_class */
	u32  lat_cri; /* Latency criticality score [0, 1024] */
	u64  last_class_check; /* Timestamp of last classification */
	bool is_scx_turbo; /* Task has SCX_TURBO=1 set */
	bool is_steam_game; /* Task is a detected Steam game */
	bool is_de_task; /* Task is a desktop environment component */
	u64  pelt_util_sum; /* Accumulated PELT utilization */
	u32  pelt_util_count; /* Count for PELT averaging */
};

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
 * Return true if @p still wants to run, false otherwise.
 */
static bool is_task_queued(const struct task_struct *p)
{
	return p->scx.flags & SCX_TASK_QUEUED;
}

/*
 * Return true if @p can only run on a single CPU, false otherwise.
 */
static inline bool is_pcpu_task(const struct task_struct *p)
{
	return p->nr_cpus_allowed == 1 || is_migration_disabled(p);
}

/*
 * Return true if the task should be forced to stay on the same CPU, false
 * otherwise.
 */
static bool is_task_sticky(const struct task_ctx *tctx)
{
	return tctx->avg_runtime < 10 * NSEC_PER_USEC;
}

/*
 * Return true if the system is considered busy (user CPU utilization is
 * above the threshold), false otherwise.
 */
static inline bool is_system_busy(void)
{
	return cpu_util >= busy_threshold;
}

/*
 * Per-class latency statistics map (extended with histogram and variance).
 */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 3); /* One entry per class */
	__type(key, u32);
	__type(value, struct class_latency_stat);
} class_latency_stats SEC(".maps");

/*
 * Per-class criticality metrics map (for LAVD-style scoring).
 */
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 3); /* One entry per class */
	__type(key, u32);
	__type(value, struct class_criticality_metrics);
} class_criticality_stats SEC(".maps");

/*
 * Per-task runtime statistics for variance calculation.
 */
struct task_runtime_stat {
	u64 total_runtime_ns;
	u64 runtime_squared_sum; /* Sum of squared runtimes */
	u32 sample_count;
	u64 last_wakeup_time;
	u32 wakeup_count;
};

struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_runtime_stat);
} task_runtime_stats SEC(".maps");

/*
 * Task hints map (set from userspace for specific PIDs).
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32); /* pid */
	__type(value, u32); /* task class hint */
} task_hints SEC(".maps");

/*
 * Return the time slice for a task based on its class.
 */
static u64 task_class_slice(const struct task_ctx *tctx)
{
	switch (tctx->task_class) {
	case AUTOLAND_CLASS_LATENCY_CRITICAL:
		return slice_ns_lc;
	case AUTOLAND_CLASS_HOG:
		return slice_ns_hog;
	default:
		return slice_ns_normal;
	}
}

/*
 * Calculate latency criticality score [0, 1024] for a task.
 */
static u32 calc_lat_cri(struct task_struct *p, struct task_ctx *tctx)
{
	u32 lat_cri = 0;

	/* Base from wakeup frequency (0 to ~64 * 8 = 512 max) */
	lat_cri += (u32)(tctx->wakeup_freq * 8);

	/* Boost for latency-critical class */
	if (tctx->task_class == AUTOLAND_CLASS_LATENCY_CRITICAL)
		lat_cri += 512;

	/* Boost for SCX_TURBO tasks */
	if (tctx->is_scx_turbo)
		lat_cri += 256;

	/* Boost for steam games */
	if (tctx->is_steam_game)
		lat_cri += 128;

	/* Boost for DE tasks */
	if (tctx->is_de_task)
		lat_cri += 64;

	return MIN(lat_cri, 1024u);
}

/*
 * Classify task based on runtime heuristics.
 */
static void classify_task_heuristics(struct task_struct *p,
				     struct task_ctx	*tctx)
{
	u64 now	  = bpf_ktime_get_ns();
	u64 delta = now - tctx->last_class_check;

	/* Re-classify every ~100ms */
	if (delta < 100 * NSEC_PER_MSEC)
		return;

	tctx->last_class_check = now;

	/* Store previous class */
	u32 prev_class = tctx->task_class;

	/* Check for userspace hint first */
	u32  pid  = p->pid;
	u32 *hint = bpf_map_lookup_elem(&task_hints, &pid);
	if (hint) {
		tctx->task_class = *hint;
	} else {
		/* High wakeup frequency + short runtime = latency critical */
		if (tctx->wakeup_freq >= 32 &&
		    tctx->avg_runtime < 100 * NSEC_PER_USEC) {
			tctx->task_class = AUTOLAND_CLASS_LATENCY_CRITICAL;
		} else {
			/*
			 * Ratio-based hog detection: calculate duty cycle
			 * (avg_runtime * wakeup_freq) = total CPU time per 100ms window.
			 *
			 * High duty cycle (> 25ms per 100ms) with long individual
			 * runtimes indicates a CPU-bound hog task.
			 *
			 * Examples:
			 * - Hog: 20ms avg × 2 wakeups = 40ms duty cycle → HOG
			 * - LC: 50us avg × 50 wakeups = 2.5ms duty cycle → NORMAL
			 * - Video worker: 5ms avg × 15 wakeups = 75ms duty cycle → HOG
			 */
			u64 duty_cycle = tctx->avg_runtime * tctx->wakeup_freq;
			if (tctx->avg_runtime > 2 * NSEC_PER_MSEC &&
			    duty_cycle > (25 * NSEC_PER_MSEC)) {
				/* High CPU utilization = hog */
				tctx->task_class = AUTOLAND_CLASS_HOG;
			} else {
				/* Default to normal */
				tctx->task_class = AUTOLAND_CLASS_NORMAL;
			}
		}
	}

	/* Update counters if class changed */
	if (prev_class != tctx->task_class) {
		/* Decrement old class */
		if (prev_class == AUTOLAND_CLASS_LATENCY_CRITICAL)
			__sync_fetch_and_sub(&nr_lc_tasks, 1);
		else if (prev_class == AUTOLAND_CLASS_NORMAL)
			__sync_fetch_and_sub(&nr_normal_tasks, 1);
		else if (prev_class == AUTOLAND_CLASS_HOG)
			__sync_fetch_and_sub(&nr_hog_tasks, 1);

		/* Increment new class */
		if (tctx->task_class == AUTOLAND_CLASS_LATENCY_CRITICAL)
			__sync_fetch_and_add(&nr_lc_tasks, 1);
		else if (tctx->task_class == AUTOLAND_CLASS_NORMAL)
			__sync_fetch_and_add(&nr_normal_tasks, 1);
		else if (tctx->task_class == AUTOLAND_CLASS_HOG)
			__sync_fetch_and_add(&nr_hog_tasks, 1);
	}
}

/*
 * Try to preempt a CPU running a lower priority task.
 *
 * Uses power-of-two random choices to find a victim CPU running a task
 * with lower latency criticality. Returns true if preemption succeeded.
 */
static bool try_preempt(struct task_struct *p, struct task_ctx *tctx)
{
	s32		    victims[2];
	struct task_struct *victim;
	struct task_ctx	   *victim_tctx;
	u32		    victim_lat_cri, lowest_lat_cri = 1024;
	s32		    victim_cpu = -1;
	int		    i;

	/* Only latency-critical tasks above threshold can preempt */
	if (tctx->lat_cri < preempt_threshold)
		return false;

	/* Select two random CPUs using power-of-two random choices */
	victims[0] = bpf_get_prandom_u32() % nr_cpu_ids;
	victims[1] = bpf_get_prandom_u32() % nr_cpu_ids;

	/* Find the victim with the lowest latency criticality */
	for (i = 0; i < 2; i++) {
		s32 cpu = victims[i];

		/* Skip if this CPU is not allowed for the task */
		if (!bpf_cpumask_test_cpu(cpu, p->cpus_ptr))
			continue;

		/* Check if the task on this CPU has lower criticality */
		victim = __COMPAT_scx_bpf_cpu_curr(cpu);
		if (!victim || victim == p)
			continue;

		victim_tctx = try_lookup_task_ctx(victim);
		if (!victim_tctx)
			continue;

		/* Calculate victim's latency criticality if not recent */
		if (victim_tctx->lat_cri == 0)
			victim_lat_cri = calc_lat_cri(victim, victim_tctx);
		else
			victim_lat_cri = victim_tctx->lat_cri;

		/*
		 * Never preempt lock holders (check if task has any lock held).
		 * We approximate this by checking if the task is in an atomic
		 * context or has disabled preemption.
		 */
		if (victim->in_execve || victim->in_iowait)
			continue;

		/* Find lowest latency criticality victim */
		if (victim_lat_cri < lowest_lat_cri) {
			lowest_lat_cri = victim_lat_cri;
			victim_cpu     = cpu;
		}
	}

	/* Preempt if we found a suitable victim with lower criticality */
	if (victim_cpu >= 0 && lowest_lat_cri < tctx->lat_cri) {
		victim = __COMPAT_scx_bpf_cpu_curr(victim_cpu);
		if (victim) {
			/*
			 * Set victim's slice to 1 (not 0) to trigger preemption
			 * without sending IPI (similar to scx_lavd).
			 */
			victim->scx.slice = 1;
			return true;
		}
	}

	return false;
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
 * Evaluate the task's time slice using load-adjusted maximum.
 *
 * Uses the controller's per-class maximum as a ceiling, but scales down
 * based on system load (number of tasks waiting). Similar to bpfland's approach.
 */
static u64 task_slice(struct task_struct *p, struct task_ctx *tctx, s32 cpu)
{
	// Get the per-class maximum from the adaptive controller
	u64 controller_max = task_class_slice(tctx);

	// Count waiting tasks on this CPU's DSQ and the global DSQ
	u64 nr_wait = scx_bpf_dsq_nr_queued(cpu) +
		      scx_bpf_dsq_nr_queued(SCX_DSQ_GLOBAL);

	// Load-adjusted slice: (weight * max) / nr_wait
	// More tasks waiting = smaller slices for fairness
	u64 slice = scale_by_task_weight(p, controller_max) / MAX(nr_wait, 1);

	// Enforce minimum floor
	if (slice < slice_min_ns)
		slice = slice_min_ns;

	return slice;
}

/*
 * Legacy task_slice for compatibility (uses default normal slice).
 */
static u64 task_slice_default(struct task_struct *p)
{
	return scale_by_task_weight(p, slice_ns);
}

/*
 * Evaluate the deadline of task @p.
 *
 * Scale the runtime according to the task's priority. Additionally, limit
 * the maximum vruntime credit accumulated while the task is sleeping based
 * on its priority, @slice_lag, and wakeup rate.
 *
 * Then, include the vruntime accumulated while the task was awake. This
 * compensates the fact that the wakeup frequency is only updated in
 * ops.runnable(): if a task never sleeps, it would retain its initial
 * wakeup frequency; by incorporating the awake vruntime into the deadline,
 * we penalize continuously running tasks even when their wakeup frequency
 * remains unchanged.
 */
static u64 task_dl(struct task_struct *p, struct task_ctx *tctx)
{
	u64 lag_scale = MAX(tctx->wakeup_freq, 1);
	u64 vtime_min =
		vtime_now - scale_by_task_weight(p, slice_lag * lag_scale);
	u64 awake_max = scale_by_task_weight_inverse(p, slice_lag);

	if (time_before(p->scx.dsq_vtime, vtime_min))
		p->scx.dsq_vtime = vtime_min;

	if (time_after(tctx->awake_vtime, awake_max))
		tctx->awake_vtime = awake_max;

	return p->scx.dsq_vtime + tctx->awake_vtime;
}

/*
 * Return true if @this_cpu and @that_cpu are in the same LLC, false
 * otherwise.
 */
static inline bool cpus_share_cache(s32 this_cpu, s32 that_cpu)
{
	if (this_cpu == that_cpu)
		return true;

	return CPU_LLC_ID(this_cpu) == CPU_LLC_ID(that_cpu);
}

/*
 * Return true if @this_cpu is faster than @that_cpu, false otherwise.
 */
static inline bool is_cpu_faster(s32 this_cpu, s32 that_cpu)
{
	if (this_cpu == that_cpu)
		return false;

	return CPU_CAPACITY(this_cpu) > CPU_CAPACITY(that_cpu);
}

/*
 * Return the SMT sibling CPU of a @cpu, or @cpu if SMT is disabled.
 */
static s32 smt_sibling(s32 cpu)
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
 * Return true if the CPU is part of a fully busy SMT core, false
 * otherwise.
 *
 * If SMT is disabled or SMT contention avoidance is disabled, always
 * return false (since there's no SMT contention or it's ignored).
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
	idle_mask    = scx_bpf_get_idle_cpumask();
	is_contended = !bpf_cpumask_test_cpu(smt_sibling(cpu), idle_mask) &&
		       !bpf_cpumask_empty(idle_mask);
	scx_bpf_put_cpumask(idle_mask);

	return is_contended;
}

/*
 * Return true if the SMT sibling of @cpu is running a latency-critical task.
 */
static bool is_smt_sibling_critical(s32 cpu)
{
	s32		sibling;
	struct cpu_ctx *sibling_cctx;

	if (!smt_enabled)
		return false;

	sibling = smt_sibling(cpu);
	if (sibling == cpu)
		return false;

	sibling_cctx = try_lookup_cpu_ctx(sibling);
	if (!sibling_cctx)
		return false;

	return sibling_cctx->current_class == AUTOLAND_CLASS_LATENCY_CRITICAL;
}

/*
 * Return true if we should attempt a task migration to an idle CPU from
 * ops.enqueue(), false otherwise.
 *
 * We want to attempt a migration on wakeup or if the CPU used by the task
 * is contended, but only if ops.select_cpu() was skipped.
 */
static bool try_migrate(const struct task_struct *p, s32 prev_cpu,
			u64 enq_flags)
{
	/*
	 * Migrate if ops.select_cpu() was skipped and one of the following
	 * conditions is true:
	 *  - migration was not attempted already via ops.select_cpu(),
	 *  - the CPU is contended by other tasks,
	 *  - SMT is enabled and the SMT core is contended by other tasks.
	 */
	return (!scx_bpf_task_running(p) &&
		!__COMPAT_is_enq_cpu_selected(enq_flags)) ||
	       __COMPAT_scx_bpf_dsq_peek(prev_cpu) ||
	       is_smt_contended(prev_cpu);
}

/*
 * Return true if the task can keep running on its current CPU from
 * ops.dispatch(), false if the task should migrate.
 */
static bool keep_running(const struct task_struct *p, s32 cpu)
{
	const struct cpumask *primary = cast_mask(primary_cpumask);

	/* Do not keep running if the task doesn't need to run */
	if (!is_task_queued(p))
		return false;

	/*
	 * Do not keep running if the CPU is not in the primary domain and
	 * the task can use the primary domain).
	 */
	if (!primary_all && primary &&
	    bpf_cpumask_intersects(primary, p->cpus_ptr) &&
	    !bpf_cpumask_test_cpu(cpu, primary))
		return false;

	return true;
}

/*
 * Initialize a new cpumask, return 0 in case of success or a negative
 * value otherwise.
 */
static int init_cpumask(struct bpf_cpumask **p_cpumask)
{
	struct bpf_cpumask *mask;

	mask = *p_cpumask;
	if (mask)
		return 0;

	mask = bpf_cpumask_create();
	if (!mask)
		return -ENOMEM;

	mask = bpf_kptr_xchg(p_cpumask, mask);
	if (mask)
		bpf_cpumask_release(mask);

	return *p_cpumask ? 0 : -ENOMEM;
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

	pmask = &cctx->smt;

	/* Make sure the target CPU mask is initialized */
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

/*
 * Called from user-space to add CPUs to the the primary domain.
 */
SEC("syscall")
int enable_primary_cpu(struct cpu_arg *input)
{
	struct bpf_cpumask *mask;
	int		    err = 0;

	err			= init_cpumask(&primary_cpumask);
	if (err)
		return err;

	bpf_rcu_read_lock();
	mask = primary_cpumask;
	if (mask)
		bpf_cpumask_set_cpu(input->cpu_id, mask);
	bpf_rcu_read_unlock();

	return err;
}

/*
 * Return the tartget @cpu if it's usable by @p, or the first CPU usable.
 */
static s32 task_cpu(const struct task_struct *p, s32 cpu)
{
	if (bpf_cpumask_test_cpu(cpu, p->cpus_ptr))
		return cpu;

	return bpf_cpumask_first(p->cpus_ptr);
}

/*
 * Check if a non-latency-critical task should avoid running on this CPU
 * because its SMT sibling is running a latency-critical task.
 */
static bool should_avoid_cpu_for_smt(struct task_struct *p, s32 cpu)
{
	struct task_ctx *tctx;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return false;

	/* Only non-latency-critical tasks should avoid these CPUs */
	if (tctx->task_class == AUTOLAND_CLASS_LATENCY_CRITICAL)
		return false;

	return is_smt_sibling_critical(cpu);
}

/*
 * Try to pick the best idle CPU based on the @preferred_cpus ranking.
 * Return a full-idle SMT core if @do_idle_smt is true, or any idle CPU if
 * @do_idle_smt is false.
 */
static s32 pick_idle_cpu_pref_smt(struct task_struct *p, s32 prev_cpu,
				  bool			is_prev_allowed,
				  const struct cpumask *primary,
				  const struct cpumask *smt)
{
	u64  max_cpus = MIN(nr_cpu_ids, MAX_CPUS);
	int  i;
	bool avoid_smt_critical;

	/* Determine if we need to avoid SMT siblings of critical tasks */
	avoid_smt_critical = should_avoid_cpu_for_smt(p, prev_cpu);

	if (is_prev_allowed &&
	    (!primary || bpf_cpumask_test_cpu(prev_cpu, primary)) &&
	    (!smt || bpf_cpumask_test_cpu(prev_cpu, smt)) &&
	    (!avoid_smt_critical || !is_smt_sibling_critical(prev_cpu)) &&
	    scx_bpf_test_and_clear_cpu_idle(prev_cpu))
		return prev_cpu;

	bpf_for(i, 0, max_cpus)
	{
		s32 cpu = preferred_cpus[i];

		if ((cpu == prev_cpu) ||
		    !bpf_cpumask_test_cpu(cpu, p->cpus_ptr))
			continue;

		/* Check SMT sibling status for this CPU */
		avoid_smt_critical = should_avoid_cpu_for_smt(p, cpu);

		if ((!primary || bpf_cpumask_test_cpu(cpu, primary)) &&
		    (!smt || bpf_cpumask_test_cpu(cpu, smt)) &&
		    (!avoid_smt_critical || !is_smt_sibling_critical(cpu)) &&
		    scx_bpf_test_and_clear_cpu_idle(cpu))
			return cpu;
	}

	return -EBUSY;
}

/*
 * Return the optimal idle CPU for task @p or -EBUSY if no idle CPU is
 * found.
 */
static s32 pick_idle_cpu_scan(struct task_struct *p, s32 prev_cpu)
{
	const struct cpumask *smt, *primary;
	bool is_prev_allowed = bpf_cpumask_test_cpu(prev_cpu, p->cpus_ptr);
	s32  cpu;

	primary = !primary_all ? cast_mask(primary_cpumask) : NULL;
	smt	= smt_enabled ? scx_bpf_get_idle_smtmask() : NULL;

	/*
	 * If the task can't migrate, there's no point looking for other
	 * CPUs.
	 */
	if (is_pcpu_task(p)) {
		if (scx_bpf_test_and_clear_cpu_idle(prev_cpu)) {
			cpu = prev_cpu;
			goto out;
		}
	}

	if (!primary_all) {
		if (smt_enabled) {
			/*
			 * Try to pick a full-idle core in the primary
			 * domain.
			 */
			cpu = pick_idle_cpu_pref_smt(
				p, prev_cpu, is_prev_allowed, primary, smt);
			if (cpu >= 0)
				goto out;
		}

		/*
		 * Try to pick any idle CPU in the primary domain.
		 */
		cpu = pick_idle_cpu_pref_smt(p, prev_cpu, is_prev_allowed,
					     primary, NULL);
		if (cpu >= 0)
			goto out;
	}

	if (smt_enabled) {
		/*
		 * Try to pick any full-idle core in the system.
		 */
		cpu = pick_idle_cpu_pref_smt(p, prev_cpu, is_prev_allowed, NULL,
					     smt);
		if (cpu >= 0)
			goto out;
	}

	/*
	 * Try to pick any idle CPU in the system.
	 */
	cpu = pick_idle_cpu_pref_smt(p, prev_cpu, is_prev_allowed, NULL, NULL);

out:
	if (smt)
		scx_bpf_put_cpumask(smt);

	return cpu;
}

/*
 * Pick an optimal idle CPU for task @p (as close as possible to
 * @prev_cpu).
 *
 * Return the CPU id or a negative value if an idle CPU can't be found.
 */
static s32 pick_idle_cpu(struct task_struct *p, s32 prev_cpu, s32 this_cpu,
			 u64 wake_flags)
{
	const struct cpumask *mask = cast_mask(primary_cpumask);
	s32		      cpu;

	/*
	 * Use lightweight idle CPU scanning when flat or preferred idle
	 * scan is enabled, unless the system is busy, in which case the
	 * cpumask-based scanning is more efficient.
	 */
	if (preferred_idle_scan)
		return pick_idle_cpu_scan(p, prev_cpu);

	/*
	 * Fallback to the old API if the kernel doesn't support
	 * scx_bpf_select_cpu_and().
	 *
	 * This is required to support kernels <= 6.16.
	 */
	if (!__COMPAT_HAS_scx_bpf_select_cpu_and) {
		bool is_idle = false;

		/*
		 * scx_bpf_select_cpu_dfl() can only be used in
		 * ops.select_cpu().
		 */
		if (this_cpu < 0)
			return -EBUSY;

		cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);

		return is_idle ? cpu : -EBUSY;
	}

	/*
	 * If a primary domain is defined, try to pick an idle CPU from
	 * there first.
	 */
	if (!primary_all && mask) {
		cpu = scx_bpf_select_cpu_and(p, prev_cpu, wake_flags, mask, 0);
		if (cpu >= 0)
			return cpu;
	}

	/*
	 * Pick any idle CPU usable by the task.
	 */
	return scx_bpf_select_cpu_and(p, prev_cpu, wake_flags, p->cpus_ptr, 0);
}

/*
 * Return true if @p can run on @cpu, false otherwise.
 */
static bool is_cpu_allowed(const struct task_struct *p, s32 cpu)
{
	return p->nr_cpus_allowed == nr_cpu_ids ||
	       bpf_cpumask_test_cpu(cpu, p->cpus_ptr);
}

/*
 * Dispatch task @p directly to @cpu, bypassing the scheduler queues.
 */
static s32 do_direct_dispatch(struct task_struct *p, s32 cpu)
{
	struct task_struct *q;
	struct task_ctx	   *tctx;
	u64		    dl;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return cpu;
	dl = task_dl(p, tctx);

	/*
	 * If there's no task waiting for the target CPU or if the first
	 * waiting task has a later deadline, dispatch to the local DSQ to
	 * save some locking overhead.
	 */
	q = __COMPAT_scx_bpf_dsq_peek(cpu);
	if (!q || q->scx.dsq_vtime >= dl)
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, task_slice(p, tctx, cpu),
				   0);
	else
		scx_bpf_dsq_insert_vtime(p, cpu, task_slice(p, tctx, cpu), dl,
					 0);

	return cpu;
}

/*
 * Called on task wakeup to give the task a chance to migrate to an idle
 * CPU.
 */
s32 BPF_STRUCT_OPS(autoland_select_cpu, struct task_struct *p, s32 prev_cpu,
		   u64 wake_flags)
{
	s32		 cpu, this_cpu = bpf_get_smp_processor_id();
	bool		 is_this_cpu_allowed = is_cpu_allowed(p, this_cpu);
	struct task_ctx *tctx;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return prev_cpu;

	/*
	 * Update latency criticality and try preemption for high-priority
	 * latency-critical tasks when no idle CPU is available.
	 */
	tctx->lat_cri = calc_lat_cri(p, tctx);

	/*
	 * On wakeup if the waker's CPU is faster than the wakee's CPU, try
	 * to move the wakee closer to the waker.
	 */
	if ((wake_flags & SCX_WAKE_TTWU) && is_cpu_faster(this_cpu, prev_cpu) &&
	    is_this_cpu_allowed) {
		/*
		 * If both the waker's CPU and the wakee's CPU are in the
		 * same LLC and the wakee's CPU is a fully idle SMT core,
		 * don't migrate.
		 */
		if (is_cpu_allowed(p, prev_cpu) &&
		    cpus_share_cache(this_cpu, prev_cpu) &&
		    (!is_smt_contended(prev_cpu)) &&
		    scx_bpf_test_and_clear_cpu_idle(prev_cpu))
			return do_direct_dispatch(p, prev_cpu);

		prev_cpu = this_cpu;
	}

	/*
	 * Try to find an optimal idle CPU for the task. If no idle CPU is
	 * found, keep using the same one.
	 */
	cpu = pick_idle_cpu(p, prev_cpu, this_cpu, wake_flags);
	if (cpu >= 0 || !is_system_busy())
		return do_direct_dispatch(p, cpu >= 0 ? cpu : prev_cpu);

	/*
	 * Try to preempt a lower priority task if this is latency-critical
	 * and we couldn't find an idle CPU.
	 */
	if (tctx->lat_cri >= preempt_threshold && try_preempt(p, tctx)) {
		/* Preemption succeeded, dispatch to previous CPU */
		return do_direct_dispatch(p, prev_cpu);
	}

	return prev_cpu;
}

/*
 * Called when a task expired its time slice and still needs to run or on
 * wakeup when there's no idle CPU available.
 */
void BPF_STRUCT_OPS(autoland_enqueue, struct task_struct *p, u64 enq_flags)
{
	s32		 prev_cpu = task_cpu(p, scx_bpf_task_cpu(p));
	struct task_ctx *tctx;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	if (is_task_sticky(tctx)) {
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL,
				   task_slice(p, tctx, prev_cpu), enq_flags);
	} else {
		struct task_struct *q;

		/*
		 * Attempt a migration to an idle CPU if possible.
		 */
		if (try_migrate(p, prev_cpu, enq_flags)) {
			s32 cpu;

			if (is_pcpu_task(p))
				cpu = scx_bpf_test_and_clear_cpu_idle(
					      prev_cpu) ?
					      prev_cpu :
					      -EBUSY;
			else
				cpu = pick_idle_cpu(p, prev_cpu, -ENOENT, 0);

			if (cpu >= 0) {
				struct task_struct *q =
					__COMPAT_scx_bpf_dsq_peek(cpu);

				if (!q || p->scx.dsq_vtime < q->scx.dsq_vtime) {
					scx_bpf_dsq_insert(
						p, SCX_DSQ_LOCAL_ON | cpu,
						task_slice(p, tctx, cpu),
						enq_flags);
					if (prev_cpu != cpu ||
					    !scx_bpf_task_running(p))
						scx_bpf_kick_cpu(cpu,
								 SCX_KICK_IDLE);
					return;
				}
				prev_cpu = cpu;
			}
		}

		/*
		 * Keep running on the same CPU.
		 */
		q = __COMPAT_scx_bpf_cpu_curr(prev_cpu);
		if (!q) {
			scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL_ON | prev_cpu,
					   task_slice(p, tctx, prev_cpu),
					   enq_flags);
		} else {
			scx_bpf_dsq_insert_vtime(p, prev_cpu,
						 task_slice(p, tctx, prev_cpu),
						 task_dl(p, tctx), enq_flags);
		}
	}
	if (!__COMPAT_is_enq_cpu_selected(enq_flags))
		scx_bpf_kick_cpu(prev_cpu, SCX_KICK_IDLE);
}

/*
 * Try to consume a task from a remote DSQ.
 */
static bool dispatch_from_any_cpu(s32 from_cpu)
{
	u64 min_vtime = ULLONG_MAX, cpu, min_cpu;

	/*
	 * Pick the task with the lowest vruntime within the same LLC.
	 *
	 * Restricting rebalancing to the LLC improves cache locality and
	 * also reduces lock contention on CPU runqueues.
	 */
	bpf_for(cpu, 0, nr_cpu_ids)
	{
		struct task_struct *p;

		p = __COMPAT_scx_bpf_dsq_peek(cpu);
		if (p && bpf_cpumask_test_cpu(from_cpu, p->cpus_ptr) &&
		    p->scx.dsq_vtime < min_vtime) {
			min_vtime = p->scx.dsq_vtime;
			min_cpu	  = cpu;
		}
	}

	return min_vtime < ULLONG_MAX && scx_bpf_dsq_move_to_local(min_cpu, 0);
}

/*
 * Called when a CPU becomes available: dispatch the next task on the CPU
 * or let the CPU go idle.
 */
void BPF_STRUCT_OPS(autoland_dispatch, s32 cpu, struct task_struct *prev)
{
	/*
	 * Immediately trigger a rebalance if the system is busy.
	 */
	if (is_system_busy() && dispatch_from_any_cpu(cpu)) {
		__sync_fetch_and_add(&nr_remote_dispatch, 1);
		return;
	}

	/*
	 * Consume from the local DSQ.
	 */
	if (scx_bpf_dsq_move_to_local(cpu, 0)) {
		__sync_fetch_and_add(&nr_local_dispatch, 1);
		return;
	}

	/*
	 * Try to consume a task from a remote CPU.
	 */
	if (dispatch_from_any_cpu(cpu)) {
		__sync_fetch_and_add(&nr_remote_dispatch, 1);
		return;
	}

	/*
	 * If no other task is contending the CPU and the previous task
	 * still wants to run, let it run by refilling its time slice.
	 */
	if (prev && keep_running(prev, cpu)) {
		struct task_ctx *prev_tctx = try_lookup_task_ctx(prev);
		if (prev_tctx)
			prev->scx.slice = task_slice(prev, prev_tctx, cpu);
		else
			prev->scx.slice = task_slice_default(prev);
		__sync_fetch_and_add(&nr_keep_running, 1);
	}
}

void BPF_STRUCT_OPS(autoland_runnable, struct task_struct *p, u64 enq_flags)
{
	u64			  now = bpf_ktime_get_ns(), delta_t;
	struct task_ctx		 *tctx;
	struct task_runtime_stat *trstat;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	tctx->awake_vtime = 0;

	/*
	 * Update the task's wakeup frequency based on the time since the
	 * last wakeup, then cap the result to avoid large spikes.
	 */
	delta_t		   = now - tctx->last_woke_at;
	tctx->wakeup_freq  = update_freq(tctx->wakeup_freq, delta_t);
	tctx->wakeup_freq  = MIN(tctx->wakeup_freq, MAX_WAKEUP_FREQ);
	tctx->last_woke_at = now;

	/* Update per-task runtime stats for criticality calculation */
	trstat = bpf_task_storage_get(&task_runtime_stats, p, 0,
				      BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (trstat) {
		/* Increment wakeup counter for this task */
		trstat->wakeup_count++;
		trstat->last_wakeup_time = now;
	}
}

/*
 * Get histogram bucket index for a latency value.
 * Uses logarithmic scale: bucket 0 = 0-1us, bucket 1 = 1-2us, ..., bucket 15 = 32768us+
 */
static u32 get_latency_bucket(u64 latency_ns)
{
	/* Convert to microseconds for bucket calculation */
	u64 latency_us = latency_ns / 1000;

	if (latency_us == 0)
		return 0;

	/* Find position of highest set bit (log2) */
	u32 bucket = 0;
	while (latency_us > 0 && bucket < LATENCY_HISTOGRAM_BUCKETS - 1) {
		latency_us >>= 1;
		bucket++;
	}

	return bucket;
}

/*
 * Update latency statistics for a task class.
 * Now includes histogram tracking for P99 calculation and variance metrics.
 */
static void update_latency_stat(u32 task_class, u64 latency_ns)
{
	struct class_latency_stat *stat;
	u32			   key = task_class;

	stat = bpf_map_lookup_elem(&class_latency_stats, &key);
	if (stat) {
		__sync_fetch_and_add(&stat->sum_ns, latency_ns);
		__sync_fetch_and_add(&stat->count, 1);

		/* Update sum of squares for variance calculation */
		/* Note: Using 64-bit multiplication, cap to prevent overflow */
		u64 squared = latency_ns;
		if (squared > 0xFFFFFFFFULL)
			squared = 0xFFFFFFFFULL; /* Cap at ~4.3s squared */
		__sync_fetch_and_add(&stat->sum_squares_ns, squared * squared);

		/* Update min/max */
		if (stat->min_ns == 0 || latency_ns < stat->min_ns)
			stat->min_ns = latency_ns;
		if (latency_ns > stat->max_ns)
			stat->max_ns = latency_ns;

		/* Update histogram bucket */
		u32 bucket = get_latency_bucket(latency_ns);
		if (bucket < LATENCY_HISTOGRAM_BUCKETS)
			__sync_fetch_and_add(&stat->histogram[bucket], 1);
	}
}

/*
 * Update per-class criticality metrics for LAVD-style scoring.
 */
static void update_criticality_metrics(u32 task_class, u64 runtime_ns,
				       u32 wakeup_delta)
{
	struct class_criticality_metrics *metrics;
	u32				  key = task_class;

	metrics = bpf_map_lookup_elem(&class_criticality_stats, &key);
	if (metrics) {
		__sync_fetch_and_add(&metrics->total_runtime_ns, runtime_ns);
		__sync_fetch_and_add(&metrics->total_wakeup_count,
				     wakeup_delta);

		/* Update runtime variance statistics */
		u64 squared = runtime_ns;
		if (squared > 0xFFFFFFFFULL)
			squared = 0xFFFFFFFFULL;
		__sync_fetch_and_add(&metrics->runtime_squared_sum,
				     squared * squared);

		__sync_fetch_and_add(&metrics->sample_count, 1);
	}
}

void BPF_STRUCT_OPS(autoland_running, struct task_struct *p)
{
	struct task_ctx *tctx;
	struct cpu_ctx	*cctx;
	s32		 cpu = bpf_get_smp_processor_id();
	u64		 now = bpf_ktime_get_ns();
	u64		 latency_ns;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	/*
	 * Calculate wake latency (time from runnable to running).
	 */
	if (tctx->last_woke_at > 0 && now > tctx->last_woke_at) {
		latency_ns = now - tctx->last_woke_at;
		update_latency_stat(tctx->task_class, latency_ns);
	}

	/*
	 * Save a timestamp when the task begins to run (used to evaluate
	 * the used time slice).
	 */
	tctx->last_run_at = now;

	/*
	 * Update current system's vruntime.
	 */
	if (time_before(vtime_now, p->scx.dsq_vtime))
		vtime_now = p->scx.dsq_vtime;

	/*
	 * Update latency criticality score.
	 */
	tctx->lat_cri = calc_lat_cri(p, tctx);

	/*
	 * Track current task class on this CPU.
	 */
	cctx = try_lookup_cpu_ctx(cpu);
	if (cctx) {
		cctx->current_class	 = tctx->task_class;
		cctx->current_task_start = tctx->last_run_at;
	}

	/*
	 * Periodic task reclassification.
	 */
	classify_task_heuristics(p, tctx);
}

void BPF_STRUCT_OPS(autoland_stopping, struct task_struct *p, bool runnable)
{
	struct task_ctx		 *tctx;
	struct task_runtime_stat *trstat;
	u64			  slice, vslice, now;
	u32			  wakeup_delta;

	tctx = try_lookup_task_ctx(p);
	if (!tctx)
		return;

	/*
	 * Evaluate the used time slice.
	 */
	now   = bpf_ktime_get_ns();
	slice = now - tctx->last_run_at;

	/*
	 * Update average runtime per scheduling cycle for sticky task detection.
	 */
	tctx->avg_runtime = calc_avg(tctx->avg_runtime, slice);

	/*
	 * Update the vruntime and the total accumulated runtime since last
	 * sleep.
	 */
	vslice = scale_by_task_weight_inverse(p, slice);
	p->scx.dsq_vtime += vslice;
	tctx->awake_vtime += vslice;

	/*
	 * Update per-task runtime statistics for variance calculation.
	 */
	trstat = bpf_task_storage_get(&task_runtime_stats, p, 0, 0);
	if (trstat) {
		trstat->total_runtime_ns += slice;

		/* Add squared runtime for variance calculation */
		u64 squared = slice;
		if (squared > 0xFFFFFFFFULL)
			squared = 0xFFFFFFFFULL;
		trstat->runtime_squared_sum += squared * squared;
		trstat->sample_count++;

		/* Get wakeup count delta and reset */
		wakeup_delta	     = trstat->wakeup_count;
		trstat->wakeup_count = 0;

		/* Update per-class criticality metrics */
		update_criticality_metrics(tctx->task_class, slice,
					   wakeup_delta);
	}
}

void BPF_STRUCT_OPS(autoland_enable, struct task_struct *p)
{
	p->scx.dsq_vtime = vtime_now;
}

s32 BPF_STRUCT_OPS(autoland_init_task, struct task_struct *p,
		   struct scx_init_task_args *args)
{
	struct task_ctx *tctx;

	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0,
				    BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx)
		return -ENOMEM;

	/* Initialize task classification */
	tctx->task_class       = AUTOLAND_CLASS_NORMAL;
	tctx->lat_cri	       = 0;
	tctx->last_class_check = 0;
	tctx->is_scx_turbo     = false;
	tctx->is_steam_game    = false;
	tctx->is_de_task       = false;
	tctx->pelt_util_sum    = 0;
	tctx->pelt_util_count  = 0;

	/* Check for userspace hints */
	u32  pid  = p->pid;
	u32 *hint = bpf_map_lookup_elem(&task_hints, &pid);
	if (hint) {
		tctx->task_class = *hint;
		/* Mark SCX_TURBO tasks for priority boost */
		if (*hint == AUTOLAND_CLASS_LATENCY_CRITICAL)
			tctx->is_scx_turbo = true;
	}

	/* Increment task class counter */
	if (tctx->task_class == AUTOLAND_CLASS_LATENCY_CRITICAL)
		__sync_fetch_and_add(&nr_lc_tasks, 1);
	else if (tctx->task_class == AUTOLAND_CLASS_HOG)
		__sync_fetch_and_add(&nr_hog_tasks, 1);
	else
		__sync_fetch_and_add(&nr_normal_tasks, 1);

	return 0;
}

/*
 * Scheduler exit callback.
 */
void BPF_STRUCT_OPS(autoland_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

/*
 * Scheduler init callback.
 */
s32 BPF_STRUCT_OPS_SLEEPABLE(autoland_init)
{
	s32				 cpu;
	int				 err;
	u32				 key;
	struct class_latency_stat	 empty_stat    = {};
	struct class_criticality_metrics empty_metrics = {};

	err					       = scx_lib_init();
	if (err)
		return err;

	nr_cpu_ids = scx_bpf_nr_cpu_ids();

	bpf_for(cpu, 0, nr_cpu_ids)
	{
		int err;

		err = scx_bpf_create_dsq(cpu, __COMPAT_scx_bpf_cpu_node(cpu));
		if (err)
			return err;
	}

	/* Initialize per-class latency statistics maps */
	bpf_for(key, 0, 3)
	{
		err = bpf_map_update_elem(&class_latency_stats, &key,
					  &empty_stat, BPF_ANY);
		if (err)
			return err;

		err = bpf_map_update_elem(&class_criticality_stats, &key,
					  &empty_metrics, BPF_ANY);
		if (err)
			return err;
	}

	return 0;
}

SCX_OPS_DEFINE(autoland_ops, .select_cpu = (void *)autoland_select_cpu,
	       .enqueue	  = (void *)autoland_enqueue,
	       .dispatch  = (void *)autoland_dispatch,
	       .runnable  = (void *)autoland_runnable,
	       .running	  = (void *)autoland_running,
	       .stopping  = (void *)autoland_stopping,
	       .enable	  = (void *)autoland_enable,
	       .init_task = (void *)autoland_init_task,
	       .init = (void *)autoland_init, .exit = (void *)autoland_exit,
	       .timeout_ms = 5000, .name = "autoland");
