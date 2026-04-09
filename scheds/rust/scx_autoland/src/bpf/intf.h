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

/*
 * Task classification levels.
 */
enum autoland_task_class {
	AUTOLAND_CLASS_NORMAL		= 0,
	AUTOLAND_CLASS_HOG		= 1,
	AUTOLAND_CLASS_LATENCY_CRITICAL = 2,
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

/*
 * Number of histogram buckets for P99 latency calculation.
 * Buckets cover latencies from 1us to ~8.3ms using logarithmic scale.
 */
#define LATENCY_HISTOGRAM_BUCKETS 16

/*
 * Per-class latency statistics (extended for P99 calculation).
 */
struct class_latency_stat {
	u64 sum_ns;
	u64 count;
	u64 sum_squares_ns; /* For variance calculation: sum of (latency^2) */
	u64 min_ns; /* Minimum observed latency */
	u64 max_ns; /* Maximum observed latency */
	/* Histogram buckets for P99 calculation (logarithmic scale) */
	u64 histogram[LATENCY_HISTOGRAM_BUCKETS];
};

/*
 * Per-class criticality metrics (for LAVD-style scoring).
 */
struct class_criticality_metrics {
	u64 total_wakeup_count; /* Total wakeups in measurement window */
	u64 total_runtime_ns; /* Total runtime in measurement window */
	u64 runtime_squared_sum; /* Sum of squared runtimes for variance */
	u32 sample_count; /* Number of samples for averaging */
};

struct cpu_arg {
	s32 cpu_id;
};

struct domain_arg {
	s32 cpu_id;
	s32 sibling_cpu_id;
};

#endif /* __INTF_H */
