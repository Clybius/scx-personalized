use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use scx_stats::prelude::*;
use scx_stats_derive::stat_doc;
use scx_stats_derive::Stats;
use serde::Deserialize;
use serde::Serialize;

#[stat_doc]
#[derive(Clone, Debug, Default, Serialize, Deserialize, Stats)]
#[stat(top)]
pub struct Metrics {
    #[stat(desc = "Number of running tasks")]
    pub nr_running: u64,
    #[stat(desc = "Number of online CPUs")]
    pub nr_cpus: u64,
    #[stat(desc = "Number of kthread direct dispatches")]
    pub nr_kthread_dispatches: u64,
    #[stat(desc = "Number of task direct dispatches")]
    pub nr_direct_dispatches: u64,
    #[stat(desc = "Number of regular task dispatches")]
    pub nr_shared_dispatches: u64,

    // Thompson Sampling specific metrics
    #[stat(desc = "Tasks in latency-critical class (games, audio, compositors)")]
    pub nr_tasks_latency_critical: u64,
    #[stat(desc = "Tasks in normal class (default interactive)")]
    pub nr_tasks_normal: u64,
    #[stat(desc = "Tasks in hog class (high CPU usage)")]
    pub nr_tasks_hog: u64,
    #[stat(desc = "Tasks in background class (low priority)")]
    pub nr_tasks_background: u64,
    #[stat(desc = "Thompson Sampling total observations processed")]
    pub thompson_updates: u64,
    #[stat(desc = "Thompson Sampling average uncertainty (x1000)")]
    pub thompson_uncertainty: u64,
}

impl Metrics {
    fn format<W: Write>(&self, w: &mut W) -> Result<()> {
        writeln!(
            w,
            "[{}] tasks -> r: {:>2}/{:<2} | dispatch -> k: {:<5} d: {:<5} s: {:<5} | thompson: updates={} unc={:.3} | classes: lc:{} n:{} h:{} bg:{}",
            crate::SCHEDULER_NAME,
            self.nr_running,
            self.nr_cpus,
            self.nr_kthread_dispatches,
            self.nr_direct_dispatches,
            self.nr_shared_dispatches,
            self.thompson_updates,
            self.thompson_uncertainty as f64 / 1000.0,
            self.nr_tasks_latency_critical,
            self.nr_tasks_normal,
            self.nr_tasks_hog,
            self.nr_tasks_background
        )?;
        Ok(())
    }

    fn delta(&self, rhs: &Self) -> Self {
        Self {
            nr_kthread_dispatches: self.nr_kthread_dispatches - rhs.nr_kthread_dispatches,
            nr_direct_dispatches: self.nr_direct_dispatches - rhs.nr_direct_dispatches,
            nr_shared_dispatches: self.nr_shared_dispatches - rhs.nr_shared_dispatches,
            thompson_updates: self.thompson_updates, // Cumulative
            thompson_uncertainty: self.thompson_uncertainty,
            nr_tasks_latency_critical: self.nr_tasks_latency_critical,
            nr_tasks_normal: self.nr_tasks_normal,
            nr_tasks_hog: self.nr_tasks_hog,
            nr_tasks_background: self.nr_tasks_background,
            nr_running: self.nr_running,
            nr_cpus: self.nr_cpus,
        }
    }
}

pub fn server_data() -> StatsServerData<(), Metrics> {
    let open: Box<dyn StatsOpener<(), Metrics>> = Box::new(move |(req_ch, res_ch)| {
        req_ch.send(())?;
        let mut prev = res_ch.recv()?;

        let read: Box<dyn StatsReader<(), Metrics>> = Box::new(move |_args, (req_ch, res_ch)| {
            req_ch.send(())?;
            let cur = res_ch.recv()?;
            let delta = cur.delta(&prev);
            prev = cur;
            delta.to_json()
        });

        Ok(read)
    });

    StatsServerData::new()
        .add_meta(Metrics::meta())
        .add_ops("top", StatsOps { open, close: None })
}

pub fn monitor(intv: Duration, shutdown: Arc<AtomicBool>) -> Result<()> {
    scx_utils::monitor_stats::<Metrics>(
        &[],
        intv,
        || shutdown.load(Ordering::Relaxed),
        |metrics| metrics.format(&mut std::io::stdout()),
    )
}

/// Thompson Sampling specific statistics for detailed monitoring
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThompsonStats {
    pub cpu: u32,
    pub class: u32,
    pub param_means: [f64; 5],    // Posterior means
    pub param_stds: [f64; 5],     // Posterior stds (uncertainty)
    pub n_observations: [f64; 5], // Number of observations per param
    pub current_params: [u64; 5], // Currently sampled params
    pub last_loss: f64,
    pub update_count: u64,
}

impl ThompsonStats {
    /// Create stats from a ThompsonSampler for a specific CPU/class
    #[allow(dead_code)] // Available for detailed Thompson stats export
    pub fn from_sampler<T>(
        cpu: u32,
        class: u32,
        thompson: &crate::optimizer_thompson::ThompsonSampler,
        current_params: [u64; 5],
        last_loss: f64,
    ) -> Option<Self> {
        let mut param_means = [0.0; 5];
        let mut param_stds = [0.0; 5];
        let mut n_observations = [0.0; 5];

        for param_idx in 0..5 {
            if let Some(posterior) = thompson.get_posterior(cpu, class, param_idx) {
                param_means[param_idx] = posterior.mean;
                param_stds[param_idx] = posterior.std;
                n_observations[param_idx] = posterior.n_observations;
            }
        }

        Some(Self {
            cpu,
            class,
            param_means,
            param_stds,
            n_observations,
            current_params,
            last_loss,
            update_count: 0, // Could be tracked separately
        })
    }
}
