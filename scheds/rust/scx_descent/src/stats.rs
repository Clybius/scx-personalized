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

    // Descent-specific metrics
    #[stat(desc = "Tasks by class (interactive,audio,batch,kernel)")]
    pub nr_tasks_interactive: u64,
    #[stat(desc = "Tasks in audio class")]
    pub nr_tasks_audio: u64,
    #[stat(desc = "Tasks in batch class")]
    pub nr_tasks_batch: u64,
    #[stat(desc = "Tasks in kernel class")]
    pub nr_tasks_kernel: u64,
    #[stat(desc = "Gradient updates processed")]
    pub gradient_updates: u64,
    #[stat(desc = "Parameter oscillations detected")]
    pub oscillations: u64,
}

impl Metrics {
    fn format<W: Write>(&self, w: &mut W) -> Result<()> {
        writeln!(
            w,
            "[{}] tasks -> r: {:>2}/{:<2} | dispatch -> k: {:<5} d: {:<5} s: {:<5} | updates: {} osc: {}",
            crate::SCHEDULER_NAME,
            self.nr_running,
            self.nr_cpus,
            self.nr_kthread_dispatches,
            self.nr_direct_dispatches,
            self.nr_shared_dispatches,
            self.gradient_updates,
            self.oscillations
        )?;
        Ok(())
    }

    fn delta(&self, rhs: &Self) -> Self {
        Self {
            nr_kthread_dispatches: self.nr_kthread_dispatches - rhs.nr_kthread_dispatches,
            nr_direct_dispatches: self.nr_direct_dispatches - rhs.nr_direct_dispatches,
            nr_shared_dispatches: self.nr_shared_dispatches - rhs.nr_shared_dispatches,
            gradient_updates: self.gradient_updates - rhs.gradient_updates,
            oscillations: self.oscillations - rhs.oscillations,
            nr_tasks_interactive: self.nr_tasks_interactive,
            nr_tasks_audio: self.nr_tasks_audio,
            nr_tasks_batch: self.nr_tasks_batch,
            nr_tasks_kernel: self.nr_tasks_kernel,
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
