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
    #[stat(desc = "local dispatches")]
    pub nr_local_dispatch: u64,
    #[stat(desc = "remote dispatches")]
    pub nr_remote_dispatch: u64,
    #[stat(desc = "keep running events")]
    pub nr_keep_running: u64,
    #[stat(desc = "latency-critical task count")]
    pub nr_lc_tasks: u64,
    #[stat(desc = "normal task count")]
    pub nr_normal_tasks: u64,
    #[stat(desc = "hog task count")]
    pub nr_hog_tasks: u64,
    #[stat(desc = "preemption events")]
    pub nr_preemptions: u64,
    #[stat(desc = "current LC slice (us)")]
    pub lc_slice_us: u64,
    #[stat(desc = "current normal slice (us)")]
    pub normal_slice_us: u64,
    #[stat(desc = "current hog slice (us)")]
    pub hog_slice_us: u64,
}

impl Metrics {
    fn format<W: Write>(&self, w: &mut W) -> Result<()> {
        writeln!(
            w,
            "[{}] dispatch: local={} remote={} running={} preempt={}",
            crate::SCHEDULER_NAME,
            self.nr_local_dispatch,
            self.nr_remote_dispatch,
            self.nr_keep_running,
            self.nr_preemptions,
        )?;
        writeln!(
            w,
            "[{}] tasks: LC={} normal={} hog={}",
            crate::SCHEDULER_NAME,
            self.nr_lc_tasks,
            self.nr_normal_tasks,
            self.nr_hog_tasks,
        )?;
        writeln!(
            w,
            "[{}] slices: LC={}us normal={}us hog={}us",
            crate::SCHEDULER_NAME,
            self.lc_slice_us,
            self.normal_slice_us,
            self.hog_slice_us,
        )?;
        Ok(())
    }

    fn delta(&self, rhs: &Self) -> Self {
        Self {
            nr_local_dispatch: self.nr_local_dispatch - rhs.nr_local_dispatch,
            nr_remote_dispatch: self.nr_remote_dispatch - rhs.nr_remote_dispatch,
            nr_keep_running: self.nr_keep_running - rhs.nr_keep_running,
            nr_preemptions: self.nr_preemptions - rhs.nr_preemptions,
            nr_lc_tasks: self.nr_lc_tasks,
            nr_normal_tasks: self.nr_normal_tasks,
            nr_hog_tasks: self.nr_hog_tasks,
            lc_slice_us: self.lc_slice_us,
            normal_slice_us: self.normal_slice_us,
            hog_slice_us: self.hog_slice_us,
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
