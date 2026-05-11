// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_astro contributors

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
    #[stat(desc = "Tasks dispatched from the interactive DSQ")]
    pub interactive_dispatches: u64,
    #[stat(desc = "Tasks dispatched from the waker-boost DSQ")]
    pub waker_boost_dispatches: u64,
    #[stat(desc = "Tasks dispatched from the normal DSQ")]
    pub normal_dispatches: u64,
    #[stat(desc = "Tasks dispatched from the compute DSQ")]
    pub compute_dispatches: u64,
    #[stat(desc = "Tasks dispatched from the background DSQ")]
    pub background_dispatches: u64,
    #[stat(desc = "Preemption events triggered")]
    pub preempts: u64,
    #[stat(desc = "Idle CPU kicks")]
    pub kicks_idle: u64,
    #[stat(desc = "Preempt kicks sent to remote CPUs")]
    pub kicks_preempt: u64,
    #[stat(desc = "Profile lane transitions")]
    pub profile_transitions: u64,
    #[stat(desc = "Waker boost grants")]
    pub waker_boosts: u64,
    #[stat(desc = "Short tasks prioritized via SRPT scaling")]
    pub srpt_short_tasks: u64,
    #[stat(desc = "Budget refill events")]
    pub budget_refills: u64,
    #[stat(desc = "Budget exhaustion events")]
    pub budget_exhaustions: u64,
    #[stat(desc = "Contained-lane starvation rounds (max across CPUs)")]
    pub contained_starvation_rounds: u64,
    #[stat(desc = "Shared-lane starvation rounds (max across CPUs)")]
    pub shared_starvation_rounds: u64,
    #[stat(desc = "High-priority burst rounds (max across CPUs)")]
    pub high_priority_burst_rounds: u64,
    #[stat(desc = "Current autotune mode (0=balanced, 1=latency, 2=throughput)")]
    pub autotune_mode: u64,
    #[stat(desc = "Autotune generation counter")]
    pub autotune_generation: u64,
    #[stat(desc = "Current interactive slice cap in us")]
    pub tune_interactive_slice_us: u64,
    #[stat(desc = "Current normal slice base in us")]
    pub tune_normal_slice_us: u64,
    #[stat(desc = "Current compute slice in us")]
    pub tune_compute_slice_us: u64,
    #[stat(desc = "Current background slice in us")]
    pub tune_background_slice_us: u64,
}

impl Metrics {
    fn autotune_mode_name(&self) -> &'static str {
        match self.autotune_mode {
            1 => "latency",
            2 => "throughput",
            _ => "balanced",
        }
    }

    fn format<W: Write>(&self, w: &mut W) -> Result<()> {
        writeln!(
            w,
            "[scx_astro] mode={} gen={} run={} int_disp={} boost_disp={} norm_disp={} comp_disp={} bg_disp={} preempt={} kick_idle={} kick_preempt={} trans={} boosts={} srpt={} refill={} exhaust={} bg_starve={} shared_starve={} hp_burst={} int_slice={}us norm_slice={}us comp_slice={}us bg_slice={}us",
            self.autotune_mode_name(),
            self.autotune_generation,
            self.nr_running,
            self.interactive_dispatches,
            self.waker_boost_dispatches,
            self.normal_dispatches,
            self.compute_dispatches,
            self.background_dispatches,
            self.preempts,
            self.kicks_idle,
            self.kicks_preempt,
            self.profile_transitions,
            self.waker_boosts,
            self.srpt_short_tasks,
            self.budget_refills,
            self.budget_exhaustions,
            self.contained_starvation_rounds,
            self.shared_starvation_rounds,
            self.high_priority_burst_rounds,
            self.tune_interactive_slice_us,
            self.tune_normal_slice_us,
            self.tune_compute_slice_us,
            self.tune_background_slice_us,
        )?;
        Ok(())
    }

    pub fn delta(&self, rhs: &Self) -> Self {
        Self {
            nr_running: self.nr_running,
            interactive_dispatches: self
                .interactive_dispatches
                .wrapping_sub(rhs.interactive_dispatches),
            waker_boost_dispatches: self
                .waker_boost_dispatches
                .wrapping_sub(rhs.waker_boost_dispatches),
            normal_dispatches: self.normal_dispatches.wrapping_sub(rhs.normal_dispatches),
            compute_dispatches: self.compute_dispatches.wrapping_sub(rhs.compute_dispatches),
            background_dispatches: self
                .background_dispatches
                .wrapping_sub(rhs.background_dispatches),
            preempts: self.preempts.wrapping_sub(rhs.preempts),
            kicks_idle: self.kicks_idle.wrapping_sub(rhs.kicks_idle),
            kicks_preempt: self.kicks_preempt.wrapping_sub(rhs.kicks_preempt),
            profile_transitions: self
                .profile_transitions
                .wrapping_sub(rhs.profile_transitions),
            waker_boosts: self.waker_boosts.wrapping_sub(rhs.waker_boosts),
            srpt_short_tasks: self.srpt_short_tasks.wrapping_sub(rhs.srpt_short_tasks),
            budget_refills: self.budget_refills.wrapping_sub(rhs.budget_refills),
            budget_exhaustions: self.budget_exhaustions.wrapping_sub(rhs.budget_exhaustions),
            contained_starvation_rounds: self.contained_starvation_rounds,
            shared_starvation_rounds: self.shared_starvation_rounds,
            high_priority_burst_rounds: self.high_priority_burst_rounds,
            autotune_mode: self.autotune_mode,
            autotune_generation: self.autotune_generation,
            tune_interactive_slice_us: self.tune_interactive_slice_us,
            tune_normal_slice_us: self.tune_normal_slice_us,
            tune_compute_slice_us: self.tune_compute_slice_us,
            tune_background_slice_us: self.tune_background_slice_us,
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
