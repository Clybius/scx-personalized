// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2024 scx_happy authors

use scx_stats::{Meta, Stat};
use serde::{Deserialize, Serialize};

#[stat_doc]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct HappyMetrics {
    #[stat(desc = "Number of LC queue dispatches")]
    pub nr_lc_dispatches: u64,

    #[stat(desc = "Number of NORMAL queue dispatches")]
    pub nr_normal_dispatches: u64,

    #[stat(desc = "Number of HOG queue dispatches")]
    pub nr_hog_dispatches: u64,

    #[stat(desc = "Number of preemptions")]
    pub nr_preemptions: u64,

    #[stat(desc = "Number of migrations")]
    pub nr_migrations: u64,

    #[stat(desc = "Number of antistall dispatches")]
    pub nr_antistall_dispatches: u64,

    #[stat(desc = "Number of SMT contention avoidances")]
    pub nr_smt_avoided: u64,

    #[stat(desc = "Number of classified tasks")]
    pub nr_classified_tasks: u64,

    #[stat(desc = "Current virtual time")]
    pub vtime_now: u64,

    #[stat(desc = "SCX_TURBO tasks count")]
    pub scx_turbo_count: u32,

    #[stat(desc = "Steam game tasks count")]
    pub steam_count: u32,

    #[stat(desc = "DE component tasks count")]
    pub de_count: u32,

    #[stat(desc = "Input tasks count")]
    pub input_count: u32,

    #[stat(desc = "Audio tasks count")]
    pub audio_count: u32,
}

impl HappyMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, other: &HappyMetrics) {
        self.nr_lc_dispatches = other.nr_lc_dispatches;
        self.nr_normal_dispatches = other.nr_normal_dispatches;
        self.nr_hog_dispatches = other.nr_hog_dispatches;
        self.nr_preemptions = other.nr_preemptions;
        self.nr_migrations = other.nr_migrations;
        self.nr_antistall_dispatches = other.nr_antistall_dispatches;
        self.nr_smt_avoided = other.nr_smt_avoided;
        self.nr_classified_tasks = other.nr_classified_tasks;
        self.vtime_now = other.vtime_now;
    }
}
