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

    /* Dynamic adjustment statistics */
    #[stat(desc = "Number of dynamic virt_nice adjustments")]
    pub nr_dynamic_adjustments: u64,

    #[stat(desc = "Number of interactive tasks detected")]
    pub nr_interactive_detected: u64,

    #[stat(desc = "Number of queue promotions")]
    pub nr_promotions: u64,

    #[stat(desc = "Number of queue demotions")]
    pub nr_demotions: u64,

    /* EEVDF statistics */
    #[stat(desc = "Number of eligible dispatches")]
    pub nr_eligible_dispatches: u64,

    #[stat(desc = "Number of ineligible dispatches")]
    pub nr_ineligible_dispatches: u64,

    #[stat(desc = "Number of deadline expirations")]
    pub nr_deadline_expired: u64,

    /* Per-queue vtime state */
    #[stat(desc = "LC queue minimum vtime")]
    pub lc_min_vtime: u64,

    #[stat(desc = "LC queue average vtime")]
    pub lc_avg_vtime: u64,

    #[stat(desc = "NORMAL queue minimum vtime")]
    pub normal_min_vtime: u64,

    #[stat(desc = "NORMAL queue average vtime")]
    pub normal_avg_vtime: u64,

    #[stat(desc = "HOG queue minimum vtime")]
    pub hog_min_vtime: u64,

    #[stat(desc = "HOG queue average vtime")]
    pub hog_avg_vtime: u64,

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
        self.nr_dynamic_adjustments = other.nr_dynamic_adjustments;
        self.nr_interactive_detected = other.nr_interactive_detected;
        self.nr_promotions = other.nr_promotions;
        self.nr_demotions = other.nr_demotions;
        self.nr_eligible_dispatches = other.nr_eligible_dispatches;
        self.nr_ineligible_dispatches = other.nr_ineligible_dispatches;
        self.nr_deadline_expired = other.nr_deadline_expired;
        self.lc_min_vtime = other.lc_min_vtime;
        self.lc_avg_vtime = other.lc_avg_vtime;
        self.normal_min_vtime = other.normal_min_vtime;
        self.normal_avg_vtime = other.normal_avg_vtime;
        self.hog_min_vtime = other.hog_min_vtime;
        self.hog_avg_vtime = other.hog_avg_vtime;
        self.vtime_now = other.vtime_now;
    }
}
