// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_astro contributors

fn add_bpf_warning_suppression(flag: &str) {
    const KEY: &str = "BPF_EXTRA_CFLAGS_POST_INCL";
    match std::env::var(KEY) {
        Ok(existing) => {
            if !existing.split_whitespace().any(|entry| entry == flag) {
                std::env::set_var(KEY, format!("{existing} {flag}"));
            }
        }
        Err(_) => std::env::set_var(KEY, flag),
    }
}

fn main() {
    add_bpf_warning_suppression("-Wno-missing-declarations");
    scx_cargo::BpfBuilder::new()
        .unwrap()
        .enable_intf("src/bpf/intf.h", "bpf_intf.rs")
        .enable_skel("src/bpf/main.bpf.c", "bpf")
        .build()
        .unwrap();
}
