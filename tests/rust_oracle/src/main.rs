use std::io::Read;

use ark_bn254::{Fq, Fr};

mod basic_oracles;
mod dory_oracle;
mod e2e_helpers;
mod field_csv;
mod hex_utils;
mod poly_mle;
mod stage1;
mod stage2;
mod stage3;
mod stage4_7;
mod sumcheck;
mod transcript;

#[cfg(feature = "e2e")]
mod e2e_verify;
#[cfg(not(feature = "e2e"))]
mod e2e_verify { pub fn main_from_cli() { panic!("e2e_verify requires: cargo run --features e2e -- ... e2e_verify"); } }

fn main() {
    let mut args = std::env::args();
    let _bin = args.next();
    let mode =
        args.next()
            .expect("pass mode: fq|fr|curve|transcript_blake2b|sumcheck_verify_blake2b|dory_pcs_eval_blake2b|lookup_table_mle_64|spartan_outer_stage1_blake2b|stage2_sumchecks_blake2b|stage3_sumchecks_blake2b|stage4_sumchecks_blake2b|stage5_sumchecks_blake2b|stage6_sumchecks_blake2b|stage7_sumchecks_blake2b");
    if mode.as_str() == "e2e_verify" {
        e2e_verify::main_from_cli();
        return;
    }
    if mode.as_str() == "e2e_transcript_pre_stage1" {
        e2e_helpers::run_e2e_transcript_pre_stage1(args);
        return;
    }
    if mode.as_str() == "e2e_stage1_debug" {
        e2e_helpers::run_e2e_stage1_debug(args);
        return;
    }
    if mode.as_str() == "e2e_stage6_breakdown" {
        e2e_helpers::run_e2e_stage6_breakdown(args);
        return;
    }
    if mode.as_str() == "e2e_bytecode_fingerprint" {
        e2e_helpers::run_e2e_bytecode_fingerprint(args);
        return;
    }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    match mode.as_str() {
        "fq" => basic_oracles::run_field::<Fq>(&input),
        "fr" => basic_oracles::run_field::<Fr>(&input),
        "curve" => basic_oracles::run_curve(&input),
        "transcript_blake2b" => basic_oracles::run_transcript_blake2b(&input),
        "sumcheck_verify_blake2b" => sumcheck::run_sumcheck_verify_blake2b(&input),
        "dory_pcs_eval_blake2b" => dory_oracle::run_dory_pcs_eval_blake2b(),
        "lookup_table_mle_64" => basic_oracles::run_lookup_table_mle_64(&input),
        "spartan_outer_stage1_blake2b" => stage1::run_spartan_outer_stage1_blake2b(),
        "stage2_sumchecks_blake2b" => stage2::run_stage2_sumchecks_blake2b(),
        "stage3_sumchecks_blake2b" => stage3::run_stage3_sumchecks_blake2b(),
        "stage4_sumchecks_blake2b" => stage4_7::run_stage4_sumchecks_blake2b(),
        "stage5_sumchecks_blake2b" => stage4_7::run_stage5_sumchecks_blake2b(),
        "stage6_sumchecks_blake2b" => stage4_7::run_stage6_sumchecks_blake2b(),
        "stage7_sumchecks_blake2b" => stage4_7::run_stage7_sumchecks_blake2b(),
        _ => panic!("unknown mode; use fq|fr|curve|transcript_blake2b|sumcheck_verify_blake2b|dory_pcs_eval_blake2b|lookup_table_mle_64|spartan_outer_stage1_blake2b|stage2_sumchecks_blake2b|stage3_sumchecks_blake2b|stage4_sumchecks_blake2b|stage5_sumchecks_blake2b|stage6_sumchecks_blake2b|stage7_sumchecks_blake2b"),
    }
}
