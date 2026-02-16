use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};

use crate::hex_utils::bytes_to_hex;
use crate::poly_mle::{
    calculate_advice_memory_evaluation_oracle, eq_mle, eq_plus_one_mle,
    eval_initial_ram_mle_oracle, fr_from_i128, lagrange_evals_symmetric,
    lagrange_kernel_symmetric,
};
use crate::sumcheck::CompressedUniPolyFr;
use crate::transcript::Blake2bTranscript;

pub(crate) fn solve_r1cs_inputs_for_target_inner(
    stage1_r0: Fr,
    outer_domain_size: usize,
    r_stream: Fr,
    target_inner: Fr,
) -> Vec<Fr> {
    // Solve for a vector of 37 R1CS input evaluations such that:
    // - A(z) = 1
    // - B(z) = target_inner
    // for the fixed stage1 transcript parameters.
    //
    // Mirrors the (duplicated) logic used in run_stage2_sumchecks_blake2b / run_stage3_sumchecks_blake2b.
    let eval_azbz = |inputs: &[Fr]| -> (Fr, Fr) {
        let idx = |i: usize| i;
        let i_left_input = idx(0);
        let i_right_input = idx(1);
        let i_product = idx(2);
        let i_write_lookup_to_rd = idx(3);
        let i_write_pc_to_rd = idx(4);
        let i_should_branch = idx(5);
        let i_pc = idx(6);
        let i_unexp_pc = idx(7);
        let i_imm = idx(8);
        let i_ram_addr = idx(9);
        let i_rs1 = idx(10);
        let i_rs2 = idx(11);
        let i_rd_write = idx(12);
        let i_ram_read = idx(13);
        let i_ram_write = idx(14);
        let i_left_lookup = idx(15);
        let i_right_lookup = idx(16);
        let i_next_unexp_pc = idx(17);
        let i_next_pc = idx(18);
        let i_next_is_virtual = idx(19);
        let i_next_is_first = idx(20);
        let i_lookup_output = idx(21);
        let i_should_jump = idx(22);
        let i_add = idx(23);
        let i_sub = idx(24);
        let i_mul = idx(25);
        let i_load = idx(26);
        let i_store = idx(27);
        let i_jump = idx(28);
        let i_virtual = idx(30);
        let i_assert = idx(31);
        let i_dnu = idx(32);
        let i_advice = idx(33);
        let i_is_compressed = idx(34);
        let i_is_last = idx(36);

        let lc_eval = |terms: &[(usize, i128)], c: i128| -> Fr {
            let mut acc = fr_from_i128(c);
            for (idx, coeff) in terms.iter() {
                acc += fr_from_i128(*coeff) * inputs[*idx];
            }
            acc
        };

        let g0: Vec<(Vec<(usize, i128)>, i128, Vec<(usize, i128)>, i128)> = vec![
            (vec![(i_load, -1), (i_store, -1)], 1, vec![(i_ram_addr, 1)], 0),
            (vec![(i_load, 1)], 0, vec![(i_ram_read, 1), (i_ram_write, -1)], 0),
            (vec![(i_load, 1)], 0, vec![(i_ram_read, 1), (i_rd_write, -1)], 0),
            (vec![(i_store, 1)], 0, vec![(i_rs2, 1), (i_ram_write, -1)], 0),
            (vec![(i_add, 1), (i_sub, 1), (i_mul, 1)], 0, vec![(i_left_lookup, 1)], 0),
            (vec![(i_add, -1), (i_sub, -1), (i_mul, -1)], 1, vec![(i_left_lookup, 1), (i_left_input, -1)], 0),
            (vec![(i_assert, 1)], 0, vec![(i_lookup_output, 1)], -1),
            (vec![(i_should_jump, 1)], 0, vec![(i_next_unexp_pc, 1), (i_lookup_output, -1)], 0),
            (vec![(i_virtual, 1), (i_is_last, -1)], 0, vec![(i_next_pc, 1), (i_pc, -1)], -1),
            (vec![(i_next_is_virtual, 1), (i_next_is_first, -1)], 0, vec![(i_dnu, -1)], 1),
        ];
        let g1: Vec<(Vec<(usize, i128)>, i128, Vec<(usize, i128)>, i128)> = vec![
            (vec![(i_load, 1), (i_store, 1)], 0, vec![(i_ram_addr, 1), (i_rs1, -1), (i_imm, -1)], 0),
            (vec![(i_add, 1)], 0, vec![(i_right_lookup, 1), (i_left_input, -1), (i_right_input, -1)], 0),
            (vec![(i_sub, 1)], 0, vec![(i_right_lookup, 1), (i_left_input, -1), (i_right_input, 1)], -(1i128 << 64)),
            (vec![(i_mul, 1)], 0, vec![(i_right_lookup, 1), (i_product, -1)], 0),
            (vec![(i_add, -1), (i_sub, -1), (i_mul, -1), (i_advice, -1)], 1, vec![(i_right_lookup, 1), (i_right_input, -1)], 0),
            (vec![(i_write_lookup_to_rd, 1)], 0, vec![(i_rd_write, 1), (i_lookup_output, -1)], 0),
            (vec![(i_write_pc_to_rd, 1)], 0, vec![(i_rd_write, 1), (i_unexp_pc, -1), (i_is_compressed, 2)], -4),
            (vec![(i_should_branch, 1)], 0, vec![(i_next_unexp_pc, 1), (i_unexp_pc, -1), (i_imm, -1)], 0),
            (vec![(i_should_branch, -1), (i_jump, -1)], 1, vec![(i_next_unexp_pc, 1), (i_unexp_pc, -1), (i_dnu, 4), (i_is_compressed, 2)], -4),
        ];

        let w = lagrange_evals_symmetric(stage1_r0, outer_domain_size);
        let mut az_g0 = Fr::zero();
        let mut bz_g0 = Fr::zero();
        for (i, (a_terms, a_c, b_terms, b_c)) in g0.iter().enumerate() {
            az_g0 += w[i] * lc_eval(a_terms, *a_c);
            bz_g0 += w[i] * lc_eval(b_terms, *b_c);
        }
        let mut az_g1 = Fr::zero();
        let mut bz_g1 = Fr::zero();
        for (i, (a_terms, a_c, b_terms, b_c)) in g1.iter().enumerate() {
            az_g1 += w[i] * lc_eval(a_terms, *a_c);
            bz_g1 += w[i] * lc_eval(b_terms, *b_c);
        }
        let az_final = az_g0 + r_stream * (az_g1 - az_g0);
        let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);
        (az_final, bz_final)
    };

    let base = vec![Fr::zero(); 37];
    let (a0, b0) = eval_azbz(&base);
    let mut az_coeff = vec![Fr::zero(); 37];
    let mut bz_coeff = vec![Fr::zero(); 37];
    for i in 0..37 {
        let mut v = base.clone();
        v[i] = Fr::one();
        let (ai, bi) = eval_azbz(&v);
        az_coeff[i] = ai - a0;
        bz_coeff[i] = bi - b0;
    }
    let rhs1 = Fr::one() - a0;
    let rhs2 = target_inner - b0;
    for i in 0..37 {
        for j in (i + 1)..37 {
            let det = az_coeff[i] * bz_coeff[j] - az_coeff[j] * bz_coeff[i];
            if det == Fr::zero() {
                continue;
            }
            let x_i = (rhs1 * bz_coeff[j] - rhs2 * az_coeff[j]) * det.inverse().unwrap();
            let x_j = (rhs2 * az_coeff[i] - rhs1 * bz_coeff[i]) * det.inverse().unwrap();
            let mut r1cs_inputs = base.clone();
            r1cs_inputs[i] = x_i;
            r1cs_inputs[j] = x_j;
            let (a, b) = eval_azbz(&r1cs_inputs);
            if a == Fr::one() && b == target_inner {
                return r1cs_inputs;
            }
        }
    }
    panic!("failed to solve r1cs input evals");
}

pub(crate) fn run_stage4_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(false, false, false);
}

pub(crate) fn run_stage5_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(true, false, false);
}

pub(crate) fn run_stage6_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(true, true, false);
}

pub(crate) fn run_stage7_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(true, true, true);
}

pub(crate) fn run_stage4_sumchecks_blake2b_inner(run_stage5: bool, run_stage6: bool, run_stage7: bool) {
    // Deterministic Rust-side oracle for Python Stage1+Stage2+Stage3+Stage4 verifier coupling.
    // Targets transcript/claim wiring parity (not soundness).
    if (run_stage6 || run_stage7) && !run_stage5 {
        panic!("stage6 oracle requires stage5");
    }
    if run_stage7 && !run_stage6 {
        panic!("stage7 oracle requires stage6");
    }
    let trace_len = 8usize;
    let num_cycles_bits = trace_len.ilog2() as usize;
    let num_rows_bits = num_cycles_bits + 2;
    let outer_domain_size = 10usize; // OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE for 19 constraints
    let outer_first_round_num_coeffs = 28usize; // OUTER_FIRST_ROUND_POLY_NUM_COEFFS

    // Stage2 parameters
    let ram_k = 8usize;
    let log_k = ram_k.ilog2() as usize;
    let log_t = trace_len.ilog2() as usize;
    let rw_phase1 = log_t; // force single advice opening behavior
    let rw_phase2 = 1usize;
    let max_rounds_stage2 = log_k + log_t;

    // Stage4 parameters
    let regs_rw_phase1 = 1usize;
    let regs_rw_phase2 = 1usize;
    // Jolt uses REGISTER_COUNT = RISCV (32) + virtual (96) = 128 => log2 = 7.
    let regs_log_k = common::constants::REGISTER_COUNT.ilog2() as usize;
    let max_rounds_stage4 = regs_log_k + log_t;

    // Memory layout (crafted so OutputSumcheck io_mask range is empty: input_start maps to RAM_START)
    let ram_start_addr: u64 = 0x8000_0000;
    let lowest_addr: u64 = ram_start_addr - 8;
    let input_start: u64 = ram_start_addr;
    let output_start: u64 = lowest_addr;
    let panic_addr: u64 = lowest_addr;
    let termination_addr: u64 = lowest_addr;

    // Advice regions within the RAM domain [lowest..lowest+K*8)
    let has_untrusted_advice_commitment = true;
    let has_trusted_advice_commitment = true;
    let untrusted_advice_start: u64 = lowest_addr; // index=0
    let max_untrusted_advice_size: u64 = if run_stage7 { 32 } else { 16 }; // bytes => 4 u64 words => num_vars=2
    let trusted_advice_start: u64 = lowest_addr + max_untrusted_advice_size; // packed after untrusted
    let max_trusted_advice_size: u64 = if run_stage7 { 32 } else { 16 };
    let advice_words = (max_untrusted_advice_size as usize / 8).max(1);
    let advice_len = advice_words.next_power_of_two().max(1);
    let advice_num_vars = advice_len.trailing_zeros() as usize; // advice_len is pow2

    // Public init-eval inputs
    let min_bytecode_address: u64 = ram_start_addr; // remaps to 1
    let bytecode_words: Vec<u64> = vec![5u64];
    let inputs_words: Vec<u64> = vec![9u64];

    // Advice opening claims (evals at advice point)
    let untrusted_advice_eval = Fr::from(11u64);
    let trusted_advice_eval = Fr::from(17u64);

    let mut t = Blake2bTranscript::new(b"Jolt");

    // ----------------
    // Stage 1: Spartan outer stage1
    // ----------------
    let mut tau: Vec<Fr> = Vec::with_capacity(num_rows_bits);
    for _ in 0..num_rows_bits {
        tau.push(t.challenge_scalar_optimized_fr());
    }

    let stage1_uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); outer_first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &stage1_uni_poly_coeffs);
    let stage1_r0 = t.challenge_scalar_optimized_fr();

    let stage1_uniskip_claim = Fr::zero();
    t.append_scalar_fr(b"opening_claim", stage1_uniskip_claim);

    t.append_scalar_fr(b"sumcheck_claim", stage1_uniskip_claim);
    let stage1_batch_coeff = t.challenge_scalar_fr();
    let mut e1 = stage1_uniskip_claim * stage1_batch_coeff;

    let num_rounds_stage1 = 1 + num_cycles_bits;
    let mut stage1_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds_stage1);
    let mut stage1_r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds_stage1);
    for j in 0..num_rounds_stage1 {
        let c0 = Fr::from((1000 + j) as u64);
        let c2 = Fr::from((2000 + j) as u64);
        let c3 = Fr::from((3000 + j) as u64);
        let poly = CompressedUniPolyFr { coeffs_except_linear_term: vec![c0, c2, c3] };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = t.challenge_scalar_optimized_fr();
        e1 = poly.eval_from_hint(e1, rj);
        stage1_polys.push(poly);
        stage1_r_sumcheck.push(rj);
    }
    let stage1_output_claim = e1;

    let mut r_cycle_be: Vec<Fr> = stage1_r_sumcheck[1..].to_vec();
    r_cycle_be.reverse();

    let tau_high = tau[tau.len() - 1];
    let tau_low = &tau[..tau.len() - 1];
    let mut r_rev = stage1_r_sumcheck.clone();
    r_rev.reverse();
    let lag = lagrange_kernel_symmetric(tau_high, stage1_r0, outer_domain_size);
    let eq = eq_mle(tau_low, &r_rev);
    let factor = lag * eq;
    let target_inner = stage1_output_claim * (stage1_batch_coeff * factor).inverse().unwrap();

    let r1cs_inputs = solve_r1cs_inputs_for_target_inner(stage1_r0, outer_domain_size, stage1_r_sumcheck[0], target_inner);

    // Seed Stage1 R1CS input openings (verifier caches them after sumcheck).
    for v in r1cs_inputs.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }

    // ----------------
    // Stage 2a: product virtualization uni-skip
    // ----------------
    let base_evals = [r1cs_inputs[2], r1cs_inputs[3], r1cs_inputs[4], r1cs_inputs[5], r1cs_inputs[22]];
    let tau_high_pv = t.challenge_scalar_optimized_fr();
    let w_tau = lagrange_evals_symmetric(tau_high_pv, 5);
    let mut pv_input_claim = Fr::zero();
    for k in 0..5 {
        pv_input_claim += w_tau[k] * base_evals[k];
    }
    let inv5 = Fr::from(5u64).inverse().unwrap();
    let c0 = pv_input_claim * inv5;
    let mut stage2_uniskip_poly_coeffs = vec![Fr::zero(); 13];
    stage2_uniskip_poly_coeffs[0] = c0;
    t.append_scalars_fr(b"uniskip_poly", &stage2_uniskip_poly_coeffs);
    let stage2_r0 = t.challenge_scalar_optimized_fr();

    let stage2_uniskip_claim = Fr::zero();
    t.append_scalar_fr(b"opening_claim", stage2_uniskip_claim);

    // ----------------
    // Stage 2b: batched sumcheck (5 instances)
    // ----------------
    let gamma_rw = t.challenge_scalar_fr();
    let gamma_instr = t.challenge_scalar_fr();
    let gamma_instr_sqr = gamma_instr.square();
    let mut stage2_r_address: Vec<Fr> = Vec::with_capacity(log_k);
    for _ in 0..log_k {
        stage2_r_address.push(t.challenge_scalar_optimized_fr());
    }

    let get = |idx: usize| r1cs_inputs[idx];
    let ramrw_input = get(13) + gamma_rw * get(14);
    let pvrem_input = stage2_uniskip_claim;
    let instr_input = get(21) + gamma_instr * get(15) + gamma_instr_sqr * get(16);
    let raf_input = get(9);
    let out_input = Fr::zero();
    let input_claims = [ramrw_input, pvrem_input, instr_input, raf_input, out_input];

    for c in input_claims.iter() {
        t.append_scalar_fr(b"sumcheck_claim", *c);
    }
    let mut batching_coeffs: Vec<Fr> = Vec::with_capacity(5);
    for _ in 0..5 {
        batching_coeffs.push(t.challenge_scalar_fr());
    }

    let num_rounds = max_rounds_stage2;
    let scale = |m: usize| Fr::from((1u64) << (num_rounds - m));
    let rounds_i = [num_rounds, log_t, log_t, log_k, log_k];
    let mut claim2 = Fr::zero();
    for i in 0..5 {
        claim2 += input_claims[i] * scale(rounds_i[i]) * batching_coeffs[i];
    }

    let mut stage2_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds);
    let mut stage2_r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds);
    let mut e2 = claim2;
    for j in 0..num_rounds {
        let c0 = Fr::from((4000 + j) as u64);
        let c2 = Fr::from((5000 + j) as u64);
        let c3 = Fr::from((6000 + j) as u64);
        let poly = CompressedUniPolyFr { coeffs_except_linear_term: vec![c0, c2, c3] };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = t.challenge_scalar_optimized_fr();
        e2 = poly.eval_from_hint(e2, rj);
        stage2_polys.push(poly);
        stage2_r_sumcheck.push(rj);
    }
    let stage2_output_claim = e2;

    // Choose InstructionClaimReduction as the only nonzero expected term.
    let instr_slice = &stage2_r_sumcheck[log_k..log_k + log_t];
    let mut opening_point_instr: Vec<Fr> = instr_slice.to_vec();
    opening_point_instr.reverse();
    let eq_instr = eq_mle(&opening_point_instr, &r_cycle_be);
    let coeff_instr = batching_coeffs[2];
    if coeff_instr == Fr::zero() || eq_instr == Fr::zero() {
        panic!("unexpected zero coefficient; tweak constants");
    }
    let instr_expected = stage2_output_claim * coeff_instr.inverse().unwrap();
    let instr_lookup_output_claim = instr_expected * eq_instr.inverse().unwrap();

    // Compute RAM init-evals at the two Stage2-derived r_address points used by Stage4.
    let normalize_ramrw_opening = |r: &[Fr]| -> Vec<Fr> {
        let p1 = &r[..rw_phase1];
        let rest1 = &r[rw_phase1..];
        let p2 = &rest1[..rw_phase2];
        let rest2 = &rest1[rw_phase2..];
        let p3_cycle = &rest2[..(log_t - rw_phase1)];
        let p3_addr = &rest2[(log_t - rw_phase1)..];
        let mut r_cycle = p3_cycle.to_vec();
        r_cycle.reverse();
        let mut p1r = p1.to_vec();
        p1r.reverse();
        r_cycle.extend_from_slice(&p1r);
        let mut r_addr = p3_addr.to_vec();
        r_addr.reverse();
        let mut p2r = p2.to_vec();
        p2r.reverse();
        r_addr.extend_from_slice(&p2r);
        let mut out = r_addr;
        out.extend_from_slice(&r_cycle);
        out
    };

    let ramrw_opening_be = normalize_ramrw_opening(&stage2_r_sumcheck);
    let r_address_rw = &ramrw_opening_be[..log_k];

    let out_slice = &stage2_r_sumcheck[log_t..log_t + log_k];
    let mut r_address_out = out_slice.to_vec();
    r_address_out.reverse();

    let public_rw = eval_initial_ram_mle_oracle(
        lowest_addr,
        min_bytecode_address,
        &bytecode_words,
        input_start,
        &inputs_words,
        r_address_rw,
    );
    let public_out = eval_initial_ram_mle_oracle(
        lowest_addr,
        min_bytecode_address,
        &bytecode_words,
        input_start,
        &inputs_words,
        &r_address_out,
    );

    let untrusted_contrib_rw = calculate_advice_memory_evaluation_oracle(
        untrusted_advice_eval,
        advice_num_vars,
        untrusted_advice_start,
        lowest_addr,
        r_address_rw,
        log_k,
    );
    let trusted_contrib_rw = calculate_advice_memory_evaluation_oracle(
        trusted_advice_eval,
        advice_num_vars,
        trusted_advice_start,
        lowest_addr,
        r_address_rw,
        log_k,
    );
    let init_eval_rw = public_rw + untrusted_contrib_rw + trusted_contrib_rw;

    let untrusted_contrib_out = calculate_advice_memory_evaluation_oracle(
        untrusted_advice_eval,
        advice_num_vars,
        untrusted_advice_start,
        lowest_addr,
        &r_address_out,
        log_k,
    );
    let trusted_contrib_out = calculate_advice_memory_evaluation_oracle(
        trusted_advice_eval,
        advice_num_vars,
        trusted_advice_start,
        lowest_addr,
        &r_address_out,
        log_k,
    );
    let init_eval_out = public_out + untrusted_contrib_out + trusted_contrib_out;

    // Stage2 opening claims (mostly zeros; set RamVal + RamValFinal to init-eval for Stage4 input_claim=0).
    let ramrw_val_claim = init_eval_rw;
    let ramrw_ra_claim = Fr::zero();
    let ramrw_raminc_claim = Fr::zero();
    let mut pv_factor_claims = vec![Fr::zero(); 9];
    if run_stage5 {
        // Stage5 checks LookupOutput parity across InstructionClaimReduction and SpartanProductVirtualization.
        // Keep ProductVirtualRemainder expected output at 0 by forcing (1 - NextIsNoop)=0.
        pv_factor_claims[5] = instr_lookup_output_claim; // VirtualPolynomial::LookupOutput
        pv_factor_claims[7] = Fr::one(); // VirtualPolynomial::NextIsNoop
    }
    let instr_left_claim = Fr::zero();
    let instr_right_claim = Fr::zero();
    let raf_ra_claim = Fr::zero();
    let out_val_final_claim = init_eval_out;
    let out_val_init_claim = Fr::zero();

    // Cache openings: append opening_claim scalars in verifier order.
    t.append_scalar_fr(b"opening_claim", ramrw_val_claim);
    t.append_scalar_fr(b"opening_claim", ramrw_ra_claim);
    t.append_scalar_fr(b"opening_claim", ramrw_raminc_claim);
    for v in pv_factor_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }
    t.append_scalar_fr(b"opening_claim", instr_lookup_output_claim);
    t.append_scalar_fr(b"opening_claim", instr_left_claim);
    t.append_scalar_fr(b"opening_claim", instr_right_claim);
    t.append_scalar_fr(b"opening_claim", raf_ra_claim);
    t.append_scalar_fr(b"opening_claim", out_val_final_claim);
    t.append_scalar_fr(b"opening_claim", out_val_init_claim);

    // Stage3 begins; need r_cycle_stage2_be for instruction virtualization expected.
    let mut r_cycle_stage2_be = stage2_r_sumcheck[log_k..log_k + log_t].to_vec();
    r_cycle_stage2_be.reverse();

    // ----------------
    // Stage 3: batched sumcheck (3 instances)
    // ----------------
    let shift_gamma_powers = t.challenge_scalar_powers_fr(5);
    let instr_gamma = t.challenge_scalar_fr();
    let instr_gamma_sqr = instr_gamma.square();
    let regs_gamma = t.challenge_scalar_fr();
    let regs_gamma_sqr = regs_gamma.square();

    let next_is_noop_claim = pv_factor_claims[7];
    let shift_input = get(17)
        + get(18) * shift_gamma_powers[1]
        + get(19) * shift_gamma_powers[2]
        + get(20) * shift_gamma_powers[3]
        + (Fr::one() - next_is_noop_claim) * shift_gamma_powers[4];
    let instr_input_stage1 = get(1) + instr_gamma * get(0);
    let instr_input_stage2 = pv_factor_claims[1] + instr_gamma * pv_factor_claims[0];
    let instr_input_total = instr_input_stage1 + instr_gamma_sqr * instr_input_stage2;
    let regs_input = get(12) + regs_gamma * get(10) + regs_gamma_sqr * get(11);

    let stage3_input_claims = [shift_input, instr_input_total, regs_input];
    for c in stage3_input_claims.iter() {
        t.append_scalar_fr(b"sumcheck_claim", *c);
    }

    let stage3_batch_coeffs = t.challenge_vector_fr(3);
    let mut stage3_claim = Fr::zero();
    for i in 0..3 {
        stage3_claim += stage3_input_claims[i] * stage3_batch_coeffs[i];
    }

    let num_rounds_stage3 = log_t;
    let mut stage3_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds_stage3);
    let mut stage3_r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds_stage3);
    let mut e3 = stage3_claim;
    for j in 0..num_rounds_stage3 {
        let c0 = Fr::from((7000 + j) as u64);
        let c2 = Fr::from((8000 + j) as u64);
        let c3 = Fr::from((9000 + j) as u64);
        let poly = CompressedUniPolyFr { coeffs_except_linear_term: vec![c0, c2, c3] };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = t.challenge_scalar_optimized_fr();
        e3 = poly.eval_from_hint(e3, rj);
        stage3_polys.push(poly);
        stage3_r_sumcheck.push(rj);
    }
    let stage3_output_claim = e3;

    let mut r_stage3_be = stage3_r_sumcheck.clone();
    r_stage3_be.reverse();

    let eq_stage1 = eq_mle(&r_stage3_be, &r_cycle_be);
    let eq_stage2 = eq_mle(&r_stage3_be, &r_cycle_stage2_be);
    let combo_instr = eq_stage1 + instr_gamma_sqr * eq_stage2;

    let coeff_shift = stage3_batch_coeffs[0];
    let coeff_instr3 = stage3_batch_coeffs[1];
    let coeff_regs = stage3_batch_coeffs[2];

    // Stage3 opening claims: shift(5), instr(8), regs(3), in verifier cache order.
    let mut stage3_shift_claims = vec![Fr::zero(); 5];
    stage3_shift_claims[4] = Fr::one(); // is_noop := 1, so shift expected = 0 by default
    let mut stage3_instr_claims = vec![Fr::zero(); 8];
    let mut stage3_regs_claims = vec![Fr::zero(); 3];

    if coeff_regs != Fr::zero() && eq_stage1 != Fr::zero() {
        // Carrier: registers claim reduction (expected = Eq(stage1)*rd_write)
        let denom = coeff_regs * eq_stage1;
        stage3_regs_claims[0] = stage3_output_claim * denom.inverse().unwrap(); // RdWriteValue
    } else if coeff_instr3 != Fr::zero() && combo_instr != Fr::zero() {
        // Carrier: instruction input virtualization (expected = combo * (right_is_imm*imm))
        let denom = coeff_instr3 * combo_instr;
        stage3_instr_claims[6] = Fr::one(); // RightOperandIsImm
        stage3_instr_claims[7] = stage3_output_claim * denom.inverse().unwrap(); // Imm
    } else if coeff_shift != Fr::zero() {
        // Fallback carrier: shift sumcheck (expected = unexpanded_pc * EqPlusOne(stage1))
        let eqp1 = eq_plus_one_mle(&r_cycle_be, &r_stage3_be);
        if eqp1 == Fr::zero() {
            panic!("all stage3 carrier denominators zero; tweak constants");
        }
        let denom = coeff_shift * eqp1;
        stage3_shift_claims[0] = stage3_output_claim * denom.inverse().unwrap(); // UnexpandedPC
    } else {
        panic!("all stage3 batch coeffs were zero; tweak constants");
    }

    // Keep UnexpandedPC consistent across SpartanShift and InstructionInputVirtualization.
    // This is required by Stage6 `BytecodeReadRafSumcheckVerifier` (it asserts equality).
    stage3_instr_claims[3] = stage3_shift_claims[0];

    for v in stage3_shift_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }
    for v in stage3_instr_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }
    for v in stage3_regs_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }

    // ----------------
    // Stage 4: advice accumulation + batched sumcheck (3 instances)
    // ----------------
    if has_untrusted_advice_commitment {
        t.append_scalar_fr(b"opening_claim", untrusted_advice_eval);
    }
    if has_trusted_advice_commitment {
        t.append_scalar_fr(b"opening_claim", trusted_advice_eval);
    }

    let stage4_regs_rw_gamma = t.challenge_scalar_fr();

    let regs_rw_input = stage3_regs_claims[0]
        + stage4_regs_rw_gamma * (stage3_regs_claims[1] + stage4_regs_rw_gamma * stage3_regs_claims[2]);
    let ram_val_eval_input = ramrw_val_claim - init_eval_rw;
    let val_final_input = out_val_final_claim - init_eval_out;

    let stage4_input_claims = [regs_rw_input, ram_val_eval_input, val_final_input];
    for c in stage4_input_claims.iter() {
        t.append_scalar_fr(b"sumcheck_claim", *c);
    }

    let stage4_batch_coeffs = t.challenge_vector_fr(3);
    let scale4 = |m: usize| Fr::from((1u64) << (max_rounds_stage4 - m));
    let rounds4 = [max_rounds_stage4, log_t, log_t];
    let mut claim4 = Fr::zero();
    for i in 0..3 {
        claim4 += stage4_input_claims[i] * scale4(rounds4[i]) * stage4_batch_coeffs[i];
    }

    let mut stage4_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(max_rounds_stage4);
    let mut stage4_r_sumcheck: Vec<Fr> = Vec::with_capacity(max_rounds_stage4);
    let mut e4 = claim4;
    for j in 0..max_rounds_stage4 {
        let c0 = Fr::from((12000 + j) as u64);
        let c2 = Fr::from((13000 + j) as u64);
        let c3 = Fr::from((14000 + j) as u64);
        let poly = CompressedUniPolyFr { coeffs_except_linear_term: vec![c0, c2, c3] };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = t.challenge_scalar_optimized_fr();
        e4 = poly.eval_from_hint(e4, rj);
        stage4_polys.push(poly);
        stage4_r_sumcheck.push(rj);
    }
    let stage4_output_claim = e4;

    // Stage4 opening claims: carry expected output with ValFinal only.
    let coeff_val_final = stage4_batch_coeffs[2];
    if coeff_val_final == Fr::zero() {
        panic!("stage4 val-final batch coeff was zero; tweak constants");
    }
    let stage4_val_final_inc_claim = stage4_output_claim * coeff_val_final.inverse().unwrap();
    let stage4_val_final_wa_claim = Fr::one();

    let stage4_regs_val_claim = Fr::zero();
    let stage4_regs_rs1_ra_claim = Fr::zero();
    let stage4_regs_rs2_ra_claim = Fr::zero();
    let stage4_regs_rd_wa_claim = Fr::zero();
    let stage4_regs_rdinc_claim = Fr::zero();
    let stage4_ram_val_eval_wa_claim = Fr::zero();
    let stage4_ram_val_eval_inc_claim = Fr::zero();

    // Cache openings: append opening_claim scalars in verifier order (Stage4 only).
    t.append_scalar_fr(b"opening_claim", stage4_regs_val_claim);
    t.append_scalar_fr(b"opening_claim", stage4_regs_rs1_ra_claim);
    t.append_scalar_fr(b"opening_claim", stage4_regs_rs2_ra_claim);
    t.append_scalar_fr(b"opening_claim", stage4_regs_rd_wa_claim);
    t.append_scalar_fr(b"opening_claim", stage4_regs_rdinc_claim);
    t.append_scalar_fr(b"opening_claim", stage4_ram_val_eval_wa_claim);
    t.append_scalar_fr(b"opening_claim", stage4_ram_val_eval_inc_claim);
    t.append_scalar_fr(b"opening_claim", stage4_val_final_inc_claim);
    t.append_scalar_fr(b"opening_claim", stage4_val_final_wa_claim);

    // ----------------
    // Stage 5: batched sumcheck (3 instances)
    // ----------------
    let lookups_ra_virtual_log_k_chunk: usize = 16;
    let mut stage5_polys: Vec<CompressedUniPolyFr> = Vec::new();
    let mut stage5_r_sumcheck: Vec<Fr> = Vec::new();
    let mut stage5_output_claim: Fr = Fr::zero();
    let mut stage5_batch_coeffs: Vec<Fr> = Vec::new();
    let mut stage5_ir_table_flag_claims: Vec<Fr> = Vec::new();
    let mut stage5_ir_instruction_ra_claims: Vec<Fr> = Vec::new();
    let mut stage5_ir_raf_flag_claim: Fr = Fr::zero();
    let mut stage5_ram_ra_reduced_claim: Fr = Fr::zero();
    let mut stage5_regs_rdinc_claim: Fr = Fr::zero();
    let mut stage5_regs_rdwa_claim: Fr = Fr::zero();

    if run_stage5 {
        // Verifier constructor order:
        // - InstructionReadRaf samples gamma
        // - RamRaClaimReduction samples gamma
        let stage5_ir_gamma = t.challenge_scalar_fr();
        let stage5_ir_gamma_sqr = stage5_ir_gamma.square();
        let stage5_ir_input =
            instr_lookup_output_claim + stage5_ir_gamma * instr_left_claim + stage5_ir_gamma_sqr * instr_right_claim;

        let stage5_ramra_gamma = t.challenge_scalar_fr();
        let stage5_ramra_gamma_sqr = stage5_ramra_gamma.square();
        let stage5_ramra_gamma_cubed = stage5_ramra_gamma_sqr * stage5_ramra_gamma;
        let stage5_ramra_input = raf_ra_claim
            + stage5_ramra_gamma * stage4_val_final_wa_claim
            + stage5_ramra_gamma_sqr * ramrw_ra_claim
            + stage5_ramra_gamma_cubed * stage4_ram_val_eval_wa_claim;

        let stage5_regs_val_input = stage4_regs_val_claim;

        let stage5_input_claims = [stage5_ir_input, stage5_ramra_input, stage5_regs_val_input];
        for c in stage5_input_claims.iter() {
            t.append_scalar_fr(b"sumcheck_claim", *c);
        }

        stage5_batch_coeffs = t.challenge_vector_fr(3);

        // BatchedSumcheck scaling uses 2^(max_rounds - m) in the field (m can be very small here).
        let max_rounds_stage5 = 128usize + log_t;
        let scale5 = |m: usize| Fr::from(2u64).pow(&[(max_rounds_stage5 - m) as u64]);
        let rounds5 = [max_rounds_stage5, log_k + log_t, log_t];
        let mut claim5 = Fr::zero();
        for i in 0..3 {
            claim5 += stage5_input_claims[i] * scale5(rounds5[i]) * stage5_batch_coeffs[i];
        }

        stage5_polys = Vec::with_capacity(max_rounds_stage5);
        stage5_r_sumcheck = Vec::with_capacity(max_rounds_stage5);
        let mut e5 = claim5;
        for j in 0..max_rounds_stage5 {
            let c0 = Fr::from((15000 + j) as u64);
            let c2 = Fr::from((16000 + j) as u64);
            let c3 = Fr::from((17000 + j) as u64);
            let poly = CompressedUniPolyFr {
                coeffs_except_linear_term: vec![c0, c2, c3],
            };
            t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
            let rj = t.challenge_scalar_optimized_fr();
            e5 = poly.eval_from_hint(e5, rj);
            stage5_polys.push(poly);
            stage5_r_sumcheck.push(rj);
        }
        stage5_output_claim = e5;

        // Carry expected output with RamRaClaimReduction only.
        let coeff_ramra = stage5_batch_coeffs[1];
        if coeff_ramra == Fr::zero() {
            panic!("stage5 ram-ra batch coeff was zero; tweak constants");
        }
        let target_ramra_expected = stage5_output_claim * coeff_ramra.inverse().unwrap();

        // Compute eq_combined at the RamRaClaimReduction slice challenges.
        let offset_ramra = max_rounds_stage5 - (log_k + log_t);
        let r_slice_ramra = &stage5_r_sumcheck[offset_ramra..];
        let mut r_address_reduced = r_slice_ramra[..log_k].to_vec();
        r_address_reduced.reverse();
        let mut r_cycle_reduced = r_slice_ramra[log_k..].to_vec();
        r_cycle_reduced.reverse();

        let r_address_1 = r_address_out.clone();
        let r_address_2 = r_address_rw.to_vec();
        let r_cycle_raf = r_cycle_be.clone();
        let r_cycle_rw = ramrw_opening_be[log_k..].to_vec();
        let mut r_cycle_val = stage4_r_sumcheck[regs_log_k..].to_vec();
        r_cycle_val.reverse();

        let eq_addr_1 = eq_mle(&r_address_1, &r_address_reduced);
        let eq_addr_2 = eq_mle(&r_address_2, &r_address_reduced);
        let eq_cycle_raf = eq_mle(&r_cycle_raf, &r_cycle_reduced);
        let eq_cycle_rw = eq_mle(&r_cycle_rw, &r_cycle_reduced);
        let eq_cycle_val = eq_mle(&r_cycle_val, &r_cycle_reduced);
        let eq_cycle_a = eq_cycle_raf + stage5_ramra_gamma * eq_cycle_val;
        let eq_cycle_b = eq_cycle_rw + stage5_ramra_gamma * eq_cycle_val;
        let eq_combined = eq_addr_1 * eq_cycle_a + stage5_ramra_gamma_sqr * eq_addr_2 * eq_cycle_b;
        if eq_combined == Fr::zero() {
            panic!("stage5 eq_combined was zero; tweak constants");
        }
        stage5_ram_ra_reduced_claim = target_ramra_expected * eq_combined.inverse().unwrap();

        // InstructionReadRaf openings: make ra_claim = 0 (all virtual RA claims zero).
        // NOTE: Keep this in sync with `LOOKUP_TABLES_64` generated in Python.
        // Rust-side count is `LookupTables::<64>::COUNT` (via strum), but the oracle avoids extra deps.
        stage5_ir_table_flag_claims = vec![Fr::zero(); 41];
        stage5_ir_instruction_ra_claims =
            vec![Fr::zero(); 128usize / lookups_ra_virtual_log_k_chunk];
        stage5_ir_raf_flag_claim = Fr::zero();

        // RegistersValEvaluation openings: force expected output to 0.
        stage5_regs_rdinc_claim = Fr::zero();
        stage5_regs_rdwa_claim = Fr::zero();

        // Cache openings: append opening_claim scalars in verifier order (Stage5 only).
        for v in stage5_ir_table_flag_claims.iter() {
            t.append_scalar_fr(b"opening_claim", *v);
        }
        for v in stage5_ir_instruction_ra_claims.iter() {
            t.append_scalar_fr(b"opening_claim", *v);
        }
        t.append_scalar_fr(b"opening_claim", stage5_ir_raf_flag_claim);
        t.append_scalar_fr(b"opening_claim", stage5_ram_ra_reduced_claim);
        t.append_scalar_fr(b"opening_claim", stage5_regs_rdinc_claim);
        t.append_scalar_fr(b"opening_claim", stage5_regs_rdwa_claim);
    }

    // Emit KV output for Python test harness.
    println!("trace_len={trace_len}");
    println!("ram_k={ram_k}");
    println!("rw_phase1={rw_phase1}");
    println!("rw_phase2={rw_phase2}");
    println!("regs_rw_phase1={regs_rw_phase1}");
    println!("regs_rw_phase2={regs_rw_phase2}");
    println!("mem_input_start={input_start}");
    println!("mem_output_start={output_start}");
    println!("mem_panic={panic_addr}");
    println!("mem_termination={termination_addr}");
    println!("mem_lowest_addr={lowest_addr}");

    println!("untrusted_advice_start={untrusted_advice_start}");
    println!("max_untrusted_advice_size={max_untrusted_advice_size}");
    println!("trusted_advice_start={trusted_advice_start}");
    println!("max_trusted_advice_size={max_trusted_advice_size}");
    println!("has_untrusted_advice_commitment={has_untrusted_advice_commitment}");
    println!("has_trusted_advice_commitment={has_trusted_advice_commitment}");
    println!("min_bytecode_address={min_bytecode_address}");
    println!(
        "bytecode_words={}",
        bytecode_words
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "inputs_words={}",
        inputs_words
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    println!("untrusted_advice_eval={untrusted_advice_eval}");
    println!("trusted_advice_eval={trusted_advice_eval}");
    println!("stage4_regs_rw_gamma={stage4_regs_rw_gamma}");

    println!("stage1_tau={}", tau.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage1_uni_poly_coeffs={}", stage1_uni_poly_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage1_uniskip_claim={stage1_uniskip_claim}");
    for j in 0..num_rounds_stage1 {
        let c = &stage1_polys[j].coeffs_except_linear_term;
        println!("stage1_sumcheck_poly_{j}={}", c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    }
    println!("stage1_r0={stage1_r0}");
    println!("stage1_r_sumcheck={}", stage1_r_sumcheck.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage1_r1cs_input_evals={}", r1cs_inputs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));

    println!("stage2_uniskip_poly_coeffs={}", stage2_uniskip_poly_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage2_uniskip_claim={stage2_uniskip_claim}");
    for j in 0..num_rounds {
        let c = &stage2_polys[j].coeffs_except_linear_term;
        println!("stage2_sumcheck_poly_{j}={}", c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    }
    println!("stage2_r0={stage2_r0}");
    println!("stage2_r_sumcheck={}", stage2_r_sumcheck.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage2_output_claim={stage2_output_claim}");
    println!("stage2_batch_coeffs={}", batching_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage2_instr_eq={eq_instr}");
    println!("stage2_instr_lookup_output_claim={instr_lookup_output_claim}");
    println!("stage2_ramrw_val_claim={ramrw_val_claim}");
    println!("stage2_ramrw_ra_claim={ramrw_ra_claim}");
    println!("stage2_ramrw_raminc_claim={ramrw_raminc_claim}");
    println!("stage2_pv_factor_claims={}", pv_factor_claims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage2_instr_left_claim={instr_left_claim}");
    println!("stage2_instr_right_claim={instr_right_claim}");
    println!("stage2_raf_ra_claim={raf_ra_claim}");
    println!("stage2_out_val_final_claim={out_val_final_claim}");
    println!("stage2_out_val_init_claim={out_val_init_claim}");

    for j in 0..num_rounds_stage3 {
        let c = &stage3_polys[j].coeffs_except_linear_term;
        println!("stage3_sumcheck_poly_{j}={}", c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    }
    println!("stage3_r_sumcheck={}", stage3_r_sumcheck.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage3_output_claim={stage3_output_claim}");
    println!("stage3_batch_coeffs={}", stage3_batch_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage3_shift_claims={}", stage3_shift_claims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage3_instr_claims={}", stage3_instr_claims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage3_regs_claims={}", stage3_regs_claims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));

    for j in 0..max_rounds_stage4 {
        let c = &stage4_polys[j].coeffs_except_linear_term;
        println!("stage4_sumcheck_poly_{j}={}", c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    }
    println!("stage4_r_sumcheck={}", stage4_r_sumcheck.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage4_output_claim={stage4_output_claim}");
    println!("stage4_batch_coeffs={}", stage4_batch_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));

    println!("stage4_regs_val_claim={stage4_regs_val_claim}");
    println!("stage4_regs_rs1_ra_claim={stage4_regs_rs1_ra_claim}");
    println!("stage4_regs_rs2_ra_claim={stage4_regs_rs2_ra_claim}");
    println!("stage4_regs_rd_wa_claim={stage4_regs_rd_wa_claim}");
    println!("stage4_regs_rdinc_claim={stage4_regs_rdinc_claim}");
    println!("stage4_ram_val_eval_wa_claim={stage4_ram_val_eval_wa_claim}");
    println!("stage4_ram_val_eval_inc_claim={stage4_ram_val_eval_inc_claim}");
    println!("stage4_val_final_inc_claim={stage4_val_final_inc_claim}");
    println!("stage4_val_final_wa_claim={stage4_val_final_wa_claim}");

    if run_stage5 {
        println!("lookups_ra_virtual_log_k_chunk={lookups_ra_virtual_log_k_chunk}");
        for j in 0..stage5_polys.len() {
            let c = &stage5_polys[j].coeffs_except_linear_term;
            println!(
                "stage5_sumcheck_poly_{j}={}",
                c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
            );
        }
        println!(
            "stage5_r_sumcheck={}",
            stage5_r_sumcheck
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("stage5_output_claim={stage5_output_claim}");
        println!(
            "stage5_batch_coeffs={}",
            stage5_batch_coeffs
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "stage5_ir_table_flag_claims={}",
            stage5_ir_table_flag_claims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "stage5_ir_instruction_ra_claims={}",
            stage5_ir_instruction_ra_claims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("stage5_ir_raf_flag_claim={stage5_ir_raf_flag_claim}");
        println!("stage5_ram_ra_reduced_claim={stage5_ram_ra_reduced_claim}");
        println!("stage5_regs_rdinc_claim={stage5_regs_rdinc_claim}");
        println!("stage5_regs_rdwa_claim={stage5_regs_rdwa_claim}");
    }

    if run_stage6 {
        // ----------------
        // Stage 6: batched sumcheck
        // Base instances order (Python): BytecodeReadRaf, Booleanity, RamHammingBooleanity, RamRaVirtual, InstructionRaVirtual, IncClaimReduction.
        // Stage7 mode adds advice cycle-phase instances (Trusted, Untrusted) at the end.
        // ----------------
        let bytecode_k: usize = 8;
        let log_k_chunk: usize = 16;
        let log_bytecode_k = bytecode_k.ilog2() as usize;
        let bytecode_d = (log_bytecode_k + log_k_chunk - 1) / log_k_chunk; // ceil_div
        let ram_d = (log_k + log_k_chunk - 1) / log_k_chunk; // ceil_div
        let instruction_d = (128usize + log_k_chunk - 1) / log_k_chunk; // ceil_div(128, log_k_chunk)

        // Constructor sampling order must match Python `verify_stage6`.
        // BytecodeReadRaf samples: gamma_powers + per-stage gamma powers.
        let stage6_bcr_gamma_powers = t.challenge_scalar_powers_fr(7);
        let stage6_bcr_stage1_gammas = t.challenge_scalar_powers_fr(2 + 14); // 2 + len(CIRCUIT_FLAGS)
        let stage6_bcr_stage2_gammas = t.challenge_scalar_powers_fr(5);
        let stage6_bcr_stage3_gammas = t.challenge_scalar_powers_fr(9);
        let stage6_bcr_stage4_gammas = t.challenge_scalar_powers_fr(3);
        let stage6_bcr_stage5_gammas = t.challenge_scalar_powers_fr(2 + 41); // 2 + len(LOOKUP_TABLES_64)

        // Booleanity samples optimized gamma (u128 -> MontU128Challenge<Fr> -> Fr).
        // (We force nonzero like Python, but keep output terms zero.)
        let mut _stage6_bool_gamma = t.challenge_scalar_optimized_fr();
        if _stage6_bool_gamma == Fr::zero() {
            _stage6_bool_gamma = Fr::one();
        }
        let total_d = instruction_d + bytecode_d + ram_d;

        // InstructionRaVirtualization samples gamma powers.
        let n_virtual_ra_polys = 128usize / lookups_ra_virtual_log_k_chunk;
        let stage6_irav_gamma_powers = t.challenge_scalar_powers_fr(n_virtual_ra_polys);

        // IncClaimReduction samples gamma.
        let stage6_inc_gamma = t.challenge_scalar_fr();
        let stage6_inc_gamma_sqr = stage6_inc_gamma.square();
        let stage6_inc_gamma_cub = stage6_inc_gamma_sqr * stage6_inc_gamma;

        // AdviceClaimReduction (cycle phase) samples gamma(s) after IncClaimReduction, before sumcheck_claims.
        // Stage 7 mode forces a non-empty address phase, but Stage 6 only runs the cycle phase.
        let stage6_advice_inputs: Vec<Fr> = if run_stage7 {
            let _advice_gamma_trusted = t.challenge_scalar_fr();
            let _advice_gamma_untrusted = t.challenge_scalar_fr();
            vec![trusted_advice_eval, untrusted_advice_eval] // single_opening=true in this oracle
        } else {
            vec![]
        };

        // Compute BytecodeReadRaf input claim (matches Python `BytecodeReadRafSumcheckVerifier.__init__`).
        let mut rv1_terms: Vec<Fr> = Vec::with_capacity(2 + 14);
        rv1_terms.push(get(7)); // UnexpandedPC (SpartanOuter)
        rv1_terms.push(get(8)); // Imm (SpartanOuter)
        // CIRCUIT_FLAGS order: AddOperands, SubtractOperands, MultiplyOperands, Load, Store, Jump, WriteLookupOutputToRD,
        // VirtualInstruction, Assert, DoNotUpdateUnexpandedPC, Advice, IsCompressed, IsFirstInSequence, IsLastInSequence
        rv1_terms.push(get(23)); // AddOperands
        rv1_terms.push(get(24)); // SubtractOperands
        rv1_terms.push(get(25)); // MultiplyOperands
        rv1_terms.push(get(26)); // Load
        rv1_terms.push(get(27)); // Store
        rv1_terms.push(get(28)); // Jump
        rv1_terms.push(get(29)); // WriteLookupOutputToRD
        rv1_terms.push(get(30)); // VirtualInstruction
        rv1_terms.push(get(31)); // Assert
        rv1_terms.push(get(32)); // DoNotUpdateUnexpandedPC
        rv1_terms.push(get(33)); // Advice
        rv1_terms.push(get(34)); // IsCompressed
        rv1_terms.push(get(35)); // IsFirstInSequence
        rv1_terms.push(get(36)); // IsLastInSequence
        let rv1 = rv1_terms
            .iter()
            .zip(stage6_bcr_stage1_gammas.iter())
            .fold(Fr::zero(), |acc, (c, g)| acc + (*c) * (*g));

        let rv2_terms = [
            pv_factor_claims[4], // OpFlags_Jump
            pv_factor_claims[6], // InstructionFlags_Branch
            pv_factor_claims[2], // InstructionFlags_IsRdNotZero
            pv_factor_claims[3], // OpFlags_WriteLookupOutputToRD
            pv_factor_claims[8], // OpFlags_VirtualInstruction
        ];
        let rv2 = rv2_terms
            .iter()
            .zip(stage6_bcr_stage2_gammas.iter())
            .fold(Fr::zero(), |acc, (c, g)| acc + (*c) * (*g));

        let imm_claim = stage3_instr_claims[7]; // Imm at InstructionInputVirtualization
        let unexpanded_pc_shift = stage3_shift_claims[0]; // UnexpandedPC at SpartanShift
        let unexpanded_pc_instr = stage3_instr_claims[3]; // UnexpandedPC at InstructionInputVirtualization
        if unexpanded_pc_shift != unexpanded_pc_instr {
            panic!("stage6: UnexpandedPC mismatch across Stage3 openings");
        }
        let rv3_terms = [
            imm_claim,
            unexpanded_pc_shift,
            stage3_instr_claims[0], // LeftOperandIsRs1Value
            stage3_instr_claims[2], // LeftOperandIsPC
            stage3_instr_claims[4], // RightOperandIsRs2Value
            stage3_instr_claims[6], // RightOperandIsImm
            stage3_shift_claims[4], // InstructionFlags_IsNoop (SpartanShift)
            stage3_shift_claims[2], // OpFlags_VirtualInstruction (SpartanShift)
            stage3_shift_claims[3], // OpFlags_IsFirstInSequence (SpartanShift)
        ];
        let rv3 = rv3_terms
            .iter()
            .zip(stage6_bcr_stage3_gammas.iter())
            .fold(Fr::zero(), |acc, (c, g)| acc + (*c) * (*g));

        let rv4_terms = [
            stage4_regs_rd_wa_claim,
            stage4_regs_rs1_ra_claim,
            stage4_regs_rs2_ra_claim,
        ];
        let rv4 = rv4_terms
            .iter()
            .zip(stage6_bcr_stage4_gammas.iter())
            .fold(Fr::zero(), |acc, (c, g)| acc + (*c) * (*g));

        let mut rv5 = stage5_regs_rdwa_claim * stage6_bcr_stage5_gammas[0];
        rv5 += stage5_ir_raf_flag_claim * stage6_bcr_stage5_gammas[1];
        for (i, v) in stage5_ir_table_flag_claims.iter().enumerate() {
            rv5 += (*v) * stage6_bcr_stage5_gammas[2 + i];
        }

        let raf_claim = get(6); // PC at SpartanOuter
        let raf_shift_claim = stage3_shift_claims[1]; // PC at SpartanShift

        let stage6_bcr_inputs = [rv1, rv2, rv3, rv4, rv5, raf_claim, raf_shift_claim];
        let stage6_bytecode_input = stage6_bcr_inputs
            .iter()
            .zip(stage6_bcr_gamma_powers.iter())
            .fold(Fr::zero(), |acc, (c, g)| acc + (*c) * (*g));

        // RamRaVirtual input claim comes from Stage5 RamRaClaimReduction.
        let stage6_ramra_virtual_input = stage5_ram_ra_reduced_claim;

        // InstructionRaVirtual input claim is a gamma-weighted sum of virtual RA claims from Stage5.
        let mut stage6_instr_ra_virtual_input = Fr::zero();
        for i in 0..n_virtual_ra_polys {
            stage6_instr_ra_virtual_input += stage5_ir_instruction_ra_claims[i] * stage6_irav_gamma_powers[i];
        }

        // IncClaimReduction input claim combines 4 committed openings from earlier stages (all zero in this oracle).
        let stage6_inc_input = ramrw_raminc_claim
            + stage6_inc_gamma * stage4_ram_val_eval_inc_claim
            + stage6_inc_gamma_sqr * stage4_regs_rdinc_claim
            + stage6_inc_gamma_cub * stage5_regs_rdinc_claim;

        // Append sumcheck_claims in instance order.
        let mut stage6_input_claims: Vec<Fr> = vec![
            stage6_bytecode_input,
            Fr::zero(), // Booleanity
            Fr::zero(), // RamHammingBooleanity
            stage6_ramra_virtual_input,
            stage6_instr_ra_virtual_input,
            stage6_inc_input,
        ];
        stage6_input_claims.extend_from_slice(&stage6_advice_inputs);
        for c in stage6_input_claims.iter() {
            t.append_scalar_fr(b"sumcheck_claim", *c);
        }

        let stage6_batch_coeffs = t.challenge_vector_fr(stage6_input_claims.len());

        // BatchedSumcheck scaling: multiply each input by 2^(max_rounds - m).
        let max_rounds_stage6 = log_k_chunk + log_t;
        let scale6 = |m: usize| Fr::from((1u64) << (max_rounds_stage6 - m));
        let mut rounds6: Vec<usize> = vec![
            (bytecode_k.ilog2() as usize) + log_t, // BytecodeReadRaf: log_K + log_T
            max_rounds_stage6,                    // Booleanity: log_k_chunk + log_T
            log_t,                                // RamHammingBooleanity
            log_t,                                // RamRaVirtual
            log_t,                                // InstructionRaVirtual
            log_t,                                // IncClaimReduction
        ];
        if run_stage7 {
            let advice_cycle_rounds = 1usize; // for advice_vars=2 (sigma=1, nu=1) and log_t=3, cycle phase binds 1 var
            rounds6.push(advice_cycle_rounds);
            rounds6.push(advice_cycle_rounds);
        }
        let mut claim6 = Fr::zero();
        for i in 0..stage6_input_claims.len() {
            claim6 += stage6_input_claims[i] * scale6(rounds6[i]) * stage6_batch_coeffs[i];
        }

        let mut stage6_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(max_rounds_stage6);
        let mut stage6_r_sumcheck: Vec<Fr> = Vec::with_capacity(max_rounds_stage6);
        let mut e6 = claim6;
        for j in 0..max_rounds_stage6 {
            let c0 = Fr::from((18000 + j) as u64);
            let c2 = Fr::from((19000 + j) as u64);
            let c3 = Fr::from((20000 + j) as u64);
            let poly = CompressedUniPolyFr {
                coeffs_except_linear_term: vec![c0, c2, c3],
            };
            t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
            let rj = t.challenge_scalar_optimized_fr();
            e6 = poly.eval_from_hint(e6, rj);
            stage6_polys.push(poly);
            stage6_r_sumcheck.push(rj);
        }
        let stage6_output_claim = e6;

        // Make all Stage6 expected-output terms zero except IncClaimReduction.
        let coeff_inc = stage6_batch_coeffs[5];
        if coeff_inc == Fr::zero() {
            panic!("stage6 inc batch coeff was zero; tweak constants");
        }
        let target_inc_expected = stage6_output_claim * coeff_inc.inverse().unwrap();

        // IncClaimReduction slice challenges are the last log_T rounds (offset = log_k_chunk).
        let r_slice_inc = &stage6_r_sumcheck[log_k_chunk..log_k_chunk + log_t];
        let mut r_cycle_stage6_be = r_slice_inc.to_vec();
        r_cycle_stage6_be.reverse();

        // r_cycle points from earlier cached openings (BE).
        let r_cycle_stage2_be = ramrw_opening_be[log_k..].to_vec();
        let mut r_cycle_stage4_be = stage4_r_sumcheck[regs_log_k..].to_vec(); // RamValEvaluation cached r_cycle_prime
        r_cycle_stage4_be.reverse();

        let eq_r2 = eq_mle(&r_cycle_stage6_be, &r_cycle_stage2_be);
        let eq_r4 = eq_mle(&r_cycle_stage6_be, &r_cycle_stage4_be);
        let eq_ram = eq_r2 + stage6_inc_gamma * eq_r4;

        // Fallback: also compute eq_rd if needed.
        let normalize_regs_rw_opening = |r: &[Fr]| -> Vec<Fr> {
            let p1 = &r[..regs_rw_phase1];
            let rest1 = &r[regs_rw_phase1..];
            let p2 = &rest1[..regs_rw_phase2];
            let rest2 = &rest1[regs_rw_phase2..];
            let p3_cycle = &rest2[..(log_t - regs_rw_phase1)];
            let p3_addr = &rest2[(log_t - regs_rw_phase1)..];
            let mut r_cycle = p3_cycle.to_vec();
            r_cycle.reverse();
            let mut p1r = p1.to_vec();
            p1r.reverse();
            r_cycle.extend_from_slice(&p1r);
            let mut r_addr = p3_addr.to_vec();
            r_addr.reverse();
            let mut p2r = p2.to_vec();
            p2r.reverse();
            r_addr.extend_from_slice(&p2r);
            let mut out = r_addr;
            out.extend_from_slice(&r_cycle);
            out
        };
        let regs_opening_be = normalize_regs_rw_opening(&stage4_r_sumcheck);
        let s_cycle_stage4_be = regs_opening_be[regs_log_k..].to_vec();

        let max_rounds_stage5 = 128usize + log_t;
        let mut s_cycle_stage5_be = stage5_r_sumcheck[max_rounds_stage5 - log_t..].to_vec();
        s_cycle_stage5_be.reverse();

        let eq_s4 = eq_mle(&r_cycle_stage6_be, &s_cycle_stage4_be);
        let eq_s5 = eq_mle(&r_cycle_stage6_be, &s_cycle_stage5_be);
        let eq_rd = eq_s4 + stage6_inc_gamma * eq_s5;
        let denom_rd = stage6_inc_gamma_sqr * eq_rd;

        let (stage6_inc_raminc_claim, stage6_inc_rdinc_claim) = if eq_ram != Fr::zero() {
            (target_inc_expected * eq_ram.inverse().unwrap(), Fr::zero())
        } else if denom_rd != Fr::zero() {
            (Fr::zero(), target_inc_expected * denom_rd.inverse().unwrap())
        } else {
            panic!("stage6 eq denominators were zero; tweak constants");
        };

        // Stage6 opening claims, appended in verifier cache order.
        let stage6_bytecode_ra_claims = vec![Fr::zero(); bytecode_d];
        let stage6_booleanity_claims = vec![Fr::zero(); total_d];
        let stage6_hamming_weight_claim = Fr::zero();
        let stage6_ram_ra_virtual_claims = vec![Fr::zero(); ram_d];
        let stage6_instruction_ra_virtual_committed_claims = vec![Fr::zero(); 128usize / log_k_chunk];

        for v in stage6_bytecode_ra_claims.iter() {
            t.append_scalar_fr(b"opening_claim", *v);
        }
        for v in stage6_booleanity_claims.iter() {
            t.append_scalar_fr(b"opening_claim", *v);
        }
        t.append_scalar_fr(b"opening_claim", stage6_hamming_weight_claim);
        for v in stage6_ram_ra_virtual_claims.iter() {
            t.append_scalar_fr(b"opening_claim", *v);
        }
        for v in stage6_instruction_ra_virtual_committed_claims.iter() {
            t.append_scalar_fr(b"opening_claim", *v);
        }
        t.append_scalar_fr(b"opening_claim", stage6_inc_raminc_claim);
        t.append_scalar_fr(b"opening_claim", stage6_inc_rdinc_claim);

        let stage6_advice_cycle_trusted_claim = Fr::zero();
        let stage6_advice_cycle_untrusted_claim = Fr::zero();
        if run_stage7 {
            // AdviceClaimReduction cycle-phase intermediate claims (SumcheckId::AdviceClaimReductionCyclePhase), in verifier cache order.
            t.append_scalar_fr(b"opening_claim", stage6_advice_cycle_trusted_claim);
            t.append_scalar_fr(b"opening_claim", stage6_advice_cycle_untrusted_claim);
        }

        // Emit Stage6 KV output for Python test harness.
        println!("bytecode_k={bytecode_k}");
        println!("log_k_chunk={log_k_chunk}");
        for j in 0..stage6_polys.len() {
            let c = &stage6_polys[j].coeffs_except_linear_term;
            println!(
                "stage6_sumcheck_poly_{j}={}",
                c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
            );
        }
        println!(
            "stage6_r_sumcheck={}",
            stage6_r_sumcheck
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("stage6_output_claim={stage6_output_claim}");
        println!(
            "stage6_batch_coeffs={}",
            stage6_batch_coeffs
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "stage6_bytecode_ra_claims={}",
            stage6_bytecode_ra_claims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "stage6_booleanity_claims={}",
            stage6_booleanity_claims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("stage6_hamming_weight_claim={stage6_hamming_weight_claim}");
        println!(
            "stage6_ram_ra_virtual_claims={}",
            stage6_ram_ra_virtual_claims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "stage6_instruction_ra_virtual_committed_claims={}",
            stage6_instruction_ra_virtual_committed_claims
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("stage6_inc_raminc_claim={stage6_inc_raminc_claim}");
        println!("stage6_inc_rdinc_claim={stage6_inc_rdinc_claim}");

        if run_stage7 {
            println!("stage6_advice_cycle_trusted_claim={stage6_advice_cycle_trusted_claim}");
            println!("stage6_advice_cycle_untrusted_claim={stage6_advice_cycle_untrusted_claim}");
        }

        if run_stage7 {
            // ----------------
            // Stage 7: HammingWeightClaimReduction + AdviceClaimReduction (address phase)
            // Instance order: HW, advice(trusted), advice(untrusted).
            // ----------------
            let n = instruction_d + bytecode_d + ram_d;
            let stage7_gamma_powers = t.challenge_scalar_powers_fr(3 * n);

            // HW claims: instruction/bytecode use 1, ram uses stage6_hamming_weight_claim (0 here).
            let mut stage7_hw_input = Fr::zero();
            for i in 0..n {
                let hw_i = if i < instruction_d + bytecode_d {
                    Fr::one()
                } else {
                    stage6_hamming_weight_claim
                };
                stage7_hw_input += stage7_gamma_powers[3 * i] * hw_i;
            }

            // Advice address-phase input claims start from cycle-phase intermediate claim (we set to 0).
            let stage7_input_claims = [stage7_hw_input, Fr::zero(), Fr::zero()];
            for c in stage7_input_claims.iter() {
                t.append_scalar_fr(b"sumcheck_claim", *c);
            }

            let stage7_batch_coeffs = t.challenge_vector_fr(3);
            let max_rounds_stage7 = log_k_chunk;
            let scale7 = |m: usize| Fr::from((1u64) << (max_rounds_stage7 - m));
            let rounds7 = [log_k_chunk, 1usize, 1usize];
            let mut claim7 = Fr::zero();
            for i in 0..3 {
                claim7 += stage7_input_claims[i] * scale7(rounds7[i]) * stage7_batch_coeffs[i];
            }

            let num_rounds_stage7 = log_k_chunk;
            let mut stage7_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds_stage7);
            let mut stage7_r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds_stage7);
            let mut e7 = claim7;
            for j in 0..num_rounds_stage7 {
                let c0 = Fr::from((21000 + j) as u64);
                let c2 = Fr::from((22000 + j) as u64);
                let poly = CompressedUniPolyFr { coeffs_except_linear_term: vec![c0, c2] };
                t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
                let rj = t.challenge_scalar_optimized_fr();
                e7 = poly.eval_from_hint(e7, rj);
                stage7_polys.push(poly);
                stage7_r_sumcheck.push(rj);
            }
            let stage7_output_claim = e7;

            let coeff_hw = stage7_batch_coeffs[0];
            if coeff_hw == Fr::zero() {
                panic!("stage7 hw batch coeff was zero; tweak constants");
            }
            let target_hw_expected = stage7_output_claim * coeff_hw.inverse().unwrap();

            // Compute eq_bool(ρ) and eq_virt(ρ) for InstructionRa(0).
            let rho_rev: Vec<Fr> = stage7_r_sumcheck.iter().cloned().rev().collect();
            let mut bool_opening = stage6_r_sumcheck.clone();
            bool_opening[..log_k_chunk].reverse();
            bool_opening[log_k_chunk..].reverse();
            let r_addr_bool = &bool_opening[..log_k_chunk];
            let eq_bool_eval = eq_mle(&rho_rev, r_addr_bool);
            let r_addr_virt0 = stage5_r_sumcheck[..log_k_chunk].to_vec(); // InstructionRa(0) address chunk
            let eq_virt_eval = eq_mle(&rho_rev, &r_addr_virt0);

            let w0 = stage7_gamma_powers[0]
                + stage7_gamma_powers[1] * eq_bool_eval
                + stage7_gamma_powers[2] * eq_virt_eval;
            if w0 == Fr::zero() {
                panic!("stage7 weight w0 was zero; tweak constants");
            }

            let mut stage7_ra_opening_claims: Vec<Fr> = vec![Fr::zero(); n];
            stage7_ra_opening_claims[0] = target_hw_expected * w0.inverse().unwrap();

            let stage7_advice_final_trusted_claim = Fr::zero();
            let stage7_advice_final_untrusted_claim = Fr::zero();

            // Cache openings: HW opens all RA polys, then advice final claims (trusted, untrusted).
            for v in stage7_ra_opening_claims.iter() {
                t.append_scalar_fr(b"opening_claim", *v);
            }
            t.append_scalar_fr(b"opening_claim", stage7_advice_final_trusted_claim);
            t.append_scalar_fr(b"opening_claim", stage7_advice_final_untrusted_claim);

            // Emit Stage7 KV output for Python test harness.
            for j in 0..num_rounds_stage7 {
                let c = &stage7_polys[j].coeffs_except_linear_term;
                println!("stage7_sumcheck_poly_{j}={}", c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            }
            println!(
                "stage7_r_sumcheck={}",
                stage7_r_sumcheck.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
            );
            println!("stage7_output_claim={stage7_output_claim}");
            println!(
                "stage7_batch_coeffs={}",
                stage7_batch_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
            );
            println!(
                "stage7_ra_opening_claims={}",
                stage7_ra_opening_claims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
            );
            println!("stage7_advice_final_trusted_claim={stage7_advice_final_trusted_claim}");
            println!("stage7_advice_final_untrusted_claim={stage7_advice_final_untrusted_claim}");
        }
    }

    let final_state = bytes_to_hex(&t.state);
    println!("final_state={final_state}");
}

