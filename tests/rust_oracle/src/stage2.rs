use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};

use crate::hex_utils::bytes_to_hex;
use crate::poly_mle::{eq_mle, fr_from_i128, lagrange_evals_symmetric, lagrange_kernel_symmetric};
use crate::sumcheck::CompressedUniPolyFr;
use crate::transcript::Blake2bTranscript;

pub(crate) fn run_stage2_sumchecks_blake2b() {
    // Deterministic Rust-side oracle for Python Stage1+Stage2 verifier coupling.
    // Targets transcript/claim wiring parity (not soundness).
    let trace_len = 8usize;
    let num_cycles_bits = trace_len.ilog2() as usize;
    let num_rows_bits = num_cycles_bits + 2;
    let outer_domain_size = 10usize; // OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE for 19 constraints
    let outer_first_round_num_coeffs = 28usize; // OUTER_FIRST_ROUND_POLY_NUM_COEFFS

    // Stage2 parameters
    let ram_k = 8usize;
    let log_k = ram_k.ilog2() as usize;
    let log_t = trace_len.ilog2() as usize;
    let rw_phase1 = 1usize;
    let rw_phase2 = 1usize;
    let max_rounds_stage2 = log_k + log_t;

    // Memory layout (crafted so OutputSumcheck io_mask range is empty: input_start maps to RAM_START)
    let ram_start_addr: u64 = 0x8000_0000;
    let lowest_addr: u64 = ram_start_addr - 8;
    let input_start: u64 = ram_start_addr;
    let output_start: u64 = lowest_addr;
    let panic_addr: u64 = lowest_addr;
    let termination_addr: u64 = lowest_addr;

    let mut t = Blake2bTranscript::new(b"Jolt");

    // ----------------
    // Stage 1 (same structure as run_spartan_outer_stage1_blake2b)
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
    let mut e = stage1_uniskip_claim * stage1_batch_coeff;

    let num_rounds_stage1 = 1 + num_cycles_bits;
    let mut stage1_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds_stage1);
    let mut stage1_r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds_stage1);
    for j in 0..num_rounds_stage1 {
        let c0 = Fr::from((1000 + j) as u64);
        let c2 = Fr::from((2000 + j) as u64);
        let c3 = Fr::from((3000 + j) as u64);
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2, c3],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = t.challenge_scalar_optimized_fr();
        e = poly.eval_from_hint(e, rj);
        stage1_polys.push(poly);
        stage1_r_sumcheck.push(rj);
    }
    let stage1_output_claim = e;

    // Factor for stage1 expected claim
    let tau_high = tau[tau.len() - 1];
    let tau_low = &tau[..tau.len() - 1];
    let mut r_rev = stage1_r_sumcheck.clone();
    r_rev.reverse();
    let lag = lagrange_kernel_symmetric(tau_high, stage1_r0, outer_domain_size);
    let eq = eq_mle(tau_low, &r_rev);
    let factor = lag * eq;
    let target_inner = stage1_output_claim * (stage1_batch_coeff * factor).inverse().unwrap();

    // Solve 37 R1CS input evals (same as stage1 oracle)
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
        let r_stream = stage1_r_sumcheck[0];
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
    let mut sol = None;
    for i in 0..37 {
        for j in (i + 1)..37 {
            let det = az_coeff[i] * bz_coeff[j] - az_coeff[j] * bz_coeff[i];
            if det != Fr::zero() {
                sol = Some((i, j, det));
                break;
            }
        }
        if sol.is_some() {
            break;
        }
    }
    let (i, j, det) = sol.expect("no invertible pair");
    let inv_det = det.inverse().unwrap();
    let x_i = (rhs1 * bz_coeff[j] - az_coeff[j] * rhs2) * inv_det;
    let x_j = (az_coeff[i] * rhs2 - rhs1 * bz_coeff[i]) * inv_det;
    let mut r1cs_inputs = base;
    r1cs_inputs[i] = x_i;
    r1cs_inputs[j] = x_j;

    for v in r1cs_inputs.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }

    let mut r_cycle_be = stage1_r_sumcheck[1..].to_vec();
    r_cycle_be.reverse();

    // ----------------
    // Stage 2a: product virtualization uni-skip
    // ----------------
    let base_evals = [
        r1cs_inputs[2],
        r1cs_inputs[3],
        r1cs_inputs[4],
        r1cs_inputs[5],
        r1cs_inputs[22],
    ];
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
    let mut claim = Fr::zero();
    for i in 0..5 {
        claim += input_claims[i] * scale(rounds_i[i]) * batching_coeffs[i];
    }

    let mut stage2_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds);
    let mut stage2_r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds);
    let mut e2 = claim;
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

    // Stage2 opening claims (mostly zeros)
    let ramrw_val_claim = Fr::zero();
    let ramrw_ra_claim = Fr::zero();
    let ramrw_raminc_claim = Fr::zero();
    let pv_factor_claims = vec![Fr::zero(); 9];
    let instr_left_claim = Fr::zero();
    let instr_right_claim = Fr::zero();
    let raf_ra_claim = Fr::zero();
    let out_val_final_claim = Fr::zero();
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

    let final_state = bytes_to_hex(&t.state);

    // Emit KV output for Python test harness.
    println!("trace_len={trace_len}");
    println!("ram_k={ram_k}");
    println!("rw_phase1={rw_phase1}");
    println!("rw_phase2={rw_phase2}");
    println!("mem_input_start={input_start}");
    println!("mem_output_start={output_start}");
    println!("mem_panic={panic_addr}");
    println!("mem_termination={termination_addr}");
    println!("mem_lowest_addr={lowest_addr}");

    println!("stage1_tau={}", tau.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage1_uni_poly_coeffs={}", stage1_uni_poly_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("stage1_uniskip_claim={stage1_uniskip_claim}");
    for j in 0..num_rounds_stage1 {
        let c = &stage1_polys[j].coeffs_except_linear_term;
        println!(
            "stage1_sumcheck_poly_{j}={}",
            c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
        );
    }
    println!("stage1_r0={stage1_r0}");
    println!(
        "stage1_r_sumcheck={}",
        stage1_r_sumcheck
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "stage1_r1cs_input_evals={}",
        r1cs_inputs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    println!(
        "stage2_uniskip_poly_coeffs={}",
        stage2_uniskip_poly_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage2_uniskip_claim={stage2_uniskip_claim}");
    for j in 0..num_rounds {
        let c = &stage2_polys[j].coeffs_except_linear_term;
        println!(
            "stage2_sumcheck_poly_{j}={}",
            c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
        );
    }
    println!("stage2_r0={stage2_r0}");
    println!(
        "stage2_r_sumcheck={}",
        stage2_r_sumcheck
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage2_output_claim={stage2_output_claim}");
    println!(
        "stage2_batch_coeffs={}",
        batching_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage2_instr_eq={eq_instr}");
    println!("stage2_instr_lookup_output_claim={instr_lookup_output_claim}");
    println!("stage2_ramrw_val_claim={ramrw_val_claim}");
    println!("stage2_ramrw_ra_claim={ramrw_ra_claim}");
    println!("stage2_ramrw_raminc_claim={ramrw_raminc_claim}");
    println!(
        "stage2_pv_factor_claims={}",
        pv_factor_claims
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage2_instr_left_claim={instr_left_claim}");
    println!("stage2_instr_right_claim={instr_right_claim}");
    println!("stage2_raf_ra_claim={raf_ra_claim}");
    println!("stage2_out_val_final_claim={out_val_final_claim}");
    println!("stage2_out_val_init_claim={out_val_init_claim}");
    println!("final_state={final_state}");
}
