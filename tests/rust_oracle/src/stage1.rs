use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};

use crate::hex_utils::bytes_to_hex;
use crate::poly_mle::{eq_mle, fr_from_i128, lagrange_evals_symmetric, lagrange_kernel_symmetric};
use crate::sumcheck::CompressedUniPolyFr;
use crate::transcript::Blake2bTranscript;

pub(crate) fn run_spartan_outer_stage1_blake2b() {
    let trace_len = 8usize;
    let num_cycles_bits = trace_len.ilog2() as usize;
    let num_rows_bits = num_cycles_bits + 2;
    let outer_domain_size = 10usize;
    let first_round_num_coeffs = 28usize;

    let mut t = Blake2bTranscript::new(b"Jolt");

    let mut tau: Vec<Fr> = Vec::with_capacity(num_rows_bits);
    for _ in 0..num_rows_bits {
        tau.push(t.challenge_scalar_optimized_fr());
    }

    let uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &uni_poly_coeffs);
    let r0 = t.challenge_scalar_optimized_fr();

    let uniskip_claim = Fr::zero();
    t.append_scalar_fr(b"opening_claim", uniskip_claim);

    t.append_scalar_fr(b"sumcheck_claim", uniskip_claim);
    let batch_coeff = t.challenge_scalar_fr();
    let mut e = uniskip_claim * batch_coeff;

    let num_rounds = 1 + num_cycles_bits;
    let mut polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(num_rounds);
    let mut r_sumcheck: Vec<Fr> = Vec::with_capacity(num_rounds);
    for j in 0..num_rounds {
        let c0 = Fr::from((1000 + j) as u64);
        let c2 = Fr::from((2000 + j) as u64);
        let c3 = Fr::from((3000 + j) as u64);
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2, c3],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = t.challenge_scalar_optimized_fr();
        e = poly.eval_from_hint(e, rj);
        polys.push(poly);
        r_sumcheck.push(rj);
    }
    let output_claim = e;

    let tau_high = tau[tau.len() - 1];
    let tau_low = &tau[..tau.len() - 1];
    let mut r_rev = r_sumcheck.clone();
    r_rev.reverse();
    let lag = lagrange_kernel_symmetric(tau_high, r0, outer_domain_size);
    let eq = eq_mle(tau_low, &r_rev);
    let factor = lag * eq;
    assert!(factor != Fr::zero(), "unexpected zero factor; tweak constants if this trips");
    let target_inner = output_claim * (batch_coeff * factor).inverse().unwrap();

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
    let _i_cf_write_lookup_to_rd = idx(29);
    let i_virtual = idx(30);
    let i_assert = idx(31);
    let i_dnu = idx(32);
    let i_advice = idx(33);
    let i_is_compressed = idx(34);
    let _i_is_first = idx(35);
    let i_is_last = idx(36);

    let lc_eval = |terms: &[(usize, i128)], c: i128, inputs: &[Fr]| -> Fr {
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

    let eval_azbz = |inputs: &[Fr]| -> (Fr, Fr) {
        let w = lagrange_evals_symmetric(r0, outer_domain_size);
        let mut az_g0 = Fr::zero();
        let mut bz_g0 = Fr::zero();
        for (i, (a_terms, a_c, b_terms, b_c)) in g0.iter().enumerate() {
            az_g0 += w[i] * lc_eval(a_terms, *a_c, inputs);
            bz_g0 += w[i] * lc_eval(b_terms, *b_c, inputs);
        }
        let mut az_g1 = Fr::zero();
        let mut bz_g1 = Fr::zero();
        for (i, (a_terms, a_c, b_terms, b_c)) in g1.iter().enumerate() {
            az_g1 += w[i] * lc_eval(a_terms, *a_c, inputs);
            bz_g1 += w[i] * lc_eval(b_terms, *b_c, inputs);
        }
        let r_stream = r_sumcheck[0];
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
    let az_des = Fr::one();
    let bz_des = target_inner;
    let rhs1 = az_des - a0;
    let rhs2 = bz_des - b0;
    let mut sol_i = None;
    for i in 0..37 {
        for j in (i + 1)..37 {
            let det = az_coeff[i] * bz_coeff[j] - az_coeff[j] * bz_coeff[i];
            if det != Fr::zero() {
                sol_i = Some((i, j, det));
                break;
            }
        }
        if sol_i.is_some() {
            break;
        }
    }
    let (i, j, det) = sol_i.expect("no invertible variable pair found");
    let inv_det = det.inverse().unwrap();
    let x_i = (rhs1 * bz_coeff[j] - az_coeff[j] * rhs2) * inv_det;
    let x_j = (az_coeff[i] * rhs2 - rhs1 * bz_coeff[i]) * inv_det;
    let mut inputs = base;
    inputs[i] = x_i;
    inputs[j] = x_j;
    let (az_chk, bz_chk) = eval_azbz(&inputs);
    let inner_sum_prod = az_chk * bz_chk;
    let instance_expected = factor * inner_sum_prod;
    let batched_expected = instance_expected * batch_coeff;
    assert_eq!(batched_expected, output_claim, "constructed instance should pass");

    for v in inputs.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }

    println!("trace_len={trace_len}");
    println!("tau={}", tau.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("uni_poly_coeffs={}", uni_poly_coeffs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("uniskip_claim={uniskip_claim}");
    for j in 0..num_rounds {
        let c = &polys[j].coeffs_except_linear_term;
        println!("sumcheck_poly_{j}={}", c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    }
    println!("r0={r0}");
    println!("r_sumcheck={}", r_sumcheck.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    println!("output_claim={output_claim}");
    println!("expected_output_claim={batched_expected}");
    println!("instance_expected_output_claim={instance_expected}");
    println!("final_state={}", bytes_to_hex(&t.state));
    println!("r1cs_input_evals={}", inputs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
}
