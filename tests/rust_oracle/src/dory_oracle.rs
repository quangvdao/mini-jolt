use ark_bn254::{Bn254, Fq, Fq2, Fq6, Fr, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, PrimeGroup};
use ark_ff::{Field, One, PrimeField, Zero};
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num};
use dory_pcs::backends::arkworks::{ArkFr as DoryFr, ArkworksPolynomial, BN254 as DoryBN254, G1Routines, G2Routines};
use dory_pcs::setup::{ProverSetup as DoryProverSetup, VerifierSetup as DoryVerifierSetup};
use dory_pcs::Polynomial;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::field_csv::{fq12_poly_basis_matrix_for, fq12_tower_csv, g1_csv, g2_csv, gt_poly_csv, gt_poly_csv_for};
use crate::hex_utils::bytes_to_hex;
use crate::transcript::JoltLikeDoryTranscript;

pub(crate) fn run_dory_pcs_eval_blake2b() {
    let nu = 2usize;
    let sigma = 3usize;
    let log_t = 2usize;
    let max_log_n = nu + sigma;
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    println!("fq_modulus={}", ark_bn254::Fq::MODULUS);
    println!("fr_modulus={}", ark_bn254::Fr::MODULUS);
    let coeffs: Vec<DoryFr> = (0..(1usize << (nu + sigma)))
        .map(|i| DoryFr(Fr::from((i as u64) + 1)))
        .collect();
    let poly = ArkworksPolynomial::new(coeffs);
    let prover_setup = DoryProverSetup::<DoryBN254>::new(&mut rng, max_log_n);
    let verifier_setup: DoryVerifierSetup<DoryBN254> = prover_setup.to_verifier_setup();
    let point_dory: Vec<DoryFr> = (0..(nu + sigma))
        .map(|i| DoryFr(Fr::from((i as u64) + 42)))
        .collect();
    let evaluation = poly.evaluate(&point_dory);
    let (commitment, row_commitments) = poly
        .commit::<DoryBN254, G1Routines>(nu, sigma, &prover_setup)
        .expect("commit");
    let mut prover_transcript = JoltLikeDoryTranscript::new(b"Jolt");
    let proof = dory_pcs::prove::<DoryFr, DoryBN254, G1Routines, G2Routines, _, _>(
        &poly,
        &point_dory,
        row_commitments,
        nu,
        sigma,
        &prover_setup,
        &mut prover_transcript,
    )
    .expect("prove");
    println!("nu={nu}");
    println!("sigma={sigma}");
    println!("dory_layout=AddressMajor");
    println!("log_T={log_t}");
    let point_fr: Vec<Fr> = point_dory.iter().map(|x| x.0).collect();
    let mut reversed: Vec<Fr> = point_fr.iter().cloned().rev().collect();
    let cycle = reversed.drain(0..log_t).collect::<Vec<_>>();
    let mut opening_point_be = reversed;
    opening_point_be.extend_from_slice(&cycle);
    let s = opening_point_be.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    println!("opening_point_be={s}");
    println!("evaluation={}", evaluation.0);
    println!("commitment={}", gt_poly_csv(commitment.0));
    println!("vmv_c={}", gt_poly_csv(proof.vmv_message.c.0));
    println!("vmv_d2={}", gt_poly_csv(proof.vmv_message.d2.0));
    println!("vmv_e1={}", g1_csv(proof.vmv_message.e1.0.into_affine()));
    for i in 0..sigma {
        let f = &proof.first_messages[i];
        let s = &proof.second_messages[i];
        println!("first_{i}_d1_left={}", gt_poly_csv(f.d1_left.0));
        println!("first_{i}_d1_right={}", gt_poly_csv(f.d1_right.0));
        println!("first_{i}_d2_left={}", gt_poly_csv(f.d2_left.0));
        println!("first_{i}_d2_right={}", gt_poly_csv(f.d2_right.0));
        println!("first_{i}_e1_beta={}", g1_csv(f.e1_beta.0.into_affine()));
        println!("first_{i}_e2_beta={}", g2_csv(f.e2_beta.0.into_affine()));
        println!("second_{i}_c_plus={}", gt_poly_csv(s.c_plus.0));
        println!("second_{i}_c_minus={}", gt_poly_csv(s.c_minus.0));
        println!("second_{i}_e1_plus={}", g1_csv(s.e1_plus.0.into_affine()));
        println!("second_{i}_e1_minus={}", g1_csv(s.e1_minus.0.into_affine()));
        println!("second_{i}_e2_plus={}", g2_csv(s.e2_plus.0.into_affine()));
        println!("second_{i}_e2_minus={}", g2_csv(s.e2_minus.0.into_affine()));
    }
    println!("final_e1={}", g1_csv(proof.final_message.e1.0.into_affine()));
    println!("final_e2={}", g2_csv(proof.final_message.e2.0.into_affine()));
    for k in 0..=sigma {
        println!("chi_{k}={}", gt_poly_csv(verifier_setup.chi[k].0));
        println!("delta_1l_{k}={}", gt_poly_csv(verifier_setup.delta_1l[k].0));
        println!("delta_1r_{k}={}", gt_poly_csv(verifier_setup.delta_1r[k].0));
        println!("delta_2l_{k}={}", gt_poly_csv(verifier_setup.delta_2l[k].0));
        println!("delta_2r_{k}={}", gt_poly_csv(verifier_setup.delta_2r[k].0));
    }
    println!("g1_0={}", g1_csv(verifier_setup.g1_0.0.into_affine()));
    println!("g2_0={}", g2_csv(verifier_setup.g2_0.0.into_affine()));
    println!("h1={}", g1_csv(verifier_setup.h1.0.into_affine()));
    println!("h2={}", g2_csv(verifier_setup.h2.0.into_affine()));
    println!("ht={}", gt_poly_csv(verifier_setup.ht.0));
    let u_in_fq12 = ark_bn254::Fq12::new(
        Fq6::new(Fq2::new(Fq::zero(), Fq::one()), Fq2::zero(), Fq2::zero()),
        Fq6::zero(),
    );
    println!("u_fq12_poly={}", gt_poly_csv(u_in_fq12));
    let w_in_fq12 = ark_bn254::Fq12::new(Fq6::zero(), Fq6::one());
    println!("w_fq12_poly={}", gt_poly_csv(w_in_fq12));
    let e_g1g2 = Bn254::pairing(G1Projective::generator().into_affine(), G2Projective::generator().into_affine()).0;
    println!("pair_g1g2={}", gt_poly_csv(e_g1g2));
    let miller = Bn254::multi_miller_loop(
        [G1Projective::generator().into_affine()],
        [G2Projective::generator().into_affine()],
    );
    println!("miller_g1g2={}", gt_poly_csv(miller.0));
    println!("miller_g1g2_tower={}", fq12_tower_csv(miller.0));
    println!("pair_g1g2_tower={}", fq12_tower_csv(e_g1g2));
    println!("miller_g1g2_sq_tower={}", fq12_tower_csv(miller.0 * miller.0));
    let q = BigUint::from_str_radix(&ark_bn254::Fq::MODULUS.to_string(), 10).unwrap();
    let r = BigUint::from_str_radix(&ark_bn254::Fr::MODULUS.to_string(), 10).unwrap();
    let q12_minus_1 = q.pow(12u32) - BigUint::from_u64(1).unwrap();
    let rem = (&q12_minus_1) % (&r);
    println!("final_exp_rem_is_zero={}", if rem == BigUint::from_u64(0).unwrap() { "1" } else { "0" });
    let exp_big = (&q12_minus_1) / (&r);
    let exp_u64s = exp_big.to_u64_digits();
    let pow_naive = miller.0.pow(exp_u64s.as_slice());
    println!("pair_g1g2_pow_naive={}", gt_poly_csv(pow_naive));
    println!(
        "pair_g1g2_pow_naive_eq={}",
        if pow_naive == e_g1g2 { "1" } else { "0" }
    );
    let exp_bits = exp_big.bits();
    println!("final_exp_bits={exp_bits}");
    println!(
        "final_exp_limbs_prefix={}",
        exp_u64s
            .iter()
            .take(6)
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut pow_manual = ark_bn254::Fq12::one();
    let mut base = miller.0;
    let mut e = exp_big.clone();
    while e > BigUint::from_u64(0).unwrap() {
        if (&e & BigUint::from_u64(1).unwrap()) == BigUint::from_u64(1).unwrap() {
            pow_manual *= base;
        }
        base *= base;
        e >>= 1usize;
    }
    println!("pair_g1g2_pow_manual={}", gt_poly_csv(pow_manual));
    println!(
        "pair_g1g2_pow_manual_eq={}",
        if pow_manual == e_g1g2 { "1" } else { "0" }
    );
    let fe = Bn254::final_exponentiation(miller).unwrap().0;
    println!("pair_g1g2_fe_eq={}", if fe == e_g1g2 { "1" } else { "0" });
    println!("pow_manual_eq_fe={}", if pow_manual == fe { "1" } else { "0" });
    let sqrt_minus3 = (-Fq::from(3u64)).sqrt().expect("sqrt(-3) exists");
    let inv2 = Fq::from(2u64).inverse().expect("2 invertible");
    let omega = (-Fq::one() + sqrt_minus3) * inv2;
    let omega2 = omega * omega;
    let zetas = [Fq::one(), -Fq::one(), omega, omega2, -omega, -omega2];
    for (i, z) in zetas.iter().enumerate() {
        let z12 = ark_bn254::Fq12::new(Fq6::new(Fq2::new(*z, Fq::zero()), Fq2::zero(), Fq2::zero()), Fq6::zero());
        let generator = w_in_fq12 * z12;
        let m = fq12_poly_basis_matrix_for(generator);
        println!("pair_g1g2_zeta_{i}={}", gt_poly_csv_for(&m, e_g1g2));
    }
    let g2_aff = G2Projective::generator().into_affine();
    let x_fq12 = ark_bn254::Fq12::new(Fq6::new(g2_aff.x, Fq2::zero(), Fq2::zero()), Fq6::zero());
    let y_fq12 = ark_bn254::Fq12::new(Fq6::new(g2_aff.y, Fq2::zero(), Fq2::zero()), Fq6::zero());
    let w_fq12 = ark_bn254::Fq12::new(Fq6::zero(), Fq6::one());
    println!("twist_g2_x={}", gt_poly_csv(x_fq12 * (w_fq12 * w_fq12)));
    println!("twist_g2_y={}", gt_poly_csv(y_fq12 * (w_fq12 * w_fq12 * w_fq12)));
    let a = verifier_setup.chi[1].0;
    let b = verifier_setup.chi[2].0;
    println!("mulcheck_a={}", gt_poly_csv(a));
    println!("mulcheck_b={}", gt_poly_csv(b));
    println!("mulcheck_ab={}", gt_poly_csv(a * b));
    println!("serde_blocks_len={}", prover_transcript.serde_blocks.len());
    for (i, b) in prover_transcript.serde_blocks.iter().enumerate() {
        println!("serde_block_{i}={}", bytes_to_hex(b));
    }
}
