use std::{fmt::Debug, io::Read, str::FromStr, sync::OnceLock};

use ark_bn254::{Bn254, Fq, Fq2, Fq6, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, PrimeGroup};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use dory_pcs::backends::arkworks::{ArkFr as DoryFr, ArkworksPolynomial, BN254 as DoryBN254, G1Routines, G2Routines};
use dory_pcs::primitives::serialization::DorySerialize;
use dory_pcs::primitives::transcript::Transcript as DoryTranscript;
use dory_pcs::setup::{ProverSetup as DoryProverSetup, VerifierSetup as DoryVerifierSetup};
use dory_pcs::Polynomial;
use jolt_core::zkvm::lookup_table::LookupTables;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Blake2b256 = Blake2b<U32>;

#[cfg(feature = "e2e")]
mod e2e_verify;
#[cfg(not(feature = "e2e"))]
mod e2e_verify { pub fn main_from_cli() { panic!("e2e_verify requires: cargo run --features e2e -- ... e2e_verify"); } }

fn pow_u64<F: Field>(mut a: F, mut e: u64) -> F {
    let mut out = F::from(1u64);
    while e > 0 {
        if e & 1 == 1 {
            out *= a;
        }
        a *= a;
        e >>= 1;
    }
    out
}

fn limbs_csv<F: PrimeField>(x: F) -> String {
    x.into_bigint()
        .as_ref()
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn fq2_csv(x: Fq2) -> String {
    format!("{}/{}", limbs_csv(x.c0), limbs_csv(x.c1))
}

fn g1_csv(p: G1Affine) -> String {
    if p.infinity {
        "inf".to_string()
    } else {
        format!("{}:{}", limbs_csv(p.x), limbs_csv(p.y))
    }
}

fn g2_csv(p: G2Affine) -> String {
    if p.infinity {
        "inf".to_string()
    } else {
        format!("{}:{}", fq2_csv(p.x), fq2_csv(p.y))
    }
}

fn fq12_tower_vec(x: ark_bn254::Fq12) -> [Fq; 12] {
    // Flatten tower element c0 + c1*w with Fq6/Fq2 coefficients into 12 base-field limbs.
    let c0 = x.c0;
    let c1 = x.c1;
    [
        c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1, c1.c0.c0, c1.c0.c1,
        c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1,
    ]
}

fn fq12_poly_basis_matrix() -> &'static [[Fq; 12]; 12] {
    // Matrix M where column j is tower_vec(w^j) for w = Fq12::new(0,1).
    static M: OnceLock<[[Fq; 12]; 12]> = OnceLock::new();
    M.get_or_init(|| {
        fq12_poly_basis_matrix_for(ark_bn254::Fq12::new(Fq6::zero(), Fq6::one()))
    })
}

fn fq12_poly_basis_matrix_for(generator: ark_bn254::Fq12) -> [[Fq; 12]; 12] {
    // Matrix M where column j is tower_vec(gen^j).
    let mut cols = [[Fq::zero(); 12]; 12];
    let mut cur = ark_bn254::Fq12::one();
    for j in 0..12 {
        cols[j] = fq12_tower_vec(cur);
        cur *= generator;
    }
    let mut rows = [[Fq::zero(); 12]; 12];
    for r in 0..12 {
        for c in 0..12 {
            rows[r][c] = cols[c][r];
        }
    }
    rows
}

fn fq12_to_poly_coeffs(x: ark_bn254::Fq12) -> [Fq; 12] {
    // Solve M * a = tower_vec(x) over Fq to get coefficients a for Σ a_i * w^i.
    let m = fq12_poly_basis_matrix();
    fq12_to_poly_coeffs_for(m, x)
}

fn fq12_to_poly_coeffs_for(m: &[[Fq; 12]; 12], x: ark_bn254::Fq12) -> [Fq; 12] {
    // Solve M * a = tower_vec(x) over Fq for a.
    let b = fq12_tower_vec(x);
    let mut aug = [[Fq::zero(); 13]; 12];
    for r in 0..12 {
        for c in 0..12 {
            aug[r][c] = m[r][c];
        }
        aug[r][12] = b[r];
    }
    for col in 0..12 {
        let mut pivot = None;
        for r in col..12 {
            if !aug[r][col].is_zero() {
                pivot = Some(r);
                break;
            }
        }
        let p = pivot.expect("singular basis matrix");
        if p != col {
            aug.swap(p, col);
        }
        let inv = aug[col][col].inverse().expect("pivot invertible");
        for c in col..=12 {
            aug[col][c] *= inv;
        }
        for r in 0..12 {
            if r == col {
                continue;
            }
            let factor = aug[r][col];
            if factor.is_zero() {
                continue;
            }
            for c in col..=12 {
                aug[r][c] -= factor * aug[col][c];
            }
        }
    }
    let mut out = [Fq::zero(); 12];
    for i in 0..12 {
        out[i] = aug[i][12];
    }
    out
}

fn gt_poly_csv(x: ark_bn254::Fq12) -> String {
    let coeffs = fq12_to_poly_coeffs(x);
    coeffs.iter().map(|c| limbs_csv(*c)).collect::<Vec<_>>().join("/")
}

fn gt_poly_csv_for(m: &[[Fq; 12]; 12], x: ark_bn254::Fq12) -> String {
    let coeffs = fq12_to_poly_coeffs_for(m, x);
    coeffs.iter().map(|c| limbs_csv(*c)).collect::<Vec<_>>().join("/")
}

fn fq12_tower_csv(x: ark_bn254::Fq12) -> String {
    fq12_tower_vec(x)
        .iter()
        .map(|c| limbs_csv(*c))
        .collect::<Vec<_>>()
        .join("/")
}

#[derive(Clone)]
struct JoltLikeDoryTranscript {
    inner: Blake2bTranscript,
    serde_blocks: Vec<Vec<u8>>,
}

impl JoltLikeDoryTranscript {
    fn new(label: &[u8]) -> Self {
        Self {
            inner: Blake2bTranscript::new(label),
            serde_blocks: vec![],
        }
    }
}

impl DoryTranscript for JoltLikeDoryTranscript {
    type Curve = DoryBN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.inner.append_bytes(b"dory_bytes", bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &DoryFr) {
        self.inner.append_scalar_fr(b"dory_field", x.0);
    }

    fn append_group<G: dory_pcs::primitives::arithmetic::Group>(&mut self, _label: &[u8], _g: &G) {
        panic!("append_group not used by this oracle mode");
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buffer = Vec::new();
        s.serialize_compressed(&mut buffer).expect("serialize_compressed");
        self.serde_blocks.push(buffer.clone());
        self.inner.append_bytes(b"dory_serde", &buffer);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> DoryFr {
        DoryFr(self.inner.challenge_scalar_fr())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        panic!("reset not supported");
    }
}

fn run_dory_pcs_eval_blake2b() {
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
    // Emit instance and setup in a line-oriented key=value format for Python tests.
    println!("nu={nu}");
    println!("sigma={sigma}");
    println!("dory_layout=AddressMajor");
    println!("log_T={log_t}");
    let point_fr: Vec<Fr> = point_dory.iter().map(|x| x.0).collect();
    let mut reversed: Vec<Fr> = point_fr.iter().cloned().rev().collect();
    // Inverse of Jolt's AddressMajor reorder: opening_point_be = [address, cycle] where reversed = [cycle, address].
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

fn hex_to_bytes(s: &str) -> Vec<u8> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        panic!("hex string must have even length");
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..s.len()).step_by(2) {
        let hi = (bytes[i] as char).to_digit(16).unwrap();
        let lo = (bytes[i + 1] as char).to_digit(16).unwrap();
        out.push(((hi << 4) | lo) as u8);
    }
    out
}

fn bytes_to_hex(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len() * 2);
    for x in b {
        out.push_str(&format!("{:02x}", x));
    }
    out
}

#[derive(Clone)]
struct Blake2bTranscript {
    state: [u8; 32],
    n_rounds: u32,
}

impl Blake2bTranscript {
    fn new(label: &[u8]) -> Self {
        if label.len() > 32 {
            panic!("label must be <= 32 bytes");
        }
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let out: [u8; 32] = Blake2b256::new().chain_update(padded).finalize().into();
        Self {
            state: out,
            n_rounds: 0,
        }
    }

    fn round_tag(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[28..32].copy_from_slice(&self.n_rounds.to_be_bytes());
        out
    }

    fn hasher(&self) -> Blake2b256 {
        Blake2b256::new()
            .chain_update(self.state)
            .chain_update(self.round_tag())
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        let out: [u8; 32] = self.hasher().chain_update(bytes).finalize().into();
        self.update_state(out);
    }

    fn raw_append_label(&mut self, label: &[u8]) {
        if label.len() > 32 {
            panic!("label must be <= 32 bytes");
        }
        let mut packed = [0u8; 32];
        packed[..label.len()].copy_from_slice(label);
        self.raw_append_bytes(&packed);
    }

    fn raw_append_label_with_len(&mut self, label: &[u8], len: u64) {
        if label.len() > 24 {
            panic!("label must be <= 24 bytes for packed format");
        }
        let mut packed = [0u8; 32];
        packed[..label.len()].copy_from_slice(label);
        packed[24..32].copy_from_slice(&len.to_be_bytes());
        self.raw_append_bytes(&packed);
    }

    fn raw_append_u64(&mut self, x: u64) {
        let mut packed = [0u8; 32];
        packed[24..32].copy_from_slice(&x.to_be_bytes());
        self.raw_append_bytes(&packed);
    }

    fn raw_append_scalar_fr(&mut self, x: Fr) {
        let mut buf = x.into_bigint().to_bytes_le();
        buf.resize(32, 0u8);
        buf.reverse();
        self.raw_append_bytes(&buf);
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.raw_append_label_with_len(label, bytes.len() as u64);
        self.raw_append_bytes(bytes);
    }

    fn append_u64(&mut self, label: &[u8], x: u64) {
        self.raw_append_label(label);
        self.raw_append_u64(x);
    }

    fn append_scalar_fr(&mut self, label: &[u8], x: Fr) {
        self.raw_append_label(label);
        self.raw_append_scalar_fr(x);
    }

    fn append_scalars_fr(&mut self, label: &[u8], xs: &[Fr]) {
        self.raw_append_label_with_len(label, xs.len() as u64);
        for &x in xs {
            self.raw_append_scalar_fr(x);
        }
    }

    fn append_serializable_uncompressed(&mut self, label: &[u8], blob: &[u8]) {
        self.raw_append_label_with_len(label, blob.len() as u64);
        let mut b = blob.to_vec();
        b.reverse();
        self.raw_append_bytes(&b);
    }

    fn append_serializable_bytes_reversed(&mut self, label: &[u8], blob_reversed: &[u8]) {
        self.append_bytes(label, blob_reversed);
    }

    fn challenge_bytes32(&mut self) -> [u8; 32] {
        let out: [u8; 32] = self.hasher().finalize().into();
        self.update_state(out);
        out
    }

    fn challenge_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        let mut remaining = n;
        let mut start = 0usize;
        while remaining > 32 {
            let block = self.challenge_bytes32();
            out[start..start + 32].copy_from_slice(&block);
            start += 32;
            remaining -= 32;
        }
        let block = self.challenge_bytes32();
        out[start..start + remaining].copy_from_slice(&block[..remaining]);
        out
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut b = self.challenge_bytes(16);
        b.reverse();
        u128::from_be_bytes(b.try_into().unwrap())
    }

    fn challenge_scalar_fr(&mut self) -> Fr {
        let b = self.challenge_bytes(16);
        let x = u128::from_be_bytes(b.try_into().unwrap());
        Fr::from(x)
    }

    fn challenge_vector_fr(&mut self, n: usize) -> Vec<Fr> {
        (0..n).map(|_| self.challenge_scalar_fr()).collect()
    }

    fn challenge_scalar_powers_fr(&mut self, n: usize) -> Vec<Fr> {
        let q = self.challenge_scalar_fr();
        let mut out = vec![Fr::one(); n];
        for i in 1..n {
            out[i] = out[i - 1] * q;
        }
        out
    }
}

fn run_field<F>(input: &str)
where
    F: PrimeField + Field + FromStr,
    <F as FromStr>::Err: Debug,
{
    for line in input.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 3 {
            panic!("expected: <a> <b> <e>");
        }
        let a = F::from_str(parts[0]).unwrap();
        let b = F::from_str(parts[1]).unwrap();
        let e: u64 = parts[2].parse().unwrap();
        let add = a + b;
        let sub = a - b;
        let mul = a * b;
        let div = a * b.inverse().unwrap();
        let p = pow_u64(a, e);
        println!(
            "{}|{}|{}|{}|{}",
            limbs_csv(add),
            limbs_csv(sub),
            limbs_csv(mul),
            limbs_csv(div),
            limbs_csv(p)
        );
    }
}

fn run_curve(input: &str) {
    let g1 = G1Projective::generator();
    let g2 = G2Projective::generator();
    for line in input.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 2 {
            panic!("expected: <a> <b>");
        }
        let a = Fr::from_str(parts[0]).unwrap();
        let b = Fr::from_str(parts[1]).unwrap();
        let p_a = (g1 * a).into_affine();
        let p_b = (g1 * b).into_affine();
        let p_ab = (G1Projective::from(p_a) + G1Projective::from(p_b)).into_affine();
        let q_a = (g2 * a).into_affine();
        let q_b = (g2 * b).into_affine();
        let q_ab = (G2Projective::from(q_a) + G2Projective::from(q_b)).into_affine();
        let pair_rel = Bn254::pairing(p_a, q_b) == Bn254::pairing((g1 * (a * b)).into_affine(), g2.into_affine());
        let mpe_is_one =
            Bn254::multi_pairing([p_a, p_a], [q_b, (-G2Projective::from(q_b)).into_affine()]).0
                == <Bn254 as Pairing>::TargetField::one();
        println!(
            "{}	{}	{}	{}	{}	{}	{}	{}",
            g1_csv(p_a),
            g1_csv(p_b),
            g1_csv(p_ab),
            g2_csv(q_a),
            g2_csv(q_b),
            g2_csv(q_ab),
            if pair_rel { "1" } else { "0" },
            if mpe_is_one { "1" } else { "0" }
        );
    }
}

fn run_transcript_blake2b(input: &str) {
    let mut transcript: Option<Blake2bTranscript> = None;
    for (line_idx, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let op = parts[0];
        match op {
            "new" => {
                if parts.len() != 2 {
                    panic!("line {}: expected: new <label_ascii>", line_idx + 1);
                }
                transcript = Some(Blake2bTranscript::new(parts[1].as_bytes()));
            }
            "append_u64" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 3 {
                    panic!("line {}: expected: append_u64 <label_ascii> <u64_dec>", line_idx + 1);
                }
                t.append_u64(parts[1].as_bytes(), parts[2].parse::<u64>().unwrap());
            }
            "append_bytes" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 3 {
                    panic!("line {}: expected: append_bytes <label_ascii> <hex>", line_idx + 1);
                }
                let b = hex_to_bytes(parts[2]);
                t.append_bytes(parts[1].as_bytes(), &b);
            }
            "append_scalar_fr" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 3 {
                    panic!(
                        "line {}: expected: append_scalar_fr <label_ascii> <int_dec>",
                        line_idx + 1
                    );
                }
                let x = Fr::from_str(parts[2]).unwrap();
                t.append_scalar_fr(parts[1].as_bytes(), x);
            }
            "append_scalars_fr" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() < 4 {
                    panic!(
                        "line {}: expected: append_scalars_fr <label_ascii> <count> <int_dec...>",
                        line_idx + 1
                    );
                }
                let count = parts[2].parse::<usize>().unwrap();
                if parts.len() != 3 + count {
                    panic!("line {}: count does not match number of scalars", line_idx + 1);
                }
                let mut xs = Vec::with_capacity(count);
                for i in 0..count {
                    xs.push(Fr::from_str(parts[3 + i]).unwrap());
                }
                t.append_scalars_fr(parts[1].as_bytes(), &xs);
            }
            "append_serializable_uncompressed" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 3 {
                    panic!(
                        "line {}: expected: append_serializable_uncompressed <label_ascii> <hex>",
                        line_idx + 1
                    );
                }
                let b = hex_to_bytes(parts[2]);
                t.append_serializable_uncompressed(parts[1].as_bytes(), &b);
            }
            "append_serializable_bytes_reversed" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 3 {
                    panic!(
                        "line {}: expected: append_serializable_bytes_reversed <label_ascii> <hex>",
                        line_idx + 1
                    );
                }
                let b = hex_to_bytes(parts[2]);
                t.append_serializable_bytes_reversed(parts[1].as_bytes(), &b);
            }
            "challenge_u128" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                let x = t.challenge_u128();
                println!("challenge_u128={x}");
            }
            "challenge_scalar_fr" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                let x = t.challenge_scalar_fr();
                println!("challenge_fr={}", x);
            }
            "challenge_vector_fr" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 2 {
                    panic!("line {}: expected: challenge_vector_fr <len>", line_idx + 1);
                }
                let n = parts[1].parse::<usize>().unwrap();
                let xs = t.challenge_vector_fr(n);
                let s = xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
                println!("challenge_vec_fr={s}");
            }
            "challenge_scalar_powers_fr" => {
                let t = transcript.as_mut().expect("transcript not initialized; call new");
                if parts.len() != 2 {
                    panic!(
                        "line {}: expected: challenge_scalar_powers_fr <len>",
                        line_idx + 1
                    );
                }
                let n = parts[1].parse::<usize>().unwrap();
                let xs = t.challenge_scalar_powers_fr(n);
                let s = xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
                println!("challenge_pows_fr={s}");
            }
            _ => panic!("line {}: unknown op: {}", line_idx + 1, op),
        }
        let t = transcript.as_ref().expect("transcript not initialized; call new");
        println!("state={}", bytes_to_hex(&t.state));
    }
}

#[derive(Clone)]
struct CompressedUniPolyFr {
    coeffs_except_linear_term: Vec<Fr>, // [c0, c2, c3, ...]
}

impl CompressedUniPolyFr {
    fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }

    fn eval_from_hint(&self, hint: Fr, x: Fr) -> Fr {
        let c0 = self.coeffs_except_linear_term[0];
        let mut linear_term = hint - c0 - c0;
        for c in self.coeffs_except_linear_term.iter().skip(1) {
            linear_term -= *c;
        }

        let mut running_point = x;
        let mut running_sum = c0 + x * linear_term;
        for c in self.coeffs_except_linear_term.iter().skip(1) {
            running_point *= x;
            running_sum += *c * running_point;
        }
        running_sum
    }
}

fn run_sumcheck_verify_blake2b(input: &str) {
    let mut transcript: Option<Blake2bTranscript> = None;
    let mut claim: Option<Fr> = None;
    let mut degree_bound: Option<usize> = None;
    let mut polys: Vec<CompressedUniPolyFr> = vec![];

    for (line_idx, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "new" => {
                if parts.len() != 2 {
                    panic!("line {}: expected: new <label_ascii>", line_idx + 1);
                }
                transcript = Some(Blake2bTranscript::new(parts[1].as_bytes()));
                let t = transcript.as_ref().unwrap();
                println!("state={}", bytes_to_hex(&t.state));
            }
            "claim_fr" => {
                if parts.len() != 2 {
                    panic!("line {}: expected: claim_fr <int_dec>", line_idx + 1);
                }
                claim = Some(Fr::from_str(parts[1]).unwrap());
            }
            "degree_bound" => {
                if parts.len() != 2 {
                    panic!("line {}: expected: degree_bound <usize>", line_idx + 1);
                }
                degree_bound = Some(parts[1].parse::<usize>().unwrap());
            }
            "poly_fr" => {
                if parts.len() < 2 {
                    panic!(
                        "line {}: expected: poly_fr <c0> <c2> <c3> ...",
                        line_idx + 1
                    );
                }
                let mut coeffs = Vec::with_capacity(parts.len() - 1);
                for p in parts.iter().skip(1) {
                    coeffs.push(Fr::from_str(p).unwrap());
                }
                polys.push(CompressedUniPolyFr {
                    coeffs_except_linear_term: coeffs,
                });
            }
            "verify" => {
                let t = transcript
                    .as_mut()
                    .expect("transcript not initialized; call new");
                let mut e = claim.expect("missing claim_fr");
                let degree_bound = degree_bound.expect("missing degree_bound");

                for (i, poly) in polys.iter().enumerate() {
                    if poly.degree() > degree_bound {
                        panic!("degree bound exceeded");
                    }
                    t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
                    let r_u128 = t.challenge_u128(); // optimized challenge uses u128 (little-endian interpretation)
                    let r_i = Fr::from(r_u128);
                    e = poly.eval_from_hint(e, r_i);
                    println!("round={i} r={r_i} e={e} state={}", bytes_to_hex(&t.state));
                }
                println!("output_claim={e}");
            }
            _ => panic!(
                "line {}: unknown op for sumcheck_verify_blake2b: {}",
                line_idx + 1,
                parts[0]
            ),
        }
    }
}

fn parse_lookup_table_64(name: &str) -> LookupTables<64> {
    match name {
        "RangeCheck" => LookupTables::RangeCheck(Default::default()),
        "RangeCheckAligned" => LookupTables::RangeCheckAligned(Default::default()),
        "And" => LookupTables::And(Default::default()),
        "Andn" => LookupTables::Andn(Default::default()),
        "Or" => LookupTables::Or(Default::default()),
        "Xor" => LookupTables::Xor(Default::default()),
        "Equal" => LookupTables::Equal(Default::default()),
        "SignedGreaterThanEqual" => LookupTables::SignedGreaterThanEqual(Default::default()),
        "UnsignedGreaterThanEqual" => LookupTables::UnsignedGreaterThanEqual(Default::default()),
        "NotEqual" => LookupTables::NotEqual(Default::default()),
        "SignedLessThan" => LookupTables::SignedLessThan(Default::default()),
        "UnsignedLessThan" => LookupTables::UnsignedLessThan(Default::default()),
        "Movsign" => LookupTables::Movsign(Default::default()),
        "UpperWord" => LookupTables::UpperWord(Default::default()),
        "LessThanEqual" => LookupTables::LessThanEqual(Default::default()),
        "ValidSignedRemainder" => LookupTables::ValidSignedRemainder(Default::default()),
        "ValidUnsignedRemainder" => LookupTables::ValidUnsignedRemainder(Default::default()),
        "ValidDiv0" => LookupTables::ValidDiv0(Default::default()),
        "HalfwordAlignment" => LookupTables::HalfwordAlignment(Default::default()),
        "WordAlignment" => LookupTables::WordAlignment(Default::default()),
        "LowerHalfWord" => LookupTables::LowerHalfWord(Default::default()),
        "SignExtendHalfWord" => LookupTables::SignExtendHalfWord(Default::default()),
        "Pow2" => LookupTables::Pow2(Default::default()),
        "Pow2W" => LookupTables::Pow2W(Default::default()),
        "ShiftRightBitmask" => LookupTables::ShiftRightBitmask(Default::default()),
        "VirtualRev8W" => LookupTables::VirtualRev8W(Default::default()),
        "VirtualSRL" => LookupTables::VirtualSRL(Default::default()),
        "VirtualSRA" => LookupTables::VirtualSRA(Default::default()),
        "VirtualROTR" => LookupTables::VirtualROTR(Default::default()),
        "VirtualROTRW" => LookupTables::VirtualROTRW(Default::default()),
        "VirtualChangeDivisor" => LookupTables::VirtualChangeDivisor(Default::default()),
        "VirtualChangeDivisorW" => LookupTables::VirtualChangeDivisorW(Default::default()),
        "MulUNoOverflow" => LookupTables::MulUNoOverflow(Default::default()),
        "VirtualXORROT32" => LookupTables::VirtualXORROT32(Default::default()),
        "VirtualXORROT24" => LookupTables::VirtualXORROT24(Default::default()),
        "VirtualXORROT16" => LookupTables::VirtualXORROT16(Default::default()),
        "VirtualXORROT63" => LookupTables::VirtualXORROT63(Default::default()),
        "VirtualXORROTW16" => LookupTables::VirtualXORROTW16(Default::default()),
        "VirtualXORROTW12" => LookupTables::VirtualXORROTW12(Default::default()),
        "VirtualXORROTW8" => LookupTables::VirtualXORROTW8(Default::default()),
        "VirtualXORROTW7" => LookupTables::VirtualXORROTW7(Default::default()),
        _ => panic!("unknown lookup table: {name}"),
    }
}

fn run_lookup_table_mle_64(input: &str) {
    for (line_idx, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Format: "<TableName> <r_csv>"
        // where r_csv is comma-separated field elements (tests pass 0/1).
        let mut parts = line.split_whitespace();
        let table_name = parts
            .next()
            .unwrap_or_else(|| panic!("line {}: expected table name", line_idx + 1));
        let r_csv = parts
            .next()
            .unwrap_or_else(|| panic!("line {}: expected r_csv", line_idx + 1));
        if parts.next().is_some() {
            panic!("line {}: too many tokens; expected: <TableName> <r_csv>", line_idx + 1);
        }

        let table = parse_lookup_table_64(table_name);
        let r: Vec<Fr> = if r_csv.trim().is_empty() {
            vec![]
        } else {
            r_csv
                .split(',')
                .map(|t| Fr::from_str(t.trim()).unwrap())
                .collect()
        };
        let out: Fr = table.evaluate_mle::<Fr, Fr>(&r);
        println!("{}", limbs_csv(out));
    }
}

fn fr_from_i128(x: i128) -> Fr {
    if x == 0 {
        return Fr::zero();
    }
    if x > 0 && x <= (u64::MAX as i128) {
        return Fr::from(x as u64);
    }
    if x < 0 && -x <= (u64::MAX as i128) {
        return -Fr::from((-x) as u64);
    }
    Fr::from_str(&x.to_string()).unwrap()
}

fn fr_factorial(n: usize) -> Fr {
    (1..=n).fold(Fr::one(), |acc, i| acc * Fr::from(i as u64))
}

fn lagrange_evals_symmetric(r: Fr, n: usize) -> Vec<Fr> {
    assert!(n > 0 && n <= 20);
    let start: i64 = -(((n - 1) / 2) as i64);
    let nodes: Vec<Fr> = (0..n).map(|i| fr_from_i128((start + i as i64) as i128)).collect();
    for (i, xi) in nodes.iter().enumerate() {
        if *xi == r {
            let mut out = vec![Fr::zero(); n];
            out[i] = Fr::one();
            return out;
        }
    }
    let weights: Vec<Fr> = (0..n)
        .map(|i| {
            let sign = if ((n - 1 - i) & 1) == 1 { -Fr::one() } else { Fr::one() };
            let denom = fr_factorial(i) * fr_factorial(n - 1 - i);
            sign * denom.inverse().unwrap()
        })
        .collect();
    let mut terms: Vec<Fr> = vec![Fr::zero(); n];
    let mut s = Fr::zero();
    for i in 0..n {
        let di = r - nodes[i];
        let t = weights[i] * di.inverse().unwrap();
        terms[i] = t;
        s += t;
    }
    let inv_s = s.inverse().unwrap();
    terms.into_iter().map(|t| t * inv_s).collect()
}

fn lagrange_kernel_symmetric(x: Fr, y: Fr, n: usize) -> Fr {
    let lx = lagrange_evals_symmetric(x, n);
    let ly = lagrange_evals_symmetric(y, n);
    lx.into_iter().zip(ly).map(|(a, b)| a * b).sum()
}

fn eq_mle(x: &[Fr], y: &[Fr]) -> Fr {
    assert_eq!(x.len(), y.len());
    let one = Fr::one();
    x.iter()
        .zip(y.iter())
        .fold(Fr::one(), |acc, (xi, yi)| acc * (*xi * *yi + (one - *xi) * (one - *yi)))
}

fn eq_plus_one_mle(x: &[Fr], y: &[Fr]) -> Fr {
    // Mirrors Python `EqPlusOnePolynomial.evaluate` (big-endian bit order).
    assert_eq!(x.len(), y.len());
    let l = x.len();
    let one = Fr::one();
    let mut total = Fr::zero();
    for k in 0..l {
        let mut lower = Fr::one();
        for i in 0..k {
            let xi = x[l - 1 - i];
            let yi = y[l - 1 - i];
            lower *= xi * (one - yi);
        }
        let kth = (one - x[l - 1 - k]) * y[l - 1 - k];
        let mut higher = Fr::one();
        for i in (k + 1)..l {
            let xi = x[l - 1 - i];
            let yi = y[l - 1 - i];
            higher *= xi * yi + (one - xi) * (one - yi);
        }
        total += lower * kth * higher;
    }
    total
}

fn remap_address_oracle(address: u64, lowest: u64) -> Option<usize> {
    if address == 0 {
        return None;
    }
    if address < lowest {
        panic!("unexpected address below lowest");
    }
    Some(((address - lowest) / 8) as usize)
}

fn eq_at_index(r: &[Fr], idx: usize) -> Fr {
    let n = r.len();
    let mut bits = Vec::with_capacity(n);
    for i in 0..n {
        let bit = ((idx >> (n - 1 - i)) & 1) == 1;
        bits.push(if bit { Fr::one() } else { Fr::zero() });
    }
    eq_mle(r, &bits)
}

fn sparse_eval_u64_block_oracle(start_index: usize, values: &[u64], r: &[Fr]) -> Fr {
    let mut acc = Fr::zero();
    for (j, v) in values.iter().enumerate() {
        if *v == 0 {
            continue;
        }
        acc += Fr::from(*v) * eq_at_index(r, start_index + j);
    }
    acc
}

fn eval_initial_ram_mle_oracle(
    lowest_addr: u64,
    min_bytecode_address: u64,
    bytecode_words: &[u64],
    input_start: u64,
    inputs_words: &[u64],
    r_address: &[Fr],
) -> Fr {
    let bytecode_start = remap_address_oracle(min_bytecode_address, lowest_addr).unwrap();
    let mut acc = sparse_eval_u64_block_oracle(bytecode_start, bytecode_words, r_address);
    if !inputs_words.is_empty() {
        let input_start_idx = remap_address_oracle(input_start, lowest_addr).unwrap();
        acc += sparse_eval_u64_block_oracle(input_start_idx, inputs_words, r_address);
    }
    acc
}

fn calculate_advice_memory_evaluation_oracle(
    eval: Fr,
    advice_num_vars: usize,
    advice_start: u64,
    lowest_addr: u64,
    r_address: &[Fr],
    total_memory_vars: usize,
) -> Fr {
    let num_missing_vars = total_memory_vars - advice_num_vars;
    let index = remap_address_oracle(advice_start, lowest_addr).unwrap();
    let mut scaling_factor = Fr::one();
    let mut index_binary = Vec::with_capacity(total_memory_vars);
    for i in (0..total_memory_vars).rev() {
        index_binary.push(((index >> i) & 1) == 1);
    }
    for (i, bit) in index_binary.iter().take(num_missing_vars).enumerate() {
        scaling_factor *= if *bit { r_address[i] } else { Fr::one() - r_address[i] };
    }
    eval * scaling_factor
}

fn run_spartan_outer_stage1_blake2b() {
    // Deterministic, Rust-side oracle for the Python SpartanOuter Stage-1 verifier prototype.
    // This intentionally mirrors the Python verifier structure (transcript coupling + expected-claim formula),
    // without pulling in the full jolt-core crate.
    let trace_len = 8usize;
    let num_cycles_bits = trace_len.ilog2() as usize;
    let num_rows_bits = num_cycles_bits + 2;
    let outer_domain_size = 10usize; // OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE for 19 constraints
    let first_round_num_coeffs = 28usize; // OUTER_FIRST_ROUND_POLY_NUM_COEFFS for degree=9

    let mut t = Blake2bTranscript::new(b"Jolt");

    // tau := challenge_vector_optimized(num_rows_bits)
    let mut tau: Vec<Fr> = Vec::with_capacity(num_rows_bits);
    for _ in 0..num_rows_bits {
        let u = t.challenge_u128();
        tau.push(Fr::from(u));
    }

    // Uni-skip polynomial: all zeros (degree <= bound, sum over base domain = 0).
    let uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &uni_poly_coeffs);
    let r0 = Fr::from(t.challenge_u128());

    // UnivariateSkip opening claim (seeded externally in real proofs). Keep 0 for simplicity.
    let uniskip_claim = Fr::zero();
    t.append_scalar_fr(b"opening_claim", uniskip_claim);

    // Stage 1 remaining sumcheck: single instance, so batch coeff is a single challenge_scalar (non-optimized).
    t.append_scalar_fr(b"sumcheck_claim", uniskip_claim);
    let batch_coeff = t.challenge_scalar_fr();
    let mut e = uniskip_claim * batch_coeff;

    // Provide 1 + log2(trace_len) compressed round polynomials (degree<=3), deterministic constants.
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
        let rj = Fr::from(t.challenge_u128());
        e = poly.eval_from_hint(e, rj);
        polys.push(poly);
        r_sumcheck.push(rj);
    }
    let output_claim = e;

    // Compute multiplicative factor: L(tau_high, r0) * Eq(tau_low, reverse(r_sumcheck))
    let tau_high = tau[tau.len() - 1];
    let tau_low = &tau[..tau.len() - 1];
    let mut r_rev = r_sumcheck.clone();
    r_rev.reverse();
    let lag = lagrange_kernel_symmetric(tau_high, r0, outer_domain_size);
    let eq = eq_mle(tau_low, &r_rev);
    let factor = lag * eq;
    assert!(factor != Fr::zero(), "unexpected zero factor; tweak constants if this trips");
    let target_inner = output_claim * (batch_coeff * factor).inverse().unwrap();

    // Indices into ALL_R1CS_INPUTS (must match Python `r1cs.py` ordering).
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
    let i_cf_write_lookup_to_rd = idx(29);
    let i_virtual = idx(30);
    let i_assert = idx(31);
    let i_dnu = idx(32);
    let i_advice = idx(33);
    let i_is_compressed = idx(34);
    let i_is_first = idx(35);
    let i_is_last = idx(36);

    // LC eval on z (inputs + const=1).
    let lc_eval = |terms: &[(usize, i128)], c: i128, inputs: &[Fr]| -> Fr {
        let mut acc = fr_from_i128(c);
        for (idx, coeff) in terms.iter() {
            acc += fr_from_i128(*coeff) * inputs[*idx];
        }
        acc
    };

    // Constraint groups (A,B) with A=condition LC and B=(left-right) LC.
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

    // Build inputs so that az_final=1 and bz_final=target_inner (solve 2x2 system).
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

    // Cache openings: append 37 opening_claims (in canonical input order) after sumcheck challenges.
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

fn run_stage2_sumchecks_blake2b() {
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
        tau.push(Fr::from(t.challenge_u128()));
    }

    let stage1_uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); outer_first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &stage1_uni_poly_coeffs);
    let stage1_r0 = Fr::from(t.challenge_u128());

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
        let rj = Fr::from(t.challenge_u128());
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
    let tau_high_pv = Fr::from(t.challenge_u128());
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
    let stage2_r0 = Fr::from(t.challenge_u128());

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
        stage2_r_address.push(Fr::from(t.challenge_u128()));
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
        let rj = Fr::from(t.challenge_u128());
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

fn run_stage3_sumchecks_blake2b() {
    // Deterministic Rust-side oracle for Python Stage1+Stage2+Stage3 verifier coupling.
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
        tau.push(Fr::from(t.challenge_u128()));
    }

    let stage1_uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); outer_first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &stage1_uni_poly_coeffs);
    let stage1_r0 = Fr::from(t.challenge_u128());

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
        let rj = Fr::from(t.challenge_u128());
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
    let tau_high_pv = Fr::from(t.challenge_u128());
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
    let stage2_r0 = Fr::from(t.challenge_u128());

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
        stage2_r_address.push(Fr::from(t.challenge_u128()));
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
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2, c3],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = Fr::from(t.challenge_u128());
        e2 = poly.eval_from_hint(e2, rj);
        stage2_polys.push(poly);
        stage2_r_sumcheck.push(rj);
    }
    let stage2_output_claim = e2;

    // r_cycle_stage2 (BE) := pv remainder's opening point (same slice as instruction claim reduction)
    let instr_slice = &stage2_r_sumcheck[log_k..log_k + log_t];
    let mut r_cycle_stage2_be: Vec<Fr> = instr_slice.to_vec();
    r_cycle_stage2_be.reverse();

    // Choose InstructionClaimReduction as the only nonzero expected term.
    let eq_instr = eq_mle(&r_cycle_stage2_be, &r_cycle_be);
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
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2, c3],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = Fr::from(t.challenge_u128());
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

    for v in stage3_shift_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }
    for v in stage3_instr_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }
    for v in stage3_regs_claims.iter() {
        t.append_scalar_fr(b"opening_claim", *v);
    }

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

    println!("final_state={final_state}");
}

fn run_stage4_sumchecks_blake2b_old() {
    // Deterministic Rust-side oracle for Python Stage1+Stage2+Stage3+Stage4 verifier coupling.
    // Targets transcript/claim wiring parity (not soundness).
    //
    // Stage4 mirrors Rust verifier ordering:
    // - verifier_accumulate_advice (optional)
    // - RegistersReadWriteChecking (samples gamma)
    // - RamValEvaluation
    // - ValFinal
    let trace_len = 8usize;
    let num_cycles_bits = trace_len.ilog2() as usize;
    let num_rows_bits = num_cycles_bits + 2;
    let outer_domain_size = 10usize; // OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE for 19 constraints
    let outer_first_round_num_coeffs = 28usize; // OUTER_FIRST_ROUND_POLY_NUM_COEFFS

    // Stage2 parameters
    let ram_k = 8usize;
    let log_k = ram_k.ilog2() as usize;
    let log_t = trace_len.ilog2() as usize;
    let rw_phase1 = log_t; // force "single advice opening" behavior
    let rw_phase2 = 1usize;
    let max_rounds_stage2 = log_k + log_t;

    // Stage4 registers RW phases
    let regs_rw_phase1 = 1usize;
    let regs_rw_phase2 = 1usize;
    let regs_log_k = 5usize;
    let max_rounds_stage4 = regs_log_k + log_t;

    // Memory layout
    let ram_start_addr: u64 = 0x8000_0000;
    let lowest_addr: u64 = ram_start_addr - 8;
    let input_start: u64 = ram_start_addr;
    let output_start: u64 = lowest_addr;
    let panic_addr: u64 = lowest_addr;
    let termination_addr: u64 = lowest_addr;

    // Advice regions (keep within the RAM domain [lowest..lowest+K*8))
    let untrusted_advice_start: u64 = lowest_addr; // index=0
    let max_untrusted_advice_size: u64 = 8 * 2; // 2 u64 words -> next_pow2=2 -> num_vars=1
    let trusted_advice_start: u64 = lowest_addr + 8; // index=1 (forces different selector)
    let max_trusted_advice_size: u64 = 8 * 2;

    let has_untrusted_advice_commitment = true;
    let has_trusted_advice_commitment = true;

    // Minimal public preprocessing for init-eval
    let min_bytecode_address: u64 = ram_start_addr; // remaps to 1
    let bytecode_words: Vec<u64> = vec![5u64];
    let inputs_words: Vec<u64> = vec![9u64];

    let mut t = Blake2bTranscript::new(b"Jolt");

    // ----------------
    // Stage 1 (same structure as run_stage3_sumchecks_blake2b)
    // ----------------
    let mut tau: Vec<Fr> = Vec::with_capacity(num_rows_bits);
    for _ in 0..num_rows_bits {
        tau.push(Fr::from(t.challenge_u128()));
    }

    let stage1_uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); outer_first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &stage1_uni_poly_coeffs);
    let stage1_r0 = Fr::from(t.challenge_u128());

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
        let rj = Fr::from(t.challenge_u128());
        e = poly.eval_from_hint(e, rj);
        stage1_polys.push(poly);
        stage1_r_sumcheck.push(rj);
    }
    let stage1_output_claim = e;

    let tau_high = tau[tau.len() - 1];
    let tau_low = &tau[..tau.len() - 1];
    let mut r_rev = stage1_r_sumcheck.clone();
    r_rev.reverse();
    let lag = lagrange_kernel_symmetric(tau_high, stage1_r0, outer_domain_size);
    let eq = eq_mle(tau_low, &r_rev);
    let factor = lag * eq;
    let target_inner = stage1_output_claim * (stage1_batch_coeff * factor).inverse().unwrap();

    // Solve 37 R1CS input evals (reuse helper from stage3 oracle)
    let r1cs_inputs = solve_r1cs_inputs_for_target_inner(stage1_r0, outer_domain_size, stage1_r_sumcheck[0], target_inner);

    // ----------------
    // Stage 2 (same structure as stage3 oracle, but with rw_phase1=log_t)
    // ----------------
    // Seed Stage2 virtual/committed openings
    let pv_factor_claims = vec![Fr::zero(); 2];
    let instr_left_claim = Fr::zero();
    let instr_right_claim = Fr::zero();
    let raf_ra_claim = Fr::zero();
    let out_val_init_claim = Fr::zero();

    // We'll set RamValFinal to match its init-eval later (so Stage4 input_claim=0).
    // For Stage2 OutputCheck expected term we keep everything zero by setting all relevant claims to 0.
    let out_val_final_claim = Fr::zero();

    // We'll set RamVal to match its init-eval later (so Stage4 input_claim=0).
    let ramrw_val_claim_placeholder = Fr::zero();
    let ramrw_ra_claim = Fr::zero();
    let ramrw_raminc_claim = Fr::zero();

    // Committed: only RamInc used in Stage2/4; other committed polys are dense.

    // Stage2 UnivariateSkip (proof shape only)
    let stage2_uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); outer_first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &stage2_uni_poly_coeffs);
    let stage2_r0 = Fr::from(t.challenge_u128());

    let stage2_uniskip_claim = Fr::zero();
    t.append_scalar_fr(b"opening_claim", stage2_uniskip_claim);

    // Stage2 batched sumcheck claim = Σ coeff_i * input_claim_i * 2^{max_rounds - m_i}
    // We make all instance input claims 0 by seeding SpartanOuter openings to 0.
    // (The verifier only checks transcript coupling; sumcheck soundness isn't targeted.)
    let stage2_num_instances = 5usize;
    let stage2_input_claims = vec![Fr::zero(); stage2_num_instances];
    for c in stage2_input_claims.iter() {
        t.append_scalar_fr(b"sumcheck_claim", *c);
    }
    let mut stage2_batch_coeffs = Vec::with_capacity(stage2_num_instances);
    for _ in 0..stage2_num_instances {
        stage2_batch_coeffs.push(t.challenge_scalar_fr());
    }
    let mut stage2_claim = Fr::zero();
    for (i, c) in stage2_input_claims.iter().enumerate() {
        // stage2 max rounds = log_k+log_t; each instance m_i differs; we keep c=0 so claim=0.
        stage2_claim += *c * stage2_batch_coeffs[i];
    }

    // Stage2 proof polys
    let mut stage2_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(max_rounds_stage2);
    let mut stage2_r_sumcheck: Vec<Fr> = Vec::with_capacity(max_rounds_stage2);
    let mut e2 = stage2_claim;
    for j in 0..max_rounds_stage2 {
        let c0 = Fr::from((4000 + j) as u64);
        let c2 = Fr::from((5000 + j) as u64);
        let c3 = Fr::from((6000 + j) as u64);
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2, c3],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = Fr::from(t.challenge_u128());
        e2 = poly.eval_from_hint(e2, rj);
        stage2_polys.push(poly);
        stage2_r_sumcheck.push(rj);
    }
    let stage2_output_claim = e2;

    // Choose Stage2 expected output to be carried by InstructionClaimReduction instance only, as in stage3 oracle.
    // This requires setting a single pv_factor_claim nonzero.
    let coeff_instr = stage2_batch_coeffs[3];
    let pv0 = stage2_output_claim * coeff_instr.inverse().unwrap();
    let pv_factor_claims = vec![pv0, Fr::zero()];

    // Cache openings coupling for Stage2 (opening_claim appends) — order matches Python verifiers.
    // For simplicity we cache zero/placeholder claims now; the actual values are printed for Python to seed.
    // Uniskip:
    // already appended stage2_uniskip_claim above.

    // ----------------
    // Stage 3 (reuse stage3 oracle structure, keeping everything zero so transcript shape matches)
    // ----------------
    // NOTE: Stage3 uses its own gamma draws. Keep openings at zero so expected outputs are 0.
    // Batch with 3 instances; input claims all 0.
    let stage3_num_rounds = log_t; // shift has log_t rounds
    let stage3_input_claims = vec![Fr::zero(); 3];
    for c in stage3_input_claims.iter() {
        t.append_scalar_fr(b"sumcheck_claim", *c);
    }
    let mut stage3_batch_coeffs = Vec::with_capacity(3);
    for _ in 0..3 {
        stage3_batch_coeffs.push(t.challenge_scalar_fr());
    }
    let mut e3 = Fr::zero();
    let mut stage3_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(stage3_num_rounds);
    let mut stage3_r_sumcheck: Vec<Fr> = Vec::with_capacity(stage3_num_rounds);
    for j in 0..stage3_num_rounds {
        let c0 = Fr::from((7000 + j) as u64);
        let c2 = Fr::from((8000 + j) as u64);
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = Fr::from(t.challenge_u128());
        e3 = poly.eval_from_hint(e3, rj);
        stage3_polys.push(poly);
        stage3_r_sumcheck.push(rj);
    }
    let stage3_output_claim = e3;
    // Make Stage3 expected carried by RegistersClaimReduction (3rd) with rd_write_value claim.
    let coeff_regs = stage3_batch_coeffs[2];
    let stage3_rd_write_claim = stage3_output_claim * coeff_regs.inverse().unwrap();

    // ----------------
    // Stage 4: advice accumulation + batched sumcheck
    // ----------------
    // Compute r_address from Stage2 RamReadWriteChecking opening point (for init-eval).
    // Stage2 r_sumcheck challenges are BE already in our oracles; normalize to BE by reversing.
    let mut stage2_r_rev = stage2_r_sumcheck.clone();
    stage2_r_rev.reverse();
    let r_address = &stage2_r_rev[..log_k];
    let total_memory_vars = log_k;

    // Advice opening points are suffixes of r_address; advice_num_vars=1 => last bit.
    let advice_num_vars = 1usize;
    let untrusted_eval = Fr::from(11u64);
    let trusted_eval = Fr::from(17u64);
    let untrusted_contrib = calculate_advice_memory_evaluation_oracle(
        untrusted_eval,
        advice_num_vars,
        untrusted_advice_start,
        lowest_addr,
        r_address,
        total_memory_vars,
    );
    let trusted_contrib = calculate_advice_memory_evaluation_oracle(
        trusted_eval,
        advice_num_vars,
        trusted_advice_start,
        lowest_addr,
        r_address,
        total_memory_vars,
    );
    let public_contrib = eval_initial_ram_mle_oracle(
        lowest_addr,
        min_bytecode_address,
        &bytecode_words,
        input_start,
        &inputs_words,
        r_address,
    );
    let init_eval = untrusted_contrib + trusted_contrib + public_contrib;

    // Seed Stage2 RamVal claim so Stage4 RamValEvaluation input_claim=0.
    let ramrw_val_claim = init_eval;
    // Seed Stage2 RamValFinal claim so Stage4 ValFinal input_claim=0.
    let out_val_final_claim = init_eval;

    // Advice accumulation transcript coupling: append opening_claim for advice claims.
    // single_opening=true due to rw_phase1==log_t => only RamValEvaluation advice openings appended.
    t.append_scalar_fr(b"opening_claim", untrusted_eval);
    t.append_scalar_fr(b"opening_claim", trusted_eval);

    // RegistersRW gamma (sampled after advice accumulation)
    let regs_gamma = t.challenge_scalar_fr();

    // Stage4 input claims: all zero (by construction)
    let regs_input_claim = Fr::zero(); // rd/rs1/rs2 claims are zero except rd in stage3, but we keep those separate from Stage4 by seeding stage3 sumcheck openings to 0 in Python test.
    let ram_val_eval_input_claim = Fr::zero();
    let val_final_input_claim = Fr::zero();

    t.append_scalar_fr(b"sumcheck_claim", regs_input_claim);
    t.append_scalar_fr(b"sumcheck_claim", ram_val_eval_input_claim);
    t.append_scalar_fr(b"sumcheck_claim", val_final_input_claim);

    let stage4_batch_coeffs = vec![t.challenge_scalar_fr(), t.challenge_scalar_fr(), t.challenge_scalar_fr()];
    let stage4_claim = Fr::zero();

    let mut stage4_polys: Vec<CompressedUniPolyFr> = Vec::with_capacity(max_rounds_stage4);
    let mut stage4_r_sumcheck: Vec<Fr> = Vec::with_capacity(max_rounds_stage4);
    let mut e4 = stage4_claim;
    for j in 0..max_rounds_stage4 {
        let c0 = Fr::from((9000 + j) as u64);
        let c2 = Fr::from((10000 + j) as u64);
        let c3 = Fr::from((11000 + j) as u64);
        let poly = CompressedUniPolyFr {
            coeffs_except_linear_term: vec![c0, c2, c3],
        };
        t.append_scalars_fr(b"sumcheck_poly", &poly.coeffs_except_linear_term);
        let rj = Fr::from(t.challenge_u128());
        e4 = poly.eval_from_hint(e4, rj);
        stage4_polys.push(poly);
        stage4_r_sumcheck.push(rj);
    }
    let stage4_output_claim = e4;

    // Choose Stage4 expected output carried by ValFinal: expected = coeff3 * (inc * wa).
    let coeff_val_final = stage4_batch_coeffs[2];
    let val_final_inc_claim = stage4_output_claim * coeff_val_final.inverse().unwrap();
    let val_final_wa_claim = Fr::one();

    // Other Stage4 cached openings are zero.
    let regs_val_claim = Fr::zero();
    let regs_rs1_ra_claim = Fr::zero();
    let regs_rs2_ra_claim = Fr::zero();
    let regs_rd_wa_claim = Fr::zero();
    let regs_rdinc_claim = Fr::zero();

    let ram_val_eval_wa_claim = Fr::zero();
    let ram_val_eval_inc_claim = Fr::zero();

    // Transcript coupling for Stage4 cache openings (opening_claim appends) in correct order:
    // registers: 4 virtual + 1 dense
    t.append_scalar_fr(b"opening_claim", regs_val_claim);
    t.append_scalar_fr(b"opening_claim", regs_rs1_ra_claim);
    t.append_scalar_fr(b"opening_claim", regs_rs2_ra_claim);
    t.append_scalar_fr(b"opening_claim", regs_rd_wa_claim);
    t.append_scalar_fr(b"opening_claim", regs_rdinc_claim);
    // ram val eval: 1 virtual + 1 dense
    t.append_scalar_fr(b"opening_claim", ram_val_eval_wa_claim);
    t.append_scalar_fr(b"opening_claim", ram_val_eval_inc_claim);
    // val final: 1 dense + 1 virtual
    t.append_scalar_fr(b"opening_claim", val_final_inc_claim);
    t.append_scalar_fr(b"opening_claim", val_final_wa_claim);

    let final_state = bytes_to_hex(&t.state);

    // ----------------
    // Print KVs consumed by Python test
    // ----------------
    println!("trace_len={trace_len}");
    println!("ram_k={ram_k}");
    println!("rw_phase1={rw_phase1}");
    println!("rw_phase2={rw_phase2}");
    println!("regs_rw_phase1={regs_rw_phase1}");
    println!("regs_rw_phase2={regs_rw_phase2}");

    println!("lowest_addr={lowest_addr}");
    println!("input_start={input_start}");
    println!("output_start={output_start}");
    println!("panic_addr={panic_addr}");
    println!("termination_addr={termination_addr}");
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

    // Stage1 artifacts (same keys as stage3 oracle)
    println!(
        "stage1_tau={}",
        tau.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
    );
    println!(
        "stage1_uniskip_poly_coeffs={}",
        stage1_uni_poly_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage1_r0={stage1_r0}");
    for (j, p) in stage1_polys.iter().enumerate() {
        println!(
            "stage1_sumcheck_poly_{j}={}",
            p.coeffs_except_linear_term
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!(
        "stage1_r_sumcheck={}",
        stage1_r_sumcheck
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage1_uniskip_claim={stage1_uniskip_claim}");
    println!("stage1_batch_coeff={stage1_batch_coeff}");
    println!("stage1_r1cs_inputs={}", r1cs_inputs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));

    // Stage2 artifacts (minimal set for Python Stage2 verifier + transcript parity)
    println!(
        "stage2_uniskip_poly_coeffs={}",
        stage2_uni_poly_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage2_r0={stage2_r0}");
    println!("stage2_uniskip_claim={stage2_uniskip_claim}");
    for (j, p) in stage2_polys.iter().enumerate() {
        println!(
            "stage2_sumcheck_poly_{j}={}",
            p.coeffs_except_linear_term
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!(
        "stage2_r_sumcheck={}",
        stage2_r_sumcheck
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "stage2_batch_coeffs={}",
        stage2_batch_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage2_r1cs_inputs={}", r1cs_inputs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
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

    // Stage3 artifacts (minimal; we only need sumcheck proof polys and rd-write claim)
    for (j, p) in stage3_polys.iter().enumerate() {
        println!(
            "stage3_sumcheck_poly_{j}={}",
            p.coeffs_except_linear_term
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!(
        "stage3_r_sumcheck={}",
        stage3_r_sumcheck
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "stage3_batch_coeffs={}",
        stage3_batch_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stage3_rd_write_claim={stage3_rd_write_claim}");
    println!("regs_gamma={regs_gamma}");
    println!("untrusted_advice_eval={untrusted_eval}");
    println!("trusted_advice_eval={trusted_eval}");

    // Stage4 sumcheck proof
    for (j, p) in stage4_polys.iter().enumerate() {
        println!(
            "stage4_sumcheck_poly_{j}={}",
            p.coeffs_except_linear_term
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!(
        "stage4_r_sumcheck={}",
        stage4_r_sumcheck
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "stage4_batch_coeffs={}",
        stage4_batch_coeffs
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    println!("stage4_regs_val_claim={regs_val_claim}");
    println!("stage4_regs_rs1_ra_claim={regs_rs1_ra_claim}");
    println!("stage4_regs_rs2_ra_claim={regs_rs2_ra_claim}");
    println!("stage4_regs_rd_wa_claim={regs_rd_wa_claim}");
    println!("stage4_regs_rdinc_claim={regs_rdinc_claim}");
    println!("stage4_ram_val_eval_wa_claim={ram_val_eval_wa_claim}");
    println!("stage4_ram_val_eval_inc_claim={ram_val_eval_inc_claim}");
    println!("stage4_val_final_inc_claim={val_final_inc_claim}");
    println!("stage4_val_final_wa_claim={val_final_wa_claim}");
    println!("final_state={final_state}");
}

fn solve_r1cs_inputs_for_target_inner(
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

fn run_stage4_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(false, false, false);
}

fn run_stage5_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(true, false, false);
}

fn run_stage6_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(true, true, false);
}

fn run_stage7_sumchecks_blake2b() {
    run_stage4_sumchecks_blake2b_inner(true, true, true);
}

fn run_stage4_sumchecks_blake2b_inner(run_stage5: bool, run_stage6: bool, run_stage7: bool) {
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
    let regs_log_k = 5usize;
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
        tau.push(Fr::from(t.challenge_u128()));
    }

    let stage1_uni_poly_coeffs: Vec<Fr> = vec![Fr::zero(); outer_first_round_num_coeffs];
    t.append_scalars_fr(b"uniskip_poly", &stage1_uni_poly_coeffs);
    let stage1_r0 = Fr::from(t.challenge_u128());

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
        let rj = Fr::from(t.challenge_u128());
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
    let tau_high_pv = Fr::from(t.challenge_u128());
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
    let stage2_r0 = Fr::from(t.challenge_u128());

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
        stage2_r_address.push(Fr::from(t.challenge_u128()));
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
        let rj = Fr::from(t.challenge_u128());
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
        let rj = Fr::from(t.challenge_u128());
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
        let rj = Fr::from(t.challenge_u128());
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
            let rj = Fr::from(t.challenge_u128());
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

        // Booleanity samples optimized gamma (u128->Fr). (We force nonzero like Python, but keep output terms zero.)
        let mut _stage6_bool_gamma = Fr::from(t.challenge_u128());
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
            let rj = Fr::from(t.challenge_u128());
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
            let N = instruction_d + bytecode_d + ram_d;
            let stage7_gamma_powers = t.challenge_scalar_powers_fr(3 * N);

            // HW claims: instruction/bytecode use 1, ram uses stage6_hamming_weight_claim (0 here).
            let mut stage7_hw_input = Fr::zero();
            for i in 0..N {
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
                let rj = Fr::from(t.challenge_u128());
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

            let mut stage7_ra_opening_claims: Vec<Fr> = vec![Fr::zero(); N];
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
        run_e2e_transcript_pre_stage1(args);
        return;
    }
    if mode.as_str() == "e2e_stage1_debug" {
        run_e2e_stage1_debug(args);
        return;
    }
    if mode.as_str() == "e2e_stage6_breakdown" {
        run_e2e_stage6_breakdown(args);
        return;
    }
    if mode.as_str() == "e2e_bytecode_fingerprint" {
        run_e2e_bytecode_fingerprint(args);
        return;
    }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    match mode.as_str() {
        "fq" => run_field::<Fq>(&input),
        "fr" => run_field::<Fr>(&input),
        "curve" => run_curve(&input),
        "transcript_blake2b" => run_transcript_blake2b(&input),
        "sumcheck_verify_blake2b" => run_sumcheck_verify_blake2b(&input),
        "dory_pcs_eval_blake2b" => run_dory_pcs_eval_blake2b(),
        "lookup_table_mle_64" => run_lookup_table_mle_64(&input),
        "spartan_outer_stage1_blake2b" => run_spartan_outer_stage1_blake2b(),
        "stage2_sumchecks_blake2b" => run_stage2_sumchecks_blake2b(),
        "stage3_sumchecks_blake2b" => run_stage3_sumchecks_blake2b(),
        "stage4_sumchecks_blake2b" => run_stage4_sumchecks_blake2b(),
        "stage5_sumchecks_blake2b" => run_stage5_sumchecks_blake2b(),
        "stage6_sumchecks_blake2b" => run_stage6_sumchecks_blake2b(),
        "stage7_sumchecks_blake2b" => run_stage7_sumchecks_blake2b(),
        _ => panic!("unknown mode; use fq|fr|curve|transcript_blake2b|sumcheck_verify_blake2b|dory_pcs_eval_blake2b|lookup_table_mle_64|spartan_outer_stage1_blake2b|stage2_sumchecks_blake2b|stage3_sumchecks_blake2b|stage4_sumchecks_blake2b|stage5_sumchecks_blake2b|stage6_sumchecks_blake2b|stage7_sumchecks_blake2b"),
    }
}

fn run_e2e_transcript_pre_stage1(mut args: std::env::Args) {
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript;
    use jolt_core::zkvm::Serializable;
    use jolt_core::zkvm::RV64IMACProof;
    use jolt_core::zkvm::fiat_shamir_preamble;
    use common::jolt_device::JoltDevice;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());
    let proof_bytes = std::fs::read(std::path::Path::new(&dir).join("proof.bin"))
        .expect("read proof.bin");
    let program_io_bytes = std::fs::read(std::path::Path::new(&dir).join("program_io.bin"))
        .expect("read program_io.bin");
    let mut program_io = JoltDevice::deserialize_from_bytes(&program_io_bytes).expect("parse program_io");
    let proof = RV64IMACProof::deserialize_from_bytes(&proof_bytes).expect("parse proof");

    // Mirror Rust verifier: truncate trailing zeros on outputs.
    program_io.outputs.truncate(
        program_io
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    let mut t = Blake2bTranscript::new(b"Jolt");
    fiat_shamir_preamble(&program_io, proof.ram_K, proof.trace_length, &mut t);

    for c in &proof.commitments {
        t.append_serializable(b"commitment", c);
    }
    if let Some(ref c) = proof.untrusted_advice_commitment {
        t.append_serializable(b"untrusted_advice", c);
    }

    let final_state = bytes_to_hex(&t.state);
    println!("transcript_pre_stage1_state={final_state}");
}

fn run_e2e_stage1_debug(mut args: std::env::Args) {
    use common::jolt_device::JoltDevice;
    use jolt_core::field::JoltField;
    use jolt_core::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript;
    use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
    use jolt_core::zkvm::spartan::outer::OuterRemainingSumcheckVerifier;
    use jolt_core::zkvm::spartan::verify_stage1_uni_skip;
    use jolt_core::zkvm::fiat_shamir_preamble;
    use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
    use jolt_core::zkvm::Serializable;
    use jolt_core::poly::opening_proof::{OpeningPoint, VerifierOpeningAccumulator};
    use jolt_core::utils::math::Math;

    type F = ark_bn254::Fr;
    type PCS = jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    type PT = jolt_core::transcripts::Blake2bTranscript;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());
    let proof_bytes = std::fs::read(std::path::Path::new(&dir).join("proof.bin"))
        .expect("read proof.bin");
    let program_io_bytes = std::fs::read(std::path::Path::new(&dir).join("program_io.bin"))
        .expect("read program_io.bin");

    let mut program_io = JoltDevice::deserialize_from_bytes(&program_io_bytes).expect("parse program_io");
    let proof: CoreJoltProof<F, PCS, PT> =
        CoreJoltProof::deserialize_from_bytes(&proof_bytes).expect("parse proof");

    // Mirror verifier: truncate trailing zeros on outputs.
    program_io.outputs.truncate(
        program_io
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new(proof.trace_length.log_2());
    for (key, (_pt, claim)) in &proof.opening_claims.0 {
        opening_accumulator
            .openings
            .insert(*key, (OpeningPoint::default(), *claim));
    }

    let mut transcript = Blake2bTranscript::new(b"Jolt");
    fiat_shamir_preamble(&program_io, proof.ram_K, proof.trace_length, &mut transcript);
    for commitment in &proof.commitments {
        transcript.append_serializable(b"commitment", commitment);
    }
    if let Some(ref c) = proof.untrusted_advice_commitment {
        transcript.append_serializable(b"untrusted_advice", c);
    }

    let key = UniformSpartanKey::<F>::new(proof.trace_length.next_power_of_two());
    let uni_skip_params = verify_stage1_uni_skip(
        &proof.stage1_uni_skip_first_round_proof,
        &key,
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage1 uniskip verify");

    let outer_remaining =
        OuterRemainingSumcheckVerifier::<F>::new(key, proof.trace_length, uni_skip_params, &opening_accumulator);

    // Manual BatchedSumcheck::verify, but also print output vs expected.
    let input_claim = <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<
        F,
        Blake2bTranscript,
    >>::input_claim(&outer_remaining, &opening_accumulator);
    transcript.append_scalar(b"sumcheck_claim", &input_claim);
    let batching_coeffs: Vec<F> = transcript.challenge_vector(1);
    let coeff = batching_coeffs[0];
    let claim = input_claim * coeff;
    let (output_claim, r_sumcheck) = proof
        .stage1_sumcheck_proof
        .verify(
            claim,
            <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::num_rounds(&outer_remaining),
            <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::degree(&outer_remaining),
            &mut transcript,
        )
        .expect("sumcheck proof verify");

    let nr = <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::num_rounds(&outer_remaining);
    let r_slice = &r_sumcheck[..nr];
    <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::cache_openings(
        &outer_remaining,
        &mut opening_accumulator,
        &mut transcript,
        r_slice,
    );
    let expected =
        <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::expected_output_claim(
            &outer_remaining,
            &opening_accumulator,
            r_slice,
        ) * coeff;

    println!("stage1_output_claim={output_claim}");
    println!("stage1_expected_claim={expected}");
    println!("stage1_claim_diff={}", output_claim - expected);
    println!("stage1_r_len={}", r_sumcheck.len());
    println!("stage1_transcript_state_end={}", bytes_to_hex(&transcript.state));
}

fn run_e2e_stage6_breakdown(mut args: std::env::Args) {
    use common::jolt_device::JoltDevice;
    use jolt_core::field::JoltField;
    use jolt_core::poly::opening_proof::{OpeningPoint, VerifierOpeningAccumulator};
    use jolt_core::subprotocols::booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier};
    use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
    use jolt_core::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript;
    use jolt_core::utils::math::Math;
    use jolt_core::zkvm::claim_reductions::{
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        RamRaClaimReductionSumcheckVerifier,
    };
    use jolt_core::zkvm::config::OneHotParams;
    use jolt_core::zkvm::fiat_shamir_preamble;
    use jolt_core::zkvm::instruction_lookups::ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier;
    use jolt_core::zkvm::instruction_lookups::read_raf_checking::InstructionReadRafSumcheckVerifier;
    use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
    use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
    use jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheckVerifier;
    use jolt_core::zkvm::ram::output_check::OutputSumcheckVerifier;
    use jolt_core::zkvm::ram::ra_virtual::RamRaVirtualSumcheckVerifier;
    use jolt_core::zkvm::ram::raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier;
    use jolt_core::zkvm::ram::read_write_checking::RamReadWriteCheckingVerifier;
    use jolt_core::zkvm::ram::val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier;
    use jolt_core::zkvm::ram::val_final::ValFinalSumcheckVerifier;
    use jolt_core::zkvm::ram::verifier_accumulate_advice;
    use jolt_core::zkvm::registers::read_write_checking::RegistersReadWriteCheckingVerifier;
    use jolt_core::zkvm::registers::val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier;
    use jolt_core::zkvm::spartan::outer::OuterRemainingSumcheckVerifier;
    use jolt_core::zkvm::spartan::product::ProductVirtualRemainderVerifier;
    use jolt_core::zkvm::spartan::shift::ShiftSumcheckVerifier;
    use jolt_core::zkvm::spartan::instruction_input::InstructionInputSumcheckVerifier;
    use jolt_core::zkvm::spartan::{verify_stage1_uni_skip, verify_stage2_uni_skip};
    use jolt_core::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier;
    use jolt_core::zkvm::Serializable;
    use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;

    type F = ark_bn254::Fr;
    type PCS = jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    type PT = jolt_core::transcripts::Blake2bTranscript;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());

    let proof_bytes = std::fs::read(std::path::Path::new(&dir).join("proof.bin"))
        .expect("read proof.bin");
    let program_io_bytes = std::fs::read(std::path::Path::new(&dir).join("program_io.bin"))
        .expect("read program_io.bin");
    let preprocessing_bytes =
        std::fs::read(std::path::Path::new(&dir).join("verifier_preprocessing.bin"))
            .expect("read verifier_preprocessing.bin");

    let mut program_io =
        JoltDevice::deserialize_from_bytes(&program_io_bytes).expect("parse program_io");
    let proof: CoreJoltProof<F, PCS, PT> =
        CoreJoltProof::deserialize_from_bytes(&proof_bytes).expect("parse proof");
    let preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::deserialize_from_bytes(&preprocessing_bytes)
            .expect("parse verifier_preprocessing");

    // Mirror verifier: truncate trailing zeros on outputs.
    program_io.outputs.truncate(
        program_io
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    // Construct OneHotParams from the validated config.
    let one_hot_params =
        OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

    // Seed opening accumulator with claims from proof.
    let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new(proof.trace_length.log_2());
    for (key, (_pt, claim)) in &proof.opening_claims.0 {
        opening_accumulator
            .openings
            .insert(*key, (OpeningPoint::default(), *claim));
    }

    // Transcript preamble + commitments, matching verifier.rs.
    let mut transcript = Blake2bTranscript::new(b"Jolt");
    fiat_shamir_preamble(&program_io, proof.ram_K, proof.trace_length, &mut transcript);
    for commitment in &proof.commitments {
        transcript.append_serializable(b"commitment", commitment);
    }
    if let Some(ref c) = proof.untrusted_advice_commitment {
        transcript.append_serializable(b"untrusted_advice", c);
    }

    // Stage 1
    let spartan_key = UniformSpartanKey::<F>::new(proof.trace_length.next_power_of_two());
    let stage1_params = verify_stage1_uni_skip(
        &proof.stage1_uni_skip_first_round_proof,
        &spartan_key,
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage1 uniskip verify");
    let outer_remaining = OuterRemainingSumcheckVerifier::<F>::new(
        spartan_key,
        proof.trace_length,
        stage1_params,
        &opening_accumulator,
    );
    BatchedSumcheck::verify(
        &proof.stage1_sumcheck_proof,
        vec![&outer_remaining],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage1 sumcheck");

    // Stage 2
    let stage2_params = verify_stage2_uni_skip(
        &proof.stage2_uni_skip_first_round_proof,
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage2 uniskip verify");
    let ram_rw = RamReadWriteCheckingVerifier::new(
        &opening_accumulator,
        &mut transcript,
        &one_hot_params,
        proof.trace_length,
        &proof.rw_config,
    );
    let spartan_product_remainder = ProductVirtualRemainderVerifier::new(
        proof.trace_length,
        stage2_params,
        &opening_accumulator,
    );
    let lookups_claim_reduction =
        InstructionLookupsClaimReductionSumcheckVerifier::new(
            proof.trace_length,
            &opening_accumulator,
            &mut transcript,
        );
    let ram_raf_eval = RamRafEvaluationSumcheckVerifier::new(
        &program_io.memory_layout,
        &one_hot_params,
        &opening_accumulator,
    );
    let ram_output_check = OutputSumcheckVerifier::new(proof.ram_K, &program_io, &mut transcript);
    BatchedSumcheck::verify(
        &proof.stage2_sumcheck_proof,
        vec![
            &ram_rw,
            &spartan_product_remainder,
            &lookups_claim_reduction,
            &ram_raf_eval,
            &ram_output_check,
        ],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage2 sumcheck");

    // Stage 3
    let spartan_shift = ShiftSumcheckVerifier::new(
        proof.trace_length.log_2(),
        &opening_accumulator,
        &mut transcript,
    );
    let spartan_instruction_input =
        InstructionInputSumcheckVerifier::new(&opening_accumulator, &mut transcript);
    let regs_claim_reduction =
        jolt_core::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier::new(
            proof.trace_length,
            &opening_accumulator,
            &mut transcript,
        );
    BatchedSumcheck::verify(
        &proof.stage3_sumcheck_proof,
        vec![&spartan_shift, &spartan_instruction_input, &regs_claim_reduction],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage3 sumcheck");

    // Stage 4
    verifier_accumulate_advice::<F>(
        proof.ram_K,
        &program_io,
        proof.untrusted_advice_commitment.is_some(),
        false, // trusted advice commitment is external in verifier; absent in these artifacts
        &mut opening_accumulator,
        &mut transcript,
        proof.rw_config.needs_single_advice_opening(proof.trace_length.log_2()),
    );
    let regs_rw = RegistersReadWriteCheckingVerifier::new(
        proof.trace_length,
        &opening_accumulator,
        &mut transcript,
        &proof.rw_config,
    );
    let ram_val_eval = RamValEvaluationSumcheckVerifier::new(
        &preprocessing.shared.ram,
        &program_io,
        proof.trace_length,
        proof.ram_K,
        &opening_accumulator,
    );
    let ram_val_final = ValFinalSumcheckVerifier::new(
        &preprocessing.shared.ram,
        &program_io,
        proof.trace_length,
        proof.ram_K,
        &opening_accumulator,
        &proof.rw_config,
    );
    BatchedSumcheck::verify(
        &proof.stage4_sumcheck_proof,
        vec![&regs_rw, &ram_val_eval, &ram_val_final],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage4 sumcheck");

    // Stage 5
    let n_cycle_vars = proof.trace_length.log_2();
    let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
        n_cycle_vars,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
        proof.trace_length,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let regs_val_eval = RegistersValEvaluationSumcheckVerifier::new(&opening_accumulator);
    BatchedSumcheck::verify(
        &proof.stage5_sumcheck_proof,
        vec![&lookups_read_raf, &ram_ra_reduction, &regs_val_eval],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage5 sumcheck");

    // Stage 6 breakdown (manual batched sumcheck).
    let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
        &preprocessing.shared.bytecode,
        n_cycle_vars,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let ram_hamming_booleanity = HammingBooleanitySumcheckVerifier::new(&opening_accumulator);
    let booleanity_params =
        BooleanitySumcheckParams::new(n_cycle_vars, &one_hot_params, &opening_accumulator, &mut transcript);
    let booleanity = BooleanitySumcheckVerifier::new(booleanity_params);
    let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
        proof.trace_length,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let lookups_ra_virtual =
        LookupsRaSumcheckVerifier::new(&one_hot_params, &opening_accumulator, &mut transcript);
    let inc_reduction =
        IncClaimReductionSumcheckVerifier::new(proof.trace_length, &opening_accumulator, &mut transcript);

    let instances: Vec<&dyn SumcheckInstanceVerifier<F, Blake2bTranscript>> = vec![
        &bytecode_read_raf,
        &booleanity,
        &ram_hamming_booleanity,
        &ram_ra_virtual,
        &lookups_ra_virtual,
        &inc_reduction,
    ];

    println!("stage6_instances={}", instances.len());
    for (i, inst) in instances.iter().enumerate() {
        let ic = inst.input_claim(&opening_accumulator);
        println!("stage6_input_claim_{i}={ic}");
        transcript.append_scalar(b"sumcheck_claim", &ic);
    }
    let batch_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
    for (i, c) in batch_coeffs.iter().enumerate() {
        println!("stage6_batch_coeff_{i}={c}");
    }

    let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();
    let max_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
    let claim: F = instances
        .iter()
        .zip(batch_coeffs.iter())
        .map(|(s, coeff)| s.input_claim(&opening_accumulator).mul_pow_2(max_rounds - s.num_rounds()) * coeff)
        .sum();
    println!("stage6_batched_input_claim={claim}");

    let (output_claim, r_sumcheck) = proof
        .stage6_sumcheck_proof
        .verify(claim, max_rounds, max_degree, &mut transcript)
        .expect("stage6 proof.verify");
    println!("stage6_output_claim={output_claim}");
    println!("stage6_r_len={}", r_sumcheck.len());

    let mut expected_total = F::zero();
    for (i, (s, coeff)) in instances.iter().zip(batch_coeffs.iter()).enumerate() {
        let offset = s.round_offset(max_rounds);
        let r_slice = &r_sumcheck[offset..offset + s.num_rounds()];
        s.cache_openings(&mut opening_accumulator, &mut transcript, r_slice);
        let exp = s.expected_output_claim(&opening_accumulator, r_slice);
        println!("stage6_expected_output_{i}={exp}");
        expected_total += exp * coeff;
    }

    println!("stage6_expected_total={expected_total}");
    println!("stage6_claim_diff={}", output_claim - expected_total);
    println!("stage6_transcript_state_end={}", bytes_to_hex(&transcript.state));
    println!("stage6_note=diff should be 0 in Rust");
}

fn run_e2e_bytecode_fingerprint(mut args: std::env::Args) {
    use jolt_core::zkvm::instruction::{Flags, InstructionLookup, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
    use jolt_core::zkvm::lookup_table::LookupTables;
    use jolt_core::zkvm::Serializable;
    use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;

    type F = ark_bn254::Fr;
    type PCS = jolt_core::poly::commitment::dory::DoryCommitmentScheme;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());
    let mut idx_opt: Option<usize> = None;
    let mut chunk_opt: Option<usize> = None;
    if let Some(s) = args.next() {
        if let Some(rest) = s.strip_prefix("chunk=") {
            chunk_opt = rest.parse::<usize>().ok();
        } else {
            idx_opt = s.parse::<usize>().ok();
        }
    }

    let preprocessing_bytes =
        std::fs::read(std::path::Path::new(&dir).join("verifier_preprocessing.bin"))
            .expect("read verifier_preprocessing.bin");
    let preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::deserialize_from_bytes(&preprocessing_bytes)
            .expect("parse verifier_preprocessing");

    let bytecode = &preprocessing.shared.bytecode.bytecode;
    println!("bytecode_len={}", bytecode.len());
    println!("bytecode_code_size={}", preprocessing.shared.bytecode.code_size);

    let mut h = Blake2b256::new();
    h.update((bytecode.len() as u64).to_le_bytes());

    for instr in bytecode.iter() {
        let n = instr.normalize();
        let cf = instr.circuit_flags();
        let inf = instr.instruction_flags();
        debug_assert_eq!(cf.len(), NUM_CIRCUIT_FLAGS);
        debug_assert_eq!(inf.len(), NUM_INSTRUCTION_FLAGS);

        let mut cf_mask: u16 = 0;
        for (i, b) in cf.iter().enumerate() {
            if *b {
                cf_mask |= 1u16 << i;
            }
        }
        let mut inf_mask: u8 = 0;
        for (i, b) in inf.iter().enumerate() {
            if *b {
                inf_mask |= 1u8 << i;
            }
        }

        let lt_idx: i16 = instr
            .lookup_table()
            .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
            .unwrap_or(-1);

        let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
        let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
        let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);

        h.update((n.address as u64).to_le_bytes());
        h.update(rd.to_le_bytes());
        h.update(rs1.to_le_bytes());
        h.update(rs2.to_le_bytes());
        h.update((n.operands.imm as i128).to_le_bytes());
        h.update(cf_mask.to_le_bytes());
        h.update([inf_mask]);
        h.update(lt_idx.to_le_bytes());
    }

    let digest = h.finalize();
    println!("bytecode_fingerprint={}", bytes_to_hex(&digest[..]));

    // Emit chunk fingerprints to quickly localize mismatches with Python.
    let chunk_size: usize = 256;
    println!("bytecode_chunk_size={chunk_size}");
    for (chunk_idx, chunk) in bytecode.chunks(chunk_size).enumerate() {
        let mut hc = Blake2b256::new();
        hc.update((chunk_idx as u64).to_le_bytes());
        hc.update((chunk.len() as u64).to_le_bytes());
        for instr in chunk.iter() {
            let n = instr.normalize();
            let cf = instr.circuit_flags();
            let inf = instr.instruction_flags();
            let mut cf_mask: u16 = 0;
            for (i, b) in cf.iter().enumerate() {
                if *b {
                    cf_mask |= 1u16 << i;
                }
            }
            let mut inf_mask: u8 = 0;
            for (i, b) in inf.iter().enumerate() {
                if *b {
                    inf_mask |= 1u8 << i;
                }
            }
            let lt_idx: i16 = instr
                .lookup_table()
                .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
                .unwrap_or(-1);
            let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
            let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
            let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);

            hc.update((n.address as u64).to_le_bytes());
            hc.update(rd.to_le_bytes());
            hc.update(rs1.to_le_bytes());
            hc.update(rs2.to_le_bytes());
            hc.update((n.operands.imm as i128).to_le_bytes());
            hc.update(cf_mask.to_le_bytes());
            hc.update([inf_mask]);
            hc.update(lt_idx.to_le_bytes());
        }
        let cd = hc.finalize();
        println!("bytecode_chunk_{chunk_idx}={}", bytes_to_hex(&cd[..]));
    }

    if let Some(chunk_idx) = chunk_opt {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(bytecode.len());
        println!("bytecode_dump_chunk={chunk_idx}");
        println!("bytecode_dump_range={start}..{end}");
        for i in start..end {
            let instr = &bytecode[i];
            let n = instr.normalize();
            let cf = instr.circuit_flags();
            let inf = instr.instruction_flags();
            let mut cf_mask: u16 = 0;
            for (j, b) in cf.iter().enumerate() {
                if *b {
                    cf_mask |= 1u16 << j;
                }
            }
            let mut inf_mask: u8 = 0;
            for (j, b) in inf.iter().enumerate() {
                if *b {
                    inf_mask |= 1u8 << j;
                }
            }
            let lt_idx: i16 = instr
                .lookup_table()
                .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
                .unwrap_or(-1);
            let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
            let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
            let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);
            println!(
                "bytecode_dump_entry i={i} addr={} rd={rd} rs1={rs1} rs2={rs2} imm={} cf={cf_mask} inf={inf_mask} lt={lt_idx}",
                n.address,
                n.operands.imm
            );
        }
        return;
    }

    if let Some(i) = idx_opt {
        let instr = bytecode.get(i).expect("index in range");
        let n = instr.normalize();
        let cf = instr.circuit_flags();
        let inf = instr.instruction_flags();
        let lt_idx: i16 = instr
            .lookup_table()
            .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
            .unwrap_or(-1);
        let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
        let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
        let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);

        let mut cf_mask: u16 = 0;
        for (j, b) in cf.iter().enumerate() {
            if *b {
                cf_mask |= 1u16 << j;
            }
        }
        let mut inf_mask: u8 = 0;
        for (j, b) in inf.iter().enumerate() {
            if *b {
                inf_mask |= 1u8 << j;
            }
        }

        println!("bytecode_index={i}");
        println!("bytecode_addr={}", n.address);
        println!("bytecode_rd={rd}");
        println!("bytecode_rs1={rs1}");
        println!("bytecode_rs2={rs2}");
        println!("bytecode_imm={}", n.operands.imm);
        println!("bytecode_cf_mask={cf_mask}");
        println!("bytecode_inf_mask={inf_mask}");
        println!("bytecode_lookup_table_index={lt_idx}");
    }
}
