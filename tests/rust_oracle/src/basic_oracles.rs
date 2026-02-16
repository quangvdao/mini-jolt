use std::fmt::Debug;
use std::str::FromStr;

use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, PrimeGroup};
use ark_ff::{Field, One, PrimeField};
use jolt_core::zkvm::lookup_table::LookupTables;

use crate::field_csv::{g1_csv, g2_csv, limbs_csv, pow_u64};
use crate::hex_utils::{bytes_to_hex, hex_to_bytes};
use crate::transcript::Blake2bTranscript;

pub(crate) fn run_field<F>(input: &str)
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

pub(crate) fn run_curve(input: &str) {
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

pub(crate) fn run_transcript_blake2b(input: &str) {
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

pub(crate) fn parse_lookup_table_64(name: &str) -> LookupTables<64> {
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

pub(crate) fn run_lookup_table_mle_64(input: &str) {
    for (line_idx, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
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
