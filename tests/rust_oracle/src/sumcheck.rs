use std::str::FromStr;

use ark_bn254::Fr;

use crate::hex_utils::bytes_to_hex;
use crate::transcript::Blake2bTranscript;

#[derive(Clone)]
pub(crate) struct CompressedUniPolyFr {
    pub(crate) coeffs_except_linear_term: Vec<Fr>,
}

impl CompressedUniPolyFr {
    pub(crate) fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }

    pub(crate) fn eval_from_hint(&self, hint: Fr, x: Fr) -> Fr {
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

pub(crate) fn run_sumcheck_verify_blake2b(input: &str) {
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
                    let r_i = t.challenge_scalar_optimized_fr();
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
