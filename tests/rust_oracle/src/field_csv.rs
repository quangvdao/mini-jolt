use std::sync::OnceLock;

use ark_bn254::{Fq, Fq2, Fq6, G1Affine, G2Affine};
use ark_ff::{Field, One, PrimeField, Zero};

pub(crate) fn pow_u64<F: Field>(mut a: F, mut e: u64) -> F {
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

pub(crate) fn limbs_csv<F: PrimeField>(x: F) -> String {
    x.into_bigint()
        .as_ref()
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn fq2_csv(x: Fq2) -> String {
    format!("{}/{}", limbs_csv(x.c0), limbs_csv(x.c1))
}

pub(crate) fn g1_csv(p: G1Affine) -> String {
    if p.infinity {
        "inf".to_string()
    } else {
        format!("{}:{}", limbs_csv(p.x), limbs_csv(p.y))
    }
}

pub(crate) fn g2_csv(p: G2Affine) -> String {
    if p.infinity {
        "inf".to_string()
    } else {
        format!("{}:{}", fq2_csv(p.x), fq2_csv(p.y))
    }
}

pub(crate) fn fq12_tower_vec(x: ark_bn254::Fq12) -> [Fq; 12] {
    let c0 = x.c0;
    let c1 = x.c1;
    [
        c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1, c1.c0.c0, c1.c0.c1,
        c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1,
    ]
}

pub(crate) fn fq12_poly_basis_matrix() -> &'static [[Fq; 12]; 12] {
    static M: OnceLock<[[Fq; 12]; 12]> = OnceLock::new();
    M.get_or_init(|| {
        fq12_poly_basis_matrix_for(ark_bn254::Fq12::new(Fq6::zero(), Fq6::one()))
    })
}

pub(crate) fn fq12_poly_basis_matrix_for(generator: ark_bn254::Fq12) -> [[Fq; 12]; 12] {
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

pub(crate) fn fq12_to_poly_coeffs(x: ark_bn254::Fq12) -> [Fq; 12] {
    let m = fq12_poly_basis_matrix();
    fq12_to_poly_coeffs_for(m, x)
}

pub(crate) fn fq12_to_poly_coeffs_for(m: &[[Fq; 12]; 12], x: ark_bn254::Fq12) -> [Fq; 12] {
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

pub(crate) fn gt_poly_csv(x: ark_bn254::Fq12) -> String {
    let coeffs = fq12_to_poly_coeffs(x);
    coeffs.iter().map(|c| limbs_csv(*c)).collect::<Vec<_>>().join("/")
}

pub(crate) fn gt_poly_csv_for(m: &[[Fq; 12]; 12], x: ark_bn254::Fq12) -> String {
    let coeffs = fq12_to_poly_coeffs_for(m, x);
    coeffs.iter().map(|c| limbs_csv(*c)).collect::<Vec<_>>().join("/")
}

pub(crate) fn fq12_tower_csv(x: ark_bn254::Fq12) -> String {
    fq12_tower_vec(x)
        .iter()
        .map(|c| limbs_csv(*c))
        .collect::<Vec<_>>()
        .join("/")
}
