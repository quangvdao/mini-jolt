use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};

pub(crate) fn fr_from_i128(x: i128) -> Fr {
    if x == 0 {
        return Fr::zero();
    }
    if x > 0 && x <= (u64::MAX as i128) {
        return Fr::from(x as u64);
    }
    if x < 0 && -x <= (u64::MAX as i128) {
        return -Fr::from((-x) as u64);
    }
    use std::str::FromStr;
    Fr::from_str(&x.to_string()).unwrap()
}

pub(crate) fn fr_factorial(n: usize) -> Fr {
    (1..=n).fold(Fr::one(), |acc, i| acc * Fr::from(i as u64))
}

pub(crate) fn lagrange_evals_symmetric(r: Fr, n: usize) -> Vec<Fr> {
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

pub(crate) fn lagrange_kernel_symmetric(x: Fr, y: Fr, n: usize) -> Fr {
    let lx = lagrange_evals_symmetric(x, n);
    let ly = lagrange_evals_symmetric(y, n);
    lx.into_iter().zip(ly).map(|(a, b)| a * b).sum()
}

pub(crate) fn eq_mle(x: &[Fr], y: &[Fr]) -> Fr {
    assert_eq!(x.len(), y.len());
    let one = Fr::one();
    x.iter()
        .zip(y.iter())
        .fold(Fr::one(), |acc, (xi, yi)| acc * (*xi * *yi + (one - *xi) * (one - *yi)))
}

pub(crate) fn eq_plus_one_mle(x: &[Fr], y: &[Fr]) -> Fr {
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

pub(crate) fn remap_address_oracle(address: u64, lowest: u64) -> Option<usize> {
    if address == 0 {
        return None;
    }
    if address < lowest {
        panic!("unexpected address below lowest");
    }
    Some(((address - lowest) / 8) as usize)
}

pub(crate) fn eq_at_index(r: &[Fr], idx: usize) -> Fr {
    let n = r.len();
    let mut bits = Vec::with_capacity(n);
    for i in 0..n {
        let bit = ((idx >> (n - 1 - i)) & 1) == 1;
        bits.push(if bit { Fr::one() } else { Fr::zero() });
    }
    eq_mle(r, &bits)
}

pub(crate) fn sparse_eval_u64_block_oracle(start_index: usize, values: &[u64], r: &[Fr]) -> Fr {
    let mut acc = Fr::zero();
    for (j, v) in values.iter().enumerate() {
        if *v == 0 {
            continue;
        }
        acc += Fr::from(*v) * eq_at_index(r, start_index + j);
    }
    acc
}

pub(crate) fn eval_initial_ram_mle_oracle(
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

pub(crate) fn calculate_advice_memory_evaluation_oracle(
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
