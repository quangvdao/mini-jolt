use ark_bn254::Fr;
use ark_ff::{BigInteger, One, PrimeField};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use dory_pcs::backends::arkworks::{ArkFr as DoryFr, BN254 as DoryBN254};
use dory_pcs::primitives::serialization::DorySerialize;
use dory_pcs::primitives::transcript::Transcript as DoryTranscript;
use jolt_core::field::challenge::MontU128Challenge;

pub(crate) type Blake2b256 = Blake2b<U32>;

#[derive(Clone)]
pub(crate) struct Blake2bTranscript {
    pub(crate) state: [u8; 32],
    pub(crate) n_rounds: u32,
}

impl Blake2bTranscript {
    pub(crate) fn new(label: &[u8]) -> Self {
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

    pub(crate) fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.raw_append_label_with_len(label, bytes.len() as u64);
        self.raw_append_bytes(bytes);
    }

    pub(crate) fn append_u64(&mut self, label: &[u8], x: u64) {
        self.raw_append_label(label);
        self.raw_append_u64(x);
    }

    pub(crate) fn append_scalar_fr(&mut self, label: &[u8], x: Fr) {
        self.raw_append_label(label);
        self.raw_append_scalar_fr(x);
    }

    pub(crate) fn append_scalars_fr(&mut self, label: &[u8], xs: &[Fr]) {
        self.raw_append_label_with_len(label, xs.len() as u64);
        for &x in xs {
            self.raw_append_scalar_fr(x);
        }
    }

    pub(crate) fn append_serializable_uncompressed(&mut self, label: &[u8], blob: &[u8]) {
        self.raw_append_label_with_len(label, blob.len() as u64);
        let mut b = blob.to_vec();
        b.reverse();
        self.raw_append_bytes(&b);
    }

    pub(crate) fn append_serializable_bytes_reversed(&mut self, label: &[u8], blob_reversed: &[u8]) {
        self.append_bytes(label, blob_reversed);
    }

    pub(crate) fn challenge_bytes32(&mut self) -> [u8; 32] {
        let out: [u8; 32] = self.hasher().finalize().into();
        self.update_state(out);
        out
    }

    pub(crate) fn challenge_bytes(&mut self, n: usize) -> Vec<u8> {
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

    pub(crate) fn challenge_u128(&mut self) -> u128 {
        let mut b = self.challenge_bytes(16);
        b.reverse();
        u128::from_be_bytes(b.try_into().unwrap())
    }

    pub(crate) fn challenge_scalar_optimized_fr(&mut self) -> Fr {
        let u = self.challenge_u128();
        let ch: MontU128Challenge<Fr> = MontU128Challenge::from(u);
        ch.into()
    }

    pub(crate) fn challenge_scalar_fr(&mut self) -> Fr {
        let b = self.challenge_bytes(16);
        let x = u128::from_be_bytes(b.try_into().unwrap());
        Fr::from(x)
    }

    pub(crate) fn challenge_vector_fr(&mut self, n: usize) -> Vec<Fr> {
        (0..n).map(|_| self.challenge_scalar_fr()).collect()
    }

    pub(crate) fn challenge_scalar_powers_fr(&mut self, n: usize) -> Vec<Fr> {
        let q = self.challenge_scalar_fr();
        let mut out = vec![Fr::one(); n];
        for i in 1..n {
            out[i] = out[i - 1] * q;
        }
        out
    }
}

#[derive(Clone)]
pub(crate) struct JoltLikeDoryTranscript {
    pub(crate) inner: Blake2bTranscript,
    pub(crate) serde_blocks: Vec<Vec<u8>>,
}

impl JoltLikeDoryTranscript {
    pub(crate) fn new(label: &[u8]) -> Self {
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
