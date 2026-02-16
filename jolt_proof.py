"""Jolt proof container (Python-native) + minimal Rust binary deserialization.

This module defines:

- A Python-native `JoltProof` dataclass that the Python verifier consumes.
- A minimal `JoltProof.from_rust_bytes(...)` reader for Rust-produced `proof.bin` (arkworks
  `CanonicalSerialize` compressed), sufficient to verify Rust proofs in Python.

## Rust canonical fields (source of truth)

Rust struct: `jolt-core/src/zkvm/proof_serialization.rs::JoltProof`.

```text
opening_claims: Claims<F>                   // map OpeningId -> claim scalar (no opening points)
commitments: Vec<PCS::Commitment>           // commitments for all main committed polynomials
stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof
stage1_sumcheck_proof: SumcheckInstanceProof
stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof
stage2_sumcheck_proof: SumcheckInstanceProof
stage3_sumcheck_proof: SumcheckInstanceProof
stage4_sumcheck_proof: SumcheckInstanceProof
stage5_sumcheck_proof: SumcheckInstanceProof
stage6_sumcheck_proof: SumcheckInstanceProof
stage7_sumcheck_proof: SumcheckInstanceProof
joint_opening_proof: PCS::Proof             // Dory joint opening proof (Stage 8)
untrusted_advice_commitment: Option<PCS::Commitment>
trace_length: usize
ram_K: usize
bytecode_K: usize
rw_config: ReadWriteConfig
one_hot_config: OneHotConfig
dory_layout: DoryLayout
```

### Commitment ordering (Rust)

`commitments` are ordered by `all_committed_polynomials(one_hot_params)` (see
`jolt-core/src/zkvm/witness.rs`):

1. `RdInc`
2. `RamInc`
3. `InstructionRa(i)` for `i=0..instruction_d-1`
4. `RamRa(i)` for `i=0..ram_d-1`
5. `BytecodeRa(i)` for `i=0..bytecode_d-1`

Advice commitments are *not* part of this list; they are appended to the transcript separately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace  # lightweight proof containers

import pathlib

import dory  # PCS verification objects (Stage 8)
from curve import Fq2, GT  # BN254 extension fields (GT is polynomial-basis Fq12)
from field import Fq, Fr  # BN254 base/scalar fields
from ids_generated import SUMCHECK_IDS, VIRTUAL_POLYS, CommittedPolynomial, SumcheckId, VirtualPolynomial  # canonical ID tables
from openings import (  # typed opening IDs + seeding API
    AdviceKind,
    CommittedId,
    OpeningId,
    VerifierOpeningAccumulator,
    VirtualId,
)
from polynomials import CompressedUniPoly, UniPoly  # sumcheck polynomial containers
from sumchecks import SumcheckInstanceProof, UniSkipFirstRoundProof  # proof containers with verify()
from zkvm_types import OneHotParams, ReadWriteConfig  # verifier-facing config objects


@dataclass(frozen=True)
class JoltProof:  # Python-native proof object consumed by `jolt_verifier.py`.
    opening_claims: dict[OpeningId, Fr]  # OpeningId -> claim (opening points filled during verification)
    trace_length: int  # trace length T (power of two)
    ram_K: int  # RAM domain size K (power of two)
    bytecode_K: int  # bytecode domain size K (power of two)

    # Stage proofs (Python-native containers, mirroring Rust structure names/roles)
    stage1_uni_skip_first_round_proof: object
    stage1_sumcheck_proof: object
    stage2_uni_skip_first_round_proof: object
    stage2_sumcheck_proof: object
    stage3_sumcheck_proof: object
    stage4_sumcheck_proof: object
    stage5_sumcheck_proof: object
    stage6_sumcheck_proof: object
    stage7_sumcheck_proof: object

    # Stage 8 (Dory joint opening). Optional while the verifier is being integrated.
    joint_opening_proof: object | None = None
    dory_verifier_setup: object | None = None
    dory_serde_blocks: list[bytes] | None = None
    dory_layout: str = "CycleMajor"

    # Commitments (main polynomials) and advice commitments (if present).
    commitments: list[object] | None = None
    commitments_serde: list[bytes] | None = None  # Rust `CanonicalSerialize` bytes for transcript absorption.
    untrusted_advice_commitment: object | None = None
    untrusted_advice_commitment_serde: bytes | None = None  # Rust `CanonicalSerialize` bytes.
    trusted_advice_commitment: object | None = None
    trusted_advice_commitment_serde: bytes | None = None  # Rust `CanonicalSerialize` bytes.

    # Config objects needed by stages (Python-native mirrors; see `zkvm_types.py`).
    rw_config: object | None = None
    one_hot_params: object | None = None

    def seed_opening_accumulator(self, acc: VerifierOpeningAccumulator) -> None:  # Seed all opening claims into accumulator.
        for key, claim in self.opening_claims.items():
            acc.seed(key, claim)

    def absorb_commitments_into_transcript(self, transcript) -> None:  # Rust verifier: append commitments before stages.
        if self.commitments_serde is None:
            return
        for b in self.commitments_serde:
            transcript.append_serializable(b"commitment", b)
        if self.untrusted_advice_commitment_serde is not None:
            transcript.append_serializable(b"untrusted_advice", self.untrusted_advice_commitment_serde)
        if self.trusted_advice_commitment_serde is not None:
            transcript.append_serializable(b"trusted_advice", self.trusted_advice_commitment_serde)

    def has_advice_commitment(self, kind: AdviceKind) -> bool:  # True if advice commitment exists for kind.
        if kind == AdviceKind.Trusted:
            return self.trusted_advice_commitment is not None
        if kind == AdviceKind.Untrusted:
            return self.untrusted_advice_commitment is not None
        raise ValueError("unknown AdviceKind")

    @classmethod
    def from_rust_bytes(
        cls,
        proof_bin: bytes,
        *,
        verifier_preprocessing_bin: bytes | None = None,
    ) -> "JoltProof":
        """Parse Rust `proof.bin` and return a verifier-ready `JoltProof`.

        This is intentionally narrow: it supports the Rust format currently produced by the
        `jolt` prover in this repo, sufficient for Python verification.
        """
        proof = _parse_jolt_proof_bytes(bytes(proof_bin))
        if verifier_preprocessing_bin is None:
            return proof
        setup = _parse_dory_verifier_setup_from_verifier_preprocessing(bytes(verifier_preprocessing_bin))
        return replace(proof, dory_verifier_setup=setup)

    @classmethod
    def from_rust_files(
        cls,
        proof_path: str | pathlib.Path,
        *,
        verifier_preprocessing_path: str | pathlib.Path | None = None,
    ) -> "JoltProof":
        proof_path = pathlib.Path(proof_path)
        pp = None if verifier_preprocessing_path is None else pathlib.Path(verifier_preprocessing_path)
        return cls.from_rust_bytes(
            proof_path.read_bytes(),
            verifier_preprocessing_bin=None if pp is None else pp.read_bytes(),
        )


class RustDeserializeError(Exception):  # Raised on Rust wire-format parsing errors.
    pass


def _u64_le(b: bytes) -> int:  # parse 8 bytes as u64 le.
    return int.from_bytes(b, "little")


def _u32_le(b: bytes) -> int:  # parse 4 bytes as u32 le.
    return int.from_bytes(b, "little")


class _Reader:  # Minimal arkworks-like little-endian reader.
    def __init__(self, data: bytes):
        self.data = bytes(data)
        self.i = 0

    def remaining(self) -> int:
        return len(self.data) - self.i

    def take(self, n: int) -> bytes:
        n = int(n)
        if n < 0 or self.i + n > len(self.data):
            raise RustDeserializeError("unexpected EOF")
        out = self.data[self.i : self.i + n]
        self.i += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "little")

    def u32(self) -> int:
        return _u32_le(self.take(4))

    def u64(self) -> int:
        return _u64_le(self.take(8))

    def usize(self) -> int:  # Matches how this repo currently serializes `usize` in proof files.
        return self.u64()

    def bool(self) -> bool:
        x = self.u8()
        if x == 0:
            return False
        if x == 1:
            return True
        raise RustDeserializeError("invalid bool")

    def vec(self, read_item):
        n = self.u64()
        out = []
        for _ in range(int(n)):
            out.append(read_item())
        return out


def _swflags_from_last_byte(last: int):  # Arkworks SWFlags parsing (bits 7 and 6).
    is_negative = (last >> 7) & 1
    is_infinity = (last >> 6) & 1
    if is_negative and is_infinity:
        raise RustDeserializeError("invalid SWFlags")
    if is_infinity:
        return ("inf", None)
    return ("sign", bool(is_negative))


def _read_fq_with_swflags(r: _Reader):  # read Fq with SWFlags embedded in last byte.
    raw = bytearray(r.take(32))
    tag, sign = _swflags_from_last_byte(raw[31])
    if tag == "inf":
        raw[31] &= 0x3F
        x = int.from_bytes(raw, "little")
        if x != 0:
            raise RustDeserializeError("nonzero x for infinity")
        return None, None, bytes(raw)
    raw[31] &= 0x3F
    x = int.from_bytes(raw, "little")
    return Fq(x), bool(sign), bytes(raw)


def _read_fr(r: _Reader) -> Fr:  # read Fr (32 bytes le).
    return Fr(int.from_bytes(r.take(32), "little"))


def _read_fq(r: _Reader) -> Fq:  # read Fq (32 bytes le).
    return Fq(int.from_bytes(r.take(32), "little"))


def _read_fq2(r: _Reader) -> Fq2:  # Read ark_bn254::Fq2 compressed (64 bytes).
    return Fq2([_read_fq(r), _read_fq(r)])


def _fq_sqrt(a: Fq):  # sqrt in Fq (p % 4 == 3).
    if int(a) == 0:
        return Fq(0)
    p = Fq.MODULUS
    x = pow(int(a), (p + 1) // 4, p)
    if (Fq(x) * Fq(x)) != a:
        raise RustDeserializeError("no sqrt in Fq")
    return Fq(x)


def _fq_sqrt_opt(a: Fq):  # Optional sqrt in Fq, returns None if non-residue.
    try:
        return _fq_sqrt(a)
    except RustDeserializeError:
        return None


def _fq_is_positive(y: Fq) -> bool:  # arkworks: y <= -y.
    yi = int(y)
    ny = (Fq.MODULUS - yi) % Fq.MODULUS
    return yi <= ny


def _fq2_is_positive(y: Fq2) -> bool:  # arkworks: lexicographic compare c1 then c0, on y and -y.
    c0, c1 = y.c
    nc0, nc1 = (-c0), (-c1)
    if int(c1) != int(nc1):
        return int(c1) < int(nc1)
    return int(c0) <= int(nc0)


def _fq2_sqrt(a: Fq2):  # sqrt in Fq2 for u^2 = -1 (BN254).
    a0, a1 = a.c
    if int(a1) == 0:
        r0 = _fq_sqrt(a0)
        return Fq2([r0, 0])
    t = _fq_sqrt(a0 * a0 + a1 * a1)
    two_inv = Fq(2).inv()
    x2 = (t + a0) * two_inv
    x = _fq_sqrt_opt(x2)
    if x is None:
        x2 = (a0 - t) * two_inv
        x = _fq_sqrt_opt(x2)
    if x is None:
        raise RustDeserializeError("no sqrt in Fq2")
    y = a1 * (Fq(2) * x).inv()
    return Fq2([x, y])


def _read_g1_projective(r: _Reader):  # parse ark_bn254::G1Projective compressed (same as affine compressed).
    x, sign_is_negative, _masked = _read_fq_with_swflags(r)
    if x is None:
        return None
    rhs = x * x * x + Fq(3)
    y = _fq_sqrt(rhs)
    want_positive = not bool(sign_is_negative)
    if _fq_is_positive(y) != want_positive:
        y = -y
    return (x, y)


def _read_g2_projective(r: _Reader):  # parse ark_bn254::G2Projective compressed.
    c0 = _read_fq(r)
    raw1 = bytearray(r.take(32))
    tag, sign = _swflags_from_last_byte(raw1[31])
    if tag == "inf":
        raw1[31] &= 0x3F
        x1 = int.from_bytes(raw1, "little")
        if int(c0) != 0 or x1 != 0:
            raise RustDeserializeError("nonzero x for infinity")
        return None
    raw1[31] &= 0x3F
    c1 = Fq(int.from_bytes(raw1, "little"))
    x = Fq2([c0, c1])
    from curve import b2  # local import to avoid cycles

    rhs = x * x * x + b2
    y = _fq2_sqrt(rhs)
    want_positive = not bool(sign)
    if _fq2_is_positive(y) != want_positive:
        y = -y
    return (x, y)


class _Fq6T:  # Minimal BN254 Fq6 tower (Fp6 over Fq2, NONRESIDUE = u+9).
    def __init__(self, c0, c1, c2):
        self.c0, self.c1, self.c2 = c0, c1, c2

    @staticmethod
    def zero():
        return _Fq6T(Fq2.zero(), Fq2.zero(), Fq2.zero())

    @staticmethod
    def one():
        return _Fq6T(Fq2.one(), Fq2.zero(), Fq2.zero())

    def __add__(self, o):
        return _Fq6T(self.c0 + o.c0, self.c1 + o.c1, self.c2 + o.c2)

    def __sub__(self, o):
        return _Fq6T(self.c0 - o.c0, self.c1 - o.c1, self.c2 - o.c2)

    def __mul__(self, o):
        nr = Fq2([9, 1])
        a0, a1, a2 = self.c0, self.c1, self.c2
        b0, b1, b2 = o.c0, o.c1, o.c2
        c0 = a0 * b0 + (a1 * b2 + a2 * b1) * nr
        c1 = a0 * b1 + a1 * b0 + (a2 * b2) * nr
        c2 = a0 * b2 + a1 * b1 + a2 * b0
        return _Fq6T(c0, c1, c2)


class _Fq12T:  # Minimal BN254 Fq12 tower (Fp12 over Fq6, NONRESIDUE = v).
    def __init__(self, c0, c1):
        self.c0, self.c1 = c0, c1

    @staticmethod
    def zero():
        return _Fq12T(_Fq6T.zero(), _Fq6T.zero())

    @staticmethod
    def one():
        return _Fq12T(_Fq6T.one(), _Fq6T.zero())

    @staticmethod
    def w():
        return _Fq12T(_Fq6T.zero(), _Fq6T.one())

    def __add__(self, o):
        return _Fq12T(self.c0 + o.c0, self.c1 + o.c1)

    def __sub__(self, o):
        return _Fq12T(self.c0 - o.c0, self.c1 - o.c1)

    def __mul__(self, o):
        a0, a1 = self.c0, self.c1
        b0, b1 = o.c0, o.c1
        t0 = a0 * b0
        t1 = a1 * b1
        c0 = t0 + _fq6_mul_by_v(t1)
        c1 = (a0 + a1) * (b0 + b1) - t0 - t1
        return _Fq12T(c0, c1)

    def pow(self, e: int):
        e = int(e)
        out = _Fq12T.one()
        base = self
        while e > 0:
            if e & 1:
                out = out * base
            base = base * base
            e >>= 1
        return out


def _fq6_mul_by_v(x: _Fq6T) -> _Fq6T:  # Multiply by v where v^3 = (u+9).
    nr = Fq2([9, 1])
    return _Fq6T(x.c2 * nr, x.c0, x.c1)


def _fq12_tower_vec(x: _Fq12T):  # Flatten tower coords into 12 base-field coeffs (Fq).
    out = []
    for f6 in (x.c0, x.c1):
        for f2 in (f6.c0, f6.c1, f6.c2):
            out.append(f2.c[0])
            out.append(f2.c[1])
    return out


_GT_INV_M: list[list[Fq]] | None = None  # cached 12x12 inverse matrix over Fq


def _gt_inv_m():  # Compute inverse map: tower_vec -> poly coeffs in {1,w,...,w^11}.
    global _GT_INV_M
    if _GT_INV_M is not None:
        return _GT_INV_M
    w = _Fq12T.w()
    cols = []
    for j in range(12):
        cols.append(_fq12_tower_vec(w.pow(j)))
    m = [[cols[j][i] for j in range(12)] for i in range(12)]
    a = [[m[i][j] for j in range(12)] + [Fq(1 if i == j else 0) for j in range(12)] for i in range(12)]
    for col in range(12):
        piv = None
        for row in range(col, 12):
            if int(a[row][col]) != 0:
                piv = row
                break
        if piv is None:
            raise RustDeserializeError("GT basis matrix is singular")
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
        inv = a[col][col].inv()
        a[col] = [x * inv for x in a[col]]
        for row in range(12):
            if row == col:
                continue
            f = a[row][col]
            if int(f) == 0:
                continue
            a[row] = [a[row][j] - f * a[col][j] for j in range(24)]
    _GT_INV_M = [row[12:] for row in a]
    return _GT_INV_M


def _read_gt(r: _Reader):  # Read ark_bn254::Fq12 compressed (384 bytes) and convert to polynomial-basis GT.
    raw = r.take(384)
    rr = _Reader(raw)
    c0 = _Fq6T(_read_fq2(rr), _read_fq2(rr), _read_fq2(rr))
    c1 = _Fq6T(_read_fq2(rr), _read_fq2(rr), _read_fq2(rr))
    v = _fq12_tower_vec(_Fq12T(c0, c1))
    inv_m = _gt_inv_m()
    coeffs = []
    for i in range(12):
        s = Fq(0)
        for j in range(12):
            s += inv_m[i][j] * v[j]
        coeffs.append(s)
    return GT(coeffs), raw


def _sumcheck_id(i: int) -> SumcheckId:  # Rust SumcheckId::from_u8.
    i = int(i)
    if i < 0 or i >= len(SUMCHECK_IDS):
        raise RustDeserializeError("bad SumcheckId")
    return SumcheckId[SUMCHECK_IDS[i]]


def _virtual_poly(i: int) -> VirtualPolynomial:  # Rust VirtualPolynomial::from_u8.
    i = int(i)
    if i < 0 or i >= len(VIRTUAL_POLYS):
        raise RustDeserializeError("bad VirtualPolynomial")
    name = VIRTUAL_POLYS[i]
    return VirtualPolynomial[name.replace(".", "_")]


def _committed_poly(r: _Reader):  # Rust CommittedPolynomial custom encoding in proof_serialization.rs.
    tag = r.u8()
    if tag == 0:
        return (CommittedPolynomial.RdInc, None)
    if tag == 1:
        return (CommittedPolynomial.RamInc, None)
    if tag == 2:
        return (CommittedPolynomial.InstructionRa, int(r.u8()))
    if tag == 3:
        return (CommittedPolynomial.BytecodeRa, int(r.u8()))
    if tag == 4:
        return (CommittedPolynomial.RamRa, int(r.u8()))
    if tag == 5:
        return (CommittedPolynomial.TrustedAdvice, None)
    if tag == 6:
        return (CommittedPolynomial.UntrustedAdvice, None)
    raise RustDeserializeError("bad CommittedPolynomial tag")


def _virtual_poly_and_index(r: _Reader):  # Rust VirtualPolynomial encoding in proof_serialization.rs.
    tag = r.u8()
    if tag == 27:
        i = int(r.u8())
        return VirtualPolynomial.InstructionRa, i
    if tag <= 37:
        name = VIRTUAL_POLYS[tag]
        return VirtualPolynomial[name.replace(".", "_")], None
    if tag == 38:
        d = int(r.u8())
        names = [
            "AddOperands",
            "SubtractOperands",
            "MultiplyOperands",
            "Load",
            "Store",
            "Jump",
            "WriteLookupOutputToRD",
            "VirtualInstruction",
            "Assert",
            "DoNotUpdateUnexpandedPC",
            "Advice",
            "IsCompressed",
            "IsFirstInSequence",
            "IsLastInSequence",
        ]
        if d < 0 or d >= len(names):
            raise RustDeserializeError("bad CircuitFlags discriminant")
        return VirtualPolynomial["OpFlags_" + names[d]], None
    if tag == 39:
        d = int(r.u8())
        names = [
            "LeftOperandIsPC",
            "RightOperandIsImm",
            "LeftOperandIsRs1Value",
            "RightOperandIsRs2Value",
            "Branch",
            "IsNoop",
            "IsRdNotZero",
        ]
        if d < 0 or d >= len(names):
            raise RustDeserializeError("bad InstructionFlags discriminant")
        return VirtualPolynomial["InstructionFlags_" + names[d]], None
    if tag == 40:
        i = int(r.u8())
        return VirtualPolynomial.LookupTableFlag, i
    raise RustDeserializeError("bad VirtualPolynomial tag")


def _opening_id(r: _Reader) -> OpeningId:  # Rust OpeningId compact encoding.
    fused = r.u8()
    n = len(SUMCHECK_IDS)
    if fused < n:
        return OpeningId(sumcheck_id=_sumcheck_id(fused), advice_kind=AdviceKind.Untrusted)
    if fused < 2 * n:
        return OpeningId(sumcheck_id=_sumcheck_id(fused - n), advice_kind=AdviceKind.Trusted)
    if fused < 3 * n:
        sid = _sumcheck_id(fused - 2 * n)
        poly, idx = _committed_poly(r)
        return OpeningId(sumcheck_id=sid, committed=CommittedId(poly, idx))
    sid = _sumcheck_id(fused - 3 * n)
    poly, idx = _virtual_poly_and_index(r)
    return OpeningId(sumcheck_id=sid, virtual=VirtualId(poly, idx))


def _read_uniskip_first_round_proof(r: _Reader):  # UniSkipFirstRoundProof = UniPoly + marker.
    coeffs = r.vec(lambda: _read_fr(r))
    return UniSkipFirstRoundProof(UniPoly(coeffs))


def _read_sumcheck_instance_proof(r: _Reader):  # SumcheckInstanceProof = Vec<CompressedUniPoly> + marker.
    polys = r.vec(lambda: CompressedUniPoly(r.vec(lambda: _read_fr(r))))
    return SumcheckInstanceProof(polys)


def _read_dory_proof(r: _Reader):  # Parse ArkDoryProof and also produce Dory transcript serde blocks.
    vmv_c, vmv_c_raw = _read_gt(r)
    vmv_d2, vmv_d2_raw = _read_gt(r)
    vmv_e1_raw = r.take(32)
    vmv_e1 = _read_g1_projective(_Reader(vmv_e1_raw))
    num_rounds = int(r.u32())
    first_msgs = []
    first_raws = []
    for _ in range(num_rounds):
        d1_left, raw1 = _read_gt(r)
        d1_right, raw2 = _read_gt(r)
        d2_left, raw3 = _read_gt(r)
        d2_right, raw4 = _read_gt(r)
        e1b_raw = r.take(32)
        e2b_raw = r.take(64)
        e1_beta = _read_g1_projective(_Reader(e1b_raw))
        e2_beta = _read_g2_projective(_Reader(e2b_raw))
        first_msgs.append(dory.FirstReduceMessage(d1_left, d1_right, d2_left, d2_right, e1_beta, e2_beta))
        first_raws.append([raw1, raw2, raw3, raw4, e1b_raw, e2b_raw])
    second_msgs = []
    second_raws = []
    for _ in range(num_rounds):
        c_plus, raw5 = _read_gt(r)
        c_minus, raw6 = _read_gt(r)
        e1p_raw = r.take(32)
        e1m_raw = r.take(32)
        e2p_raw = r.take(64)
        e2m_raw = r.take(64)
        e1_plus = _read_g1_projective(_Reader(e1p_raw))
        e1_minus = _read_g1_projective(_Reader(e1m_raw))
        e2_plus = _read_g2_projective(_Reader(e2p_raw))
        e2_minus = _read_g2_projective(_Reader(e2m_raw))
        second_msgs.append(dory.SecondReduceMessage(c_plus, c_minus, e1_plus, e1_minus, e2_plus, e2_minus))
        second_raws.append([raw5, raw6, e1p_raw, e1m_raw, e2p_raw, e2m_raw])
    fe1_raw = r.take(32)
    fe2_raw = r.take(64)
    fe1 = _read_g1_projective(_Reader(fe1_raw))
    fe2 = _read_g2_projective(_Reader(fe2_raw))
    nu = int(r.u32())
    sigma = int(r.u32())
    blocks = [vmv_c_raw, vmv_d2_raw, vmv_e1_raw]
    for i in range(num_rounds):
        blocks += first_raws[i]
        blocks += second_raws[i]
    blocks += [fe1_raw, fe2_raw]
    proof = dory.DoryProof(
        dory.VMVMessage(vmv_c, vmv_d2, vmv_e1),
        first_msgs,
        second_msgs,
        dory.ScalarProductMessage(fe1, fe2),
        nu,
        sigma,
    )
    return proof, blocks


def _parse_jolt_proof_bytes(data: bytes) -> JoltProof:
    r = _Reader(data)

    opening_claims: dict[OpeningId, Fr] = {}
    n = r.usize()
    for _ in range(int(n)):
        oid = _opening_id(r)
        opening_claims[oid] = _read_fr(r)

    commitments = []
    commitments_serde = []
    for _ in range(int(r.u64())):
        gt, raw = _read_gt(r)
        commitments.append(gt)
        commitments_serde.append(raw)

    stage1_uni = _read_uniskip_first_round_proof(r)
    stage1_sc = _read_sumcheck_instance_proof(r)
    stage2_uni = _read_uniskip_first_round_proof(r)
    stage2_sc = _read_sumcheck_instance_proof(r)
    stage3_sc = _read_sumcheck_instance_proof(r)
    stage4_sc = _read_sumcheck_instance_proof(r)
    stage5_sc = _read_sumcheck_instance_proof(r)
    stage6_sc = _read_sumcheck_instance_proof(r)
    stage7_sc = _read_sumcheck_instance_proof(r)

    joint_opening_proof, dory_blocks = _read_dory_proof(r)

    untrusted_commitment = None
    untrusted_commitment_serde = None
    if r.bool():
        untrusted_commitment, raw = _read_gt(r)
        untrusted_commitment_serde = raw

    trace_length = int(r.usize())
    ram_k = int(r.usize())
    bytecode_k = int(r.usize())
    rw_config = ReadWriteConfig(
        ram_rw_phase1_num_rounds=int(r.u8()),
        ram_rw_phase2_num_rounds=int(r.u8()),
        registers_rw_phase1_num_rounds=int(r.u8()),
        registers_rw_phase2_num_rounds=int(r.u8()),
    )
    one_hot_config = {"log_k_chunk": int(r.u8()), "lookups_ra_virtual_log_k_chunk": int(r.u8())}
    one_hot_params = OneHotParams(
        ram_k=ram_k,
        bytecode_k=bytecode_k,
        log_k_chunk=int(one_hot_config["log_k_chunk"]),
        lookups_ra_virtual_log_k_chunk=int(one_hot_config["lookups_ra_virtual_log_k_chunk"]),
    )
    dory_layout = int(r.u8())
    if dory_layout not in (0, 1):
        raise RustDeserializeError("bad dory_layout")
    dory_layout_s = "CycleMajor" if dory_layout == 0 else "AddressMajor"
    if r.remaining() != 0:
        raise RustDeserializeError("trailing bytes in JoltProof")

    return JoltProof(
        opening_claims=opening_claims,
        trace_length=trace_length,
        ram_K=ram_k,
        bytecode_K=bytecode_k,
        stage1_uni_skip_first_round_proof=stage1_uni,
        stage1_sumcheck_proof=stage1_sc,
        stage2_uni_skip_first_round_proof=stage2_uni,
        stage2_sumcheck_proof=stage2_sc,
        stage3_sumcheck_proof=stage3_sc,
        stage4_sumcheck_proof=stage4_sc,
        stage5_sumcheck_proof=stage5_sc,
        stage6_sumcheck_proof=stage6_sc,
        stage7_sumcheck_proof=stage7_sc,
        joint_opening_proof=joint_opening_proof,
        dory_serde_blocks=dory_blocks,
        dory_layout=dory_layout_s,
        commitments=commitments,
        commitments_serde=commitments_serde,
        untrusted_advice_commitment=untrusted_commitment,
        untrusted_advice_commitment_serde=untrusted_commitment_serde,
        trusted_advice_commitment=None,
        trusted_advice_commitment_serde=None,
        rw_config=rw_config,
        one_hot_params=one_hot_params,
    )


def _parse_dory_verifier_setup_from_verifier_preprocessing(data: bytes):
    r = _Reader(data)
    delta_1l = r.vec(lambda: _read_gt(r)[0])
    delta_1r = r.vec(lambda: _read_gt(r)[0])
    delta_2l = r.vec(lambda: _read_gt(r)[0])
    delta_2r = r.vec(lambda: _read_gt(r)[0])
    chi = r.vec(lambda: _read_gt(r)[0])
    g1_0 = _read_g1_projective(r)
    g2_0 = _read_g2_projective(r)
    h1 = _read_g1_projective(r)
    h2 = _read_g2_projective(r)
    ht = _read_gt(r)[0]
    _max_log_n = int.from_bytes(r.take(8), "little")
    return dory.DoryVerifierSetup(delta_1l, delta_1r, delta_2l, delta_2r, chi, g1_0, g2_0, h1, h2, ht)
