from dataclasses import dataclass  # minimal typed IDs (Rust-like ADT ergonomics)
from enum import StrEnum  # typed string identifiers

from ids_generated import CommittedPolynomial, SumcheckId, VirtualPolynomial  # generated canonical IDs

class OpeningError(Exception):  # Raised when openings are missing/malformed.
    pass


BIG_ENDIAN = "big"  # Big-endian opening point bit order (matches Rust `BIG_ENDIAN=false`).
LITTLE_ENDIAN = "little"  # Little-endian opening point bit order (matches Rust `LITTLE_ENDIAN=true`).


class OpeningPoint:  # Opening point wrapper with explicit endianness.
    def __init__(self, r, endianness=BIG_ENDIAN):  # Store challenge vector `r` + endianness.
        self.r = list(r)
        if endianness not in (BIG_ENDIAN, LITTLE_ENDIAN):
            raise ValueError("invalid endianness")
        self.endianness = endianness

    def __len__(self):  # Length of opening point.
        return len(self.r)

    def __getitem__(self, idx):  # Index into opening point coordinates.
        return self.r[idx]

    def clone(self):  # Cheap copy for callers expecting Rust-like clone.
        return OpeningPoint(self.r, self.endianness)

    def match_endianness(self, target_endianness):  # Reverse coordinates if endianness differs.
        if target_endianness not in (BIG_ENDIAN, LITTLE_ENDIAN):
            raise ValueError("invalid target endianness")
        if self.endianness == target_endianness:
            return OpeningPoint(self.r, self.endianness)
        return OpeningPoint(list(reversed(self.r)), target_endianness)


class AdviceKind(StrEnum):  # Advice opening kind identifiers.
    Untrusted = "untrusted"
    Trusted = "trusted"


@dataclass(frozen=True, slots=True)
class VirtualId:  # Virtual polynomial ID (optionally indexed family).
    polynomial: VirtualPolynomial  # Base virtual polynomial name (family head if indexed).
    index: int | None = None  # Optional family index (e.g. InstructionRa(i)).


@dataclass(frozen=True, slots=True)
class CommittedId:  # Committed polynomial ID (optionally indexed family).
    polynomial: CommittedPolynomial  # Base committed polynomial name (family head if indexed).
    index: int | None = None  # Optional family index (e.g. RamRa(i)).


@dataclass(frozen=True, slots=True)
class OpeningId:  # Typed opening ID (Rust-like discriminated union).
    sumcheck_id: SumcheckId  # Sumcheck instance that owns the opening.
    virtual: VirtualId | None = None  # Virtual polynomial opening, if any.
    committed: CommittedId | None = None  # Committed polynomial opening, if any.
    advice_kind: AdviceKind | None = None  # Advice opening kind, if any.


def _opening_virtual(sumcheck_id, virtual_polynomial, index=None):  # Construct OpeningId for a virtual opening.
    return OpeningId(sumcheck_id=sumcheck_id, virtual=VirtualId(virtual_polynomial, index=index))


def _opening_committed(sumcheck_id, committed_polynomial, index=None):  # Construct OpeningId for a committed opening.
    return OpeningId(sumcheck_id=sumcheck_id, committed=CommittedId(committed_polynomial, index=index))


def _opening_advice(sumcheck_id, advice_kind):  # Construct OpeningId for an advice opening.
    return OpeningId(sumcheck_id=sumcheck_id, advice_kind=advice_kind)


class VerifierOpeningAccumulator:  # Verifier-side opening accumulator (typed keys, minimal API).
    def __init__(self):  # Initialize empty opening map.
        self._openings = {}  # OpeningId -> (OpeningPoint|None, claim)

    def seed(self, opening_id, claim):  # Pre-seed claim from proof parsing.
        self._openings[opening_id] = (None, claim)

    def get(self, opening_id):  # Return (OpeningPoint, claim) for seeded opening.
        if opening_id not in self._openings:
            raise OpeningError(f"no opening for {opening_id}")
        point, claim = self._openings[opening_id]
        if point is None:
            point = OpeningPoint([], BIG_ENDIAN)
        return point, claim

    def maybe_get(self, opening_id):  # Return Optional[(OpeningPoint, claim)].
        if opening_id not in self._openings:
            return None
        point, claim = self._openings[opening_id]
        if point is None:
            point = OpeningPoint([], BIG_ENDIAN)
        return point, claim

    def append(self, transcript, opening_id, opening_point):  # Populate opening point + transcript-couple claim.
        if opening_id not in self._openings:
            raise OpeningError(f"append for missing claim key {opening_id}")
        _, claim = self._openings[opening_id]
        transcript.append_scalar(b"opening_claim", claim)
        self._openings[opening_id] = (opening_point, claim)

    def set_virtual_claim(self, virtual_polynomial, sumcheck_id, claim):  # Back-compat: seed virtual claim.
        self.seed(_opening_virtual(sumcheck_id, virtual_polynomial), claim)

    def set_virtual_claim_i(self, virtual_polynomial, index, sumcheck_id, claim):  # Seed indexed virtual family claim.
        self.seed(_opening_virtual(sumcheck_id, virtual_polynomial, index=int(index)), claim)

    def set_committed_claim(self, committed_polynomial, sumcheck_id, claim):  # Back-compat: seed committed claim.
        self.seed(_opening_committed(sumcheck_id, committed_polynomial), claim)

    def set_committed_claim_i(self, committed_polynomial, index, sumcheck_id, claim):  # Seed indexed committed family claim.
        self.seed(_opening_committed(sumcheck_id, committed_polynomial, index=int(index)), claim)

    def set_untrusted_advice_claim(self, sumcheck_id, claim):  # Back-compat: seed untrusted advice claim.
        self.seed(_opening_advice(sumcheck_id, AdviceKind.Untrusted), claim)

    def set_trusted_advice_claim(self, sumcheck_id, claim):  # Back-compat: seed trusted advice claim.
        self.seed(_opening_advice(sumcheck_id, AdviceKind.Trusted), claim)

    def get_virtual_polynomial_opening(self, virtual_polynomial, sumcheck_id):  # Back-compat: get (OpeningPoint, claim).
        return self.get(_opening_virtual(sumcheck_id, virtual_polynomial))

    def get_virtual_polynomial_opening_i(self, virtual_polynomial, index, sumcheck_id):  # Get (OpeningPoint, claim) for indexed virtual.
        return self.get(_opening_virtual(sumcheck_id, virtual_polynomial, index=int(index)))

    def get_committed_polynomial_opening(self, committed_polynomial, sumcheck_id):  # Back-compat: get (OpeningPoint, claim).
        return self.get(_opening_committed(sumcheck_id, committed_polynomial))

    def get_committed_polynomial_opening_i(self, committed_polynomial, index, sumcheck_id):  # Get (OpeningPoint, claim) for indexed committed.
        return self.get(_opening_committed(sumcheck_id, committed_polynomial, index=int(index)))

    def get_advice_opening(self, kind, sumcheck_id):  # Back-compat: Optional[(OpeningPoint, claim)].
        if kind in (AdviceKind.Untrusted, AdviceKind.Trusted):
            advice_kind = kind
        else:
            advice_kind = AdviceKind(str(kind))
        return self.maybe_get(_opening_advice(sumcheck_id, advice_kind))

    def append_virtual(self, transcript, virtual_polynomial, sumcheck_id, opening_point):  # Back-compat: append virtual opening.
        self.append(transcript, _opening_virtual(sumcheck_id, virtual_polynomial), opening_point)

    def append_virtual_i(self, transcript, virtual_polynomial, index, sumcheck_id, opening_point):  # Append indexed virtual opening.
        self.append(transcript, _opening_virtual(sumcheck_id, virtual_polynomial, index=int(index)), opening_point)

    def append_dense(self, transcript, committed_polynomial, sumcheck_id, opening_point):  # Back-compat: append committed (dense) opening.
        self.append(
            transcript,
            _opening_committed(sumcheck_id, committed_polynomial),
            OpeningPoint(list(opening_point), BIG_ENDIAN),
        )

    def append_dense_i(self, transcript, committed_polynomial, index, sumcheck_id, opening_point):  # Append indexed committed opening.
        self.append(
            transcript,
            _opening_committed(sumcheck_id, committed_polynomial, index=int(index)),
            OpeningPoint(list(opening_point), BIG_ENDIAN),
        )

    def append_sparse(self, transcript, committed_polynomials, sumcheck_id, opening_point):  # Future: append committed (sparse) openings.
        for committed_polynomial in list(committed_polynomials):
            self.append_dense(transcript, committed_polynomial, sumcheck_id, opening_point)

    def append_untrusted_advice(self, transcript, sumcheck_id, opening_point):  # Back-compat: append untrusted advice opening.
        self.append(transcript, _opening_advice(sumcheck_id, AdviceKind.Untrusted), opening_point)

    def append_trusted_advice(self, transcript, sumcheck_id, opening_point):  # Back-compat: append trusted advice opening.
        self.append(transcript, _opening_advice(sumcheck_id, AdviceKind.Trusted), opening_point)
