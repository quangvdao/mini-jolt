from polynomials import CompressedUniPoly  # sumcheck message type
from polynomials import EqPlusOnePolynomial, EqPolynomial, LagrangePolynomial, RangeMaskPolynomial, UniPoly, UnmapRamAddressPolynomial, log2_pow2  # verifier-minimal polynomial helpers
from polynomials import IdentityPolynomial, OperandPolynomial  # instruction-lookup helper polynomials
from openings import (  # opening accumulator types
    AdviceKind,
    BIG_ENDIAN,
    LITTLE_ENDIAN,
    CommittedPolynomial,
    OpeningPoint,
    SumcheckId,
    VirtualPolynomial,
)
from ids_generated import LOOKUP_TABLES_64  # canonical Rust LookupTables<64> order
from lookup_tables import evaluate_mle  # lookup table MLE evaluation
from ram_io import (
    calculate_advice_memory_evaluation,
    eval_initial_ram_mle,
    eval_io_mle,
    remap_address,
    verifier_accumulate_advice,
)  # RAM helper MLEs
from r1cs import (  # Spartan outer key + constants
    ALL_R1CS_INPUTS,
    OUTER_FIRST_ROUND_POLY_DEGREE_BOUND,
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
    OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    UniformSpartanKey,
)
from rv64imac.constants import RAM_START_ADDRESS, REGISTER_COUNT  # shared constants
from zkvm_types import JoltDevice, MemoryLayout, OneHotParams, RAMPreprocessing, ReadWriteConfig  # Stage 2/4 public types

class SumcheckVerifyError(Exception):  # Raised on sumcheck verification failure.
    pass

class SumcheckInstanceProof:  # Sumcheck proof consisting of per-round compressed univariates.
    def __init__(self, compressed_polys):  # Store a list of CompressedUniPoly.
        self.compressed_polys = list(compressed_polys)

    def verify(self, claim, num_rounds, degree_bound, transcript):  # Mirror Rust `SumcheckInstanceProof::verify`.
        e = claim
        r = []
        num_rounds = int(num_rounds)
        degree_bound = int(degree_bound)
        if len(self.compressed_polys) != num_rounds:
            raise SumcheckVerifyError("invalid proof length for num_rounds")
        for poly in self.compressed_polys:
            if poly.degree() > degree_bound:
                raise SumcheckVerifyError("degree bound exceeded")
            transcript.append_scalars(b"sumcheck_poly", poly.coeffs_except_linear_term)
            r_i = transcript.challenge_scalar_optimized()
            r.append(r_i)
            e = poly.eval_from_hint(e, r_i)
        return e, r

# ---------------------------------------------------------------------------
# Shared helpers — reduce boilerplate across sumcheck verifiers
# ---------------------------------------------------------------------------

def _normalize_le_to_be(challenges):  # Flip little-endian sumcheck challenges to big-endian OpeningPoint.
    return OpeningPoint(list(challenges), LITTLE_ENDIAN).match_endianness(BIG_ENDIAN)

def _cache_virtual_batch(accumulator, transcript, names, sumcheck_id, opening_point):  # Enqueue several virtual openings at the same point.
    for name in names:
        accumulator.append_virtual(transcript, name, sumcheck_id, opening_point.clone())

class SumcheckInstanceVerifier:  # Interface for a sumcheck instance used in batched verification.
    def degree(self):  # Return max degree of this instance.
        raise NotImplementedError()

    def num_rounds(self):  # Return number of rounds/variables in this instance.
        raise NotImplementedError()

    def round_offset(self, max_num_rounds):  # Front-loaded batching: offset = max - self.num_rounds().
        return int(max_num_rounds) - int(self.num_rounds())

    def input_claim(self, opening_accumulator):  # Return initial claim for this instance.
        raise NotImplementedError()

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Return expected final claim.
        raise NotImplementedError()

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Enqueue openings.
        raise NotImplementedError()

class BatchedSumcheck:  # Rust-style front-loaded batched sumcheck verification.
    @staticmethod
    def verify(proof, instances, opening_accumulator, transcript):  # Verify batched sumcheck.
        instances = list(instances)
        if not instances:
            raise ValueError("instances must be non-empty")
        max_degree = max(int(s.degree()) for s in instances)
        max_num_rounds = max(int(s.num_rounds()) for s in instances)

        for s in instances:
            transcript.append_scalar(b"sumcheck_claim", s.input_claim(opening_accumulator))

        batching_coeffs = transcript.challenge_vector(len(instances))

        claim = None
        for s, coeff in zip(instances, batching_coeffs):
            input_claim = s.input_claim(opening_accumulator)
            scaled = input_claim * (1 << (max_num_rounds - int(s.num_rounds())))
            term = scaled * coeff
            claim = term if claim is None else (claim + term)

        output_claim, r_sumcheck = proof.verify(claim, max_num_rounds, max_degree, transcript)

        expected = None
        for s, coeff in zip(instances, batching_coeffs):
            offset = int(s.round_offset(max_num_rounds))
            r_slice = r_sumcheck[offset : offset + int(s.num_rounds())]
            s.cache_openings(opening_accumulator, transcript, r_slice)
            term = s.expected_output_claim(opening_accumulator, r_slice) * coeff
            expected = term if expected is None else (expected + term)

        if output_claim != expected:
            raise SumcheckVerifyError("sumcheck output claim mismatch")
        return r_sumcheck

class UniSkipVerifyError(Exception):  # Raised on univariate-skip first-round verification failure.
    pass

class UniSkipFirstRoundProof:  # Univariate-skip first-round proof: full uncompressed univariate polynomial.
    def __init__(self, uni_poly):  # Store `UniPoly` with full coefficients.
        if not isinstance(uni_poly, UniPoly):
            uni_poly = UniPoly(uni_poly)
        self.uni_poly = uni_poly

    def verify(self, domain_size, expected_num_coeffs, verifier, opening_accumulator, transcript):  # Mirror Rust `UniSkipFirstRoundProof::verify`.
        degree_bound = int(verifier.degree())
        if self.uni_poly.degree() > degree_bound:
            raise UniSkipVerifyError("degree bound exceeded")
        if len(self.uni_poly.coeffs) != int(expected_num_coeffs):
            raise UniSkipVerifyError("invalid first-round poly length")
        transcript.append_scalars(b"uniskip_poly", self.uni_poly.coeffs)
        r0 = transcript.challenge_scalar_optimized()
        input_claim = verifier.input_claim(opening_accumulator)
        ok = self.uni_poly.check_sum_evals_symmetric_domain(
            int(domain_size),
            input_claim,
            expected_num_coeffs=int(expected_num_coeffs),
        )
        verifier.cache_openings(opening_accumulator, transcript, [r0])
        if not ok:
            raise UniSkipVerifyError("uniskip symmetric-domain sum check failed")
        return r0


# ===========================================================================
# Backward-compatible re-exports from stages/ package
# ===========================================================================

from stages.stage1 import SpartanOuterUniSkipParams, SpartanOuterUniSkipVerifier, SpartanOuterRemainingSumcheckVerifier
from stages.stage2 import (
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_UNIQUE_FACTOR_VIRTUALS,
    ProductVirtualUniSkipParams,
    ProductVirtualUniSkipVerifier,
    ProductVirtualRemainderVerifier,
    InstructionLookupsClaimReductionSumcheckVerifier,
    RamRafEvaluationSumcheckVerifier,
    OutputSumcheckVerifier,
    RamReadWriteCheckingVerifier,
)
from stages.stage3 import ShiftSumcheckVerifier, InstructionInputSumcheckVerifier, RegistersClaimReductionSumcheckVerifier
from stages.stage4 import RegistersReadWriteCheckingVerifier, RamValEvaluationSumcheckVerifier, ValFinalSumcheckVerifier
from stages.stage5 import InstructionReadRafSumcheckVerifier, RamRaClaimReductionSumcheckVerifier, RegistersValEvaluationSumcheckVerifier
from stages.stage6 import (
    HammingBooleanitySumcheckVerifier,
    RamRaVirtualSumcheckVerifier,
    InstructionRaVirtualSumcheckVerifier,
    IncClaimReductionSumcheckVerifier,
    AdviceClaimReductionVerifier,
    BooleanitySumcheckVerifier,
    BytecodeReadRafSumcheckVerifier,
)
from stages.stage7 import HammingWeightClaimReductionSumcheckVerifier
