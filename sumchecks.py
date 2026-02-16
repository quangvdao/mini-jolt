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
# Stage 1: Spartan outer sumcheck
# ===========================================================================

class SpartanOuterUniSkipParams:  # Params for Spartan outer uni-skip (stores tau only).
    def __init__(self, tau):  # Store tau vector (challenge-vector-optimized).
        self.tau = list(tau)

class SpartanOuterUniSkipVerifier(SumcheckInstanceVerifier):  # Verifier instance for Spartan outer uni-skip.
    def __init__(self, key, transcript):  # Sample tau from transcript, matching Rust `OuterUniSkipParams::new`.
        if not isinstance(key, UniformSpartanKey):
            key = UniformSpartanKey(int(key))
        self.key = key
        self.params = SpartanOuterUniSkipParams(transcript.challenge_vector_optimized(key.num_rows_bits()))

    def degree(self):  # First-round uniskip poly degree bound.
        return OUTER_FIRST_ROUND_POLY_DEGREE_BOUND

    def num_rounds(self):  # Uniskip is one round.
        return 1

    def input_claim(self, _opening_accumulator):  # Outer uniskip input claim is zero.
        from field import Fr  # local import to keep module dependency minimal
        return Fr.zero()

    def expected_output_claim(self, _opening_accumulator, _sumcheck_challenges):  # Unused for uni-skip.
        raise NotImplementedError("uniskip verifier does not use expected_output_claim")

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Populate r0 opening for UnivariateSkip.
        opening_point = OpeningPoint(list(sumcheck_challenges), BIG_ENDIAN)
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanOuter,
            opening_point,
        )

class SpartanOuterRemainingSumcheckVerifier(SumcheckInstanceVerifier):  # Spartan outer remaining sumcheck verifier (Stage 1b).
    OUTER_REMAINING_DEGREE_BOUND = 3  # Rust: OUTER_REMAINING_DEGREE_BOUND.

    def __init__(self, key, trace_len, uni_skip_params, opening_accumulator):  # Build verifier params from key + accumulator.
        if not isinstance(key, UniformSpartanKey):
            key = UniformSpartanKey(int(key))
        self.key = key
        self.trace_len = int(trace_len)
        self.num_cycles_bits = log2_pow2(self.trace_len)
        self.tau = list(uni_skip_params.tau)
        r_uni_skip, _ = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanOuter,
        )
        if len(r_uni_skip) != 1:
            raise ValueError("expected uniskip opening point length 1")
        self.r0 = r_uni_skip[0]

    def degree(self):  # Degree bound for remaining outer round polynomials.
        return self.OUTER_REMAINING_DEGREE_BOUND

    def num_rounds(self):  # Total rounds = 1 + num_cycle_bits (streaming round + cycle vars).
        return 1 + int(self.num_cycles_bits)

    def input_claim(self, opening_accumulator):  # Input claim is the UnivariateSkip opening claim.
        return opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanOuter,
        )[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust `OuterRemainingSumcheckVerifier::expected_output_claim`.
        r1cs_input_evals = [
            opening_accumulator.get_virtual_polynomial_opening(inp, SumcheckId.SpartanOuter)[1]
            for inp in ALL_R1CS_INPUTS
        ]
        rx_constr = [sumcheck_challenges[0], self.r0]
        inner_sum_prod = self.key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals)
        tau_high = self.tau[-1]
        tau_low = self.tau[:-1]
        tau_high_bound_r0 = LagrangePolynomial.lagrange_kernel(
            tau_high,
            self.r0,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        )
        r_tail_reversed = list(reversed(sumcheck_challenges))
        tau_bound_r_tail_reversed = EqPolynomial.mle(tau_low, r_tail_reversed)
        return tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Enqueue/cache all R1CS input openings at r_cycle.
        r_cycle = _normalize_le_to_be(sumcheck_challenges[1:])
        _cache_virtual_batch(opening_accumulator, transcript, ALL_R1CS_INPUTS, SumcheckId.SpartanOuter, r_cycle)

## NOTE: stage orchestration functions moved to `jolt_verifier.py`.

# ===========================================================================
# Stage 2: Product virtualization + batched sumchecks
# ===========================================================================

class ProductVirtualUniSkipParams:  # Params for product virtualization uni-skip (Stage 2a).
    def __init__(self, tau, base_evals):  # Store tau=[tau_low||tau_high] and 5 base evals.
        self.tau = list(tau)
        self.base_evals = list(base_evals)

PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE = 5  # Rust: PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE.
PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS = 13  # Rust: PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS.
PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND = 12  # Rust: PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND.

class ProductVirtualUniSkipVerifier(SumcheckInstanceVerifier):  # Stage 2a: univariate-skip verifier for product virtualization.

    def __init__(self, opening_accumulator, transcript):  # Build params from Stage1 openings + transcript.
        (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.Product,
            SumcheckId.SpartanOuter,
        )
        tau_high = transcript.challenge_scalar_optimized()
        tau = list(r_cycle.r) + [tau_high]
        base_names = [
            VirtualPolynomial.Product,
            VirtualPolynomial.WriteLookupOutputToRD,
            VirtualPolynomial.WritePCtoRD,
            VirtualPolynomial.ShouldBranch,
            VirtualPolynomial.ShouldJump,
        ]
        base_evals = [opening_accumulator.get_virtual_polynomial_opening(n, SumcheckId.SpartanOuter)[1] for n in base_names]
        self.params = ProductVirtualUniSkipParams(tau, base_evals)

    def degree(self):  # First-round poly degree bound.
        return PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND

    def num_rounds(self):  # One uniskip round.
        return 1

    def input_claim(self, _opening_accumulator):  # claim = Σ_i L_i(tau_high) * base_evals[i].
        tau_high = self.params.tau[-1]
        w = LagrangePolynomial.evals(tau_high, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE)
        out = None
        for wi, bi in zip(w, self.params.base_evals):
            term = wi * bi
            out = term if out is None else (out + term)
        return out

    def expected_output_claim(self, _opening_accumulator, _sumcheck_challenges):  # Unused for univariate skip.
        raise NotImplementedError("uniskip verifier does not use expected_output_claim")

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Populate UnivariateSkip opening point.
        opening_point = OpeningPoint(list(sumcheck_challenges), BIG_ENDIAN)
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanProductVirtualization,
            opening_point,
        )

# Product virtualization factor virtual polynomials (Rust: PRODUCT_UNIQUE_FACTOR_VIRTUALS).
PRODUCT_UNIQUE_FACTOR_VIRTUALS = [
    VirtualPolynomial.LeftInstructionInput,
    VirtualPolynomial.RightInstructionInput,
    VirtualPolynomial.InstructionFlags_IsRdNotZero,
    VirtualPolynomial.OpFlags_WriteLookupOutputToRD,
    VirtualPolynomial.OpFlags_Jump,
    VirtualPolynomial.LookupOutput,
    VirtualPolynomial.InstructionFlags_Branch,
    VirtualPolynomial.NextIsNoop,
    VirtualPolynomial.OpFlags_VirtualInstruction,
]

class ProductVirtualRemainderVerifier(SumcheckInstanceVerifier):  # Stage 2: product virtualization remainder sumcheck verifier.
    PRODUCT_VIRTUAL_REMAINDER_DEGREE = 3  # Rust: PRODUCT_VIRTUAL_REMAINDER_DEGREE.

    def __init__(self, trace_len, uni_skip_params, opening_accumulator):  # Build params from uniskip + accumulator.
        self.trace_len = int(trace_len)
        self.n_cycle_vars = log2_pow2(self.trace_len)
        self.tau = list(uni_skip_params.tau)
        r0_point, _ = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanProductVirtualization,
        )
        if len(r0_point) != 1:
            raise ValueError("expected product uniskip opening point length 1")
        self.r0 = r0_point[0]

    def degree(self):  # Degree bound for remainder.
        return self.PRODUCT_VIRTUAL_REMAINDER_DEGREE

    def num_rounds(self):  # Rounds = log2(T).
        return self.n_cycle_vars

    def input_claim(self, opening_accumulator):  # Input claim is uniskip claim.
        return opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanProductVirtualization,
        )[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: product.rs:666-750.
        w = LagrangePolynomial.evals(self.r0, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE)
        def g(name):
            return opening_accumulator.get_virtual_polynomial_opening(name, SumcheckId.SpartanProductVirtualization)[1]
        l_inst = g(VirtualPolynomial.LeftInstructionInput)
        r_inst = g(VirtualPolynomial.RightInstructionInput)
        is_rd_not_zero = g(VirtualPolynomial.InstructionFlags_IsRdNotZero)
        wl_flag = g(VirtualPolynomial.OpFlags_WriteLookupOutputToRD)
        j_flag = g(VirtualPolynomial.OpFlags_Jump)
        lookup_out = g(VirtualPolynomial.LookupOutput)
        branch_flag = g(VirtualPolynomial.InstructionFlags_Branch)
        next_is_noop = g(VirtualPolynomial.NextIsNoop)
        fused_left = w[0] * l_inst + w[1] * is_rd_not_zero + w[2] * is_rd_not_zero + w[3] * lookup_out + w[4] * j_flag
        from field import Fr  # local import for one()
        fused_right = w[0] * r_inst + w[1] * wl_flag + w[2] * j_flag + w[3] * branch_flag + w[4] * (Fr.one() - next_is_noop)
        tau_high = self.tau[-1]
        tau_low = self.tau[:-1]
        tau_high_bound_r0 = LagrangePolynomial.lagrange_kernel(tau_high, self.r0, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE)
        r_tail_reversed = list(reversed(sumcheck_challenges))
        tau_bound_r_tail_reversed = EqPolynomial.mle(tau_low, r_tail_reversed)
        return tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Cache factor openings at r_cycle.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        _cache_virtual_batch(opening_accumulator, transcript, PRODUCT_UNIQUE_FACTOR_VIRTUALS, SumcheckId.SpartanProductVirtualization, opening_point)

class InstructionLookupsClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 2: instruction lookups claim reduction.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, opening_accumulator, transcript):  # Sample gamma, record r_spartan.
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.r_spartan = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.SpartanOuter,
        )[0]

    def degree(self):  # Degree bound.
        return self.DEGREE_BOUND

    def num_rounds(self):  # log2(T).
        return self.n_cycle_vars

    def input_claim(self, opening_accumulator):  # LookupOutput + gamma*Left + gamma^2*Right at SpartanOuter.
        lo = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.LookupOutput, SumcheckId.SpartanOuter)[1]
        l = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.LeftLookupOperand, SumcheckId.SpartanOuter)[1]
        r = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RightLookupOperand, SumcheckId.SpartanOuter)[1]
        return lo + self.gamma * l + self.gamma_sqr * r

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: instruction_lookups.rs:486-490.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        lo = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.LookupOutput, SumcheckId.InstructionClaimReduction)[1]
        l = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.LeftLookupOperand, SumcheckId.InstructionClaimReduction)[1]
        r = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RightLookupOperand, SumcheckId.InstructionClaimReduction)[1]
        return EqPolynomial.mle(opening_point.r, self.r_spartan.r) * (lo + self.gamma * l + self.gamma_sqr * r)

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Cache 3 virtual openings.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        _cache_virtual_batch(
            opening_accumulator, transcript,
            [VirtualPolynomial.LookupOutput, VirtualPolynomial.LeftLookupOperand, VirtualPolynomial.RightLookupOperand],
            SumcheckId.InstructionClaimReduction, opening_point,
        )

class RamRafEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 2: RAM RAF evaluation.
    DEGREE_BOUND = 2

    def __init__(self, memory_layout, one_hot_params, opening_accumulator):  # Capture log_K, start_address, and r_cycle.
        self.start_address = int(memory_layout.get_lowest_address())
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.r_cycle = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamAddress, SumcheckId.SpartanOuter)[0]

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.log_K

    def input_claim(self, opening_accumulator):  # RamAddress claim at SpartanOuter.
        return opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamAddress, SumcheckId.SpartanOuter)[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: raf_evaluation.rs:289-301.
        r = _normalize_le_to_be(sumcheck_challenges)
        unmap_eval = UnmapRamAddressPolynomial(self.log_K, self.start_address).evaluate(r.r)
        ra_claim = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamRa, SumcheckId.RamRafEvaluation)[1]
        return unmap_eval * ra_claim

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Cache RamRa at r_address||r_cycle.
        r_address = _normalize_le_to_be(sumcheck_challenges)
        ra_opening_point = OpeningPoint(list(r_address.r) + list(self.r_cycle.r), BIG_ENDIAN)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamRa, SumcheckId.RamRafEvaluation, ra_opening_point)

class OutputSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 2: RAM output check.
    DEGREE_BOUND = 3

    def __init__(self, ram_K, program_io, transcript):  # Sample r_address as in Rust params.
        self.K = int(ram_K)
        self.r_address = transcript.challenge_vector_optimized(log2_pow2(self.K))
        self.program_io = program_io

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return log2_pow2(self.K)

    def input_claim(self, _opening_accumulator):
        from field import Fr  # local import
        return Fr.zero()

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: output_check.rs:261-293.
        val_final_claim = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamValFinal, SumcheckId.RamOutputCheck)[1]
        r_address_prime = _normalize_le_to_be(sumcheck_challenges).r
        io_start = int(remap_address(self.program_io.memory_layout.input_start, self.program_io.memory_layout))
        io_end = int(remap_address(RAM_START_ADDRESS, self.program_io.memory_layout))
        io_mask = RangeMaskPolynomial(io_start, io_end)
        eq_eval = EqPolynomial.mle(self.r_address, r_address_prime)
        io_mask_eval = io_mask.evaluate_mle(r_address_prime)
        val_io_eval = eval_io_mle(self.program_io, r_address_prime)
        return eq_eval * io_mask_eval * (val_final_claim - val_io_eval)

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Cache RamValFinal and RamValInit.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        _cache_virtual_batch(
            opening_accumulator, transcript,
            [VirtualPolynomial.RamValFinal, VirtualPolynomial.RamValInit],
            SumcheckId.RamOutputCheck, opening_point,
        )

class RamReadWriteCheckingVerifier(SumcheckInstanceVerifier):  # Stage 2: RAM read/write checking.
    DEGREE_BOUND = 3

    def __init__(self, opening_accumulator, transcript, one_hot_params, trace_length, config):  # Capture params + sample gamma.
        self.gamma = transcript.challenge_scalar()
        self.K = int(one_hot_params.ram_k)
        self.T = int(trace_length)
        self.r_cycle_stage1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamReadValue, SumcheckId.SpartanOuter)[0]
        self.phase1 = int(config.ram_rw_phase1_num_rounds)
        self.phase2 = int(config.ram_rw_phase2_num_rounds)

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return log2_pow2(self.K) + log2_pow2(self.T)

    def input_claim(self, opening_accumulator):  # rv + gamma*wv at SpartanOuter.
        rv = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamReadValue, SumcheckId.SpartanOuter)[1]
        wv = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamWriteValue, SumcheckId.SpartanOuter)[1]
        return rv + self.gamma * wv

    def _normalize_opening_point(self, sumcheck_challenges):  # Rust: 3-phase reversal (read_write_checking.rs:119-151).
        logT = log2_pow2(self.T)
        logK = log2_pow2(self.K)
        p1, rest = list(sumcheck_challenges[: self.phase1]), list(sumcheck_challenges[self.phase1 :])
        p2, rest = list(rest[: self.phase2]), list(rest[self.phase2 :])
        p3_cycle = list(rest[: logT - self.phase1])
        p3_addr = list(rest[logT - self.phase1 :])
        r_cycle = list(reversed(p3_cycle)) + list(reversed(p1))
        r_addr = list(reversed(p3_addr)) + list(reversed(p2))
        if len(r_cycle) != logT or len(r_addr) != logK:
            raise ValueError("normalize_opening_point: bad split sizes")
        return OpeningPoint(r_addr + r_cycle, BIG_ENDIAN)

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: read_write_checking.rs:679-707.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        logK = log2_pow2(self.K)
        r_address = OpeningPoint(opening_point.r[:logK], BIG_ENDIAN)
        r_cycle = OpeningPoint(opening_point.r[logK:], BIG_ENDIAN)
        eq_eval_cycle = EqPolynomial.mle(self.r_cycle_stage1.r, r_cycle.r)
        ra_claim = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamRa, SumcheckId.RamReadWriteChecking)[1]
        val_claim = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamVal, SumcheckId.RamReadWriteChecking)[1]
        inc_claim = opening_accumulator.get_committed_polynomial_opening(CommittedPolynomial.RamInc, SumcheckId.RamReadWriteChecking)[1]
        return eq_eval_cycle * ra_claim * (val_claim + self.gamma * (val_claim + inc_claim))

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Cache RamVal/RamRa and RamInc.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamVal, SumcheckId.RamReadWriteChecking, opening_point.clone())
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamRa, SumcheckId.RamReadWriteChecking, opening_point.clone())
        logK = log2_pow2(self.K)
        r_cycle = opening_point.r[logK:]
        opening_accumulator.append_dense(transcript, CommittedPolynomial.RamInc, SumcheckId.RamReadWriteChecking, r_cycle)

## NOTE: stage orchestration functions moved to `jolt_verifier.py`.

# ===========================================================================
# Stage 3: Shift + instruction input virtualization + registers claim reduction
# ===========================================================================

class ShiftSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 3: Spartan shift sumcheck.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, opening_accumulator, transcript):
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.gamma_powers = list(transcript.challenge_scalar_powers(5))
        r_outer, _ = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.NextPC,
            SumcheckId.SpartanOuter,
        )
        r_product, _ = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.NextIsNoop,
            SumcheckId.SpartanProductVirtualization,
        )
        self.r_outer = r_outer
        self.r_product = r_product

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, opening_accumulator):  # Rust: shift.rs:93-119.
        from field import Fr  # local import for one()

        next_pc = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.NextPC, SumcheckId.SpartanOuter)[1]
        next_unexpanded_pc = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.NextUnexpandedPC, SumcheckId.SpartanOuter)[1]
        next_is_virtual = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.NextIsVirtual, SumcheckId.SpartanOuter)[1]
        next_is_first = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.NextIsFirstInSequence, SumcheckId.SpartanOuter)[1]
        next_is_noop = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.NextIsNoop, SumcheckId.SpartanProductVirtualization)[1]

        return (
            next_unexpanded_pc
            + next_pc * self.gamma_powers[1]
            + next_is_virtual * self.gamma_powers[2]
            + next_is_first * self.gamma_powers[3]
            + (Fr.one() - next_is_noop) * self.gamma_powers[4]
        )

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: shift.rs:293-352.
        from field import Fr  # local import for one()

        r = _normalize_le_to_be(sumcheck_challenges)

        unexpanded_pc = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.UnexpandedPC, SumcheckId.SpartanShift)[1]
        pc = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.PC, SumcheckId.SpartanShift)[1]
        is_virtual = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.OpFlags_VirtualInstruction, SumcheckId.SpartanShift)[1]
        is_first = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.OpFlags_IsFirstInSequence, SumcheckId.SpartanShift)[1]
        is_noop = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.InstructionFlags_IsNoop, SumcheckId.SpartanShift)[1]

        eq_plus_one_outer = EqPlusOnePolynomial(list(self.r_outer.r)).evaluate(r.r)
        eq_plus_one_product = EqPlusOnePolynomial(list(self.r_product.r)).evaluate(r.r)

        left = (
            self.gamma_powers[0] * unexpanded_pc
            + self.gamma_powers[1] * pc
            + self.gamma_powers[2] * is_virtual
            + self.gamma_powers[3] * is_first
        )
        return left * eq_plus_one_outer + self.gamma_powers[4] * (Fr.one() - is_noop) * eq_plus_one_product

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: shift.rs:354-391.
        r = _normalize_le_to_be(sumcheck_challenges)
        _cache_virtual_batch(
            opening_accumulator,
            transcript,
            [
                VirtualPolynomial.UnexpandedPC,
                VirtualPolynomial.PC,
                VirtualPolynomial.OpFlags_VirtualInstruction,
                VirtualPolynomial.OpFlags_IsFirstInSequence,
                VirtualPolynomial.InstructionFlags_IsNoop,
            ],
            SumcheckId.SpartanShift,
            r,
        )

class InstructionInputSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 3: instruction input virtualization.
    DEGREE_BOUND = 3

    def __init__(self, opening_accumulator, transcript):
        self.r_cycle_stage_1 = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LeftInstructionInput,
            SumcheckId.SpartanOuter,
        )[0]
        self.r_cycle_stage_2 = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LeftInstructionInput,
            SumcheckId.SpartanProductVirtualization,
        )[0]
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return len(self.r_cycle_stage_1)

    def input_claim(self, opening_accumulator):  # Rust: instruction_input.rs:75-97.
        left1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.LeftInstructionInput, SumcheckId.SpartanOuter)[1]
        right1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RightInstructionInput, SumcheckId.SpartanOuter)[1]
        left2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.LeftInstructionInput, SumcheckId.SpartanProductVirtualization)[1]
        right2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RightInstructionInput, SumcheckId.SpartanProductVirtualization)[1]

        claim1 = right1 + self.gamma * left1
        claim2 = right2 + self.gamma * left2
        return claim1 + self.gamma_sqr * claim2

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: instruction_input.rs:517-582.
        r = _normalize_le_to_be(sumcheck_challenges)

        eq1 = EqPolynomial.mle(r.r, self.r_cycle_stage_1.r)
        eq2 = EqPolynomial.mle(r.r, self.r_cycle_stage_2.r)

        rs1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs1Value, SumcheckId.InstructionInputVirtualization)[1]
        left_is_rs1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.InstructionFlags_LeftOperandIsRs1Value, SumcheckId.InstructionInputVirtualization)[1]
        unexpanded_pc = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.UnexpandedPC, SumcheckId.InstructionInputVirtualization)[1]
        left_is_pc = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.InstructionFlags_LeftOperandIsPC, SumcheckId.InstructionInputVirtualization)[1]

        rs2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs2Value, SumcheckId.InstructionInputVirtualization)[1]
        right_is_rs2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.InstructionFlags_RightOperandIsRs2Value, SumcheckId.InstructionInputVirtualization)[1]
        imm = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Imm, SumcheckId.InstructionInputVirtualization)[1]
        right_is_imm = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.InstructionFlags_RightOperandIsImm, SumcheckId.InstructionInputVirtualization)[1]

        left_input = left_is_rs1 * rs1 + left_is_pc * unexpanded_pc
        right_input = right_is_rs2 * rs2 + right_is_imm * imm

        return (eq1 + self.gamma_sqr * eq2) * (right_input + self.gamma * left_input)

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: instruction_input.rs:584-639.
        r = _normalize_le_to_be(sumcheck_challenges)
        _cache_virtual_batch(
            opening_accumulator,
            transcript,
            [
                VirtualPolynomial.InstructionFlags_LeftOperandIsRs1Value,
                VirtualPolynomial.Rs1Value,
                VirtualPolynomial.InstructionFlags_LeftOperandIsPC,
                VirtualPolynomial.UnexpandedPC,
                VirtualPolynomial.InstructionFlags_RightOperandIsRs2Value,
                VirtualPolynomial.Rs2Value,
                VirtualPolynomial.InstructionFlags_RightOperandIsImm,
                VirtualPolynomial.Imm,
            ],
            SumcheckId.InstructionInputVirtualization,
            r,
        )

class RegistersClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 3: registers claim reduction.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, opening_accumulator, transcript):
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.r_spartan = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.SpartanOuter,
        )[0]

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, opening_accumulator):  # Rust: registers.rs:58-68.
        rd = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RdWriteValue, SumcheckId.SpartanOuter)[1]
        rs1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs1Value, SumcheckId.SpartanOuter)[1]
        rs2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs2Value, SumcheckId.SpartanOuter)[1]
        return rd + self.gamma * rs1 + self.gamma_sqr * rs2

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: registers.rs:442-472.
        r = _normalize_le_to_be(sumcheck_challenges)
        rd = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RdWriteValue, SumcheckId.RegistersClaimReduction)[1]
        rs1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs1Value, SumcheckId.RegistersClaimReduction)[1]
        rs2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs2Value, SumcheckId.RegistersClaimReduction)[1]
        return EqPolynomial.mle(r.r, self.r_spartan.r) * (rd + self.gamma * rs1 + self.gamma_sqr * rs2)

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: registers.rs:474-501.
        r = _normalize_le_to_be(sumcheck_challenges)
        _cache_virtual_batch(
            opening_accumulator,
            transcript,
            [VirtualPolynomial.RdWriteValue, VirtualPolynomial.Rs1Value, VirtualPolynomial.Rs2Value],
            SumcheckId.RegistersClaimReduction,
            r,
        )

## NOTE: stage orchestration functions moved to `jolt_verifier.py`.

# ===========================================================================
# Stage 4: Registers read/write checking + RAM val evaluation + RAM val final
# ===========================================================================

class RegistersReadWriteCheckingVerifier(SumcheckInstanceVerifier):  # Stage 4: registers read/write checking.
    DEGREE_BOUND = 3
    LOG_K = int(REGISTER_COUNT).bit_length() - 1  # Rust: log2(REGISTER_COUNT)=7 for Jolt (128 regs)

    def __init__(self, trace_len, opening_accumulator, transcript, rw_config):
        self.gamma = transcript.challenge_scalar()
        self.T = int(trace_len)
        self.phase1 = int(rw_config.registers_rw_phase1_num_rounds)
        self.phase2 = int(rw_config.registers_rw_phase2_num_rounds)
        self.r_cycle = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RdWriteValue,
            SumcheckId.RegistersClaimReduction,
        )[0]

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.LOG_K + log2_pow2(self.T)

    def input_claim(self, opening_accumulator):
        rd = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RdWriteValue, SumcheckId.RegistersClaimReduction)[1]
        rs1 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs1Value, SumcheckId.RegistersClaimReduction)[1]
        rs2 = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs2Value, SumcheckId.RegistersClaimReduction)[1]
        return rd + self.gamma * (rs1 + self.gamma * rs2)

    def _normalize_opening_point(self, sumcheck_challenges):  # Rust: registers/read_write_checking.rs:141-174.
        logT = log2_pow2(self.T)
        p1, rest = list(sumcheck_challenges[: self.phase1]), list(sumcheck_challenges[self.phase1 :])
        p2, rest = list(rest[: self.phase2]), list(rest[self.phase2 :])
        p3_cycle = list(rest[: logT - self.phase1])
        p3_addr = list(rest[logT - self.phase1 :])
        r_cycle = list(reversed(p3_cycle)) + list(reversed(p1))
        r_addr = list(reversed(p3_addr)) + list(reversed(p2))
        if len(r_cycle) != logT or len(r_addr) != self.LOG_K:
            raise ValueError("registers normalize_opening_point: bad split sizes")
        return OpeningPoint(r_addr + r_cycle, BIG_ENDIAN)

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: read_write_checking.rs:828-878.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        r_cycle = OpeningPoint(opening_point.r[self.LOG_K :], BIG_ENDIAN)

        val = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RegistersVal, SumcheckId.RegistersReadWriteChecking)[1]
        rs1_ra = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs1Ra, SumcheckId.RegistersReadWriteChecking)[1]
        rs2_ra = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs2Ra, SumcheckId.RegistersReadWriteChecking)[1]
        rd_wa = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RdWa, SumcheckId.RegistersReadWriteChecking)[1]
        inc = opening_accumulator.get_committed_polynomial_opening(CommittedPolynomial.RdInc, SumcheckId.RegistersReadWriteChecking)[1]

        rd_write_value = rd_wa * (inc + val)
        rs1_value = rs1_ra * val
        rs2_value = rs2_ra * val
        eq_eval = EqPolynomial.mle(r_cycle.r, self.r_cycle.r)
        return eq_eval * (rd_write_value + self.gamma * (rs1_value + self.gamma * rs2_value))

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: read_write_checking.rs:880-919.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RegistersVal, SumcheckId.RegistersReadWriteChecking, opening_point.clone())
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.Rs1Ra, SumcheckId.RegistersReadWriteChecking, opening_point.clone())
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.Rs2Ra, SumcheckId.RegistersReadWriteChecking, opening_point.clone())
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RdWa, SumcheckId.RegistersReadWriteChecking, opening_point.clone())
        r_cycle = opening_point.r[self.LOG_K :]
        opening_accumulator.append_dense(transcript, CommittedPolynomial.RdInc, SumcheckId.RegistersReadWriteChecking, r_cycle)

class RamValEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 4: RAM val evaluation.
    DEGREE_BOUND = 3

    def __init__(self, ram_preprocessing, program_io, trace_len, ram_K, opening_accumulator):
        self.T = int(trace_len)
        self.K = int(ram_K)
        r_full = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamVal, SumcheckId.RamReadWriteChecking)[0]
        self.r_address = OpeningPoint(r_full.r[: log2_pow2(self.K)], BIG_ENDIAN)
        self.r_cycle = OpeningPoint(r_full.r[log2_pow2(self.K) :], BIG_ENDIAN)

        n_memory_vars = log2_pow2(self.K)
        def _advice_num_vars(max_size_bytes):
            n_words = max(1, int(max_size_bytes) // 8)
            pow2 = 1 << (n_words - 1).bit_length()
            return (pow2.bit_length() - 1) if pow2 > 1 else 0

        untrusted = calculate_advice_memory_evaluation(
            opening_accumulator.get_advice_opening(AdviceKind.Untrusted, SumcheckId.RamValEvaluation),
            _advice_num_vars(program_io.memory_layout.max_untrusted_advice_size),
            program_io.memory_layout.untrusted_advice_start,
            program_io.memory_layout,
            self.r_address.r,
            n_memory_vars,
        )
        trusted = calculate_advice_memory_evaluation(
            opening_accumulator.get_advice_opening(AdviceKind.Trusted, SumcheckId.RamValEvaluation),
            _advice_num_vars(program_io.memory_layout.max_trusted_advice_size),
            program_io.memory_layout.trusted_advice_start,
            program_io.memory_layout,
            self.r_address.r,
            n_memory_vars,
        )
        public = eval_initial_ram_mle(ram_preprocessing, program_io, self.r_address.r)
        self.init_eval = untrusted + trusted + public

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return log2_pow2(self.T)

    def input_claim(self, opening_accumulator):
        claimed = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamVal, SumcheckId.RamReadWriteChecking)[1]
        return claimed - self.init_eval

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: ram/val_evaluation.rs:350-384.
        r = _normalize_le_to_be(sumcheck_challenges)
        lt_eval = None
        eq_term = None
        from field import Fr  # local import
        lt_eval = Fr.zero()
        eq_term = Fr.one()
        for x, y in zip(r.r, self.r_cycle.r):
            lt_eval += (Fr.one() - x) * y * eq_term
            eq_term *= Fr.one() - x - y + x * y + x * y

        inc = opening_accumulator.get_committed_polynomial_opening(CommittedPolynomial.RamInc, SumcheckId.RamValEvaluation)[1]
        wa = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamRa, SumcheckId.RamValEvaluation)[1]
        return inc * wa * lt_eval

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ram/val_evaluation.rs:386-415.
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        r_full = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamVal, SumcheckId.RamReadWriteChecking)[0]
        r_address = r_full.r[: len(r_full.r) - len(r_cycle_prime.r)]
        wa_opening_point = OpeningPoint(list(r_address) + list(r_cycle_prime.r), BIG_ENDIAN)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamRa, SumcheckId.RamValEvaluation, wa_opening_point)
        opening_accumulator.append_dense(transcript, CommittedPolynomial.RamInc, SumcheckId.RamValEvaluation, r_cycle_prime.r)

class ValFinalSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 4: RAM val final.
    DEGREE_BOUND = 2

    def __init__(self, ram_preprocessing, program_io, trace_len, ram_K, opening_accumulator, rw_config):
        self.T = int(trace_len)
        self.K = int(ram_K)
        self.r_address = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamValFinal,
            SumcheckId.RamOutputCheck,
        )[0]

        n_memory_vars = log2_pow2(self.K)
        def _advice_num_vars(max_size_bytes):
            n_words = max(1, int(max_size_bytes) // 8)
            pow2 = 1 << (n_words - 1).bit_length()
            return (pow2.bit_length() - 1) if pow2 > 1 else 0

        log_T = log2_pow2(self.T)
        advice_sumcheck_id = SumcheckId.RamValEvaluation if rw_config.needs_single_advice_opening(log_T) else SumcheckId.RamValFinalEvaluation

        untrusted = calculate_advice_memory_evaluation(
            opening_accumulator.get_advice_opening(AdviceKind.Untrusted, advice_sumcheck_id),
            _advice_num_vars(program_io.memory_layout.max_untrusted_advice_size),
            program_io.memory_layout.untrusted_advice_start,
            program_io.memory_layout,
            self.r_address.r,
            n_memory_vars,
        )
        trusted = calculate_advice_memory_evaluation(
            opening_accumulator.get_advice_opening(AdviceKind.Trusted, advice_sumcheck_id),
            _advice_num_vars(program_io.memory_layout.max_trusted_advice_size),
            program_io.memory_layout.trusted_advice_start,
            program_io.memory_layout,
            self.r_address.r,
            n_memory_vars,
        )
        public = eval_initial_ram_mle(ram_preprocessing, program_io, self.r_address.r)
        self.val_init_eval = untrusted + trusted + public

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return log2_pow2(self.T)

    def input_claim(self, opening_accumulator):
        val_final = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamValFinal, SumcheckId.RamOutputCheck)[1]
        return val_final - self.val_init_eval

    def expected_output_claim(self, opening_accumulator, _sumcheck_challenges):  # Rust: ram/val_final.rs:331-349.
        inc = opening_accumulator.get_committed_polynomial_opening(CommittedPolynomial.RamInc, SumcheckId.RamValFinalEvaluation)[1]
        wa = opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RamRa, SumcheckId.RamValFinalEvaluation)[1]
        return inc * wa

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ram/val_final.rs:351-378.
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        wa_opening_point = OpeningPoint(list(self.r_address.r) + list(r_cycle_prime.r), BIG_ENDIAN)
        opening_accumulator.append_dense(transcript, CommittedPolynomial.RamInc, SumcheckId.RamValFinalEvaluation, r_cycle_prime.r)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamRa, SumcheckId.RamValFinalEvaluation, wa_opening_point)

## NOTE: stage orchestration functions moved to `jolt_verifier.py`.


# ===========================================================================
# Stage 5: InstructionReadRaf + RamRaClaimReduction + RegistersValEvaluation
# ===========================================================================

class InstructionReadRafSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: instruction lookups read+RAF checking.
    LOG_K = 128  # Rust: instruction_lookups::LOG_K = XLEN*2 = 128 for RV64.

    def __init__(self, n_cycle_vars, one_hot_params, opening_accumulator, transcript):  # Sample gamma; store r_reduction + chunking.
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.log_T = int(n_cycle_vars)
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)
        self.r_reduction = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.InstructionClaimReduction,
        )[0]

    def degree(self):  # Rust: n_virtual_ra_polys + 2.
        return (self.LOG_K // self.ra_virtual_log_k_chunk) + 2

    def num_rounds(self):  # Rust: LOG_K + log_T.
        return self.LOG_K + self.log_T

    def input_claim(self, opening_accumulator):  # Rust: read_raf_checking.rs:145-165.
        rv_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.InstructionClaimReduction,
        )[1]
        rv_claim_branch = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.SpartanProductVirtualization,
        )[1]
        if rv_claim != rv_claim_branch:
            raise SumcheckVerifyError("LookupOutput claim mismatch across stages")
        left_operand_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LeftLookupOperand,
            SumcheckId.InstructionClaimReduction,
        )[1]
        right_operand_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RightLookupOperand,
            SumcheckId.InstructionClaimReduction,
        )[1]
        return rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim

    def _normalize_opening_point(self, sumcheck_challenges):  # Rust: read_raf_checking.rs:172-180.
        r_address_prime = list(sumcheck_challenges[: self.LOG_K])
        r_cycle_prime = list(reversed(sumcheck_challenges[self.LOG_K :]))
        return OpeningPoint(r_address_prime + r_cycle_prime, BIG_ENDIAN)

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: read_raf_checking.rs:1195-1266.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        r_address_prime = opening_point.r[: self.LOG_K]
        r_cycle_prime = opening_point.r[self.LOG_K :]

        left_operand_eval = OperandPolynomial(self.LOG_K, OperandPolynomial.LEFT).evaluate(r_address_prime)
        right_operand_eval = OperandPolynomial(self.LOG_K, OperandPolynomial.RIGHT).evaluate(r_address_prime)
        identity_poly_eval = IdentityPolynomial(self.LOG_K).evaluate(r_address_prime)

        val_evals = [evaluate_mle(t, r_address_prime, xlen=64) for t in LOOKUP_TABLES_64]
        eq_eval_r_reduction = EqPolynomial.mle(self.r_reduction.r, r_cycle_prime)

        n_virtual_ra_polys = self.LOG_K // self.ra_virtual_log_k_chunk
        ra_claim = None
        for i in range(n_virtual_ra_polys):
            c = opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionReadRaf,
            )[1]
            ra_claim = c if ra_claim is None else (ra_claim * c)

        table_flag_claims = [
            opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.LookupTableFlag,
                i,
                SumcheckId.InstructionReadRaf,
            )[1]
            for i in range(len(LOOKUP_TABLES_64))
        ]
        raf_flag_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.InstructionRafFlag,
            SumcheckId.InstructionReadRaf,
        )[1]

        val_claim = None
        for ev, flag in zip(val_evals, table_flag_claims):
            term = ev * flag
            val_claim = term if val_claim is None else (val_claim + term)

        from field import Fr  # local import for one()
        raf_claim = (Fr.one() - raf_flag_claim) * (left_operand_eval + self.gamma * right_operand_eval) + raf_flag_claim * self.gamma * identity_poly_eval
        return eq_eval_r_reduction * ra_claim * (val_claim + self.gamma * raf_claim)

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: read_raf_checking.rs:1268-1309.
        r_sumcheck = self._normalize_opening_point(sumcheck_challenges)
        r_address = OpeningPoint(r_sumcheck.r[: self.LOG_K], BIG_ENDIAN)
        r_cycle = OpeningPoint(r_sumcheck.r[self.LOG_K :], BIG_ENDIAN)

        for i in range(len(LOOKUP_TABLES_64)):
            opening_accumulator.append_virtual_i(
                transcript,
                VirtualPolynomial.LookupTableFlag,
                i,
                SumcheckId.InstructionReadRaf,
                r_cycle.clone(),
            )

        for i, r_address_chunk in enumerate(
            [r_address.r[j : j + self.ra_virtual_log_k_chunk] for j in range(0, len(r_address.r), self.ra_virtual_log_k_chunk)]
        ):
            opening_point = OpeningPoint(list(r_address_chunk) + list(r_cycle.r), BIG_ENDIAN)
            opening_accumulator.append_virtual_i(
                transcript,
                VirtualPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionReadRaf,
                opening_point,
            )

        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.InstructionRafFlag,
            SumcheckId.InstructionReadRaf,
            r_cycle.clone(),
        )


class RamRaClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: reduce RAM RA claims for virtualization.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, one_hot_params, opening_accumulator, transcript):  # Extract points/claims; sample gamma.
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.log_T = log2_pow2(int(trace_len))

        r_raf, claim_raf = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamRafEvaluation,
        )
        r_rw, claim_rw = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamReadWriteChecking,
        )
        r_val_eval, claim_val_eval = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamValEvaluation,
        )
        r_val_final, claim_val_final = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamValFinalEvaluation,
        )

        self.r_address_1 = list(r_raf.r[: self.log_K])
        self.r_cycle_raf = list(r_raf.r[self.log_K :])
        self.r_address_2 = list(r_rw.r[: self.log_K])
        self.r_cycle_rw = list(r_rw.r[self.log_K :])
        self.r_cycle_val = list(r_val_eval.r[self.log_K :])

        if self.r_address_1 != list(r_val_final.r[: self.log_K]):
            raise ValueError("unexpected r_address mismatch for RAM RA reduction")
        if self.r_address_2 != list(r_val_eval.r[: self.log_K]):
            raise ValueError("unexpected r_address mismatch for RAM RA reduction")
        if self.r_cycle_val != list(r_val_final.r[self.log_K :]):
            raise ValueError("unexpected r_cycle mismatch for RAM RA reduction")

        self.claim_raf = claim_raf
        self.claim_val_final = claim_val_final
        self.claim_rw = claim_rw
        self.claim_val_eval = claim_val_eval

        self.gamma = transcript.challenge_scalar()
        self.gamma_squared = self.gamma * self.gamma
        self.gamma_cubed = self.gamma_squared * self.gamma

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.log_K + self.log_T

    def input_claim(self, _opening_accumulator):  # Rust: ram_ra.rs:1000-1005.
        return self.claim_raf + self.gamma * self.claim_val_final + self.gamma_squared * self.claim_rw + self.gamma_cubed * self.claim_val_eval

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: ram_ra.rs:1046-1083.
        r_address_reduced = list(reversed(sumcheck_challenges[: self.log_K]))
        r_cycle_reduced = list(reversed(sumcheck_challenges[self.log_K :]))

        eq_addr_1 = EqPolynomial.mle(self.r_address_1, r_address_reduced)
        eq_addr_2 = EqPolynomial.mle(self.r_address_2, r_address_reduced)

        eq_cycle_raf = EqPolynomial.mle(self.r_cycle_raf, r_cycle_reduced)
        eq_cycle_rw = EqPolynomial.mle(self.r_cycle_rw, r_cycle_reduced)
        eq_cycle_val = EqPolynomial.mle(self.r_cycle_val, r_cycle_reduced)

        eq_cycle_A = eq_cycle_raf + self.gamma * eq_cycle_val
        eq_cycle_B = eq_cycle_rw + self.gamma * eq_cycle_val
        eq_combined = eq_addr_1 * eq_cycle_A + self.gamma_squared * eq_addr_2 * eq_cycle_B

        ra_claim_reduced = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamRaClaimReduction,
        )[1]
        return eq_combined * ra_claim_reduced

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ram_ra.rs:1085-1109.
        r_address_be = list(reversed(sumcheck_challenges[: self.log_K]))
        r_cycle_be = list(reversed(sumcheck_challenges[self.log_K :]))
        opening_point = OpeningPoint(r_address_be + r_cycle_be, BIG_ENDIAN)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamRa, SumcheckId.RamRaClaimReduction, opening_point)


class RegistersValEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: registers val evaluation.
    DEGREE_BOUND = 3
    LOG_K = int(REGISTER_COUNT).bit_length() - 1  # Rust: log2(REGISTER_COUNT)=7 for Jolt (128 regs)

    def __init__(self, opening_accumulator):  # Extract r_address||r_cycle from RegistersVal opening.
        r_full = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RegistersVal,
            SumcheckId.RegistersReadWriteChecking,
        )[0]
        self.r_cycle = OpeningPoint(r_full.r[self.LOG_K :], BIG_ENDIAN)
        self.r_address = OpeningPoint(r_full.r[: self.LOG_K], BIG_ENDIAN)

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return len(self.r_cycle.r)

    def input_claim(self, opening_accumulator):  # Rust: val_evaluation.rs:80-86.
        return opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RegistersVal,
            SumcheckId.RegistersReadWriteChecking,
        )[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: registers/val_evaluation.rs:250-283.
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        from field import Fr  # local import for one()
        lt_eval = Fr.zero()
        eq_term = Fr.one()
        for x, y in zip(r_cycle_prime.r, self.r_cycle.r):
            lt_eval += (Fr.one() - x) * y * eq_term
            eq_term *= Fr.one() - x - y + x * y + x * y

        inc_claim = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersValEvaluation,
        )[1]
        wa_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RdWa,
            SumcheckId.RegistersValEvaluation,
        )[1]
        return inc_claim * wa_claim * lt_eval

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: registers/val_evaluation.rs:285-314.
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        opening_accumulator.append_dense(
            transcript,
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersValEvaluation,
            r_cycle_prime.r,
        )
        opening_point = OpeningPoint(list(self.r_address.r) + list(r_cycle_prime.r), BIG_ENDIAN)
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.RdWa,
            SumcheckId.RegistersValEvaluation,
            opening_point,
        )


## NOTE: stage orchestration functions moved to `jolt_verifier.py`.


# ===========================================================================
# Stage 6: BytecodeReadRaf + Booleanity + HammingBooleanity + RA virtualization + reductions
# ===========================================================================

class HammingBooleanitySumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: RAM Hamming booleanity.
    DEGREE_BOUND = 3

    def __init__(self, opening_accumulator):  # Extract r_cycle from SpartanOuter (LookupOutput opening point).
        self.r_cycle = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.SpartanOuter,
        )[0]

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return len(self.r_cycle.r)

    def input_claim(self, _opening_accumulator):  # Booleanity sumcheck input claim is 0.
        from field import Fr  # local import for zero()
        return Fr.zero()

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: ram/hamming_booleanity.rs:182-220.
        H_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamHammingWeight,
            SumcheckId.RamHammingBooleanity,
        )[1]
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        eq = EqPolynomial.mle(sumcheck_challenges, list(reversed(self.r_cycle.r)))
        return (H_claim * H_claim - H_claim) * eq

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ram/hamming_booleanity.rs:222-234.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.RamHammingWeight,
            SumcheckId.RamHammingBooleanity,
            opening_point,
        )


class RamRaVirtualSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: RAM RA virtualization.
    def __init__(self, trace_len, one_hot_params, opening_accumulator, transcript):  # Extract reduced (r_addr||r_cycle) and sample nothing.
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.log_T = log2_pow2(int(trace_len))
        r_reduced = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamRaClaimReduction,
        )[0]
        self.r_cycle_reduced = OpeningPoint(list(r_reduced.r[self.log_K :]), BIG_ENDIAN)
        r_address_reduced = OpeningPoint(list(r_reduced.r[: self.log_K]), BIG_ENDIAN)
        self.r_address_chunks = one_hot_params.compute_r_address_chunks(r_address_reduced.r)
        self.d = int(one_hot_params.ram_d)

    def degree(self):  # Rust: d + 1.
        return self.d + 1

    def num_rounds(self):  # Rust: log_T.
        return self.log_T

    def input_claim(self, opening_accumulator):  # Rust: ra_virtual.rs:132-138.
        return opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamRaClaimReduction,
        )[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: ra_virtual.rs:273-295.
        r_cycle_final = _normalize_le_to_be(sumcheck_challenges)
        eq_eval = EqPolynomial.mle(self.r_cycle_reduced.r, r_cycle_final.r)
        ra_prod = None
        for i in range(self.d):
            c = opening_accumulator.get_committed_polynomial_opening_i(
                CommittedPolynomial.RamRa,
                i,
                SumcheckId.RamRaVirtualization,
            )[1]
            ra_prod = c if ra_prod is None else (ra_prod * c)
        return eq_eval * ra_prod

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ra_virtual.rs:297-315.
        r_cycle_final = _normalize_le_to_be(sumcheck_challenges)
        for i in range(self.d):
            opening_point = list(self.r_address_chunks[i]) + list(r_cycle_final.r)
            opening_accumulator.append_dense_i(
                transcript,
                CommittedPolynomial.RamRa,
                i,
                SumcheckId.RamRaVirtualization,
                opening_point,
            )


class InstructionRaVirtualSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: instruction RA virtualization.
    def __init__(self, one_hot_params, opening_accumulator, transcript):  # Extract r_address/r_cycle from Stage5 + sample gamma powers.
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)
        self.log_k_chunk = int(one_hot_params.log_k_chunk)
        self.n_committed_per_virtual = self.ra_virtual_log_k_chunk // self.log_k_chunk
        self.n_virtual_ra_polys = 128 // self.ra_virtual_log_k_chunk
        self.n_committed_ra_polys = 128 // self.log_k_chunk

        r_address = []
        for i in range(self.n_virtual_ra_polys):
            r, _ = opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionReadRaf,
            )
            r_address.extend(r.r[: self.ra_virtual_log_k_chunk])
        r0, _ = opening_accumulator.get_virtual_polynomial_opening_i(
            VirtualPolynomial.InstructionRa,
            0,
            SumcheckId.InstructionReadRaf,
        )
        self.r_cycle = OpeningPoint(list(r0.r[self.ra_virtual_log_k_chunk :]), BIG_ENDIAN)
        self.r_address = OpeningPoint(list(r_address), BIG_ENDIAN)
        self.gamma_powers = list(transcript.challenge_scalar_powers(self.n_virtual_ra_polys))
        self.one_hot_params = one_hot_params

    def degree(self):  # Rust: n_committed_per_virtual + 1.
        return self.n_committed_per_virtual + 1

    def num_rounds(self):  # Rust: log_T.
        return len(self.r_cycle.r)

    def input_claim(self, opening_accumulator):  # Rust: ra_virtual.rs:101-113.
        acc = None
        for i in range(self.n_virtual_ra_polys):
            c = opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionReadRaf,
            )[1]
            term = self.gamma_powers[i] * c
            acc = term if acc is None else (acc + term)
        return acc

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: ra_virtual.rs:314-341.
        r_cycle_final = _normalize_le_to_be(sumcheck_challenges)
        eq_eval = EqPolynomial.mle(self.r_cycle.r, r_cycle_final.r)
        committed_claims = [
            opening_accumulator.get_committed_polynomial_opening_i(
                CommittedPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionRaVirtualization,
            )[1]
            for i in range(self.n_committed_ra_polys)
        ]
        ra_acc = None
        idx = 0
        for i in range(self.n_virtual_ra_polys):
            prod = None
            for _ in range(self.n_committed_per_virtual):
                c = committed_claims[idx]
                idx += 1
                prod = c if prod is None else (prod * c)
            term = self.gamma_powers[i] * prod
            ra_acc = term if ra_acc is None else (ra_acc + term)
        return eq_eval * ra_acc

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ra_virtual.rs:343-367.
        r_cycle_final = _normalize_le_to_be(sumcheck_challenges)
        r_address_chunks = self.one_hot_params.compute_r_address_chunks(self.r_address.r)
        for i, r_address_chunk in enumerate(r_address_chunks):
            opening_point = list(r_address_chunk) + list(r_cycle_final.r)
            opening_accumulator.append_dense_i(
                transcript,
                CommittedPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionRaVirtualization,
                opening_point,
            )


class IncClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: reduce RamInc/RdInc multi-openings.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, opening_accumulator, transcript):  # Sample gamma; capture the 4 opening points.
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.gamma_cub = self.gamma_sqr * self.gamma
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.r_cycle_stage2 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RamInc,
            SumcheckId.RamReadWriteChecking,
        )[0]
        self.r_cycle_stage4 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RamInc,
            SumcheckId.RamValEvaluation,
        )[0]
        self.s_cycle_stage4 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersReadWriteChecking,
        )[0]
        self.s_cycle_stage5 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersValEvaluation,
        )[0]

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, opening_accumulator):  # Rust: increments.rs:140-163.
        v1 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RamInc,
            SumcheckId.RamReadWriteChecking,
        )[1]
        v2 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RamInc,
            SumcheckId.RamValEvaluation,
        )[1]
        w1 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersReadWriteChecking,
        )[1]
        w2 = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersValEvaluation,
        )[1]
        return v1 + self.gamma * v2 + self.gamma_sqr * w1 + self.gamma_cub * w2

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: increments.rs:663-693.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        eq_r2 = EqPolynomial.mle(opening_point.r, self.r_cycle_stage2.r)
        eq_r4 = EqPolynomial.mle(opening_point.r, self.r_cycle_stage4.r)
        eq_s4 = EqPolynomial.mle(opening_point.r, self.s_cycle_stage4.r)
        eq_s5 = EqPolynomial.mle(opening_point.r, self.s_cycle_stage5.r)
        eq_ram = eq_r2 + self.gamma * eq_r4
        eq_rd = eq_s4 + self.gamma * eq_s5
        ram_inc_claim = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RamInc,
            SumcheckId.IncClaimReduction,
        )[1]
        rd_inc_claim = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.IncClaimReduction,
        )[1]
        return ram_inc_claim * eq_ram + self.gamma_sqr * rd_inc_claim * eq_rd

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: increments.rs:695-716.
        opening_point = _normalize_le_to_be(sumcheck_challenges)
        opening_accumulator.append_dense(
            transcript,
            CommittedPolynomial.RamInc,
            SumcheckId.IncClaimReduction,
            opening_point.r,
        )
        opening_accumulator.append_dense(
            transcript,
            CommittedPolynomial.RdInc,
            SumcheckId.IncClaimReduction,
            opening_point.r,
        )


class AdviceClaimReductionVerifier(SumcheckInstanceVerifier):  # Stage 6/7: two-phase advice claim reduction.
    DEGREE_BOUND = 2

    PHASE_CYCLE = "cycle"
    PHASE_ADDRESS = "address"

    def __init__(self, kind, memory_layout, trace_len, log_k_chunk, opening_accumulator, transcript, single_opening):  # Mirror Rust AdviceClaimReductionParams::new (CycleMajor only).
        from field import Fr  # local import for field ops

        self.kind = AdviceKind(kind) if not isinstance(kind, AdviceKind) else kind
        self.phase = self.PHASE_CYCLE
        self.single_opening = bool(single_opening)

        self.log_t = log2_pow2(int(trace_len))
        self.log_k_chunk = int(log_k_chunk)

        self.r_val_eval = opening_accumulator.get_advice_opening(self.kind, SumcheckId.RamValEvaluation)[0]
        self.r_val_final = None if self.single_opening else opening_accumulator.get_advice_opening(self.kind, SumcheckId.RamValFinalEvaluation)[0]

        self.gamma = transcript.challenge_scalar()

        max_advice_size_bytes = int(
            memory_layout.max_trusted_advice_size if self.kind == AdviceKind.Trusted else memory_layout.max_untrusted_advice_size
        )
        advice_col_vars, advice_row_vars = self._advice_sigma_nu_from_max_bytes(max_advice_size_bytes)
        self.advice_col_vars = advice_col_vars
        self.advice_row_vars = advice_row_vars

        main_col_vars, main_row_vars = self._main_sigma_nu(self.log_k_chunk, self.log_t)
        self.main_col_vars = main_col_vars
        self.main_row_vars = main_row_vars

        self.cycle_phase_col_rounds, self.cycle_phase_row_rounds = self._cycle_phase_round_schedule_cycle_major(
            self.log_t,
            self.log_k_chunk,
            self.main_col_vars,
            self.advice_row_vars,
            self.advice_col_vars,
        )
        self.cycle_var_challenges = []  # populated during cycle-phase cache_openings (LE)

        self._two_inv = Fr(2).inv()

    @staticmethod
    def _balanced_sigma_nu(total_vars: int) -> tuple[int, int]:
        total_vars = int(total_vars)
        sigma = (total_vars + 1) // 2
        nu = total_vars - sigma
        return sigma, nu

    @classmethod
    def _main_sigma_nu(cls, log_k_chunk: int, log_t: int) -> tuple[int, int]:
        return cls._balanced_sigma_nu(int(log_k_chunk) + int(log_t))

    @classmethod
    def _advice_sigma_nu_from_max_bytes(cls, max_advice_size_bytes: int) -> tuple[int, int]:
        words = int(max_advice_size_bytes) // 8
        if words <= 0:
            words = 1
        pow2 = 1 << ((words - 1).bit_length())
        advice_vars = log2_pow2(pow2) if pow2 > 1 else 0
        return cls._balanced_sigma_nu(advice_vars)

    @staticmethod
    def _cycle_phase_round_schedule_cycle_major(log_t: int, _log_k_chunk: int, main_col_vars: int, advice_row_vars: int, advice_col_vars: int):  # Rust: cycle_phase_round_schedule (CycleMajor).
        col_rounds = range(0, min(int(log_t), int(advice_col_vars)))
        row_start = min(int(log_t), int(main_col_vars))
        row_end = min(int(log_t), int(main_col_vars) + int(advice_row_vars))
        row_rounds = range(row_start, row_end)
        return col_rounds, row_rounds

    def num_address_phase_rounds(self) -> int:
        return (self.advice_col_vars + self.advice_row_vars) - (len(self.cycle_phase_col_rounds) + len(self.cycle_phase_row_rounds))

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        if self.phase == self.PHASE_CYCLE:
            if len(self.cycle_phase_row_rounds) != 0:
                return int(self.cycle_phase_row_rounds.stop) - int(self.cycle_phase_col_rounds.start)
            return len(self.cycle_phase_col_rounds)
        return self.num_address_phase_rounds()

    def round_offset(self, max_num_rounds):  # Rust: advice.rs:656-665.
        if self.phase == self.PHASE_CYCLE:
            booleanity_rounds = int(self.log_k_chunk) + int(self.log_t)
            booleanity_offset = int(max_num_rounds) - booleanity_rounds
            return booleanity_offset + int(self.log_k_chunk)
        return 0

    def input_claim(self, opening_accumulator):  # Rust: advice.rs:193-220.
        claim = opening_accumulator.get_advice_opening(self.kind, SumcheckId.RamValEvaluation)
        from field import Fr  # local import
        out = Fr.zero() if claim is None else claim[1]
        if not self.single_opening:
            final = opening_accumulator.get_advice_opening(self.kind, SumcheckId.RamValFinalEvaluation)
            if final is not None:
                out += self.gamma * final[1]
        if self.phase == self.PHASE_ADDRESS:
            mid = opening_accumulator.get_advice_opening(self.kind, SumcheckId.AdviceClaimReductionCyclePhase)
            if mid is None:
                raise SumcheckVerifyError("Cycle phase intermediate claim not found")
            out = mid[1]
        return out

    def _normalize_opening_point_cycle(self, challenges):  # Rust: advice.rs normalize_opening_point CycleVariables.
        advice_vars = self.advice_col_vars + self.advice_row_vars
        out = []
        for i in self.cycle_phase_col_rounds:
            out.append(challenges[i])
        for i in self.cycle_phase_row_rounds:
            out.append(challenges[i])
        if len(out) != advice_vars:
            # In cycle phase we only expose the variables actually bound in this phase.
            pass
        return OpeningPoint(out, LITTLE_ENDIAN).match_endianness(BIG_ENDIAN)

    def normalize_opening_point(self, challenges):  # Public normalize to BE for cache/eq.
        if self.phase == self.PHASE_CYCLE:
            return self._normalize_opening_point_cycle(list(challenges))
        # CycleMajor: advice_point_le = [cycle_var_challenges || address_challenges]
        return OpeningPoint(list(self.cycle_var_challenges) + list(challenges), LITTLE_ENDIAN).match_endianness(BIG_ENDIAN)

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: advice.rs:564-609.
        if self.phase == self.PHASE_CYCLE:
            mid = opening_accumulator.get_advice_opening(self.kind, SumcheckId.AdviceClaimReductionCyclePhase)
            if mid is None:
                raise SumcheckVerifyError("Cycle phase intermediate claim not found")
            return mid[1]

        opening_point = self.normalize_opening_point(sumcheck_challenges)
        advice = opening_accumulator.get_advice_opening(self.kind, SumcheckId.AdviceClaimReduction)
        if advice is None:
            raise SumcheckVerifyError("Final advice claim not found")
        advice_claim = advice[1]

        eq_eval = EqPolynomial.mle(opening_point.r, self.r_val_eval.r)
        eq_combined = eq_eval
        if not self.single_opening:
            eq_final = EqPolynomial.mle(opening_point.r, self.r_val_final.r)
            eq_combined = eq_eval + self.gamma * eq_final

        if len(self.cycle_phase_row_rounds) == 0 or len(self.cycle_phase_col_rounds) == 0:
            gap_len = 0
        else:
            gap_len = int(self.cycle_phase_row_rounds.start) - int(self.cycle_phase_col_rounds.stop)
        scale = None
        from field import Fr  # local import
        scale = Fr.one()
        for _ in range(int(gap_len)):
            scale *= self._two_inv
        return advice_claim * eq_combined * scale

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: advice.rs:612-653.
        opening_point = self.normalize_opening_point(sumcheck_challenges)
        if self.phase == self.PHASE_CYCLE:
            if self.kind == AdviceKind.Trusted:
                opening_accumulator.append_trusted_advice(transcript, SumcheckId.AdviceClaimReductionCyclePhase, opening_point.clone())
            else:
                opening_accumulator.append_untrusted_advice(transcript, SumcheckId.AdviceClaimReductionCyclePhase, opening_point.clone())
            self.cycle_var_challenges = opening_point.match_endianness(LITTLE_ENDIAN).r

        if self.num_address_phase_rounds() == 0 or self.phase == self.PHASE_ADDRESS:
            if self.kind == AdviceKind.Trusted:
                opening_accumulator.append_trusted_advice(transcript, SumcheckId.AdviceClaimReduction, opening_point)
            else:
                opening_accumulator.append_untrusted_advice(transcript, SumcheckId.AdviceClaimReduction, opening_point)


class BooleanitySumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: booleanity for all committed RA families.
    DEGREE_BOUND = 3

    def __init__(self, log_t, one_hot_params, opening_accumulator, transcript):  # Extract r_address/r_cycle from Stage5; sample gamma.
        self.log_t = int(log_t)
        self.log_k_chunk = int(one_hot_params.log_k_chunk)
        self.one_hot_params = one_hot_params
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)

        stage5_point = opening_accumulator.get_virtual_polynomial_opening_i(
            VirtualPolynomial.InstructionRa,
            0,
            SumcheckId.InstructionReadRaf,
        )[0]
        stage5_addr = list(stage5_point.r[: self.ra_virtual_log_k_chunk])
        stage5_addr.reverse()  # BE -> LE
        r_cycle = list(stage5_point.r[self.ra_virtual_log_k_chunk :])
        r_cycle.reverse()  # BE -> LE

        if len(stage5_addr) >= self.log_k_chunk:
            r_address = stage5_addr[len(stage5_addr) - self.log_k_chunk :]
        else:
            extra = transcript.challenge_vector_optimized(self.log_k_chunk - len(stage5_addr))
            r_address = stage5_addr + list(extra)

        self.r_address = list(r_address)  # LE
        self.r_cycle = list(r_cycle)  # LE

        total_d = int(one_hot_params.instruction_d) + int(one_hot_params.bytecode_d) + int(one_hot_params.ram_d)
        self.poly_types = []
        for i in range(int(one_hot_params.instruction_d)):
            self.poly_types.append((CommittedPolynomial.InstructionRa, i))
        for i in range(int(one_hot_params.bytecode_d)):
            self.poly_types.append((CommittedPolynomial.BytecodeRa, i))
        for i in range(int(one_hot_params.ram_d)):
            self.poly_types.append((CommittedPolynomial.RamRa, i))

        gamma = transcript.challenge_scalar_optimized()
        from field import Fr  # local import for one()/zero()
        if gamma == Fr.zero():
            gamma = Fr.one()
        gamma_sq = gamma * gamma
        self.gamma_powers_square = []
        g = Fr.one()
        for _ in range(total_d):
            self.gamma_powers_square.append(g)
            g *= gamma_sq

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.log_k_chunk + self.log_t

    def input_claim(self, _opening_accumulator):  # Rust: booleanity.rs:90-92.
        from field import Fr  # local import for zero()
        return Fr.zero()

    def _normalize_opening_point(self, sumcheck_challenges):  # Rust: booleanity.rs:94-102.
        r = list(sumcheck_challenges)
        r[: self.log_k_chunk] = list(reversed(r[: self.log_k_chunk]))
        r[self.log_k_chunk :] = list(reversed(r[self.log_k_chunk :]))
        return OpeningPoint(r, BIG_ENDIAN)

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: booleanity.rs:502-531.
        ra_claims = [
            opening_accumulator.get_committed_polynomial_opening_i(poly, i, SumcheckId.Booleanity)[1]
            for (poly, i) in self.poly_types
        ]
        combined_r = list(reversed(self.r_address)) + list(reversed(self.r_cycle))
        eq = EqPolynomial.mle(sumcheck_challenges, combined_r)
        acc = None
        for g2i, ra in zip(self.gamma_powers_square, ra_claims):
            term = (ra * ra - ra) * g2i
            acc = term if acc is None else (acc + term)
        return eq * acc

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: booleanity.rs:533-546.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        for poly, i in self.poly_types:
            opening_accumulator.append_dense_i(
                transcript,
                poly,
                i,
                SumcheckId.Booleanity,
                opening_point.r,
            )


class BytecodeReadRafSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: bytecode read+RAF checking.
    def __init__(self, bytecode_preprocessing, n_cycle_vars, one_hot_params, opening_accumulator, transcript):  # Sample gammas and build LHS params.
        from field import Fr  # local import for field operations
        from rv64imac.isa import CIRCUIT_FLAGS, INSTRUCTION_FLAGS  # canonical flag orders

        self.bytecode = list(bytecode_preprocessing.bytecode)
        self.one_hot_params = one_hot_params
        self.K = int(one_hot_params.bytecode_k)
        self.log_K = log2_pow2(self.K)
        self.log_T = int(n_cycle_vars)
        self.d = int(one_hot_params.bytecode_d)
        self.log_reg_K = int(REGISTER_COUNT).bit_length() - 1  # Rust: log2(REGISTER_COUNT)=7 (128 regs)

        self.gamma_powers = list(transcript.challenge_scalar_powers(7))
        stage1_gammas = list(transcript.challenge_scalar_powers(2 + len(CIRCUIT_FLAGS)))
        stage2_gammas = list(transcript.challenge_scalar_powers(5))
        stage3_gammas = list(transcript.challenge_scalar_powers(9))
        stage4_gammas = list(transcript.challenge_scalar_powers(3))
        stage5_gammas = list(transcript.challenge_scalar_powers(2 + len(LOOKUP_TABLES_64)))

        def v(name, sumcheck):  # Read a virtual claim from accumulator.
            return opening_accumulator.get_virtual_polynomial_opening(name, sumcheck)[1]

        rv1_terms = [v(VirtualPolynomial.UnexpandedPC, SumcheckId.SpartanOuter), v(VirtualPolynomial.Imm, SumcheckId.SpartanOuter)]
        for flag in CIRCUIT_FLAGS:
            rv1_terms.append(v(getattr(VirtualPolynomial, f"OpFlags_{flag}"), SumcheckId.SpartanOuter))
        rv1 = Fr.zero()
        for c, g in zip(rv1_terms, stage1_gammas):
            rv1 += c * g

        rv2_terms = [
            v(VirtualPolynomial.OpFlags_Jump, SumcheckId.SpartanProductVirtualization),
            v(VirtualPolynomial.InstructionFlags_Branch, SumcheckId.SpartanProductVirtualization),
            v(VirtualPolynomial.InstructionFlags_IsRdNotZero, SumcheckId.SpartanProductVirtualization),
            v(VirtualPolynomial.OpFlags_WriteLookupOutputToRD, SumcheckId.SpartanProductVirtualization),
            v(VirtualPolynomial.OpFlags_VirtualInstruction, SumcheckId.SpartanProductVirtualization),
        ]
        rv2 = Fr.zero()
        for c, g in zip(rv2_terms, stage2_gammas):
            rv2 += c * g

        imm_claim = v(VirtualPolynomial.Imm, SumcheckId.InstructionInputVirtualization)
        unexpanded_pc_shift = v(VirtualPolynomial.UnexpandedPC, SumcheckId.SpartanShift)
        unexpanded_pc_instr = v(VirtualPolynomial.UnexpandedPC, SumcheckId.InstructionInputVirtualization)
        if unexpanded_pc_shift != unexpanded_pc_instr:
            raise SumcheckVerifyError("UnexpandedPC claim mismatch across stages")
        rv3_terms = [
            imm_claim,
            unexpanded_pc_shift,
            v(VirtualPolynomial.InstructionFlags_LeftOperandIsRs1Value, SumcheckId.InstructionInputVirtualization),
            v(VirtualPolynomial.InstructionFlags_LeftOperandIsPC, SumcheckId.InstructionInputVirtualization),
            v(VirtualPolynomial.InstructionFlags_RightOperandIsRs2Value, SumcheckId.InstructionInputVirtualization),
            v(VirtualPolynomial.InstructionFlags_RightOperandIsImm, SumcheckId.InstructionInputVirtualization),
            v(VirtualPolynomial.InstructionFlags_IsNoop, SumcheckId.SpartanShift),
            v(VirtualPolynomial.OpFlags_VirtualInstruction, SumcheckId.SpartanShift),
            v(VirtualPolynomial.OpFlags_IsFirstInSequence, SumcheckId.SpartanShift),
        ]
        rv3 = Fr.zero()
        for c, g in zip(rv3_terms, stage3_gammas):
            rv3 += c * g

        rv4_terms = [
            v(VirtualPolynomial.RdWa, SumcheckId.RegistersReadWriteChecking),
            v(VirtualPolynomial.Rs1Ra, SumcheckId.RegistersReadWriteChecking),
            v(VirtualPolynomial.Rs2Ra, SumcheckId.RegistersReadWriteChecking),
        ]
        rv4 = Fr.zero()
        for c, g in zip(rv4_terms, stage4_gammas):
            rv4 += c * g

        rv5 = v(VirtualPolynomial.RdWa, SumcheckId.RegistersValEvaluation) * stage5_gammas[0]
        rv5 += v(VirtualPolynomial.InstructionRafFlag, SumcheckId.InstructionReadRaf) * stage5_gammas[1]
        for i in range(len(LOOKUP_TABLES_64)):
            rv5 += opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.LookupTableFlag,
                i,
                SumcheckId.InstructionReadRaf,
            )[1] * stage5_gammas[2 + i]

        raf_claim = v(VirtualPolynomial.PC, SumcheckId.SpartanOuter)
        raf_shift_claim = v(VirtualPolynomial.PC, SumcheckId.SpartanShift)

        self.rv_claims = [rv1, rv2, rv3, rv4, rv5]
        self.raf_claim = raf_claim
        self.raf_shift_claim = raf_shift_claim
        self.input_claim_cached = Fr.zero()
        for c, g in zip([rv1, rv2, rv3, rv4, rv5, raf_claim, raf_shift_claim], self.gamma_powers):
            self.input_claim_cached += c * g

        self.int_poly = IdentityPolynomial(self.log_K)
        self.stage_gammas = [stage1_gammas, stage2_gammas, stage3_gammas, stage4_gammas, stage5_gammas]

        self.r_cycles = [
            opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Imm, SumcheckId.SpartanOuter)[0].r,
            opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.OpFlags_Jump, SumcheckId.SpartanProductVirtualization)[0].r,
            opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.UnexpandedPC, SumcheckId.SpartanShift)[0].r,
            opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.Rs1Ra, SumcheckId.RegistersReadWriteChecking)[0].r[self.log_reg_K :],
            opening_accumulator.get_virtual_polynomial_opening(VirtualPolynomial.RdWa, SumcheckId.RegistersValEvaluation)[0].r[self.log_reg_K :],
        ]

        r_register_4 = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RdWa,
            SumcheckId.RegistersReadWriteChecking,
        )[0].r[: self.log_reg_K]
        r_register_5 = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RdWa,
            SumcheckId.RegistersValEvaluation,
        )[0].r[: self.log_reg_K]
        self.eq_r_register_4 = EqPolynomial.evals(r_register_4)
        self.eq_r_register_5 = EqPolynomial.evals(r_register_5)

    def degree(self):  # Rust: d + 1.
        return self.d + 1

    def num_rounds(self):  # Rust: log_K + log_T.
        return self.log_K + self.log_T

    def input_claim(self, _opening_accumulator):  # Rust: params caches input_claim.
        return self.input_claim_cached

    def _normalize_opening_point(self, sumcheck_challenges):  # Rust: reverse address + cycle segments.
        r = list(sumcheck_challenges)
        r[: self.log_K] = list(reversed(r[: self.log_K]))
        r[self.log_K :] = list(reversed(r[self.log_K :]))
        return OpeningPoint(r, BIG_ENDIAN)

    def _val_eval_stage(self, stage_idx: int, eq_r_addr: list[Fr]) -> Fr:  # Evaluate stage Val(k) MLE at r_address.
        from rv64imac.isa import circuit_flags, instruction_flags, lookup_table  # ISA metadata
        from rv64imac.types import Xlen  # RV64 tag
        from field import Fr  # local import for zero()/one()

        gammas = self.stage_gammas[stage_idx]
        acc = Fr.zero()
        for k, inst in enumerate(self.bytecode):
            inst = inst.normalize()
            cf = circuit_flags(inst)
            inf = instruction_flags(inst)
            v = Fr.zero()
            if stage_idx == 0:
                v = Fr(int(inst.address)) + Fr(int(inst.operands.imm)) * gammas[1]
                for flag_val, gp in zip(cf, gammas[2:]):
                    if flag_val:
                        v += gp
            elif stage_idx == 1:
                if cf[5]:
                    v += gammas[0]  # Jump
                if inf[4]:
                    v += gammas[1]  # Branch
                if inf[6]:
                    v += gammas[2]  # IsRdNotZero
                if cf[6]:
                    v += gammas[3]  # WriteLookupOutputToRD
                if cf[7]:
                    v += gammas[4]  # VirtualInstruction
            elif stage_idx == 2:
                v = Fr(int(inst.operands.imm)) + gammas[1] * Fr(int(inst.address))
                if inf[2]:
                    v += gammas[2]
                if inf[0]:
                    v += gammas[3]
                if inf[3]:
                    v += gammas[4]
                if inf[1]:
                    v += gammas[5]
                if inf[5]:
                    v += gammas[6]
                if cf[7]:
                    v += gammas[7]
                if cf[12]:
                    v += gammas[8]
            elif stage_idx == 3:
                rd = inst.operands.rd
                rs1 = inst.operands.rs1
                rs2 = inst.operands.rs2
                rd_eq = self.eq_r_register_4[int(rd)] if rd is not None else Fr.zero()
                rs1_eq = self.eq_r_register_4[int(rs1)] if rs1 is not None else Fr.zero()
                rs2_eq = self.eq_r_register_4[int(rs2)] if rs2 is not None else Fr.zero()
                v = rd_eq * gammas[0] + rs1_eq * gammas[1] + rs2_eq * gammas[2]
            elif stage_idx == 4:
                rd = inst.operands.rd
                rd_eq = self.eq_r_register_5[int(rd)] if rd is not None else Fr.zero()
                v = rd_eq * gammas[0]
                is_interleaved = (not cf[0]) and (not cf[1]) and (not cf[2]) and (not cf[10])
                if not is_interleaved:
                    v += gammas[1]
                t = lookup_table(inst, Xlen.Bit64)
                if t is not None:
                    idx = LOOKUP_TABLES_64.index(t)
                    v += gammas[2 + idx]
            acc += v * eq_r_addr[k]
        return acc

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: read_raf_checking.rs:619-670 (verifier).
        from field import Fr  # local import for zero()
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        r_address_prime = opening_point.r[: self.log_K]
        r_cycle_prime = opening_point.r[self.log_K :]

        int_eval = self.int_poly.evaluate(r_address_prime)
        ra_prod = None
        for i in range(self.d):
            c = opening_accumulator.get_committed_polynomial_opening_i(
                CommittedPolynomial.BytecodeRa,
                i,
                SumcheckId.BytecodeReadRaf,
            )[1]
            ra_prod = c if ra_prod is None else (ra_prod * c)

        eq_r_addr = EqPolynomial.evals(r_address_prime)
        val = Fr.zero()
        for stage_idx in range(5):
            val_eval = self._val_eval_stage(stage_idx, eq_r_addr)
            inj = Fr.zero()
            if stage_idx == 0:
                inj = int_eval * self.gamma_powers[5]
            if stage_idx == 2:
                inj = int_eval * self.gamma_powers[4]
            eq_cycle = EqPolynomial.mle(self.r_cycles[stage_idx], r_cycle_prime)
            val += (val_eval + inj) * eq_cycle * self.gamma_powers[stage_idx]

        return ra_prod * val

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: read_raf_checking.rs:672-697.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        r_address = opening_point.r[: self.log_K]
        r_cycle = opening_point.r[self.log_K :]
        r_address_chunks = self.one_hot_params.compute_r_address_chunks(r_address)
        for i in range(self.d):
            opening_accumulator.append_dense_i(
                transcript,
                CommittedPolynomial.BytecodeRa,
                i,
                SumcheckId.BytecodeReadRaf,
                list(r_address_chunks[i]) + list(r_cycle),
            )


class HammingWeightClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 7: fused HW+address reduction.
    DEGREE_BOUND = 2

    def __init__(self, one_hot_params, opening_accumulator, transcript):  # Rust: HammingWeightClaimReductionParams::new.
        from field import Fr  # local import

        self.one_hot_params = one_hot_params
        self.log_k_chunk = int(one_hot_params.log_k_chunk)
        self.instruction_d = int(one_hot_params.instruction_d)
        self.bytecode_d = int(one_hot_params.bytecode_d)
        self.ram_d = int(one_hot_params.ram_d)
        self.N = self.instruction_d + self.bytecode_d + self.ram_d

        self.poly_types = []
        for i in range(self.instruction_d):
            self.poly_types.append((CommittedPolynomial.InstructionRa, i))
        for i in range(self.bytecode_d):
            self.poly_types.append((CommittedPolynomial.BytecodeRa, i))
        for i in range(self.ram_d):
            self.poly_types.append((CommittedPolynomial.RamRa, i))

        self.gamma_powers = list(transcript.challenge_scalar_powers(3 * self.N))

        unified_bool_point, _ = opening_accumulator.get_committed_polynomial_opening_i(
            CommittedPolynomial.InstructionRa,
            0,
            SumcheckId.Booleanity,
        )
        self.r_addr_bool = list(unified_bool_point.r[: self.log_k_chunk])  # BE
        self.r_cycle = list(unified_bool_point.r[self.log_k_chunk :])  # BE

        ram_hw_factor = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamHammingWeight,
            SumcheckId.RamHammingBooleanity,
        )[1]

        self.r_addr_virt = []
        self.claims_hw = []
        self.claims_bool = []
        self.claims_virt = []
        for poly, idx in self.poly_types:
            if poly == CommittedPolynomial.InstructionRa:
                virt_sumcheck_id = SumcheckId.InstructionRaVirtualization
                hw_claim = Fr.one()
            elif poly == CommittedPolynomial.BytecodeRa:
                virt_sumcheck_id = SumcheckId.BytecodeReadRaf
                hw_claim = Fr.one()
            else:
                virt_sumcheck_id = SumcheckId.RamRaVirtualization
                hw_claim = ram_hw_factor
            self.claims_hw.append(hw_claim)
            self.claims_bool.append(
                opening_accumulator.get_committed_polynomial_opening_i(poly, idx, SumcheckId.Booleanity)[1]
            )
            virt_point, virt_claim = opening_accumulator.get_committed_polynomial_opening_i(poly, idx, virt_sumcheck_id)
            self.r_addr_virt.append(list(virt_point.r[: self.log_k_chunk]))  # BE
            self.claims_virt.append(virt_claim)

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.log_k_chunk

    def input_claim(self, _opening_accumulator):  # Rust: hamming_weight.rs:251-261.
        acc = None
        for i in range(self.N):
            term = self.gamma_powers[3 * i] * self.claims_hw[i]
            term += self.gamma_powers[3 * i + 1] * self.claims_bool[i]
            term += self.gamma_powers[3 * i + 2] * self.claims_virt[i]
            acc = term if acc is None else (acc + term)
        return acc

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: hamming_weight.rs:476-514.
        rho_rev = list(reversed(sumcheck_challenges))
        eq_bool_eval = EqPolynomial.mle(rho_rev, self.r_addr_bool)
        out = None
        for i, (poly, idx) in enumerate(self.poly_types):
            eq_virt_eval = EqPolynomial.mle(rho_rev, self.r_addr_virt[i])
            g_i = opening_accumulator.get_committed_polynomial_opening_i(
                poly,
                idx,
                SumcheckId.HammingWeightClaimReduction,
            )[1]
            w = self.gamma_powers[3 * i]
            w += self.gamma_powers[3 * i + 1] * eq_bool_eval
            w += self.gamma_powers[3 * i + 2] * eq_virt_eval
            term = g_i * w
            out = term if out is None else (out + term)
        return out

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: hamming_weight.rs:516-538.
        r_address = OpeningPoint(list(sumcheck_challenges), LITTLE_ENDIAN).match_endianness(BIG_ENDIAN).r
        full_point = list(r_address) + list(self.r_cycle)
        for poly, idx in self.poly_types:
            opening_accumulator.append_dense_i(
                transcript,
                poly,
                idx,
                SumcheckId.HammingWeightClaimReduction,
                full_point,
            )
