"""Stage 2: Product virtualization + batched sumchecks."""
from openings import BIG_ENDIAN, CommittedPolynomial, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial, LagrangePolynomial, RangeMaskPolynomial, UnmapRamAddressPolynomial, log2_pow2
from ram_io import eval_io_mle, remap_address
from rv64imac.constants import RAM_START_ADDRESS
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be, _cache_virtual_batch

PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE = 5  # Rust: PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE.
PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS = 13  # Rust: PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS.
PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND = 12  # Rust: PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND.


class ProductVirtualUniSkipParams:  # Params for product virtualization uni-skip (Stage 2a).
    def __init__(self, tau, base_evals):  # Store tau=[tau_low||tau_high] and 5 base evals.
        self.tau = list(tau)
        self.base_evals = list(base_evals)

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
