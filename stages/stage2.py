"""Stage 2: Product virtualization + batched sumchecks."""
from field import Fr
from openings import BIG_ENDIAN, CommittedPolynomial as CP, OpeningPoint, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial, LagrangePolynomial, RangeMaskPolynomial, UnmapRamAddressPolynomial, log2_pow2
from ram_io import eval_io_mle, remap_address
from rv64imac.constants import RAM_START_ADDRESS
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be, _cache_virtual_batch

PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE = 5
PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS = 13
PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND = 12

class ProductVirtualUniSkipParams:  # Params for product virtualization uni-skip (Stage 2a).
    def __init__(self, tau, base_evals):
        self.tau = list(tau)
        self.base_evals = list(base_evals)

class ProductVirtualUniSkipVerifier(SumcheckInstanceVerifier):  # Stage 2a: univariate-skip verifier.
    DEGREE_BOUND = PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND

    def __init__(self, acc, transcript):
        r_cycle, _ = acc.vp(VP.Product, SC.SpartanOuter)
        tau_high = transcript.challenge_scalar_optimized()
        tau = list(r_cycle.r) + [tau_high]
        base_names = [VP.Product, VP.WriteLookupOutputToRD, VP.WritePCtoRD, VP.ShouldBranch, VP.ShouldJump]
        base_evals = [acc.vp(n, SC.SpartanOuter)[1] for n in base_names]
        self.params = ProductVirtualUniSkipParams(tau, base_evals)

    def num_rounds(self):
        return 1

    def input_claim(self, _acc):
        tau_high = self.params.tau[-1]
        w = LagrangePolynomial.evals(tau_high, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE)
        out = None
        for wi, bi in zip(w, self.params.base_evals):
            term = wi * bi
            out = term if out is None else (out + term)
        return out

    def expected_output_claim(self, _acc, _r):
        raise NotImplementedError("uniskip verifier does not use expected_output_claim")

    def cache_openings(self, acc, transcript, r):
        acc.append_virtual(transcript, VP.UnivariateSkip, SC.SpartanProductVirtualization, OpeningPoint(list(r), BIG_ENDIAN))

PRODUCT_UNIQUE_FACTOR_VIRTUALS = [
    VP.LeftInstructionInput, VP.RightInstructionInput, VP.InstructionFlags_IsRdNotZero,
    VP.OpFlags_WriteLookupOutputToRD, VP.OpFlags_Jump, VP.LookupOutput,
    VP.InstructionFlags_Branch, VP.NextIsNoop, VP.OpFlags_VirtualInstruction,
]

class ProductVirtualRemainderVerifier(SumcheckInstanceVerifier):  # Stage 2: product virtualization remainder.
    DEGREE_BOUND = 3

    def __init__(self, trace_len, uni_skip_params, acc):
        self.trace_len = int(trace_len)
        self.n_cycle_vars = log2_pow2(self.trace_len)
        self.tau = list(uni_skip_params.tau)
        r0_point, _ = acc.vp(VP.UnivariateSkip, SC.SpartanProductVirtualization)
        if len(r0_point) != 1:
            raise ValueError("expected product uniskip opening point length 1")
        self.r0 = r0_point[0]

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, acc):
        return acc.vp(VP.UnivariateSkip, SC.SpartanProductVirtualization)[1]

    def expected_output_claim(self, acc, r):
        w = LagrangePolynomial.evals(self.r0, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE)
        def g(name):
            return acc.vp(name, SC.SpartanProductVirtualization)[1]
        l_inst, r_inst = g(VP.LeftInstructionInput), g(VP.RightInstructionInput)
        is_rd_not_zero = g(VP.InstructionFlags_IsRdNotZero)
        wl_flag, j_flag = g(VP.OpFlags_WriteLookupOutputToRD), g(VP.OpFlags_Jump)
        lookup_out, branch_flag = g(VP.LookupOutput), g(VP.InstructionFlags_Branch)
        next_is_noop = g(VP.NextIsNoop)
        fused_left = w[0] * l_inst + w[1] * is_rd_not_zero + w[2] * is_rd_not_zero + w[3] * lookup_out + w[4] * j_flag
        fused_right = w[0] * r_inst + w[1] * wl_flag + w[2] * j_flag + w[3] * branch_flag + w[4] * (Fr.one() - next_is_noop)
        tau_high, tau_low = self.tau[-1], self.tau[:-1]
        tau_high_bound = LagrangePolynomial.lagrange_kernel(tau_high, self.r0, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE)
        tau_bound = EqPolynomial.mle(tau_low, list(reversed(r)))
        return tau_high_bound * tau_bound * fused_left * fused_right

    def cache_openings(self, acc, transcript, r):
        _cache_virtual_batch(acc, transcript, PRODUCT_UNIQUE_FACTOR_VIRTUALS, SC.SpartanProductVirtualization, _normalize_le_to_be(r))

class InstructionLookupsClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 2: instruction lookups claim reduction.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, acc, transcript):
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.r_spartan = acc.vp(VP.LookupOutput, SC.SpartanOuter)[0]

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, acc):
        lo = acc.vp(VP.LookupOutput, SC.SpartanOuter)[1]
        l = acc.vp(VP.LeftLookupOperand, SC.SpartanOuter)[1]
        r = acc.vp(VP.RightLookupOperand, SC.SpartanOuter)[1]
        return lo + self.gamma * l + self.gamma_sqr * r

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        lo = acc.vp(VP.LookupOutput, SC.InstructionClaimReduction)[1]
        l = acc.vp(VP.LeftLookupOperand, SC.InstructionClaimReduction)[1]
        r_ = acc.vp(VP.RightLookupOperand, SC.InstructionClaimReduction)[1]
        return EqPolynomial.mle(pt.r, self.r_spartan.r) * (lo + self.gamma * l + self.gamma_sqr * r_)

    def cache_openings(self, acc, transcript, r):
        pt = _normalize_le_to_be(r)
        _cache_virtual_batch(acc, transcript, [VP.LookupOutput, VP.LeftLookupOperand, VP.RightLookupOperand], SC.InstructionClaimReduction, pt)

class RamRafEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 2: RAM RAF evaluation.
    DEGREE_BOUND = 2

    def __init__(self, memory_layout, one_hot_params, acc):
        self.start_address = int(memory_layout.get_lowest_address())
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.r_cycle = acc.vp(VP.RamAddress, SC.SpartanOuter)[0]

    def num_rounds(self):
        return self.log_K

    def input_claim(self, acc):
        return acc.vp(VP.RamAddress, SC.SpartanOuter)[1]

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        unmap_eval = UnmapRamAddressPolynomial(self.log_K, self.start_address).evaluate(pt.r)
        return unmap_eval * acc.vp(VP.RamRa, SC.RamRafEvaluation)[1]

    def cache_openings(self, acc, transcript, r):
        r_addr = _normalize_le_to_be(r)
        acc.append_virtual(transcript, VP.RamRa, SC.RamRafEvaluation, OpeningPoint(list(r_addr.r) + list(self.r_cycle.r), BIG_ENDIAN))

class OutputSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 2: RAM output check.
    DEGREE_BOUND = 3

    def __init__(self, ram_K, program_io, transcript):
        self.K = int(ram_K)
        self.r_address = transcript.challenge_vector_optimized(log2_pow2(self.K))
        self.program_io = program_io

    def num_rounds(self):
        return log2_pow2(self.K)

    def input_claim(self, _acc):
        return Fr.zero()

    def expected_output_claim(self, acc, r):
        val_final = acc.vp(VP.RamValFinal, SC.RamOutputCheck)[1]
        r_addr = _normalize_le_to_be(r).r
        io_start = int(remap_address(self.program_io.memory_layout.input_start, self.program_io.memory_layout))
        io_end = int(remap_address(RAM_START_ADDRESS, self.program_io.memory_layout))
        eq_eval = EqPolynomial.mle(self.r_address, r_addr)
        io_mask_eval = RangeMaskPolynomial(io_start, io_end).evaluate_mle(r_addr)
        return eq_eval * io_mask_eval * (val_final - eval_io_mle(self.program_io, r_addr))

    def cache_openings(self, acc, transcript, r):
        pt = _normalize_le_to_be(r)
        _cache_virtual_batch(acc, transcript, [VP.RamValFinal, VP.RamValInit], SC.RamOutputCheck, pt)

class RamReadWriteCheckingVerifier(SumcheckInstanceVerifier):  # Stage 2: RAM read/write checking.
    DEGREE_BOUND = 3

    def __init__(self, acc, transcript, one_hot_params, trace_length, config):
        self.gamma = transcript.challenge_scalar()
        self.K, self.T = int(one_hot_params.ram_k), int(trace_length)
        self.r_cycle_stage1 = acc.vp(VP.RamReadValue, SC.SpartanOuter)[0]
        self.phase1, self.phase2 = int(config.ram_rw_phase1_num_rounds), int(config.ram_rw_phase2_num_rounds)

    def num_rounds(self):
        return log2_pow2(self.K) + log2_pow2(self.T)

    def input_claim(self, acc):
        rv = acc.vp(VP.RamReadValue, SC.SpartanOuter)[1]
        wv = acc.vp(VP.RamWriteValue, SC.SpartanOuter)[1]
        return rv + self.gamma * wv

    def _normalize_opening_point(self, r):  # 3-phase reversal (read_write_checking.rs:119-151).
        logT, logK = log2_pow2(self.T), log2_pow2(self.K)
        p1, rest = list(r[:self.phase1]), list(r[self.phase1:])
        p2, rest = list(rest[:self.phase2]), list(rest[self.phase2:])
        p3_cycle, p3_addr = list(rest[:logT - self.phase1]), list(rest[logT - self.phase1:])
        r_cycle = list(reversed(p3_cycle)) + list(reversed(p1))
        r_addr = list(reversed(p3_addr)) + list(reversed(p2))
        if len(r_cycle) != logT or len(r_addr) != logK:
            raise ValueError("normalize_opening_point: bad split sizes")
        return OpeningPoint(r_addr + r_cycle, BIG_ENDIAN)

    def expected_output_claim(self, acc, r):
        pt = self._normalize_opening_point(r)
        logK = log2_pow2(self.K)
        eq_eval = EqPolynomial.mle(self.r_cycle_stage1.r, pt.r[logK:])
        ra = acc.vp(VP.RamRa, SC.RamReadWriteChecking)[1]
        val = acc.vp(VP.RamVal, SC.RamReadWriteChecking)[1]
        inc = acc.cp(CP.RamInc, SC.RamReadWriteChecking)[1]
        return eq_eval * ra * (val + self.gamma * (val + inc))

    def cache_openings(self, acc, transcript, r):
        pt = self._normalize_opening_point(r)
        acc.append_virtual(transcript, VP.RamVal, SC.RamReadWriteChecking, pt.clone())
        acc.append_virtual(transcript, VP.RamRa, SC.RamReadWriteChecking, pt.clone())
        acc.append_dense(transcript, CP.RamInc, SC.RamReadWriteChecking, pt.r[log2_pow2(self.K):])
