"""Stage 5: InstructionReadRaf + RamRaClaimReduction + RegistersValEvaluation."""
from field import Fr
from ids_generated import LOOKUP_TABLES_64
from lookup_tables import evaluate_mle
from openings import BIG_ENDIAN, CommittedPolynomial as CP, OpeningPoint, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial, IdentityPolynomial, OperandPolynomial, log2_pow2
from rv64imac.constants import REGISTER_COUNT
from sumchecks import SumcheckInstanceVerifier, SumcheckVerifyError, _normalize_le_to_be


class InstructionReadRafSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: instruction lookups read+RAF checking.
    LOG_K = 128  # instruction_lookups::LOG_K = XLEN*2 = 128 for RV64.

    def __init__(self, n_cycle_vars, one_hot_params, acc, transcript):
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.log_T = int(n_cycle_vars)
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)
        self.r_reduction = acc.vp(VP.LookupOutput, SC.InstructionClaimReduction)[0]

    def degree(self):  # n_virtual_ra_polys + 2.
        return (self.LOG_K // self.ra_virtual_log_k_chunk) + 2

    def num_rounds(self):
        return self.LOG_K + self.log_T

    def input_claim(self, acc):
        rv = acc.vp(VP.LookupOutput, SC.InstructionClaimReduction)[1]
        rv_branch = acc.vp(VP.LookupOutput, SC.SpartanProductVirtualization)[1]
        if rv != rv_branch:
            raise SumcheckVerifyError("LookupOutput claim mismatch across stages")
        left = acc.vp(VP.LeftLookupOperand, SC.InstructionClaimReduction)[1]
        right = acc.vp(VP.RightLookupOperand, SC.InstructionClaimReduction)[1]
        return rv + self.gamma * left + self.gamma_sqr * right

    def _normalize_opening_point(self, r):
        r_addr = list(r[:self.LOG_K])
        r_cycle = list(reversed(r[self.LOG_K:]))
        return OpeningPoint(r_addr + r_cycle, BIG_ENDIAN)

    def expected_output_claim(self, acc, r):
        pt = self._normalize_opening_point(r)
        r_addr, r_cycle = pt.r[:self.LOG_K], pt.r[self.LOG_K:]
        left_op = OperandPolynomial(self.LOG_K, OperandPolynomial.LEFT).evaluate(r_addr)
        right_op = OperandPolynomial(self.LOG_K, OperandPolynomial.RIGHT).evaluate(r_addr)
        id_eval = IdentityPolynomial(self.LOG_K).evaluate(r_addr)
        val_evals = [evaluate_mle(t, r_addr, xlen=64) for t in LOOKUP_TABLES_64]
        eq_eval = EqPolynomial.mle(self.r_reduction.r, r_cycle)
        n_virt = self.LOG_K // self.ra_virtual_log_k_chunk
        ra_claim = None
        for i in range(n_virt):
            c = acc.vpi(VP.InstructionRa, i, SC.InstructionReadRaf)[1]
            ra_claim = c if ra_claim is None else (ra_claim * c)
        table_flags = [acc.vpi(VP.LookupTableFlag, i, SC.InstructionReadRaf)[1] for i in range(len(LOOKUP_TABLES_64))]
        raf_flag = acc.vp(VP.InstructionRafFlag, SC.InstructionReadRaf)[1]
        val_claim = None
        for ev, flag in zip(val_evals, table_flags):
            term = ev * flag
            val_claim = term if val_claim is None else (val_claim + term)
        raf_claim = (Fr.one() - raf_flag) * (left_op + self.gamma * right_op) + raf_flag * self.gamma * id_eval
        return eq_eval * ra_claim * (val_claim + self.gamma * raf_claim)

    def cache_openings(self, acc, transcript, r):
        pt = self._normalize_opening_point(r)
        r_addr_pt = OpeningPoint(pt.r[:self.LOG_K], BIG_ENDIAN)
        r_cycle_pt = OpeningPoint(pt.r[self.LOG_K:], BIG_ENDIAN)
        for i in range(len(LOOKUP_TABLES_64)):
            acc.append_virtual_i(transcript, VP.LookupTableFlag, i, SC.InstructionReadRaf, r_cycle_pt.clone())
        chunk = self.ra_virtual_log_k_chunk
        for i, j in enumerate(range(0, len(r_addr_pt.r), chunk)):
            op = OpeningPoint(list(r_addr_pt.r[j:j+chunk]) + list(r_cycle_pt.r), BIG_ENDIAN)
            acc.append_virtual_i(transcript, VP.InstructionRa, i, SC.InstructionReadRaf, op)
        acc.append_virtual(transcript, VP.InstructionRafFlag, SC.InstructionReadRaf, r_cycle_pt.clone())


class RamRaClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: reduce RAM RA claims.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, one_hot_params, acc, transcript):
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.log_T = log2_pow2(int(trace_len))
        r_raf, self.claim_raf = acc.vp(VP.RamRa, SC.RamRafEvaluation)
        r_rw, self.claim_rw = acc.vp(VP.RamRa, SC.RamReadWriteChecking)
        r_val_eval, self.claim_val_eval = acc.vp(VP.RamRa, SC.RamValEvaluation)
        r_val_final, self.claim_val_final = acc.vp(VP.RamRa, SC.RamValFinalEvaluation)
        K = self.log_K
        self.r_address_1, self.r_cycle_raf = list(r_raf.r[:K]), list(r_raf.r[K:])
        self.r_address_2, self.r_cycle_rw = list(r_rw.r[:K]), list(r_rw.r[K:])
        self.r_cycle_val = list(r_val_eval.r[K:])
        if self.r_address_1 != list(r_val_final.r[:K]):
            raise ValueError("unexpected r_address mismatch for RAM RA reduction")
        if self.r_address_2 != list(r_val_eval.r[:K]):
            raise ValueError("unexpected r_address mismatch for RAM RA reduction")
        if self.r_cycle_val != list(r_val_final.r[K:]):
            raise ValueError("unexpected r_cycle mismatch for RAM RA reduction")
        self.gamma = transcript.challenge_scalar()
        self.gamma_squared = self.gamma * self.gamma
        self.gamma_cubed = self.gamma_squared * self.gamma

    def num_rounds(self):
        return self.log_K + self.log_T

    def input_claim(self, _acc):
        return self.claim_raf + self.gamma * self.claim_val_final + self.gamma_squared * self.claim_rw + self.gamma_cubed * self.claim_val_eval

    def expected_output_claim(self, acc, r):
        r_addr = list(reversed(r[:self.log_K]))
        r_cycle = list(reversed(r[self.log_K:]))
        eq_a1 = EqPolynomial.mle(self.r_address_1, r_addr)
        eq_a2 = EqPolynomial.mle(self.r_address_2, r_addr)
        eq_c_raf = EqPolynomial.mle(self.r_cycle_raf, r_cycle)
        eq_c_rw = EqPolynomial.mle(self.r_cycle_rw, r_cycle)
        eq_c_val = EqPolynomial.mle(self.r_cycle_val, r_cycle)
        eq_A = eq_c_raf + self.gamma * eq_c_val
        eq_B = eq_c_rw + self.gamma * eq_c_val
        eq_combined = eq_a1 * eq_A + self.gamma_squared * eq_a2 * eq_B
        return eq_combined * acc.vp(VP.RamRa, SC.RamRaClaimReduction)[1]

    def cache_openings(self, acc, transcript, r):
        r_addr_be = list(reversed(r[:self.log_K]))
        r_cycle_be = list(reversed(r[self.log_K:]))
        acc.append_virtual(transcript, VP.RamRa, SC.RamRaClaimReduction, OpeningPoint(r_addr_be + r_cycle_be, BIG_ENDIAN))


class RegistersValEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: registers val evaluation.
    DEGREE_BOUND = 3
    LOG_K = int(REGISTER_COUNT).bit_length() - 1  # log2(128)=7

    def __init__(self, acc):
        r_full = acc.vp(VP.RegistersVal, SC.RegistersReadWriteChecking)[0]
        self.r_cycle = OpeningPoint(r_full.r[self.LOG_K:], BIG_ENDIAN)
        self.r_address = OpeningPoint(r_full.r[:self.LOG_K], BIG_ENDIAN)

    def num_rounds(self):
        return len(self.r_cycle.r)

    def input_claim(self, acc):
        return acc.vp(VP.RegistersVal, SC.RegistersReadWriteChecking)[1]

    def expected_output_claim(self, acc, r):
        r_cycle_prime = _normalize_le_to_be(r)
        lt_eval, eq_term = Fr.zero(), Fr.one()
        for x, y in zip(r_cycle_prime.r, self.r_cycle.r):
            lt_eval += (Fr.one() - x) * y * eq_term
            eq_term *= Fr.one() - x - y + x * y + x * y
        inc = acc.cp(CP.RdInc, SC.RegistersValEvaluation)[1]
        wa = acc.vp(VP.RdWa, SC.RegistersValEvaluation)[1]
        return inc * wa * lt_eval

    def cache_openings(self, acc, transcript, r):
        r_cycle_prime = _normalize_le_to_be(r)
        acc.append_dense(transcript, CP.RdInc, SC.RegistersValEvaluation, r_cycle_prime.r)
        acc.append_virtual(transcript, VP.RdWa, SC.RegistersValEvaluation, OpeningPoint(list(self.r_address.r) + list(r_cycle_prime.r), BIG_ENDIAN))
