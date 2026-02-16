"""Stage 4: Registers read/write checking + RAM val evaluation + RAM val final."""
from field import Fr
from openings import AdviceKind, BIG_ENDIAN, CommittedPolynomial as CP, OpeningPoint, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial, log2_pow2
from ram_io import calculate_advice_memory_evaluation, eval_initial_ram_mle
from rv64imac.constants import REGISTER_COUNT
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be


class RegistersReadWriteCheckingVerifier(SumcheckInstanceVerifier):  # Stage 4: registers read/write checking.
    DEGREE_BOUND = 3
    LOG_K = int(REGISTER_COUNT).bit_length() - 1  # log2(128)=7

    def __init__(self, trace_len, acc, transcript, rw_config):
        self.gamma = transcript.challenge_scalar()
        self.T = int(trace_len)
        self.phase1, self.phase2 = int(rw_config.registers_rw_phase1_num_rounds), int(rw_config.registers_rw_phase2_num_rounds)
        self.r_cycle = acc.vp(VP.RdWriteValue, SC.RegistersClaimReduction)[0]

    def num_rounds(self):
        return self.LOG_K + log2_pow2(self.T)

    def input_claim(self, acc):
        rd = acc.vp(VP.RdWriteValue, SC.RegistersClaimReduction)[1]
        rs1 = acc.vp(VP.Rs1Value, SC.RegistersClaimReduction)[1]
        rs2 = acc.vp(VP.Rs2Value, SC.RegistersClaimReduction)[1]
        return rd + self.gamma * (rs1 + self.gamma * rs2)

    def _normalize_opening_point(self, r):  # 3-phase reversal (registers/read_write_checking.rs:141-174).
        logT = log2_pow2(self.T)
        p1, rest = list(r[:self.phase1]), list(r[self.phase1:])
        p2, rest = list(rest[:self.phase2]), list(rest[self.phase2:])
        p3_cycle, p3_addr = list(rest[:logT - self.phase1]), list(rest[logT - self.phase1:])
        r_cycle = list(reversed(p3_cycle)) + list(reversed(p1))
        r_addr = list(reversed(p3_addr)) + list(reversed(p2))
        if len(r_cycle) != logT or len(r_addr) != self.LOG_K:
            raise ValueError("registers normalize_opening_point: bad split sizes")
        return OpeningPoint(r_addr + r_cycle, BIG_ENDIAN)

    def expected_output_claim(self, acc, r):
        pt = self._normalize_opening_point(r)
        r_cycle = pt.r[self.LOG_K:]
        val = acc.vp(VP.RegistersVal, SC.RegistersReadWriteChecking)[1]
        rs1_ra = acc.vp(VP.Rs1Ra, SC.RegistersReadWriteChecking)[1]
        rs2_ra = acc.vp(VP.Rs2Ra, SC.RegistersReadWriteChecking)[1]
        rd_wa = acc.vp(VP.RdWa, SC.RegistersReadWriteChecking)[1]
        inc = acc.cp(CP.RdInc, SC.RegistersReadWriteChecking)[1]
        eq_eval = EqPolynomial.mle(r_cycle, self.r_cycle.r)
        return eq_eval * (rd_wa * (inc + val) + self.gamma * (rs1_ra * val + self.gamma * rs2_ra * val))

    def cache_openings(self, acc, transcript, r):
        pt = self._normalize_opening_point(r)
        for vp in [VP.RegistersVal, VP.Rs1Ra, VP.Rs2Ra, VP.RdWa]:
            acc.append_virtual(transcript, vp, SC.RegistersReadWriteChecking, pt.clone())
        acc.append_dense(transcript, CP.RdInc, SC.RegistersReadWriteChecking, pt.r[self.LOG_K:])

def _advice_num_vars(max_size_bytes):  # Shared helper: compute advice num_vars from max byte size.
    n_words = max(1, int(max_size_bytes) // 8)
    pow2 = 1 << (n_words - 1).bit_length()
    return (pow2.bit_length() - 1) if pow2 > 1 else 0

class RamValEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 4: RAM val evaluation.
    DEGREE_BOUND = 3

    def __init__(self, ram_preprocessing, program_io, trace_len, ram_K, acc):
        self.T, self.K = int(trace_len), int(ram_K)
        r_full = acc.vp(VP.RamVal, SC.RamReadWriteChecking)[0]
        logK = log2_pow2(self.K)
        self.r_address = OpeningPoint(r_full.r[:logK], BIG_ENDIAN)
        self.r_cycle = OpeningPoint(r_full.r[logK:], BIG_ENDIAN)
        n_mem = logK
        ml = program_io.memory_layout
        untrusted = calculate_advice_memory_evaluation(acc.adv(AdviceKind.Untrusted, SC.RamValEvaluation),
            _advice_num_vars(ml.max_untrusted_advice_size), ml.untrusted_advice_start, ml, self.r_address.r, n_mem)
        trusted = calculate_advice_memory_evaluation(acc.adv(AdviceKind.Trusted, SC.RamValEvaluation),
            _advice_num_vars(ml.max_trusted_advice_size), ml.trusted_advice_start, ml, self.r_address.r, n_mem)
        self.init_eval = untrusted + trusted + eval_initial_ram_mle(ram_preprocessing, program_io, self.r_address.r)

    def num_rounds(self):
        return log2_pow2(self.T)

    def input_claim(self, acc):
        return acc.vp(VP.RamVal, SC.RamReadWriteChecking)[1] - self.init_eval

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        lt_eval, eq_term = Fr.zero(), Fr.one()
        for x, y in zip(pt.r, self.r_cycle.r):
            lt_eval += (Fr.one() - x) * y * eq_term
            eq_term *= Fr.one() - x - y + x * y + x * y
        inc = acc.cp(CP.RamInc, SC.RamValEvaluation)[1]
        wa = acc.vp(VP.RamRa, SC.RamValEvaluation)[1]
        return inc * wa * lt_eval

    def cache_openings(self, acc, transcript, r):
        r_cycle_prime = _normalize_le_to_be(r)
        r_full = acc.vp(VP.RamVal, SC.RamReadWriteChecking)[0]
        r_addr = r_full.r[:len(r_full.r) - len(r_cycle_prime.r)]
        acc.append_virtual(transcript, VP.RamRa, SC.RamValEvaluation, OpeningPoint(list(r_addr) + list(r_cycle_prime.r), BIG_ENDIAN))
        acc.append_dense(transcript, CP.RamInc, SC.RamValEvaluation, r_cycle_prime.r)

class ValFinalSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 4: RAM val final.
    DEGREE_BOUND = 2

    def __init__(self, ram_preprocessing, program_io, trace_len, ram_K, acc, rw_config):
        self.T, self.K = int(trace_len), int(ram_K)
        self.r_address = acc.vp(VP.RamValFinal, SC.RamOutputCheck)[0]
        n_mem = log2_pow2(self.K)
        log_T = log2_pow2(self.T)
        adv_sc = SC.RamValEvaluation if rw_config.needs_single_advice_opening(log_T) else SC.RamValFinalEvaluation
        ml = program_io.memory_layout
        untrusted = calculate_advice_memory_evaluation(acc.adv(AdviceKind.Untrusted, adv_sc),
            _advice_num_vars(ml.max_untrusted_advice_size), ml.untrusted_advice_start, ml, self.r_address.r, n_mem)
        trusted = calculate_advice_memory_evaluation(acc.adv(AdviceKind.Trusted, adv_sc),
            _advice_num_vars(ml.max_trusted_advice_size), ml.trusted_advice_start, ml, self.r_address.r, n_mem)
        self.val_init_eval = untrusted + trusted + eval_initial_ram_mle(ram_preprocessing, program_io, self.r_address.r)

    def num_rounds(self):
        return log2_pow2(self.T)

    def input_claim(self, acc):
        return acc.vp(VP.RamValFinal, SC.RamOutputCheck)[1] - self.val_init_eval

    def expected_output_claim(self, acc, _r):
        inc = acc.cp(CP.RamInc, SC.RamValFinalEvaluation)[1]
        wa = acc.vp(VP.RamRa, SC.RamValFinalEvaluation)[1]
        return inc * wa

    def cache_openings(self, acc, transcript, r):
        r_cycle_prime = _normalize_le_to_be(r)
        acc.append_dense(transcript, CP.RamInc, SC.RamValFinalEvaluation, r_cycle_prime.r)
        acc.append_virtual(transcript, VP.RamRa, SC.RamValFinalEvaluation, OpeningPoint(list(self.r_address.r) + list(r_cycle_prime.r), BIG_ENDIAN))
