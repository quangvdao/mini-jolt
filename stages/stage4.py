"""Stage 4: Registers read/write checking + RAM val evaluation + RAM val final."""
from openings import AdviceKind, BIG_ENDIAN, CommittedPolynomial, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial, log2_pow2
from ram_io import calculate_advice_memory_evaluation, eval_initial_ram_mle
from rv64imac.constants import REGISTER_COUNT
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be


class RegistersReadWriteCheckingVerifier(SumcheckInstanceVerifier):  # Stage 4: registers read/write checking.
    DEGREE_BOUND = 3
    LOG_K = int(REGISTER_COUNT).bit_length() - 1  # Rust: log2(REGISTER_COUNT)=7 (128 regs)

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
