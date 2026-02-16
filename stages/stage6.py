"""Stage 6: BytecodeReadRaf + Booleanity + HammingBooleanity + RA virtualization + reductions."""
from ids_generated import LOOKUP_TABLES_64
from openings import AdviceKind, BIG_ENDIAN, LITTLE_ENDIAN, CommittedPolynomial, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial, IdentityPolynomial, log2_pow2
from rv64imac.constants import REGISTER_COUNT
from sumchecks import SumcheckInstanceVerifier, SumcheckVerifyError, _normalize_le_to_be


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
            pass
        return OpeningPoint(out, LITTLE_ENDIAN).match_endianness(BIG_ENDIAN)

    def normalize_opening_point(self, challenges):  # Public normalize to BE for cache/eq.
        if self.phase == self.PHASE_CYCLE:
            return self._normalize_opening_point_cycle(list(challenges))
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

    def _val_eval_stage(self, stage_idx, eq_r_addr):  # Evaluate stage Val(k) MLE at r_address.
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
