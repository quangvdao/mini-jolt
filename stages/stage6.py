"""Stage 6: BytecodeReadRaf + Booleanity + HammingBooleanity + RA virtualization + reductions."""
from field import Fr
from ids_generated import LOOKUP_TABLES_64
from openings import AdviceKind, BIG_ENDIAN, LITTLE_ENDIAN, CommittedPolynomial as CP, OpeningPoint, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial, IdentityPolynomial, log2_pow2
from rv64imac.constants import REGISTER_COUNT
from sumchecks import SumcheckInstanceVerifier, SumcheckVerifyError, _normalize_le_to_be


class HammingBooleanitySumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: RAM Hamming booleanity.
    DEGREE_BOUND = 3

    def __init__(self, acc):
        self.r_cycle = acc.vp(VP.LookupOutput, SC.SpartanOuter)[0]

    def num_rounds(self):
        return len(self.r_cycle.r)

    def input_claim(self, _acc):
        return Fr.zero()

    def expected_output_claim(self, acc, r):
        H = acc.vp(VP.RamHammingWeight, SC.RamHammingBooleanity)[1]
        eq = EqPolynomial.mle(r, list(reversed(self.r_cycle.r)))
        return (H * H - H) * eq

    def cache_openings(self, acc, transcript, r):
        acc.append_virtual(transcript, VP.RamHammingWeight, SC.RamHammingBooleanity, _normalize_le_to_be(r))


class RamRaVirtualSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: RAM RA virtualization.
    def __init__(self, trace_len, one_hot_params, acc, transcript):
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.log_T = log2_pow2(int(trace_len))
        r_reduced = acc.vp(VP.RamRa, SC.RamRaClaimReduction)[0]
        self.r_cycle_reduced = OpeningPoint(list(r_reduced.r[self.log_K:]), BIG_ENDIAN)
        self.r_address_chunks = one_hot_params.compute_r_address_chunks(list(r_reduced.r[:self.log_K]))
        self.d = int(one_hot_params.ram_d)

    def degree(self):
        return self.d + 1

    def num_rounds(self):
        return self.log_T

    def input_claim(self, acc):
        return acc.vp(VP.RamRa, SC.RamRaClaimReduction)[1]

    def expected_output_claim(self, acc, r):
        r_cycle_final = _normalize_le_to_be(r)
        eq_eval = EqPolynomial.mle(self.r_cycle_reduced.r, r_cycle_final.r)
        ra_prod = None
        for i in range(self.d):
            c = acc.cpi(CP.RamRa, i, SC.RamRaVirtualization)[1]
            ra_prod = c if ra_prod is None else (ra_prod * c)
        return eq_eval * ra_prod

    def cache_openings(self, acc, transcript, r):
        r_cycle_final = _normalize_le_to_be(r)
        for i in range(self.d):
            acc.append_dense_i(transcript, CP.RamRa, i, SC.RamRaVirtualization, list(self.r_address_chunks[i]) + list(r_cycle_final.r))


class InstructionRaVirtualSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: instruction RA virtualization.
    def __init__(self, one_hot_params, acc, transcript):
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)
        self.log_k_chunk = int(one_hot_params.log_k_chunk)
        self.n_committed_per_virtual = self.ra_virtual_log_k_chunk // self.log_k_chunk
        self.n_virtual_ra_polys = 128 // self.ra_virtual_log_k_chunk
        self.n_committed_ra_polys = 128 // self.log_k_chunk
        r_address = []
        for i in range(self.n_virtual_ra_polys):
            r, _ = acc.vpi(VP.InstructionRa, i, SC.InstructionReadRaf)
            r_address.extend(r.r[:self.ra_virtual_log_k_chunk])
        r0, _ = acc.vpi(VP.InstructionRa, 0, SC.InstructionReadRaf)
        self.r_cycle = OpeningPoint(list(r0.r[self.ra_virtual_log_k_chunk:]), BIG_ENDIAN)
        self.r_address = OpeningPoint(list(r_address), BIG_ENDIAN)
        self.gamma_powers = list(transcript.challenge_scalar_powers(self.n_virtual_ra_polys))
        self.one_hot_params = one_hot_params

    def degree(self):
        return self.n_committed_per_virtual + 1

    def num_rounds(self):
        return len(self.r_cycle.r)

    def input_claim(self, acc):
        out = None
        for i in range(self.n_virtual_ra_polys):
            term = self.gamma_powers[i] * acc.vpi(VP.InstructionRa, i, SC.InstructionReadRaf)[1]
            out = term if out is None else (out + term)
        return out

    def expected_output_claim(self, acc, r):
        r_cycle_final = _normalize_le_to_be(r)
        eq_eval = EqPolynomial.mle(self.r_cycle.r, r_cycle_final.r)
        committed = [acc.cpi(CP.InstructionRa, i, SC.InstructionRaVirtualization)[1] for i in range(self.n_committed_ra_polys)]
        ra_acc, idx = None, 0
        for i in range(self.n_virtual_ra_polys):
            prod = None
            for _ in range(self.n_committed_per_virtual):
                c = committed[idx]; idx += 1
                prod = c if prod is None else (prod * c)
            term = self.gamma_powers[i] * prod
            ra_acc = term if ra_acc is None else (ra_acc + term)
        return eq_eval * ra_acc

    def cache_openings(self, acc, transcript, r):
        r_cycle_final = _normalize_le_to_be(r)
        r_address_chunks = self.one_hot_params.compute_r_address_chunks(self.r_address.r)
        for i, chunk in enumerate(r_address_chunks):
            acc.append_dense_i(transcript, CP.InstructionRa, i, SC.InstructionRaVirtualization, list(chunk) + list(r_cycle_final.r))


class IncClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: reduce RamInc/RdInc multi-openings.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, acc, transcript):
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.gamma_cub = self.gamma_sqr * self.gamma
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.r_cycle_s2 = acc.cp(CP.RamInc, SC.RamReadWriteChecking)[0]
        self.r_cycle_s4 = acc.cp(CP.RamInc, SC.RamValEvaluation)[0]
        self.s_cycle_s4 = acc.cp(CP.RdInc, SC.RegistersReadWriteChecking)[0]
        self.s_cycle_s5 = acc.cp(CP.RdInc, SC.RegistersValEvaluation)[0]

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, acc):
        v1 = acc.cp(CP.RamInc, SC.RamReadWriteChecking)[1]
        v2 = acc.cp(CP.RamInc, SC.RamValEvaluation)[1]
        w1 = acc.cp(CP.RdInc, SC.RegistersReadWriteChecking)[1]
        w2 = acc.cp(CP.RdInc, SC.RegistersValEvaluation)[1]
        return v1 + self.gamma * v2 + self.gamma_sqr * w1 + self.gamma_cub * w2

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        eq_r2 = EqPolynomial.mle(pt.r, self.r_cycle_s2.r)
        eq_r4 = EqPolynomial.mle(pt.r, self.r_cycle_s4.r)
        eq_s4 = EqPolynomial.mle(pt.r, self.s_cycle_s4.r)
        eq_s5 = EqPolynomial.mle(pt.r, self.s_cycle_s5.r)
        ram_inc = acc.cp(CP.RamInc, SC.IncClaimReduction)[1]
        rd_inc = acc.cp(CP.RdInc, SC.IncClaimReduction)[1]
        return ram_inc * (eq_r2 + self.gamma * eq_r4) + self.gamma_sqr * rd_inc * (eq_s4 + self.gamma * eq_s5)

    def cache_openings(self, acc, transcript, r):
        pt = _normalize_le_to_be(r)
        acc.append_dense(transcript, CP.RamInc, SC.IncClaimReduction, pt.r)
        acc.append_dense(transcript, CP.RdInc, SC.IncClaimReduction, pt.r)


class AdviceClaimReductionVerifier(SumcheckInstanceVerifier):  # Stage 6/7: two-phase advice claim reduction.
    DEGREE_BOUND = 2
    PHASE_CYCLE = "cycle"
    PHASE_ADDRESS = "address"

    def __init__(self, kind, memory_layout, trace_len, log_k_chunk, acc, transcript, single_opening):
        self.kind = AdviceKind(kind) if not isinstance(kind, AdviceKind) else kind
        self.phase = self.PHASE_CYCLE
        self.single_opening = bool(single_opening)
        self.log_t = log2_pow2(int(trace_len))
        self.log_k_chunk = int(log_k_chunk)
        self.r_val_eval = acc.adv(self.kind, SC.RamValEvaluation)[0]
        self.r_val_final = None if self.single_opening else acc.adv(self.kind, SC.RamValFinalEvaluation)[0]
        self.gamma = transcript.challenge_scalar()
        max_bytes = int(memory_layout.max_trusted_advice_size if self.kind == AdviceKind.Trusted else memory_layout.max_untrusted_advice_size)
        self.advice_col_vars, self.advice_row_vars = self._advice_sigma_nu(max_bytes)
        self.main_col_vars, self.main_row_vars = self._balanced_sigma_nu(self.log_k_chunk + self.log_t)
        self.cycle_phase_col_rounds, self.cycle_phase_row_rounds = self._cycle_phase_rounds(
            self.log_t, self.main_col_vars, self.advice_row_vars, self.advice_col_vars)
        self.cycle_var_challenges = []
        self._two_inv = Fr(2).inv()

    @staticmethod
    def _balanced_sigma_nu(total_vars):
        total_vars = int(total_vars)
        sigma = (total_vars + 1) // 2
        return sigma, total_vars - sigma

    @classmethod
    def _advice_sigma_nu(cls, max_bytes):
        words = max(1, int(max_bytes) // 8)
        pow2 = 1 << ((words - 1).bit_length())
        advice_vars = log2_pow2(pow2) if pow2 > 1 else 0
        return cls._balanced_sigma_nu(advice_vars)

    @staticmethod
    def _cycle_phase_rounds(log_t, main_col_vars, advice_row_vars, advice_col_vars):
        col_rounds = range(0, min(int(log_t), int(advice_col_vars)))
        row_start = min(int(log_t), int(main_col_vars))
        row_end = min(int(log_t), int(main_col_vars) + int(advice_row_vars))
        return col_rounds, range(row_start, row_end)

    def num_address_phase_rounds(self):
        return (self.advice_col_vars + self.advice_row_vars) - (len(self.cycle_phase_col_rounds) + len(self.cycle_phase_row_rounds))

    def num_rounds(self):
        if self.phase == self.PHASE_CYCLE:
            if len(self.cycle_phase_row_rounds) != 0:
                return int(self.cycle_phase_row_rounds.stop) - int(self.cycle_phase_col_rounds.start)
            return len(self.cycle_phase_col_rounds)
        return self.num_address_phase_rounds()

    def round_offset(self, max_num_rounds):
        if self.phase == self.PHASE_CYCLE:
            return int(max_num_rounds) - (self.log_k_chunk + self.log_t) + self.log_k_chunk
        return 0

    def input_claim(self, acc):
        claim = acc.adv(self.kind, SC.RamValEvaluation)
        out = Fr.zero() if claim is None else claim[1]
        if not self.single_opening:
            final = acc.adv(self.kind, SC.RamValFinalEvaluation)
            if final is not None:
                out += self.gamma * final[1]
        if self.phase == self.PHASE_ADDRESS:
            mid = acc.adv(self.kind, SC.AdviceClaimReductionCyclePhase)
            if mid is None:
                raise SumcheckVerifyError("Cycle phase intermediate claim not found")
            out = mid[1]
        return out

    def normalize_opening_point(self, challenges):
        if self.phase == self.PHASE_CYCLE:
            out = [challenges[i] for i in self.cycle_phase_col_rounds] + [challenges[i] for i in self.cycle_phase_row_rounds]
            return OpeningPoint(out, LITTLE_ENDIAN).match_endianness(BIG_ENDIAN)
        return OpeningPoint(list(self.cycle_var_challenges) + list(challenges), LITTLE_ENDIAN).match_endianness(BIG_ENDIAN)

    def expected_output_claim(self, acc, r):
        if self.phase == self.PHASE_CYCLE:
            mid = acc.adv(self.kind, SC.AdviceClaimReductionCyclePhase)
            if mid is None:
                raise SumcheckVerifyError("Cycle phase intermediate claim not found")
            return mid[1]
        pt = self.normalize_opening_point(r)
        advice = acc.adv(self.kind, SC.AdviceClaimReduction)
        if advice is None:
            raise SumcheckVerifyError("Final advice claim not found")
        eq_combined = EqPolynomial.mle(pt.r, self.r_val_eval.r)
        if not self.single_opening:
            eq_combined = eq_combined + self.gamma * EqPolynomial.mle(pt.r, self.r_val_final.r)
        cr, rr = self.cycle_phase_col_rounds, self.cycle_phase_row_rounds
        gap_len = 0 if (len(rr) == 0 or len(cr) == 0) else int(rr.start) - int(cr.stop)
        scale = Fr.one()
        for _ in range(gap_len):
            scale *= self._two_inv
        return advice[1] * eq_combined * scale

    def cache_openings(self, acc, transcript, r):
        pt = self.normalize_opening_point(r)
        if self.phase == self.PHASE_CYCLE:
            if self.kind == AdviceKind.Trusted:
                acc.append_trusted_advice(transcript, SC.AdviceClaimReductionCyclePhase, pt.clone())
            else:
                acc.append_untrusted_advice(transcript, SC.AdviceClaimReductionCyclePhase, pt.clone())
            self.cycle_var_challenges = pt.match_endianness(LITTLE_ENDIAN).r
        if self.num_address_phase_rounds() == 0 or self.phase == self.PHASE_ADDRESS:
            if self.kind == AdviceKind.Trusted:
                acc.append_trusted_advice(transcript, SC.AdviceClaimReduction, pt)
            else:
                acc.append_untrusted_advice(transcript, SC.AdviceClaimReduction, pt)


class BooleanitySumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: booleanity for all committed RA families.
    DEGREE_BOUND = 3

    def __init__(self, log_t, one_hot_params, acc, transcript):
        self.log_t = int(log_t)
        self.log_k_chunk = int(one_hot_params.log_k_chunk)
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)
        stage5_point = acc.vpi(VP.InstructionRa, 0, SC.InstructionReadRaf)[0]
        stage5_addr = list(reversed(stage5_point.r[:self.ra_virtual_log_k_chunk]))  # BE->LE
        r_cycle = list(reversed(stage5_point.r[self.ra_virtual_log_k_chunk:]))  # BE->LE
        if len(stage5_addr) >= self.log_k_chunk:
            r_address = stage5_addr[len(stage5_addr) - self.log_k_chunk:]
        else:
            r_address = stage5_addr + list(transcript.challenge_vector_optimized(self.log_k_chunk - len(stage5_addr)))
        self.r_address, self.r_cycle = list(r_address), list(r_cycle)  # LE
        total_d = int(one_hot_params.instruction_d) + int(one_hot_params.bytecode_d) + int(one_hot_params.ram_d)
        self.poly_types = ([(CP.InstructionRa, i) for i in range(int(one_hot_params.instruction_d))]
            + [(CP.BytecodeRa, i) for i in range(int(one_hot_params.bytecode_d))]
            + [(CP.RamRa, i) for i in range(int(one_hot_params.ram_d))])
        gamma = transcript.challenge_scalar_optimized()
        if gamma == Fr.zero():
            gamma = Fr.one()
        g, self.gamma_powers_square = Fr.one(), []
        for _ in range(total_d):
            self.gamma_powers_square.append(g)
            g *= gamma * gamma

    def num_rounds(self):
        return self.log_k_chunk + self.log_t

    def input_claim(self, _acc):
        return Fr.zero()

    def _normalize_opening_point(self, r):
        out = list(r)
        out[:self.log_k_chunk] = list(reversed(out[:self.log_k_chunk]))
        out[self.log_k_chunk:] = list(reversed(out[self.log_k_chunk:]))
        return OpeningPoint(out, BIG_ENDIAN)

    def expected_output_claim(self, acc, r):
        ra_claims = [acc.cpi(poly, i, SC.Booleanity)[1] for poly, i in self.poly_types]
        eq = EqPolynomial.mle(r, list(reversed(self.r_address)) + list(reversed(self.r_cycle)))
        out = None
        for g2i, ra in zip(self.gamma_powers_square, ra_claims):
            term = (ra * ra - ra) * g2i
            out = term if out is None else (out + term)
        return eq * out

    def cache_openings(self, acc, transcript, r):
        pt = self._normalize_opening_point(r)
        for poly, i in self.poly_types:
            acc.append_dense_i(transcript, poly, i, SC.Booleanity, pt.r)


class BytecodeReadRafSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 6: bytecode read+RAF checking.
    def __init__(self, bytecode_preprocessing, n_cycle_vars, one_hot_params, acc, transcript):
        from rv64imac.isa import CIRCUIT_FLAGS, INSTRUCTION_FLAGS
        self.bytecode = list(bytecode_preprocessing.bytecode)
        self.one_hot_params = one_hot_params
        self.K = int(one_hot_params.bytecode_k)
        self.log_K = log2_pow2(self.K)
        self.log_T = int(n_cycle_vars)
        self.d = int(one_hot_params.bytecode_d)
        self.log_reg_K = int(REGISTER_COUNT).bit_length() - 1

        self.gamma_powers = list(transcript.challenge_scalar_powers(7))
        s1g = list(transcript.challenge_scalar_powers(2 + len(CIRCUIT_FLAGS)))
        s2g = list(transcript.challenge_scalar_powers(5))
        s3g = list(transcript.challenge_scalar_powers(9))
        s4g = list(transcript.challenge_scalar_powers(3))
        s5g = list(transcript.challenge_scalar_powers(2 + len(LOOKUP_TABLES_64)))

        def v(name, sc):
            return acc.vp(name, sc)[1]

        rv1_terms = [v(VP.UnexpandedPC, SC.SpartanOuter), v(VP.Imm, SC.SpartanOuter)]
        for flag in CIRCUIT_FLAGS:
            rv1_terms.append(v(getattr(VP, f"OpFlags_{flag}"), SC.SpartanOuter))
        rv1 = sum((c * g for c, g in zip(rv1_terms, s1g)), Fr.zero())

        rv2_terms = [v(VP.OpFlags_Jump, SC.SpartanProductVirtualization), v(VP.InstructionFlags_Branch, SC.SpartanProductVirtualization),
            v(VP.InstructionFlags_IsRdNotZero, SC.SpartanProductVirtualization), v(VP.OpFlags_WriteLookupOutputToRD, SC.SpartanProductVirtualization),
            v(VP.OpFlags_VirtualInstruction, SC.SpartanProductVirtualization)]
        rv2 = sum((c * g for c, g in zip(rv2_terms, s2g)), Fr.zero())

        imm_claim = v(VP.Imm, SC.InstructionInputVirtualization)
        upc_shift = v(VP.UnexpandedPC, SC.SpartanShift)
        upc_instr = v(VP.UnexpandedPC, SC.InstructionInputVirtualization)
        if upc_shift != upc_instr:
            raise SumcheckVerifyError("UnexpandedPC claim mismatch across stages")
        IIV = SC.InstructionInputVirtualization
        rv3_terms = [imm_claim, upc_shift,
            v(VP.InstructionFlags_LeftOperandIsRs1Value, IIV), v(VP.InstructionFlags_LeftOperandIsPC, IIV),
            v(VP.InstructionFlags_RightOperandIsRs2Value, IIV), v(VP.InstructionFlags_RightOperandIsImm, IIV),
            v(VP.InstructionFlags_IsNoop, SC.SpartanShift), v(VP.OpFlags_VirtualInstruction, SC.SpartanShift),
            v(VP.OpFlags_IsFirstInSequence, SC.SpartanShift)]
        rv3 = sum((c * g for c, g in zip(rv3_terms, s3g)), Fr.zero())

        rv4 = sum((c * g for c, g in zip(
            [v(VP.RdWa, SC.RegistersReadWriteChecking), v(VP.Rs1Ra, SC.RegistersReadWriteChecking), v(VP.Rs2Ra, SC.RegistersReadWriteChecking)], s4g)), Fr.zero())

        rv5 = v(VP.RdWa, SC.RegistersValEvaluation) * s5g[0] + v(VP.InstructionRafFlag, SC.InstructionReadRaf) * s5g[1]
        for i in range(len(LOOKUP_TABLES_64)):
            rv5 += acc.vpi(VP.LookupTableFlag, i, SC.InstructionReadRaf)[1] * s5g[2 + i]

        raf_claim = v(VP.PC, SC.SpartanOuter)
        raf_shift_claim = v(VP.PC, SC.SpartanShift)

        self.input_claim_cached = sum((c * g for c, g in zip(
            [rv1, rv2, rv3, rv4, rv5, raf_claim, raf_shift_claim], self.gamma_powers)), Fr.zero())
        self.int_poly = IdentityPolynomial(self.log_K)
        self.stage_gammas = [s1g, s2g, s3g, s4g, s5g]

        RRW = SC.RegistersReadWriteChecking
        RVE = SC.RegistersValEvaluation
        self.r_cycles = [
            acc.vp(VP.Imm, SC.SpartanOuter)[0].r,
            acc.vp(VP.OpFlags_Jump, SC.SpartanProductVirtualization)[0].r,
            acc.vp(VP.UnexpandedPC, SC.SpartanShift)[0].r,
            acc.vp(VP.Rs1Ra, RRW)[0].r[self.log_reg_K:],
            acc.vp(VP.RdWa, RVE)[0].r[self.log_reg_K:],
        ]
        self.eq_r_register_4 = EqPolynomial.evals(acc.vp(VP.RdWa, RRW)[0].r[:self.log_reg_K])
        self.eq_r_register_5 = EqPolynomial.evals(acc.vp(VP.RdWa, RVE)[0].r[:self.log_reg_K])

    def degree(self):
        return self.d + 1

    def num_rounds(self):
        return self.log_K + self.log_T

    def input_claim(self, _acc):
        return self.input_claim_cached

    def _normalize_opening_point(self, r):
        out = list(r)
        out[:self.log_K] = list(reversed(out[:self.log_K]))
        out[self.log_K:] = list(reversed(out[self.log_K:]))
        return OpeningPoint(out, BIG_ENDIAN)

    def _val_eval_stage(self, stage_idx, eq_r_addr):
        from rv64imac.isa import circuit_flags, instruction_flags, lookup_table
        from rv64imac.types import Xlen
        gammas = self.stage_gammas[stage_idx]
        acc = Fr.zero()
        for k, inst in enumerate(self.bytecode):
            inst = inst.normalize()
            cf, inf = circuit_flags(inst), instruction_flags(inst)
            v = Fr.zero()
            if stage_idx == 0:
                v = Fr(int(inst.address)) + Fr(int(inst.operands.imm)) * gammas[1]
                for flag_val, gp in zip(cf, gammas[2:]):
                    if flag_val: v += gp
            elif stage_idx == 1:
                if cf[5]: v += gammas[0]
                if inf[4]: v += gammas[1]
                if inf[6]: v += gammas[2]
                if cf[6]: v += gammas[3]
                if cf[7]: v += gammas[4]
            elif stage_idx == 2:
                v = Fr(int(inst.operands.imm)) + gammas[1] * Fr(int(inst.address))
                for flag_val, gi in [(inf[2], 2), (inf[0], 3), (inf[3], 4), (inf[1], 5), (inf[5], 6), (cf[7], 7), (cf[12], 8)]:
                    if flag_val: v += gammas[gi]
            elif stage_idx == 3:
                rd, rs1, rs2 = inst.operands.rd, inst.operands.rs1, inst.operands.rs2
                v = ((self.eq_r_register_4[int(rd)] if rd is not None else Fr.zero()) * gammas[0]
                    + (self.eq_r_register_4[int(rs1)] if rs1 is not None else Fr.zero()) * gammas[1]
                    + (self.eq_r_register_4[int(rs2)] if rs2 is not None else Fr.zero()) * gammas[2])
            elif stage_idx == 4:
                rd = inst.operands.rd
                v = (self.eq_r_register_5[int(rd)] if rd is not None else Fr.zero()) * gammas[0]
                if not ((not cf[0]) and (not cf[1]) and (not cf[2]) and (not cf[10])):
                    v += gammas[1]
                t = lookup_table(inst, Xlen.Bit64)
                if t is not None:
                    v += gammas[2 + LOOKUP_TABLES_64.index(t)]
            acc += v * eq_r_addr[k]
        return acc

    def expected_output_claim(self, acc, r):
        pt = self._normalize_opening_point(r)
        r_addr, r_cycle = pt.r[:self.log_K], pt.r[self.log_K:]
        int_eval = self.int_poly.evaluate(r_addr)
        ra_prod = None
        for i in range(self.d):
            c = acc.cpi(CP.BytecodeRa, i, SC.BytecodeReadRaf)[1]
            ra_prod = c if ra_prod is None else (ra_prod * c)
        eq_r_addr = EqPolynomial.evals(r_addr)
        val = Fr.zero()
        for stage_idx in range(5):
            inj = Fr.zero()
            if stage_idx == 0: inj = int_eval * self.gamma_powers[5]
            if stage_idx == 2: inj = int_eval * self.gamma_powers[4]
            eq_cycle = EqPolynomial.mle(self.r_cycles[stage_idx], r_cycle)
            val += (self._val_eval_stage(stage_idx, eq_r_addr) + inj) * eq_cycle * self.gamma_powers[stage_idx]
        return ra_prod * val

    def cache_openings(self, acc, transcript, r):
        pt = self._normalize_opening_point(r)
        r_addr, r_cycle = pt.r[:self.log_K], pt.r[self.log_K:]
        chunks = self.one_hot_params.compute_r_address_chunks(r_addr)
        for i in range(self.d):
            acc.append_dense_i(transcript, CP.BytecodeRa, i, SC.BytecodeReadRaf, list(chunks[i]) + list(r_cycle))
