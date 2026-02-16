"""Stage 3: Shift + instruction input virtualization + registers claim reduction."""
from field import Fr
from openings import BIG_ENDIAN, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial, EqPlusOnePolynomial, log2_pow2
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be, _cache_virtual_batch


class ShiftSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 3: Spartan shift sumcheck.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, acc, transcript):
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.gamma_powers = list(transcript.challenge_scalar_powers(5))
        self.r_outer = acc.vp(VP.NextPC, SC.SpartanOuter)[0]
        self.r_product = acc.vp(VP.NextIsNoop, SC.SpartanProductVirtualization)[0]

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, acc):
        next_pc = acc.vp(VP.NextPC, SC.SpartanOuter)[1]
        next_unexpanded_pc = acc.vp(VP.NextUnexpandedPC, SC.SpartanOuter)[1]
        next_is_virtual = acc.vp(VP.NextIsVirtual, SC.SpartanOuter)[1]
        next_is_first = acc.vp(VP.NextIsFirstInSequence, SC.SpartanOuter)[1]
        next_is_noop = acc.vp(VP.NextIsNoop, SC.SpartanProductVirtualization)[1]
        return (next_unexpanded_pc + next_pc * self.gamma_powers[1] + next_is_virtual * self.gamma_powers[2]
                + next_is_first * self.gamma_powers[3] + (Fr.one() - next_is_noop) * self.gamma_powers[4])

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        unexpanded_pc = acc.vp(VP.UnexpandedPC, SC.SpartanShift)[1]
        pc = acc.vp(VP.PC, SC.SpartanShift)[1]
        is_virtual = acc.vp(VP.OpFlags_VirtualInstruction, SC.SpartanShift)[1]
        is_first = acc.vp(VP.OpFlags_IsFirstInSequence, SC.SpartanShift)[1]
        is_noop = acc.vp(VP.InstructionFlags_IsNoop, SC.SpartanShift)[1]
        eq1 = EqPlusOnePolynomial(list(self.r_outer.r)).evaluate(pt.r)
        eq2 = EqPlusOnePolynomial(list(self.r_product.r)).evaluate(pt.r)
        gp = self.gamma_powers
        left = gp[0] * unexpanded_pc + gp[1] * pc + gp[2] * is_virtual + gp[3] * is_first
        return left * eq1 + gp[4] * (Fr.one() - is_noop) * eq2

    def cache_openings(self, acc, transcript, r):
        pt = _normalize_le_to_be(r)
        _cache_virtual_batch(acc, transcript,
            [VP.UnexpandedPC, VP.PC, VP.OpFlags_VirtualInstruction, VP.OpFlags_IsFirstInSequence, VP.InstructionFlags_IsNoop],
            SC.SpartanShift, pt)

class InstructionInputSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 3: instruction input virtualization.
    DEGREE_BOUND = 3

    def __init__(self, acc, transcript):
        self.r_cycle_1 = acc.vp(VP.LeftInstructionInput, SC.SpartanOuter)[0]
        self.r_cycle_2 = acc.vp(VP.LeftInstructionInput, SC.SpartanProductVirtualization)[0]
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma

    def num_rounds(self):
        return len(self.r_cycle_1)

    def input_claim(self, acc):
        left1 = acc.vp(VP.LeftInstructionInput, SC.SpartanOuter)[1]
        right1 = acc.vp(VP.RightInstructionInput, SC.SpartanOuter)[1]
        left2 = acc.vp(VP.LeftInstructionInput, SC.SpartanProductVirtualization)[1]
        right2 = acc.vp(VP.RightInstructionInput, SC.SpartanProductVirtualization)[1]
        return (right1 + self.gamma * left1) + self.gamma_sqr * (right2 + self.gamma * left2)

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        eq1 = EqPolynomial.mle(pt.r, self.r_cycle_1.r)
        eq2 = EqPolynomial.mle(pt.r, self.r_cycle_2.r)
        IIV = SC.InstructionInputVirtualization
        rs1 = acc.vp(VP.Rs1Value, IIV)[1]
        left_is_rs1 = acc.vp(VP.InstructionFlags_LeftOperandIsRs1Value, IIV)[1]
        unexpanded_pc = acc.vp(VP.UnexpandedPC, IIV)[1]
        left_is_pc = acc.vp(VP.InstructionFlags_LeftOperandIsPC, IIV)[1]
        rs2 = acc.vp(VP.Rs2Value, IIV)[1]
        right_is_rs2 = acc.vp(VP.InstructionFlags_RightOperandIsRs2Value, IIV)[1]
        imm = acc.vp(VP.Imm, IIV)[1]
        right_is_imm = acc.vp(VP.InstructionFlags_RightOperandIsImm, IIV)[1]
        left_input = left_is_rs1 * rs1 + left_is_pc * unexpanded_pc
        right_input = right_is_rs2 * rs2 + right_is_imm * imm
        return (eq1 + self.gamma_sqr * eq2) * (right_input + self.gamma * left_input)

    def cache_openings(self, acc, transcript, r):
        pt = _normalize_le_to_be(r)
        _cache_virtual_batch(acc, transcript,
            [VP.InstructionFlags_LeftOperandIsRs1Value, VP.Rs1Value, VP.InstructionFlags_LeftOperandIsPC, VP.UnexpandedPC,
             VP.InstructionFlags_RightOperandIsRs2Value, VP.Rs2Value, VP.InstructionFlags_RightOperandIsImm, VP.Imm],
            SC.InstructionInputVirtualization, pt)

class RegistersClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 3: registers claim reduction.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, acc, transcript):
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.n_cycle_vars = log2_pow2(int(trace_len))
        self.r_spartan = acc.vp(VP.LookupOutput, SC.SpartanOuter)[0]

    def num_rounds(self):
        return self.n_cycle_vars

    def input_claim(self, acc):
        rd = acc.vp(VP.RdWriteValue, SC.SpartanOuter)[1]
        rs1 = acc.vp(VP.Rs1Value, SC.SpartanOuter)[1]
        rs2 = acc.vp(VP.Rs2Value, SC.SpartanOuter)[1]
        return rd + self.gamma * rs1 + self.gamma_sqr * rs2

    def expected_output_claim(self, acc, r):
        pt = _normalize_le_to_be(r)
        rd = acc.vp(VP.RdWriteValue, SC.RegistersClaimReduction)[1]
        rs1 = acc.vp(VP.Rs1Value, SC.RegistersClaimReduction)[1]
        rs2 = acc.vp(VP.Rs2Value, SC.RegistersClaimReduction)[1]
        return EqPolynomial.mle(pt.r, self.r_spartan.r) * (rd + self.gamma * rs1 + self.gamma_sqr * rs2)

    def cache_openings(self, acc, transcript, r):
        pt = _normalize_le_to_be(r)
        _cache_virtual_batch(acc, transcript, [VP.RdWriteValue, VP.Rs1Value, VP.Rs2Value], SC.RegistersClaimReduction, pt)
