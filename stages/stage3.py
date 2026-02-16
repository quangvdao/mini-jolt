"""Stage 3: Shift + instruction input virtualization + registers claim reduction."""
from openings import BIG_ENDIAN, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial, EqPlusOnePolynomial, log2_pow2
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be, _cache_virtual_batch


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
