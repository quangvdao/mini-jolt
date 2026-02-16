"""Stage 7: Hamming weight claim reduction."""
from openings import BIG_ENDIAN, LITTLE_ENDIAN, CommittedPolynomial, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial
from sumchecks import SumcheckInstanceVerifier


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
