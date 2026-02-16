"""Stage 7: Hamming weight claim reduction."""
from field import Fr
from openings import BIG_ENDIAN, LITTLE_ENDIAN, CommittedPolynomial as CP, OpeningPoint, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial
from sumchecks import SumcheckInstanceVerifier


class HammingWeightClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 7: fused HW+address reduction.
    DEGREE_BOUND = 2

    def __init__(self, one_hot_params, acc, transcript):
        self.log_k_chunk = int(one_hot_params.log_k_chunk)
        self.instruction_d = int(one_hot_params.instruction_d)
        self.bytecode_d = int(one_hot_params.bytecode_d)
        self.ram_d = int(one_hot_params.ram_d)
        self.N = self.instruction_d + self.bytecode_d + self.ram_d
        self.poly_types = ([(CP.InstructionRa, i) for i in range(self.instruction_d)]
            + [(CP.BytecodeRa, i) for i in range(self.bytecode_d)]
            + [(CP.RamRa, i) for i in range(self.ram_d)])
        self.gamma_powers = list(transcript.challenge_scalar_powers(3 * self.N))
        unified_bool_point, _ = acc.cpi(CP.InstructionRa, 0, SC.Booleanity)
        self.r_addr_bool = list(unified_bool_point.r[:self.log_k_chunk])
        self.r_cycle = list(unified_bool_point.r[self.log_k_chunk:])
        ram_hw_factor = acc.vp(VP.RamHammingWeight, SC.RamHammingBooleanity)[1]
        self.r_addr_virt, self.claims_hw, self.claims_bool, self.claims_virt = [], [], [], []
        for poly, idx in self.poly_types:
            if poly == CP.InstructionRa:
                virt_sc, hw = SC.InstructionRaVirtualization, Fr.one()
            elif poly == CP.BytecodeRa:
                virt_sc, hw = SC.BytecodeReadRaf, Fr.one()
            else:
                virt_sc, hw = SC.RamRaVirtualization, ram_hw_factor
            self.claims_hw.append(hw)
            self.claims_bool.append(acc.cpi(poly, idx, SC.Booleanity)[1])
            virt_pt, virt_claim = acc.cpi(poly, idx, virt_sc)
            self.r_addr_virt.append(list(virt_pt.r[:self.log_k_chunk]))
            self.claims_virt.append(virt_claim)

    def num_rounds(self):
        return self.log_k_chunk

    def input_claim(self, _acc):
        out = None
        for i in range(self.N):
            gp = self.gamma_powers
            term = gp[3*i] * self.claims_hw[i] + gp[3*i+1] * self.claims_bool[i] + gp[3*i+2] * self.claims_virt[i]
            out = term if out is None else (out + term)
        return out

    def expected_output_claim(self, acc, r):
        rho_rev = list(reversed(r))
        eq_bool = EqPolynomial.mle(rho_rev, self.r_addr_bool)
        out = None
        for i, (poly, idx) in enumerate(self.poly_types):
            eq_virt = EqPolynomial.mle(rho_rev, self.r_addr_virt[i])
            g_i = acc.cpi(poly, idx, SC.HammingWeightClaimReduction)[1]
            gp = self.gamma_powers
            w = gp[3*i] + gp[3*i+1] * eq_bool + gp[3*i+2] * eq_virt
            term = g_i * w
            out = term if out is None else (out + term)
        return out

    def cache_openings(self, acc, transcript, r):
        r_addr = OpeningPoint(list(r), LITTLE_ENDIAN).match_endianness(BIG_ENDIAN).r
        full_point = list(r_addr) + list(self.r_cycle)
        for poly, idx in self.poly_types:
            acc.append_dense_i(transcript, poly, idx, SC.HammingWeightClaimReduction, full_point)
