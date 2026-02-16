"""Stage 1: Spartan outer sumcheck verifiers."""
from field import Fr
from openings import BIG_ENDIAN, OpeningPoint, SumcheckId as SC, VirtualPolynomial as VP
from polynomials import EqPolynomial, LagrangePolynomial, log2_pow2
from r1cs import ALL_R1CS_INPUTS, OUTER_FIRST_ROUND_POLY_DEGREE_BOUND, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, UniformSpartanKey
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be, _cache_virtual_batch


class SpartanOuterUniSkipParams:  # Params for Spartan outer uni-skip (stores tau only).
    def __init__(self, tau):
        self.tau = list(tau)

class SpartanOuterUniSkipVerifier(SumcheckInstanceVerifier):  # Verifier instance for Spartan outer uni-skip.
    DEGREE_BOUND = OUTER_FIRST_ROUND_POLY_DEGREE_BOUND

    def __init__(self, key, transcript):
        if not isinstance(key, UniformSpartanKey):
            key = UniformSpartanKey(int(key))
        self.key = key
        self.params = SpartanOuterUniSkipParams(transcript.challenge_vector_optimized(key.num_rows_bits()))

    def num_rounds(self):
        return 1

    def input_claim(self, _acc):
        return Fr.zero()

    def expected_output_claim(self, _acc, _r):
        raise NotImplementedError("uniskip verifier does not use expected_output_claim")

    def cache_openings(self, acc, transcript, r):
        acc.append_virtual(transcript, VP.UnivariateSkip, SC.SpartanOuter, OpeningPoint(list(r), BIG_ENDIAN))

class SpartanOuterRemainingSumcheckVerifier(SumcheckInstanceVerifier):  # Spartan outer remaining sumcheck (Stage 1b).
    DEGREE_BOUND = 3

    def __init__(self, key, trace_len, uni_skip_params, acc):
        if not isinstance(key, UniformSpartanKey):
            key = UniformSpartanKey(int(key))
        self.key = key
        self.trace_len = int(trace_len)
        self.num_cycles_bits = log2_pow2(self.trace_len)
        self.tau = list(uni_skip_params.tau)
        r_uni_skip, _ = acc.vp(VP.UnivariateSkip, SC.SpartanOuter)
        if len(r_uni_skip) != 1:
            raise ValueError("expected uniskip opening point length 1")
        self.r0 = r_uni_skip[0]

    def num_rounds(self):
        return 1 + int(self.num_cycles_bits)

    def input_claim(self, acc):
        return acc.vp(VP.UnivariateSkip, SC.SpartanOuter)[1]

    def expected_output_claim(self, acc, r):
        r1cs_input_evals = [acc.vp(inp, SC.SpartanOuter)[1] for inp in ALL_R1CS_INPUTS]
        rx_constr = [r[0], self.r0]
        inner_sum_prod = self.key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals)
        tau_high = self.tau[-1]
        tau_low = self.tau[:-1]
        tau_high_bound_r0 = LagrangePolynomial.lagrange_kernel(tau_high, self.r0, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE)
        tau_bound = EqPolynomial.mle(tau_low, list(reversed(r)))
        return tau_high_bound_r0 * tau_bound * inner_sum_prod

    def cache_openings(self, acc, transcript, r):
        r_cycle = _normalize_le_to_be(r[1:])
        _cache_virtual_batch(acc, transcript, ALL_R1CS_INPUTS, SC.SpartanOuter, r_cycle)
