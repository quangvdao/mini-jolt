"""Stage 1: Spartan outer sumcheck verifiers."""
from openings import BIG_ENDIAN, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial, LagrangePolynomial, log2_pow2
from r1cs import ALL_R1CS_INPUTS, OUTER_FIRST_ROUND_POLY_DEGREE_BOUND, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, UniformSpartanKey
from sumchecks import SumcheckInstanceVerifier, _normalize_le_to_be, _cache_virtual_batch


class SpartanOuterUniSkipParams:  # Params for Spartan outer uni-skip (stores tau only).
    def __init__(self, tau):  # Store tau vector (challenge-vector-optimized).
        self.tau = list(tau)

class SpartanOuterUniSkipVerifier(SumcheckInstanceVerifier):  # Verifier instance for Spartan outer uni-skip.
    def __init__(self, key, transcript):  # Sample tau from transcript, matching Rust `OuterUniSkipParams::new`.
        if not isinstance(key, UniformSpartanKey):
            key = UniformSpartanKey(int(key))
        self.key = key
        self.params = SpartanOuterUniSkipParams(transcript.challenge_vector_optimized(key.num_rows_bits()))

    def degree(self):  # First-round uniskip poly degree bound.
        return OUTER_FIRST_ROUND_POLY_DEGREE_BOUND

    def num_rounds(self):  # Uniskip is one round.
        return 1

    def input_claim(self, _opening_accumulator):  # Outer uniskip input claim is zero.
        from field import Fr  # local import to keep module dependency minimal
        return Fr.zero()

    def expected_output_claim(self, _opening_accumulator, _sumcheck_challenges):  # Unused for uni-skip.
        raise NotImplementedError("uniskip verifier does not use expected_output_claim")

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Populate r0 opening for UnivariateSkip.
        opening_point = OpeningPoint(list(sumcheck_challenges), BIG_ENDIAN)
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanOuter,
            opening_point,
        )

class SpartanOuterRemainingSumcheckVerifier(SumcheckInstanceVerifier):  # Spartan outer remaining sumcheck verifier (Stage 1b).
    OUTER_REMAINING_DEGREE_BOUND = 3  # Rust: OUTER_REMAINING_DEGREE_BOUND.

    def __init__(self, key, trace_len, uni_skip_params, opening_accumulator):  # Build verifier params from key + accumulator.
        if not isinstance(key, UniformSpartanKey):
            key = UniformSpartanKey(int(key))
        self.key = key
        self.trace_len = int(trace_len)
        self.num_cycles_bits = log2_pow2(self.trace_len)
        self.tau = list(uni_skip_params.tau)
        r_uni_skip, _ = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanOuter,
        )
        if len(r_uni_skip) != 1:
            raise ValueError("expected uniskip opening point length 1")
        self.r0 = r_uni_skip[0]

    def degree(self):  # Degree bound for remaining outer round polynomials.
        return self.OUTER_REMAINING_DEGREE_BOUND

    def num_rounds(self):  # Total rounds = 1 + num_cycle_bits (streaming round + cycle vars).
        return 1 + int(self.num_cycles_bits)

    def input_claim(self, opening_accumulator):  # Input claim is the UnivariateSkip opening claim.
        return opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.UnivariateSkip,
            SumcheckId.SpartanOuter,
        )[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust `OuterRemainingSumcheckVerifier::expected_output_claim`.
        r1cs_input_evals = [
            opening_accumulator.get_virtual_polynomial_opening(inp, SumcheckId.SpartanOuter)[1]
            for inp in ALL_R1CS_INPUTS
        ]
        rx_constr = [sumcheck_challenges[0], self.r0]
        inner_sum_prod = self.key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals)
        tau_high = self.tau[-1]
        tau_low = self.tau[:-1]
        tau_high_bound_r0 = LagrangePolynomial.lagrange_kernel(
            tau_high,
            self.r0,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        )
        r_tail_reversed = list(reversed(sumcheck_challenges))
        tau_bound_r_tail_reversed = EqPolynomial.mle(tau_low, r_tail_reversed)
        return tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Enqueue/cache all R1CS input openings at r_cycle.
        r_cycle = _normalize_le_to_be(sumcheck_challenges[1:])
        _cache_virtual_batch(opening_accumulator, transcript, ALL_R1CS_INPUTS, SumcheckId.SpartanOuter, r_cycle)
