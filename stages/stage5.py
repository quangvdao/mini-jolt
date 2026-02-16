"""Stage 5: InstructionReadRaf + RamRaClaimReduction + RegistersValEvaluation."""
from ids_generated import LOOKUP_TABLES_64
from lookup_tables import evaluate_mle
from openings import BIG_ENDIAN, CommittedPolynomial, OpeningPoint, SumcheckId, VirtualPolynomial
from polynomials import EqPolynomial, IdentityPolynomial, OperandPolynomial, log2_pow2
from rv64imac.constants import REGISTER_COUNT
from sumchecks import SumcheckInstanceVerifier, SumcheckVerifyError, _normalize_le_to_be


class InstructionReadRafSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: instruction lookups read+RAF checking.
    LOG_K = 128  # Rust: instruction_lookups::LOG_K = XLEN*2 = 128 for RV64.

    def __init__(self, n_cycle_vars, one_hot_params, opening_accumulator, transcript):  # Sample gamma; store r_reduction + chunking.
        self.gamma = transcript.challenge_scalar()
        self.gamma_sqr = self.gamma * self.gamma
        self.log_T = int(n_cycle_vars)
        self.ra_virtual_log_k_chunk = int(one_hot_params.lookups_ra_virtual_log_k_chunk)
        self.r_reduction = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.InstructionClaimReduction,
        )[0]

    def degree(self):  # Rust: n_virtual_ra_polys + 2.
        return (self.LOG_K // self.ra_virtual_log_k_chunk) + 2

    def num_rounds(self):  # Rust: LOG_K + log_T.
        return self.LOG_K + self.log_T

    def input_claim(self, opening_accumulator):  # Rust: read_raf_checking.rs:145-165.
        rv_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.InstructionClaimReduction,
        )[1]
        rv_claim_branch = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LookupOutput,
            SumcheckId.SpartanProductVirtualization,
        )[1]
        if rv_claim != rv_claim_branch:
            raise SumcheckVerifyError("LookupOutput claim mismatch across stages")
        left_operand_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.LeftLookupOperand,
            SumcheckId.InstructionClaimReduction,
        )[1]
        right_operand_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RightLookupOperand,
            SumcheckId.InstructionClaimReduction,
        )[1]
        return rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim

    def _normalize_opening_point(self, sumcheck_challenges):  # Rust: read_raf_checking.rs:172-180.
        r_address_prime = list(sumcheck_challenges[: self.LOG_K])
        r_cycle_prime = list(reversed(sumcheck_challenges[self.LOG_K :]))
        return OpeningPoint(r_address_prime + r_cycle_prime, BIG_ENDIAN)

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: read_raf_checking.rs:1195-1266.
        opening_point = self._normalize_opening_point(sumcheck_challenges)
        r_address_prime = opening_point.r[: self.LOG_K]
        r_cycle_prime = opening_point.r[self.LOG_K :]

        left_operand_eval = OperandPolynomial(self.LOG_K, OperandPolynomial.LEFT).evaluate(r_address_prime)
        right_operand_eval = OperandPolynomial(self.LOG_K, OperandPolynomial.RIGHT).evaluate(r_address_prime)
        identity_poly_eval = IdentityPolynomial(self.LOG_K).evaluate(r_address_prime)

        val_evals = [evaluate_mle(t, r_address_prime, xlen=64) for t in LOOKUP_TABLES_64]
        eq_eval_r_reduction = EqPolynomial.mle(self.r_reduction.r, r_cycle_prime)

        n_virtual_ra_polys = self.LOG_K // self.ra_virtual_log_k_chunk
        ra_claim = None
        for i in range(n_virtual_ra_polys):
            c = opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionReadRaf,
            )[1]
            ra_claim = c if ra_claim is None else (ra_claim * c)

        table_flag_claims = [
            opening_accumulator.get_virtual_polynomial_opening_i(
                VirtualPolynomial.LookupTableFlag,
                i,
                SumcheckId.InstructionReadRaf,
            )[1]
            for i in range(len(LOOKUP_TABLES_64))
        ]
        raf_flag_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.InstructionRafFlag,
            SumcheckId.InstructionReadRaf,
        )[1]

        val_claim = None
        for ev, flag in zip(val_evals, table_flag_claims):
            term = ev * flag
            val_claim = term if val_claim is None else (val_claim + term)

        from field import Fr  # local import for one()
        raf_claim = (Fr.one() - raf_flag_claim) * (left_operand_eval + self.gamma * right_operand_eval) + raf_flag_claim * self.gamma * identity_poly_eval
        return eq_eval_r_reduction * ra_claim * (val_claim + self.gamma * raf_claim)

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: read_raf_checking.rs:1268-1309.
        r_sumcheck = self._normalize_opening_point(sumcheck_challenges)
        r_address = OpeningPoint(r_sumcheck.r[: self.LOG_K], BIG_ENDIAN)
        r_cycle = OpeningPoint(r_sumcheck.r[self.LOG_K :], BIG_ENDIAN)

        for i in range(len(LOOKUP_TABLES_64)):
            opening_accumulator.append_virtual_i(
                transcript,
                VirtualPolynomial.LookupTableFlag,
                i,
                SumcheckId.InstructionReadRaf,
                r_cycle.clone(),
            )

        for i, r_address_chunk in enumerate(
            [r_address.r[j : j + self.ra_virtual_log_k_chunk] for j in range(0, len(r_address.r), self.ra_virtual_log_k_chunk)]
        ):
            opening_point = OpeningPoint(list(r_address_chunk) + list(r_cycle.r), BIG_ENDIAN)
            opening_accumulator.append_virtual_i(
                transcript,
                VirtualPolynomial.InstructionRa,
                i,
                SumcheckId.InstructionReadRaf,
                opening_point,
            )

        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.InstructionRafFlag,
            SumcheckId.InstructionReadRaf,
            r_cycle.clone(),
        )


class RamRaClaimReductionSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: reduce RAM RA claims for virtualization.
    DEGREE_BOUND = 2

    def __init__(self, trace_len, one_hot_params, opening_accumulator, transcript):  # Extract points/claims; sample gamma.
        self.log_K = log2_pow2(int(one_hot_params.ram_k))
        self.log_T = log2_pow2(int(trace_len))

        r_raf, claim_raf = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamRafEvaluation,
        )
        r_rw, claim_rw = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamReadWriteChecking,
        )
        r_val_eval, claim_val_eval = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamValEvaluation,
        )
        r_val_final, claim_val_final = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamValFinalEvaluation,
        )

        self.r_address_1 = list(r_raf.r[: self.log_K])
        self.r_cycle_raf = list(r_raf.r[self.log_K :])
        self.r_address_2 = list(r_rw.r[: self.log_K])
        self.r_cycle_rw = list(r_rw.r[self.log_K :])
        self.r_cycle_val = list(r_val_eval.r[self.log_K :])

        if self.r_address_1 != list(r_val_final.r[: self.log_K]):
            raise ValueError("unexpected r_address mismatch for RAM RA reduction")
        if self.r_address_2 != list(r_val_eval.r[: self.log_K]):
            raise ValueError("unexpected r_address mismatch for RAM RA reduction")
        if self.r_cycle_val != list(r_val_final.r[self.log_K :]):
            raise ValueError("unexpected r_cycle mismatch for RAM RA reduction")

        self.claim_raf = claim_raf
        self.claim_val_final = claim_val_final
        self.claim_rw = claim_rw
        self.claim_val_eval = claim_val_eval

        self.gamma = transcript.challenge_scalar()
        self.gamma_squared = self.gamma * self.gamma
        self.gamma_cubed = self.gamma_squared * self.gamma

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return self.log_K + self.log_T

    def input_claim(self, _opening_accumulator):  # Rust: ram_ra.rs:1000-1005.
        return self.claim_raf + self.gamma * self.claim_val_final + self.gamma_squared * self.claim_rw + self.gamma_cubed * self.claim_val_eval

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: ram_ra.rs:1046-1083.
        r_address_reduced = list(reversed(sumcheck_challenges[: self.log_K]))
        r_cycle_reduced = list(reversed(sumcheck_challenges[self.log_K :]))

        eq_addr_1 = EqPolynomial.mle(self.r_address_1, r_address_reduced)
        eq_addr_2 = EqPolynomial.mle(self.r_address_2, r_address_reduced)

        eq_cycle_raf = EqPolynomial.mle(self.r_cycle_raf, r_cycle_reduced)
        eq_cycle_rw = EqPolynomial.mle(self.r_cycle_rw, r_cycle_reduced)
        eq_cycle_val = EqPolynomial.mle(self.r_cycle_val, r_cycle_reduced)

        eq_cycle_A = eq_cycle_raf + self.gamma * eq_cycle_val
        eq_cycle_B = eq_cycle_rw + self.gamma * eq_cycle_val
        eq_combined = eq_addr_1 * eq_cycle_A + self.gamma_squared * eq_addr_2 * eq_cycle_B

        ra_claim_reduced = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RamRa,
            SumcheckId.RamRaClaimReduction,
        )[1]
        return eq_combined * ra_claim_reduced

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: ram_ra.rs:1085-1109.
        r_address_be = list(reversed(sumcheck_challenges[: self.log_K]))
        r_cycle_be = list(reversed(sumcheck_challenges[self.log_K :]))
        opening_point = OpeningPoint(r_address_be + r_cycle_be, BIG_ENDIAN)
        opening_accumulator.append_virtual(transcript, VirtualPolynomial.RamRa, SumcheckId.RamRaClaimReduction, opening_point)


class RegistersValEvaluationSumcheckVerifier(SumcheckInstanceVerifier):  # Stage 5: registers val evaluation.
    DEGREE_BOUND = 3
    LOG_K = int(REGISTER_COUNT).bit_length() - 1  # Rust: log2(REGISTER_COUNT)=7 (128 regs)

    def __init__(self, opening_accumulator):  # Extract r_address||r_cycle from RegistersVal opening.
        r_full = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RegistersVal,
            SumcheckId.RegistersReadWriteChecking,
        )[0]
        self.r_cycle = OpeningPoint(r_full.r[self.LOG_K :], BIG_ENDIAN)
        self.r_address = OpeningPoint(r_full.r[: self.LOG_K], BIG_ENDIAN)

    def degree(self):
        return self.DEGREE_BOUND

    def num_rounds(self):
        return len(self.r_cycle.r)

    def input_claim(self, opening_accumulator):  # Rust: val_evaluation.rs:80-86.
        return opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RegistersVal,
            SumcheckId.RegistersReadWriteChecking,
        )[1]

    def expected_output_claim(self, opening_accumulator, sumcheck_challenges):  # Rust: registers/val_evaluation.rs:250-283.
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        from field import Fr  # local import for one()
        lt_eval = Fr.zero()
        eq_term = Fr.one()
        for x, y in zip(r_cycle_prime.r, self.r_cycle.r):
            lt_eval += (Fr.one() - x) * y * eq_term
            eq_term *= Fr.one() - x - y + x * y + x * y

        inc_claim = opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersValEvaluation,
        )[1]
        wa_claim = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial.RdWa,
            SumcheckId.RegistersValEvaluation,
        )[1]
        return inc_claim * wa_claim * lt_eval

    def cache_openings(self, opening_accumulator, transcript, sumcheck_challenges):  # Rust: registers/val_evaluation.rs:285-314.
        r_cycle_prime = _normalize_le_to_be(sumcheck_challenges)
        opening_accumulator.append_dense(
            transcript,
            CommittedPolynomial.RdInc,
            SumcheckId.RegistersValEvaluation,
            r_cycle_prime.r,
        )
        opening_point = OpeningPoint(list(self.r_address.r) + list(r_cycle_prime.r), BIG_ENDIAN)
        opening_accumulator.append_virtual(
            transcript,
            VirtualPolynomial.RdWa,
            SumcheckId.RegistersValEvaluation,
            opening_point,
        )
