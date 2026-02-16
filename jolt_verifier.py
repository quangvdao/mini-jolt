from dataclasses import replace  # frozen dataclass updates

from field import Fr  # BN254 scalar field for claims/challenges
from openings import AdviceKind, CommittedPolynomial, SumcheckId, VerifierOpeningAccumulator  # typed IDs + opening accumulator
from polynomials import log2_pow2  # verifier-minimal helpers
from r1cs import UniformSpartanKey  # Spartan verifier key
from transcript import Blake2bTranscript  # verifier transcript
from zkvm_types import JoltDevice, OneHotParams  # public inputs + one-hot params

import dory  # Stage8 Dory joint opening verification
from jolt_preprocessing import JoltPreprocessing  # explicit verifier preprocessing
from jolt_proof import JoltProof  # canonical proof container
from r1cs import OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE  # Spartan outer constants
from ram_io import verifier_accumulate_advice  # advice accumulation helper
from sumchecks import BatchedSumcheck, SumcheckInstanceProof, UniSkipFirstRoundProof  # core proof containers
from stages.stage1 import SpartanOuterRemainingSumcheckVerifier, SpartanOuterUniSkipVerifier
from stages.stage2 import (PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
    ProductVirtualUniSkipVerifier, ProductVirtualRemainderVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
    RamRafEvaluationSumcheckVerifier, OutputSumcheckVerifier, RamReadWriteCheckingVerifier)
from stages.stage3 import ShiftSumcheckVerifier, InstructionInputSumcheckVerifier, RegistersClaimReductionSumcheckVerifier
from stages.stage4 import RegistersReadWriteCheckingVerifier, RamValEvaluationSumcheckVerifier, ValFinalSumcheckVerifier
from stages.stage5 import InstructionReadRafSumcheckVerifier, RamRaClaimReductionSumcheckVerifier, RegistersValEvaluationSumcheckVerifier
from stages.stage6 import (HammingBooleanitySumcheckVerifier, RamRaVirtualSumcheckVerifier, InstructionRaVirtualSumcheckVerifier,
    IncClaimReductionSumcheckVerifier, AdviceClaimReductionVerifier, BooleanitySumcheckVerifier, BytecodeReadRafSumcheckVerifier)
from stages.stage7 import HammingWeightClaimReductionSumcheckVerifier


class JoltVerifyError(Exception):  # Raised on top-level Jolt verification failure.
    pass


def fiat_shamir_preamble(program_io: JoltDevice, ram_K: int, trace_length: int, transcript: Blake2bTranscript):  # Rust: zkvm/mod.rs fiat_shamir_preamble.
    ml = program_io.memory_layout
    max_input_size = int(getattr(ml, "max_input_size", len(program_io.inputs)))
    max_output_size = int(getattr(ml, "max_output_size", len(program_io.outputs)))
    heap_size = int(getattr(ml, "heap_size", 0))
    transcript.append_u64(b"max_input_size", max_input_size)
    transcript.append_u64(b"max_output_size", max_output_size)
    transcript.append_u64(b"heap_size", heap_size)
    transcript.append_bytes(b"inputs", program_io.inputs)
    transcript.append_bytes(b"outputs", program_io.outputs)
    transcript.append_u64(b"panic", 1 if bool(program_io.panic_flag) else 0)
    transcript.append_u64(b"ram_K", int(ram_K))
    transcript.append_u64(b"trace_length", int(trace_length))


def compute_advice_lagrange_factor(opening_point, advice_opening_point):  # Rust: poly/opening_proof.rs compute_advice_lagrange_factor.
    opening_point = list(opening_point)
    advice_opening_point = list(advice_opening_point)
    out = Fr.one()
    for r in opening_point:
        out *= Fr.one() if r in advice_opening_point else (Fr.one() - r)
    return out


class JoltVerifier:  # Top-level verifier orchestrator mirroring Rust `JoltVerifier`.
    def __init__(self, program_bytecode, program_io: JoltDevice, proof: JoltProof, *, preprocessing: JoltPreprocessing | None = None, memory_init=None):  # Store inputs; allocate transcript+accumulator.
        self.program_bytecode = list(program_bytecode)
        # Rust verifier truncates trailing zeros on device outputs before transcript coupling.
        outs = bytes(program_io.outputs)
        trimmed = outs.rstrip(b"\x00")
        if trimmed != outs:
            program_io = replace(program_io, outputs=trimmed)
        self.program_io = program_io
        self.proof = proof
        if preprocessing is None:
            if memory_init is None:
                raise JoltVerifyError("missing preprocessing: provide `preprocessing` or `memory_init`")
            preprocessing = JoltPreprocessing.preprocess(self.program_bytecode, memory_init)
        self.preprocessing = preprocessing
        self.transcript = Blake2bTranscript.new(b"Jolt")
        self.opening_accumulator = VerifierOpeningAccumulator()
        self.one_hot_params = (
            proof.one_hot_params
            if proof.one_hot_params is not None
            else OneHotParams(ram_k=int(proof.ram_K), bytecode_k=int(proof.bytecode_K))
        )

    def verify_stages_1_to_7(self, *, absorb_preamble=False):  # Verify through Stage 7 (sumchecks + reductions).
        if absorb_preamble:
            fiat_shamir_preamble(self.program_io, self.proof.ram_K, self.proof.trace_length, self.transcript)
            self.proof.absorb_commitments_into_transcript(self.transcript)

        self.proof.seed_opening_accumulator(self.opening_accumulator)

        trace_len = int(self.proof.trace_length)
        bytecode_preprocessing = self.preprocessing.bytecode

        # Stage 1
        key = UniformSpartanKey(trace_len)
        _ = verify_spartan_outer_stage1(
            self.proof.stage1_uni_skip_first_round_proof,
            self.proof.stage1_sumcheck_proof,
            key,
            trace_len,
            self.opening_accumulator,
            self.transcript,
        )

        # Stage 2
        if self.proof.rw_config is None:
            raise JoltVerifyError("proof.rw_config is required for Stage 2/4/6 advice behavior")
        _ = verify_stage2(
            self.proof.stage2_uni_skip_first_round_proof,
            self.proof.stage2_sumcheck_proof,
            trace_len,
            self.one_hot_params,
            self.proof.rw_config,
            self.program_io.memory_layout,
            self.program_io,
            self.opening_accumulator,
            self.transcript,
        )

        # Stage 3
        _ = verify_stage3(
            self.proof.stage3_sumcheck_proof,
            trace_len,
            self.opening_accumulator,
            self.transcript,
        )

        # Stage 4
        _ = verify_stage4(
            self.proof.stage4_sumcheck_proof,
            trace_len,
            self.one_hot_params,
            self.proof.rw_config,
            self.preprocessing.ram,
            self.program_io,
            self.proof.untrusted_advice_commitment is not None,
            self.proof.trusted_advice_commitment is not None,
            self.opening_accumulator,
            self.transcript,
        )

        # Stage 5
        _ = verify_stage5(
            self.proof.stage5_sumcheck_proof,
            trace_len,
            self.one_hot_params,
            self.opening_accumulator,
            self.transcript,
        )

        # Stage 6
        advice_verifiers = {}
        _ = verify_stage6(
            self.proof.stage6_sumcheck_proof,
            trace_len,
            self.one_hot_params,
            bytecode_preprocessing,
            self.opening_accumulator,
            self.transcript,
            has_untrusted_advice_commitment=(self.proof.untrusted_advice_commitment is not None),
            has_trusted_advice_commitment=(self.proof.trusted_advice_commitment is not None),
            memory_layout=self.program_io.memory_layout,
            rw_config=self.proof.rw_config,
            advice_reduction_verifiers_out=advice_verifiers,
        )

        # Stage 7
        _ = verify_stage7(
            self.proof.stage7_sumcheck_proof,
            trace_len,
            self.one_hot_params,
            self.opening_accumulator,
            self.transcript,
            advice_reduction_verifiers=advice_verifiers,
        )

    def verify_stage8_joint_opening(self):  # Verify the Stage 8 Dory joint opening.
        if self.proof.joint_opening_proof is None:
            raise JoltVerifyError("missing joint_opening_proof (Stage 8)")
        if self.proof.dory_verifier_setup is None:
            raise JoltVerifyError("missing dory_verifier_setup (Stage 8)")
        if self.proof.dory_serde_blocks is None:
            raise JoltVerifyError("missing dory_serde_blocks (Stage 8)")
        if self.proof.commitments is None:
            raise JoltVerifyError("missing commitments (Stage 8)")

        log_t = log2_pow2(int(self.proof.trace_length))
        log_k_chunk = int(self.one_hot_params.log_k_chunk)

        # Unified opening point (ρ_address || r_cycle_stage6), BE.
        opening_point, _ = self.opening_accumulator.get_committed_polynomial_opening_i(
            CommittedPolynomial.InstructionRa,
            0,
            SumcheckId.HammingWeightClaimReduction,
        )
        r_address_stage7 = opening_point.r[:log_k_chunk]

        # Dense polynomials: apply lagrange factor for embedding into top-left block.
        lagrange_factor = Fr.one()
        for r in r_address_stage7:
            lagrange_factor *= Fr.one() - r

        _, ram_inc_claim = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RamInc,
            SumcheckId.IncClaimReduction,
        )
        _, rd_inc_claim = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial.RdInc,
            SumcheckId.IncClaimReduction,
        )

        # Commitments are ordered by Rust `all_committed_polynomials(one_hot_params)`:
        # [RdInc, RamInc, InstructionRa..., RamRa..., BytecodeRa...]. Build a lookup map.
        d_instr = int(self.one_hot_params.instruction_d)
        d_ram = int(self.one_hot_params.ram_d)
        d_bytecode = int(self.one_hot_params.bytecode_d)
        expected_commitments = 2 + d_instr + d_ram + d_bytecode
        if len(self.proof.commitments) != expected_commitments:
            raise JoltVerifyError("unexpected commitments length for Stage 8")
        c = list(self.proof.commitments)
        committed_map = {
            CommittedPolynomial.RdInc: c[0],
            CommittedPolynomial.RamInc: c[1],
        }
        idx = 2
        for i in range(d_instr):
            committed_map[(CommittedPolynomial.InstructionRa, i)] = c[idx]
            idx += 1
        for i in range(d_ram):
            committed_map[(CommittedPolynomial.RamRa, i)] = c[idx]
            idx += 1
        for i in range(d_bytecode):
            committed_map[(CommittedPolynomial.BytecodeRa, i)] = c[idx]
            idx += 1

        # Rust Stage 8 polynomial_claims order:
        # 1) RamInc, 2) RdInc, 3) InstructionRa*, 4) BytecodeRa*, 5) RamRa*, then advice (optional).
        claims = [ram_inc_claim * lagrange_factor, rd_inc_claim * lagrange_factor]
        commitments = [committed_map[CommittedPolynomial.RamInc], committed_map[CommittedPolynomial.RdInc]]

        for i in range(d_instr):
            _, ci = self.opening_accumulator.get_committed_polynomial_opening_i(
                CommittedPolynomial.InstructionRa,
                i,
                SumcheckId.HammingWeightClaimReduction,
            )
            claims.append(ci)
            commitments.append(committed_map[(CommittedPolynomial.InstructionRa, i)])
        for i in range(d_bytecode):
            _, ci = self.opening_accumulator.get_committed_polynomial_opening_i(
                CommittedPolynomial.BytecodeRa,
                i,
                SumcheckId.HammingWeightClaimReduction,
            )
            claims.append(ci)
            commitments.append(committed_map[(CommittedPolynomial.BytecodeRa, i)])
        for i in range(d_ram):
            _, ci = self.opening_accumulator.get_committed_polynomial_opening_i(
                CommittedPolynomial.RamRa,
                i,
                SumcheckId.HammingWeightClaimReduction,
            )
            claims.append(ci)
            commitments.append(committed_map[(CommittedPolynomial.RamRa, i)])

        # Advice polynomials: embed into main matrix via advice lagrange factor.
        advice = self.opening_accumulator.get_advice_opening(AdviceKind.Trusted, SumcheckId.AdviceClaimReduction)
        if advice is not None and self.proof.trusted_advice_commitment is not None:
            advice_point, advice_claim = advice
            lf = compute_advice_lagrange_factor(opening_point.r, advice_point.r)
            claims.append(advice_claim * lf)
            commitments.append(self.proof.trusted_advice_commitment)
        advice = self.opening_accumulator.get_advice_opening(AdviceKind.Untrusted, SumcheckId.AdviceClaimReduction)
        if advice is not None and self.proof.untrusted_advice_commitment is not None:
            advice_point, advice_claim = advice
            lf = compute_advice_lagrange_factor(opening_point.r, advice_point.r)
            claims.append(advice_claim * lf)
            commitments.append(self.proof.untrusted_advice_commitment)

        dory.verify_rlc_joint_opening(
            self.proof.joint_opening_proof,
            self.proof.dory_verifier_setup,
            self.transcript,
            opening_point.r,
            claims,
            commitments,
            dory_layout=self.proof.dory_layout,
            log_T=log_t,
            serde_blocks=self.proof.dory_serde_blocks,
            rlc_label=b"rlc_claims",
        )

    def verify(self, *, absorb_preamble=True, verify_stage8=True):  # Full verifier entrypoint (Rust-like defaults).
        self.verify_stages_1_to_7(absorb_preamble=absorb_preamble)
        if verify_stage8:
            self.verify_stage8_joint_opening()


def verify_jolt(program_bytecode, program_io: JoltDevice, proof: JoltProof, *, preprocessing: JoltPreprocessing | None = None, memory_init=None):  # Convenience wrapper for full verification.
    v = JoltVerifier(program_bytecode, program_io, proof, preprocessing=preprocessing, memory_init=memory_init)
    return v.verify()


# ---------------------------------------------------------------------------
# Stage orchestration (hoisted from `sumchecks.py`)
# ---------------------------------------------------------------------------

def verify_spartan_outer_stage1(uni_skip_proof, outer_sumcheck_proof, key, trace_len, opening_accumulator, transcript):  # Verify Spartan outer Stage 1.
    uni_skip_verifier = SpartanOuterUniSkipVerifier(key, transcript)
    _r0 = uni_skip_proof.verify(
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        uni_skip_verifier,
        opening_accumulator,
        transcript,
    )
    outer_remaining = SpartanOuterRemainingSumcheckVerifier(
        key,
        trace_len,
        uni_skip_verifier.params,
        opening_accumulator,
    )
    return BatchedSumcheck.verify(
        outer_sumcheck_proof,
        [outer_remaining],
        opening_accumulator,
        transcript,
    )


def verify_stage2(stage2_uni_skip_proof, stage2_sumcheck_proof, trace_len, one_hot_params, rw_config, memory_layout, program_io, opening_accumulator, transcript):  # Verify Stage 2.
    # Stage 2a: product virtualization uni-skip
    pv_uniskip = ProductVirtualUniSkipVerifier(opening_accumulator, transcript)
    _ = stage2_uni_skip_proof.verify(
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        pv_uniskip,
        opening_accumulator,
        transcript,
    )
    # Stage 2b: batched sumcheck
    ram_rw = RamReadWriteCheckingVerifier(opening_accumulator, transcript, one_hot_params, trace_len, rw_config)
    pv_rem = ProductVirtualRemainderVerifier(trace_len, pv_uniskip.params, opening_accumulator)
    instr_red = InstructionLookupsClaimReductionSumcheckVerifier(trace_len, opening_accumulator, transcript)
    ram_raf = RamRafEvaluationSumcheckVerifier(memory_layout, one_hot_params, opening_accumulator)
    out_chk = OutputSumcheckVerifier(one_hot_params.ram_k, program_io, transcript)
    return BatchedSumcheck.verify(
        stage2_sumcheck_proof,
        [ram_rw, pv_rem, instr_red, ram_raf, out_chk],
        opening_accumulator,
        transcript,
    )


def verify_stage3(stage3_sumcheck_proof, trace_len, opening_accumulator, transcript):  # Verify Stage 3.
    shift = ShiftSumcheckVerifier(trace_len, opening_accumulator, transcript)
    instr_in = InstructionInputSumcheckVerifier(opening_accumulator, transcript)
    regs = RegistersClaimReductionSumcheckVerifier(trace_len, opening_accumulator, transcript)
    return BatchedSumcheck.verify(stage3_sumcheck_proof, [shift, instr_in, regs], opening_accumulator, transcript)


def verify_stage4(stage4_sumcheck_proof, trace_len, one_hot_params, rw_config, ram_preprocessing, program_io, has_untrusted_advice_commitment, has_trusted_advice_commitment, opening_accumulator, transcript):  # Verify Stage 4.
    log_T = log2_pow2(int(trace_len))
    verifier_accumulate_advice(
        one_hot_params.ram_k,
        program_io,
        has_untrusted_advice_commitment,
        has_trusted_advice_commitment,
        opening_accumulator,
        transcript,
        rw_config.needs_single_advice_opening(log_T),
    )
    regs_rw = RegistersReadWriteCheckingVerifier(trace_len, opening_accumulator, transcript, rw_config)
    ram_val_eval = RamValEvaluationSumcheckVerifier(ram_preprocessing, program_io, trace_len, one_hot_params.ram_k, opening_accumulator)
    val_final = ValFinalSumcheckVerifier(ram_preprocessing, program_io, trace_len, one_hot_params.ram_k, opening_accumulator, rw_config)
    return BatchedSumcheck.verify(stage4_sumcheck_proof, [regs_rw, ram_val_eval, val_final], opening_accumulator, transcript)


def verify_stage5(stage5_sumcheck_proof, trace_len, one_hot_params, opening_accumulator, transcript):  # Verify Stage 5.
    n_cycle_vars = log2_pow2(int(trace_len))
    lookups_read_raf = InstructionReadRafSumcheckVerifier(
        n_cycle_vars,
        one_hot_params,
        opening_accumulator,
        transcript,
    )
    ram_ra_reduction = RamRaClaimReductionSumcheckVerifier(
        trace_len,
        one_hot_params,
        opening_accumulator,
        transcript,
    )
    registers_val_evaluation = RegistersValEvaluationSumcheckVerifier(opening_accumulator)
    return BatchedSumcheck.verify(
        stage5_sumcheck_proof,
        [lookups_read_raf, ram_ra_reduction, registers_val_evaluation],
        opening_accumulator,
        transcript,
    )


def verify_stage6(stage6_sumcheck_proof, trace_len, one_hot_params, bytecode_preprocessing, opening_accumulator, transcript, *, has_untrusted_advice_commitment=False, has_trusted_advice_commitment=False, memory_layout=None, rw_config=None, advice_reduction_verifiers_out=None):  # Verify Stage 6.
    n_cycle_vars = log2_pow2(int(trace_len))
    bytecode_read_raf = BytecodeReadRafSumcheckVerifier(
        bytecode_preprocessing,
        n_cycle_vars,
        one_hot_params,
        opening_accumulator,
        transcript,
    )
    ram_hamming_booleanity = HammingBooleanitySumcheckVerifier(opening_accumulator)
    booleanity = BooleanitySumcheckVerifier(n_cycle_vars, one_hot_params, opening_accumulator, transcript)
    ram_ra_virtual = RamRaVirtualSumcheckVerifier(trace_len, one_hot_params, opening_accumulator, transcript)
    lookups_ra_virtual = InstructionRaVirtualSumcheckVerifier(one_hot_params, opening_accumulator, transcript)
    inc_reduction = IncClaimReductionSumcheckVerifier(trace_len, opening_accumulator, transcript)
    instances = [bytecode_read_raf, booleanity, ram_hamming_booleanity, ram_ra_virtual, lookups_ra_virtual, inc_reduction]
    advice_trusted = None
    advice_untrusted = None
    if has_trusted_advice_commitment or has_untrusted_advice_commitment:
        if memory_layout is None or rw_config is None:
            raise ValueError("Stage 6 advice reduction requires memory_layout and rw_config")
        log_t = log2_pow2(int(trace_len))
        single_opening = rw_config.needs_single_advice_opening(log_t)
        log_k_chunk = int(one_hot_params.log_k_chunk)
        if has_trusted_advice_commitment:
            advice_trusted = AdviceClaimReductionVerifier(
                AdviceKind.Trusted,
                memory_layout,
                trace_len,
                log_k_chunk,
                opening_accumulator,
                transcript,
                single_opening,
            )
            instances.append(advice_trusted)
        if has_untrusted_advice_commitment:
            advice_untrusted = AdviceClaimReductionVerifier(
                AdviceKind.Untrusted,
                memory_layout,
                trace_len,
                log_k_chunk,
                opening_accumulator,
                transcript,
                single_opening,
            )
            instances.append(advice_untrusted)
    r = BatchedSumcheck.verify(stage6_sumcheck_proof, instances, opening_accumulator, transcript)
    if advice_reduction_verifiers_out is not None:
        if advice_trusted is not None:
            advice_reduction_verifiers_out[AdviceKind.Trusted] = advice_trusted
        if advice_untrusted is not None:
            advice_reduction_verifiers_out[AdviceKind.Untrusted] = advice_untrusted
    return r


def verify_stage7(stage7_sumcheck_proof, trace_len, one_hot_params, opening_accumulator, transcript, *, advice_reduction_verifiers=None):  # Verify Stage 7.
    hw = HammingWeightClaimReductionSumcheckVerifier(one_hot_params, opening_accumulator, transcript)
    instances = [hw]
    if advice_reduction_verifiers:
        for kind in (AdviceKind.Trusted, AdviceKind.Untrusted):
            v = advice_reduction_verifiers.get(kind)
            if v is None:
                continue
            if v.num_address_phase_rounds() > 0:
                v.phase = v.PHASE_ADDRESS
                instances.append(v)
    return BatchedSumcheck.verify(stage7_sumcheck_proof, instances, opening_accumulator, transcript)
