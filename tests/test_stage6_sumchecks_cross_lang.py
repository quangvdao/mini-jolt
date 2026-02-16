import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow importing local verifier modules

from field import Fr  # BN254 Fr field elements
from openings import CommittedPolynomial, SumcheckId, VerifierOpeningAccumulator, VirtualPolynomial  # opening accumulator types
from polynomials import CompressedUniPoly, UniPoly  # polynomial containers
from r1cs import ALL_R1CS_INPUTS, UniformSpartanKey  # Spartan key + canonical input order
from rv64imac.bytecode import BytecodePreprocessing  # Stage 6 bytecode preprocessing
from rv64imac.constants import REGISTER_COUNT  # Rust register domain (arch + virtual)
from jolt_verifier import verify_spartan_outer_stage1, verify_stage2, verify_stage3, verify_stage4, verify_stage5, verify_stage6  # stage orchestration
from sumchecks import SumcheckInstanceProof, UniSkipFirstRoundProof  # proof containers
from tests.oracle import run_rust_oracle  # rust oracle harness
from transcript import Blake2bTranscript  # transcript for parity
from zkvm_types import JoltDevice, MemoryLayout, OneHotParams, RAMPreprocessing, ReadWriteConfig  # public types


def _parse_kv_lines(text):  # Parse oracle stdout with key=value lines into dict.
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"bad oracle line: {line!r}")
        k, v = line.split("=", 1)
        out[k] = v
    return out


def _csv_fr(s):  # CSV of ints -> list[Fr]
    return [Fr(int(x)) for x in s.split(",") if x]


def _csv_u64(s):  # CSV of ints -> list[int]
    return [int(x) for x in s.split(",") if x]


def _u64_words_to_le_bytes(words):  # list[u64] -> bytes, little-endian packing
    out = bytearray()
    for w in words:
        out += int(w).to_bytes(8, "little", signed=False)
    return bytes(out)


class Stage6SumchecksCrossLangTests(unittest.TestCase):  # Python↔Rust parity: Stage1 → Stage2 → Stage3 → Stage4 → Stage5 → Stage6.
    def test_stage6_sumchecks_match_rust_oracle(self):
        kv = _parse_kv_lines(run_rust_oracle("stage6_sumchecks_blake2b", "").strip())

        trace_len = int(kv["trace_len"])
        ram_k = int(kv["ram_k"])
        rw_phase1 = int(kv["rw_phase1"])
        rw_phase2 = int(kv["rw_phase2"])
        regs_rw_phase1 = int(kv["regs_rw_phase1"])
        regs_rw_phase2 = int(kv["regs_rw_phase2"])

        has_untrusted = kv["has_untrusted_advice_commitment"].lower() == "true"
        has_trusted = kv["has_trusted_advice_commitment"].lower() == "true"

        memory_layout = MemoryLayout(
            input_start=int(kv["mem_input_start"]),
            output_start=int(kv["mem_output_start"]),
            panic=int(kv["mem_panic"]),
            termination=int(kv["mem_termination"]),
            untrusted_advice_start=int(kv["untrusted_advice_start"]),
            max_untrusted_advice_size=int(kv["max_untrusted_advice_size"]),
            trusted_advice_start=int(kv["trusted_advice_start"]),
            max_trusted_advice_size=int(kv["max_trusted_advice_size"]),
        )
        program_inputs_words = _csv_u64(kv["inputs_words"])
        program_io = JoltDevice(
            memory_layout=memory_layout,
            inputs=_u64_words_to_le_bytes(program_inputs_words),
            outputs=b"",
            panic_flag=False,
        )
        ram_preprocessing = RAMPreprocessing(
            min_bytecode_address=int(kv["min_bytecode_address"]),
            bytecode_words=_csv_u64(kv["bytecode_words"]),
        )

        one_hot_params = OneHotParams(
            ram_k=ram_k,
            bytecode_k=int(kv["bytecode_k"]),
            log_k_chunk=int(kv["log_k_chunk"]),
            lookups_ra_virtual_log_k_chunk=int(kv["lookups_ra_virtual_log_k_chunk"]),
        )
        rw_config = ReadWriteConfig(
            ram_rw_phase1_num_rounds=rw_phase1,
            ram_rw_phase2_num_rounds=rw_phase2,
            registers_rw_phase1_num_rounds=regs_rw_phase1,
            registers_rw_phase2_num_rounds=regs_rw_phase2,
        )

        # ---- Stage 1 inputs/proofs ----
        stage1_uni_poly_coeffs = _csv_fr(kv["stage1_uni_poly_coeffs"])
        stage1_uniskip_claim = Fr(int(kv["stage1_uniskip_claim"]))
        r1cs_input_evals = _csv_fr(kv["stage1_r1cs_input_evals"])

        stage1_num_rounds = 1 + (trace_len.bit_length() - 1)
        stage1_polys = []
        for j in range(stage1_num_rounds):
            stage1_polys.append(CompressedUniPoly(_csv_fr(kv[f"stage1_sumcheck_poly_{j}"])))

        t = Blake2bTranscript.new(b"Jolt")
        acc = VerifierOpeningAccumulator()
        acc.set_virtual_claim(VirtualPolynomial.UnivariateSkip, SumcheckId.SpartanOuter, stage1_uniskip_claim)
        for inp, v in zip(ALL_R1CS_INPUTS, r1cs_input_evals):
            acc.set_virtual_claim(inp, SumcheckId.SpartanOuter, v)

        stage1_uni_skip_proof = UniSkipFirstRoundProof(UniPoly(stage1_uni_poly_coeffs))
        stage1_sumcheck_proof = SumcheckInstanceProof(stage1_polys)
        key = UniformSpartanKey(trace_len)
        _ = verify_spartan_outer_stage1(
            stage1_uni_skip_proof,
            stage1_sumcheck_proof,
            key,
            trace_len,
            acc,
            t,
        )

        # ---- Stage 2 claim seeding ----
        stage2_uniskip_claim = Fr(int(kv["stage2_uniskip_claim"]))
        acc.set_virtual_claim(VirtualPolynomial.UnivariateSkip, SumcheckId.SpartanProductVirtualization, stage2_uniskip_claim)

        acc.set_virtual_claim(VirtualPolynomial.RamVal, SumcheckId.RamReadWriteChecking, Fr(int(kv["stage2_ramrw_val_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RamRa, SumcheckId.RamReadWriteChecking, Fr(int(kv["stage2_ramrw_ra_claim"])))
        acc.set_committed_claim(CommittedPolynomial.RamInc, SumcheckId.RamReadWriteChecking, Fr(int(kv["stage2_ramrw_raminc_claim"])))

        pv_factor_names = [
            VirtualPolynomial.LeftInstructionInput,
            VirtualPolynomial.RightInstructionInput,
            VirtualPolynomial.InstructionFlags_IsRdNotZero,
            VirtualPolynomial.OpFlags_WriteLookupOutputToRD,
            VirtualPolynomial.OpFlags_Jump,
            VirtualPolynomial.LookupOutput,
            VirtualPolynomial.InstructionFlags_Branch,
            VirtualPolynomial.NextIsNoop,
            VirtualPolynomial.OpFlags_VirtualInstruction,
        ]
        pv_factor_claims = _csv_fr(kv["stage2_pv_factor_claims"])
        self.assertEqual(len(pv_factor_claims), len(pv_factor_names))
        for name, v in zip(pv_factor_names, pv_factor_claims):
            acc.set_virtual_claim(name, SumcheckId.SpartanProductVirtualization, v)

        acc.set_virtual_claim(VirtualPolynomial.LookupOutput, SumcheckId.InstructionClaimReduction, Fr(int(kv["stage2_instr_lookup_output_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.LeftLookupOperand, SumcheckId.InstructionClaimReduction, Fr(int(kv["stage2_instr_left_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RightLookupOperand, SumcheckId.InstructionClaimReduction, Fr(int(kv["stage2_instr_right_claim"])))

        acc.set_virtual_claim(VirtualPolynomial.RamRa, SumcheckId.RamRafEvaluation, Fr(int(kv["stage2_raf_ra_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RamValFinal, SumcheckId.RamOutputCheck, Fr(int(kv["stage2_out_val_final_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RamValInit, SumcheckId.RamOutputCheck, Fr(int(kv["stage2_out_val_init_claim"])))

        # ---- Stage 2 proofs ----
        stage2_uni_poly_coeffs = _csv_fr(kv["stage2_uniskip_poly_coeffs"])
        stage2_num_rounds = (ram_k.bit_length() - 1) + (trace_len.bit_length() - 1)
        stage2_polys = []
        for j in range(stage2_num_rounds):
            stage2_polys.append(CompressedUniPoly(_csv_fr(kv[f"stage2_sumcheck_poly_{j}"])))

        stage2_uni_skip_proof = UniSkipFirstRoundProof(UniPoly(stage2_uni_poly_coeffs))
        stage2_sumcheck_proof = SumcheckInstanceProof(stage2_polys)
        _ = verify_stage2(
            stage2_uni_skip_proof,
            stage2_sumcheck_proof,
            trace_len,
            one_hot_params,
            rw_config,
            memory_layout,
            program_io,
            acc,
            t,
        )

        # ---- Stage 3 claim seeding + proof ----
        stage3_shift_claims = _csv_fr(kv["stage3_shift_claims"])
        stage3_instr_claims = _csv_fr(kv["stage3_instr_claims"])
        stage3_regs_claims = _csv_fr(kv["stage3_regs_claims"])
        self.assertEqual(len(stage3_shift_claims), 5)
        self.assertEqual(len(stage3_instr_claims), 8)
        self.assertEqual(len(stage3_regs_claims), 3)

        shift_names = [
            VirtualPolynomial.UnexpandedPC,
            VirtualPolynomial.PC,
            VirtualPolynomial.OpFlags_VirtualInstruction,
            VirtualPolynomial.OpFlags_IsFirstInSequence,
            VirtualPolynomial.InstructionFlags_IsNoop,
        ]
        for name, v in zip(shift_names, stage3_shift_claims):
            acc.set_virtual_claim(name, SumcheckId.SpartanShift, v)

        instr_names = [
            VirtualPolynomial.InstructionFlags_LeftOperandIsRs1Value,
            VirtualPolynomial.Rs1Value,
            VirtualPolynomial.InstructionFlags_LeftOperandIsPC,
            VirtualPolynomial.UnexpandedPC,
            VirtualPolynomial.InstructionFlags_RightOperandIsRs2Value,
            VirtualPolynomial.Rs2Value,
            VirtualPolynomial.InstructionFlags_RightOperandIsImm,
            VirtualPolynomial.Imm,
        ]
        for name, v in zip(instr_names, stage3_instr_claims):
            acc.set_virtual_claim(name, SumcheckId.InstructionInputVirtualization, v)

        regs_names = [VirtualPolynomial.RdWriteValue, VirtualPolynomial.Rs1Value, VirtualPolynomial.Rs2Value]
        for name, v in zip(regs_names, stage3_regs_claims):
            acc.set_virtual_claim(name, SumcheckId.RegistersClaimReduction, v)

        stage3_num_rounds = trace_len.bit_length() - 1
        stage3_polys = []
        for j in range(stage3_num_rounds):
            stage3_polys.append(CompressedUniPoly(_csv_fr(kv[f"stage3_sumcheck_poly_{j}"])))
        stage3_sumcheck_proof = SumcheckInstanceProof(stage3_polys)
        _ = verify_stage3(stage3_sumcheck_proof, trace_len, acc, t)

        # ---- Stage 4 claim seeding + proof ----
        if has_untrusted:
            acc.set_untrusted_advice_claim(SumcheckId.RamValEvaluation, Fr(int(kv["untrusted_advice_eval"])))
        if has_trusted:
            acc.set_trusted_advice_claim(SumcheckId.RamValEvaluation, Fr(int(kv["trusted_advice_eval"])))

        acc.set_virtual_claim(VirtualPolynomial.RegistersVal, SumcheckId.RegistersReadWriteChecking, Fr(int(kv["stage4_regs_val_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.Rs1Ra, SumcheckId.RegistersReadWriteChecking, Fr(int(kv["stage4_regs_rs1_ra_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.Rs2Ra, SumcheckId.RegistersReadWriteChecking, Fr(int(kv["stage4_regs_rs2_ra_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RdWa, SumcheckId.RegistersReadWriteChecking, Fr(int(kv["stage4_regs_rd_wa_claim"])))
        acc.set_committed_claim(CommittedPolynomial.RdInc, SumcheckId.RegistersReadWriteChecking, Fr(int(kv["stage4_regs_rdinc_claim"])))

        acc.set_virtual_claim(VirtualPolynomial.RamRa, SumcheckId.RamValEvaluation, Fr(int(kv["stage4_ram_val_eval_wa_claim"])))
        acc.set_committed_claim(CommittedPolynomial.RamInc, SumcheckId.RamValEvaluation, Fr(int(kv["stage4_ram_val_eval_inc_claim"])))

        acc.set_committed_claim(CommittedPolynomial.RamInc, SumcheckId.RamValFinalEvaluation, Fr(int(kv["stage4_val_final_inc_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RamRa, SumcheckId.RamValFinalEvaluation, Fr(int(kv["stage4_val_final_wa_claim"])))

        stage4_num_rounds = (REGISTER_COUNT.bit_length() - 1) + (trace_len.bit_length() - 1)
        stage4_polys = []
        for j in range(stage4_num_rounds):
            stage4_polys.append(CompressedUniPoly(_csv_fr(kv[f"stage4_sumcheck_poly_{j}"])))
        stage4_sumcheck_proof = SumcheckInstanceProof(stage4_polys)

        _ = verify_stage4(
            stage4_sumcheck_proof,
            trace_len,
            one_hot_params,
            rw_config,
            ram_preprocessing,
            program_io,
            has_untrusted,
            has_trusted,
            acc,
            t,
        )

        # ---- Stage 5 claim seeding + proof ----
        table_flags = _csv_fr(kv["stage5_ir_table_flag_claims"])
        instr_ra_claims = _csv_fr(kv["stage5_ir_instruction_ra_claims"])
        for i, v in enumerate(table_flags):
            acc.set_virtual_claim_i(VirtualPolynomial.LookupTableFlag, i, SumcheckId.InstructionReadRaf, v)
        for i, v in enumerate(instr_ra_claims):
            acc.set_virtual_claim_i(VirtualPolynomial.InstructionRa, i, SumcheckId.InstructionReadRaf, v)
        acc.set_virtual_claim(VirtualPolynomial.InstructionRafFlag, SumcheckId.InstructionReadRaf, Fr(int(kv["stage5_ir_raf_flag_claim"])))

        acc.set_virtual_claim(VirtualPolynomial.RamRa, SumcheckId.RamRaClaimReduction, Fr(int(kv["stage5_ram_ra_reduced_claim"])))
        acc.set_committed_claim(CommittedPolynomial.RdInc, SumcheckId.RegistersValEvaluation, Fr(int(kv["stage5_regs_rdinc_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RdWa, SumcheckId.RegistersValEvaluation, Fr(int(kv["stage5_regs_rdwa_claim"])))

        stage5_num_rounds = 128 + (trace_len.bit_length() - 1)
        stage5_polys = []
        for j in range(stage5_num_rounds):
            stage5_polys.append(CompressedUniPoly(_csv_fr(kv[f"stage5_sumcheck_poly_{j}"])))
        stage5_sumcheck_proof = SumcheckInstanceProof(stage5_polys)

        r_sumcheck_stage5 = verify_stage5(stage5_sumcheck_proof, trace_len, one_hot_params, acc, t)
        oracle_r5 = _csv_fr(kv["stage5_r_sumcheck"])
        self.assertEqual(r_sumcheck_stage5, oracle_r5)

        # ---- Stage 6 claim seeding + proof ----
        bytecode_ra_claims = _csv_fr(kv["stage6_bytecode_ra_claims"])
        for i, v in enumerate(bytecode_ra_claims):
            acc.set_committed_claim_i(CommittedPolynomial.BytecodeRa, i, SumcheckId.BytecodeReadRaf, v)

        booleanity_claims = _csv_fr(kv["stage6_booleanity_claims"])
        self.assertEqual(len(booleanity_claims), one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d)
        off = 0
        for i in range(one_hot_params.instruction_d):
            acc.set_committed_claim_i(CommittedPolynomial.InstructionRa, i, SumcheckId.Booleanity, booleanity_claims[off + i])
        off += one_hot_params.instruction_d
        for i in range(one_hot_params.bytecode_d):
            acc.set_committed_claim_i(CommittedPolynomial.BytecodeRa, i, SumcheckId.Booleanity, booleanity_claims[off + i])
        off += one_hot_params.bytecode_d
        for i in range(one_hot_params.ram_d):
            acc.set_committed_claim_i(CommittedPolynomial.RamRa, i, SumcheckId.Booleanity, booleanity_claims[off + i])

        acc.set_virtual_claim(
            VirtualPolynomial.RamHammingWeight,
            SumcheckId.RamHammingBooleanity,
            Fr(int(kv["stage6_hamming_weight_claim"])),
        )

        ram_ra_virtual_claims = _csv_fr(kv["stage6_ram_ra_virtual_claims"])
        for i, v in enumerate(ram_ra_virtual_claims):
            acc.set_committed_claim_i(CommittedPolynomial.RamRa, i, SumcheckId.RamRaVirtualization, v)

        instr_ra_virtual_committed = _csv_fr(kv["stage6_instruction_ra_virtual_committed_claims"])
        for i, v in enumerate(instr_ra_virtual_committed):
            acc.set_committed_claim_i(CommittedPolynomial.InstructionRa, i, SumcheckId.InstructionRaVirtualization, v)

        acc.set_committed_claim(CommittedPolynomial.RamInc, SumcheckId.IncClaimReduction, Fr(int(kv["stage6_inc_raminc_claim"])))
        acc.set_committed_claim(CommittedPolynomial.RdInc, SumcheckId.IncClaimReduction, Fr(int(kv["stage6_inc_rdinc_claim"])))

        stage6_num_rounds = int(kv["log_k_chunk"]) + (trace_len.bit_length() - 1)
        stage6_polys = []
        for j in range(stage6_num_rounds):
            stage6_polys.append(CompressedUniPoly(_csv_fr(kv[f"stage6_sumcheck_poly_{j}"])))
        stage6_sumcheck_proof = SumcheckInstanceProof(stage6_polys)

        bytecode_preprocessing = BytecodePreprocessing.preprocess([])
        r_sumcheck_stage6 = verify_stage6(
            stage6_sumcheck_proof,
            trace_len,
            one_hot_params,
            bytecode_preprocessing,
            acc,
            t,
            has_untrusted_advice_commitment=False,
            has_trusted_advice_commitment=False,
        )
        oracle_r6 = _csv_fr(kv["stage6_r_sumcheck"])
        self.assertEqual(r_sumcheck_stage6, oracle_r6)
        self.assertEqual(t.state_hex(), kv["final_state"])

