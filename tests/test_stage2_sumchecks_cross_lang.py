import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow importing local verifier modules

from field import Fr  # BN254 Fr field elements
from openings import CommittedPolynomial, SumcheckId, VerifierOpeningAccumulator, VirtualPolynomial  # opening accumulator types
from polynomials import CompressedUniPoly, UniPoly  # polynomial containers
from r1cs import ALL_R1CS_INPUTS, UniformSpartanKey  # Spartan key + canonical input order
from jolt_verifier import verify_spartan_outer_stage1, verify_stage2  # stage orchestration (final source of truth)
from sumchecks import SumcheckInstanceProof, UniSkipFirstRoundProof  # proof containers
from tests.oracle import run_rust_oracle  # rust oracle harness
from transcript import Blake2bTranscript  # transcript for parity
from zkvm_types import JoltDevice, MemoryLayout, OneHotParams, ReadWriteConfig  # Stage-2 public types


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


class Stage2SumchecksCrossLangTests(unittest.TestCase):  # Python↔Rust full-proof parity: Stage1 then Stage2.
    def test_stage2_sumchecks_match_rust_oracle(self):
        kv = _parse_kv_lines(run_rust_oracle("stage2_sumchecks_blake2b", "").strip())

        trace_len = int(kv["trace_len"])
        ram_k = int(kv["ram_k"])
        rw_phase1 = int(kv["rw_phase1"])
        rw_phase2 = int(kv["rw_phase2"])

        memory_layout = MemoryLayout(
            input_start=int(kv["mem_input_start"]),
            output_start=int(kv["mem_output_start"]),
            panic=int(kv["mem_panic"]),
            termination=int(kv["mem_termination"]),
        )
        program_io = JoltDevice(memory_layout=memory_layout, inputs=b"", outputs=b"", panic_flag=False)

        # ---- Stage 1 inputs/proofs ----
        stage1_uni_poly_coeffs = [Fr(int(x)) for x in kv["stage1_uni_poly_coeffs"].split(",") if x]
        stage1_uniskip_claim = Fr(int(kv["stage1_uniskip_claim"]))
        r1cs_input_evals = [Fr(int(x)) for x in kv["stage1_r1cs_input_evals"].split(",") if x]

        stage1_num_rounds = 1 + (trace_len.bit_length() - 1)
        stage1_polys = []
        for j in range(stage1_num_rounds):
            coeffs = [Fr(int(x)) for x in kv[f"stage1_sumcheck_poly_{j}"].split(",") if x]
            stage1_polys.append(CompressedUniPoly(coeffs))

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
        pv_factor_claims = [Fr(int(x)) for x in kv["stage2_pv_factor_claims"].split(",") if x]
        self.assertEqual(len(pv_factor_claims), len(pv_factor_names))
        for name, v in zip(pv_factor_names, pv_factor_claims):
            acc.set_virtual_claim(name, SumcheckId.SpartanProductVirtualization, v)

        acc.set_virtual_claim(
            VirtualPolynomial.LookupOutput,
            SumcheckId.InstructionClaimReduction,
            Fr(int(kv["stage2_instr_lookup_output_claim"])),
        )
        acc.set_virtual_claim(
            VirtualPolynomial.LeftLookupOperand,
            SumcheckId.InstructionClaimReduction,
            Fr(int(kv["stage2_instr_left_claim"])),
        )
        acc.set_virtual_claim(
            VirtualPolynomial.RightLookupOperand,
            SumcheckId.InstructionClaimReduction,
            Fr(int(kv["stage2_instr_right_claim"])),
        )

        acc.set_virtual_claim(VirtualPolynomial.RamRa, SumcheckId.RamRafEvaluation, Fr(int(kv["stage2_raf_ra_claim"])))

        acc.set_virtual_claim(VirtualPolynomial.RamValFinal, SumcheckId.RamOutputCheck, Fr(int(kv["stage2_out_val_final_claim"])))
        acc.set_virtual_claim(VirtualPolynomial.RamValInit, SumcheckId.RamOutputCheck, Fr(int(kv["stage2_out_val_init_claim"])))

        # ---- Stage 2 proofs ----
        stage2_uni_poly_coeffs = [Fr(int(x)) for x in kv["stage2_uniskip_poly_coeffs"].split(",") if x]
        stage2_uni_skip_proof = UniSkipFirstRoundProof(UniPoly(stage2_uni_poly_coeffs))

        stage2_num_rounds = (ram_k.bit_length() - 1) + (trace_len.bit_length() - 1)
        stage2_polys = []
        for j in range(stage2_num_rounds):
            coeffs = [Fr(int(x)) for x in kv[f"stage2_sumcheck_poly_{j}"].split(",") if x]
            stage2_polys.append(CompressedUniPoly(coeffs))
        stage2_sumcheck_proof = SumcheckInstanceProof(stage2_polys)

        one_hot = OneHotParams(ram_k=ram_k)
        rw_cfg = ReadWriteConfig(ram_rw_phase1_num_rounds=rw_phase1, ram_rw_phase2_num_rounds=rw_phase2)

        r2 = verify_stage2(
            stage2_uni_skip_proof,
            stage2_sumcheck_proof,
            trace_len,
            one_hot,
            rw_cfg,
            memory_layout,
            program_io,
            acc,
            t,
        )

        want_stage2_r_sumcheck = [int(Fr(int(x))) for x in kv["stage2_r_sumcheck"].split(",") if x]
        got_stage2_r_sumcheck = [int(x) for x in r2]
        self.assertEqual(got_stage2_r_sumcheck, want_stage2_r_sumcheck)

        want_stage2_r0 = int(Fr(int(kv["stage2_r0"])))
        got_stage2_r0 = int(
            acc.get_virtual_polynomial_opening(VirtualPolynomial.UnivariateSkip, SumcheckId.SpartanProductVirtualization)[0][0]
        )
        self.assertEqual(got_stage2_r0, want_stage2_r0)

        self.assertEqual(t.state_hex(), kv["final_state"])


if __name__ == "__main__":
    unittest.main()

