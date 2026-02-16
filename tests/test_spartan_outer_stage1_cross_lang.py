import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow importing local verifier modules

from field import Fr  # BN254 Fr field elements
from openings import SumcheckId, VerifierOpeningAccumulator, VirtualPolynomial  # opening accumulator types
from polynomials import CompressedUniPoly, UniPoly  # polynomial containers
from r1cs import ALL_R1CS_INPUTS, UniformSpartanKey  # Spartan key + canonical input order
from jolt_verifier import verify_spartan_outer_stage1  # Stage-1 verifier entrypoint (final source of truth)
from sumchecks import SumcheckInstanceProof, UniSkipFirstRoundProof  # proof containers
from tests.oracle import run_rust_oracle  # rust oracle harness
from transcript import Blake2bTranscript  # transcript for parity

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

class SpartanOuterStage1CrossLangTests(unittest.TestCase):  # Python↔Rust parity tests for Spartan outer Stage 1.
    def test_spartan_outer_stage1_matches_rust_oracle(self):  # Parity with `spartan_outer_stage1_blake2b` oracle mode.
        kv = _parse_kv_lines(run_rust_oracle("spartan_outer_stage1_blake2b", "").strip())
        trace_len = int(kv["trace_len"])
        uni_poly_coeffs = [Fr(int(x)) for x in kv["uni_poly_coeffs"].split(",") if x]
        uniskip_claim = Fr(int(kv["uniskip_claim"]))
        r1cs_input_evals = [Fr(int(x)) for x in kv["r1cs_input_evals"].split(",") if x]
        polys = []
        num_rounds = 1 + (trace_len.bit_length() - 1)
        for j in range(num_rounds):
            coeffs = [Fr(int(x)) for x in kv[f"sumcheck_poly_{j}"].split(",") if x]
            polys.append(CompressedUniPoly(coeffs))
        t = Blake2bTranscript.new(b"Jolt")
        acc = VerifierOpeningAccumulator()
        acc.set_virtual_claim(VirtualPolynomial.UnivariateSkip, SumcheckId.SpartanOuter, uniskip_claim)
        for inp, v in zip(ALL_R1CS_INPUTS, r1cs_input_evals):
            acc.set_virtual_claim(inp, SumcheckId.SpartanOuter, v)
        uni_skip_proof = UniSkipFirstRoundProof(UniPoly(uni_poly_coeffs))
        outer_sumcheck_proof = SumcheckInstanceProof(polys)
        key = UniformSpartanKey(trace_len)
        r_sumcheck = verify_spartan_outer_stage1(
            uni_skip_proof,
            outer_sumcheck_proof,
            key,
            trace_len,
            acc,
            t,
        )
        want_r0 = int(Fr(int(kv["r0"])))
        got_r0 = int(acc.get_virtual_polynomial_opening(VirtualPolynomial.UnivariateSkip, SumcheckId.SpartanOuter)[0][0])
        self.assertEqual(got_r0, want_r0)
        want_r_sumcheck = [int(Fr(int(x))) for x in kv["r_sumcheck"].split(",") if x]
        got_r_sumcheck = [int(x) for x in r_sumcheck]
        self.assertEqual(got_r_sumcheck, want_r_sumcheck)
        self.assertEqual(t.state_hex(), kv["final_state"])

if __name__ == "__main__":
    unittest.main()

