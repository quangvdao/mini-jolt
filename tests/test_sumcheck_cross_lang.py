import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow importing local verifier modules

from field import Fr  # BN254 Fr field elements
from polynomials import CompressedUniPoly  # compressed univariate messages
from sumchecks import SumcheckInstanceProof  # sumcheck verifier proof container
from tests.oracle import run_rust_oracle  # rust oracle harness
from transcript import Blake2bTranscript  # transcript for parity


def run_python_sumcheck_script(script_text):  # Execute the same script as the rust oracle and print trace.
    t = None
    claim = None
    degree_bound = None
    polys = []
    out_lines = []
    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        op = parts[0]
        if op == "new":
            t = Blake2bTranscript.new(parts[1].encode())
            out_lines.append("state=" + t.state_hex())
        elif op == "claim_fr":
            claim = Fr(int(parts[1]))
        elif op == "degree_bound":
            degree_bound = int(parts[1])
        elif op == "poly_fr":
            coeffs = [Fr(int(x)) for x in parts[1:]]
            polys.append(CompressedUniPoly(coeffs))
        elif op == "verify":
            _ = degree_bound  # kept for script parity with rust oracle even if unused here
            proof = SumcheckInstanceProof(polys)
            e = claim
            rs = []
            for i, poly in enumerate(proof.compressed_polys):
                t.append_scalars(b"sumcheck_poly", poly.coeffs_except_linear_term)
                r_i = t.challenge_scalar_optimized()
                rs.append(r_i)
                e = poly.eval_from_hint(e, r_i)
                out_lines.append(
                    f"round={i} r={int(r_i)} e={int(e)} state={t.state_hex()}"
                )
            out_lines.append(f"output_claim={int(e)}")
        else:
            raise ValueError(f"unknown op: {op}")
    return "\n".join(out_lines).strip()


class SumcheckCrossLangTests(unittest.TestCase):  # Python↔Rust parity tests for sumcheck verification.
    def test_sumcheck_verify_matches_rust_oracle(self):  # Parity with `sumcheck_verify_blake2b` oracle mode.
        script = "\n".join(
            [
                "new Jolt",
                "claim_fr 123",
                "degree_bound 3",
                "poly_fr 9 1",
                "poly_fr 2 3 4",
                "verify",
            ]
        )
        rust_out = run_rust_oracle("sumcheck_verify_blake2b", script).strip()
        py_out = run_python_sumcheck_script(script).strip()
        self.assertEqual(py_out, rust_out)


if __name__ == "__main__":
    unittest.main()
