import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow importing local verifier modules

from field import Fr  # BN254 Fr field elements
from polynomials import CompressedUniPoly  # sumcheck message poly type
from sumchecks import SumcheckInstanceProof, SumcheckVerifyError  # verifier + error type
from transcript import Blake2bTranscript  # transcript for challenge sampling


class SumcheckTests(unittest.TestCase):  # Tests for the sumcheck verifier template.
    def test_sumcheck_instance_proof_verify_smoke(self):  # Smoke test: runs transcript-coupled verification.
        # This tests the verifier-side mechanics: transcript coupling + eval_from_hint chain.
        t = Blake2bTranscript.new(b"Jolt")
        claim = Fr(123)
        polys = [
            CompressedUniPoly([Fr(9), Fr(1)]),  # degree=2 (c0,c2)
            CompressedUniPoly([Fr(2), Fr(3), Fr(4)]),  # degree=3 (c0,c2,c3)
        ]
        proof = SumcheckInstanceProof(polys)
        out_claim, r = proof.verify(claim, num_rounds=2, degree_bound=3, transcript=t)
        self.assertEqual(len(r), 2)
        self.assertIsInstance(out_claim, Fr)

    def test_degree_bound_exceeded(self):  # Enforces degree bound check.
        t = Blake2bTranscript.new(b"Jolt")
        claim = Fr(0)
        poly = CompressedUniPoly([Fr(1), Fr(2), Fr(3), Fr(4)])  # degree=4
        proof = SumcheckInstanceProof([poly])
        with self.assertRaises(SumcheckVerifyError):
            proof.verify(claim, num_rounds=1, degree_bound=3, transcript=t)


if __name__ == "__main__":
    unittest.main()
