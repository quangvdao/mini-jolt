import pathlib  # locate repo root
import sys  # import from repo root
import unittest  # test harness

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow `import dory`, `import curve`, etc

from curve import Fq, Fq2, GT  # group/field types for parsed oracle values
from field import Fr  # BN254 scalar field
from tests.oracle import limbs_csv_to_int, parse_g1_csv, parse_g2_csv, run_rust_oracle  # rust oracle helpers
from transcript import Blake2bTranscript  # Jolt transcript

import dory  # Dory PCS verifier under test

def _g1_from_csv(s):  # Parse a G1 CSV string into curve.py point type.
    t = parse_g1_csv(s)
    return None if t is None else (Fq(t[0]), Fq(t[1]))

def _g2_from_csv(s):  # Parse a G2 CSV string into curve.py point type.
    t = parse_g2_csv(s)
    return None if t is None else (Fq2([t[0], t[1]]), Fq2([t[2], t[3]]))

def _gt_from_poly_csv(s):  # Parse GT as 12 Fq coeffs in polynomial basis (c0..c11 separated by '/').
    parts = s.split("/")
    if len(parts) != 12:
        raise ValueError("expected 12 GT coefficients")
    coeffs = [limbs_csv_to_int(p) for p in parts]
    return GT(coeffs)

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

class DoryVerifyCrossLangTests(unittest.TestCase):  # Cross-language Dory verifier tests.
    def test_dory_verify_accepts_oracle_proof(self):  # Python verifier accepts Rust oracle proof.
        kv = _parse_kv_lines(run_rust_oracle("dory_pcs_eval_blake2b", "").strip())
        nu = int(kv["nu"])
        sigma = int(kv["sigma"])
        opening_point_be = [Fr(int(x)) for x in kv["opening_point_be"].split(",") if x]
        evaluation = Fr(int(kv["evaluation"]))
        commitment = _gt_from_poly_csv(kv["commitment"])
        vmv = dory.VMVMessage(_gt_from_poly_csv(kv["vmv_c"]), _gt_from_poly_csv(kv["vmv_d2"]), _g1_from_csv(kv["vmv_e1"]))
        first_messages = []
        second_messages = []
        for i in range(sigma):
            first_messages.append(
                dory.FirstReduceMessage(
                    _gt_from_poly_csv(kv[f"first_{i}_d1_left"]),
                    _gt_from_poly_csv(kv[f"first_{i}_d1_right"]),
                    _gt_from_poly_csv(kv[f"first_{i}_d2_left"]),
                    _gt_from_poly_csv(kv[f"first_{i}_d2_right"]),
                    _g1_from_csv(kv[f"first_{i}_e1_beta"]),
                    _g2_from_csv(kv[f"first_{i}_e2_beta"]),
                )
            )
            second_messages.append(
                dory.SecondReduceMessage(
                    _gt_from_poly_csv(kv[f"second_{i}_c_plus"]),
                    _gt_from_poly_csv(kv[f"second_{i}_c_minus"]),
                    _g1_from_csv(kv[f"second_{i}_e1_plus"]),
                    _g1_from_csv(kv[f"second_{i}_e1_minus"]),
                    _g2_from_csv(kv[f"second_{i}_e2_plus"]),
                    _g2_from_csv(kv[f"second_{i}_e2_minus"]),
                )
            )
        final_msg = dory.ScalarProductMessage(_g1_from_csv(kv["final_e1"]), _g2_from_csv(kv["final_e2"]))
        proof = dory.DoryProof(vmv, first_messages, second_messages, final_msg, nu, sigma)
        chi = [_gt_from_poly_csv(kv[f"chi_{k}"]) for k in range(sigma + 1)]
        delta_1l = [_gt_from_poly_csv(kv[f"delta_1l_{k}"]) for k in range(sigma + 1)]
        delta_1r = [_gt_from_poly_csv(kv[f"delta_1r_{k}"]) for k in range(sigma + 1)]
        delta_2l = [_gt_from_poly_csv(kv[f"delta_2l_{k}"]) for k in range(sigma + 1)]
        delta_2r = [_gt_from_poly_csv(kv[f"delta_2r_{k}"]) for k in range(sigma + 1)]
        setup = dory.DoryVerifierSetup(
            delta_1l,
            delta_1r,
            delta_2l,
            delta_2r,
            chi,
            _g1_from_csv(kv["g1_0"]),
            _g2_from_csv(kv["g2_0"]),
            _g1_from_csv(kv["h1"]),
            _g2_from_csv(kv["h2"]),
            _gt_from_poly_csv(kv["ht"]),
        )
        serde_blocks_len = int(kv["serde_blocks_len"])
        serde_blocks = [bytes.fromhex(kv[f"serde_block_{i}"]) for i in range(serde_blocks_len)]
        t = Blake2bTranscript.new(b"Jolt")
        dory.verify(
            proof,
            setup,
            t,
            opening_point_be,
            evaluation,
            commitment,
            dory_layout=kv["dory_layout"],
            log_T=int(kv["log_T"]),
            serde_blocks=serde_blocks,
        )

    def test_dory_verify_rejects_wrong_evaluation(self):  # Wrong evaluation should fail final check.
        kv = _parse_kv_lines(run_rust_oracle("dory_pcs_eval_blake2b", "").strip())
        nu = int(kv["nu"])
        sigma = int(kv["sigma"])
        opening_point_be = [Fr(int(x)) for x in kv["opening_point_be"].split(",") if x]
        evaluation = Fr(int(kv["evaluation"]))
        commitment = _gt_from_poly_csv(kv["commitment"])
        vmv = dory.VMVMessage(_gt_from_poly_csv(kv["vmv_c"]), _gt_from_poly_csv(kv["vmv_d2"]), _g1_from_csv(kv["vmv_e1"]))
        first_messages = []
        second_messages = []
        for i in range(sigma):
            first_messages.append(
                dory.FirstReduceMessage(
                    _gt_from_poly_csv(kv[f"first_{i}_d1_left"]),
                    _gt_from_poly_csv(kv[f"first_{i}_d1_right"]),
                    _gt_from_poly_csv(kv[f"first_{i}_d2_left"]),
                    _gt_from_poly_csv(kv[f"first_{i}_d2_right"]),
                    _g1_from_csv(kv[f"first_{i}_e1_beta"]),
                    _g2_from_csv(kv[f"first_{i}_e2_beta"]),
                )
            )
            second_messages.append(
                dory.SecondReduceMessage(
                    _gt_from_poly_csv(kv[f"second_{i}_c_plus"]),
                    _gt_from_poly_csv(kv[f"second_{i}_c_minus"]),
                    _g1_from_csv(kv[f"second_{i}_e1_plus"]),
                    _g1_from_csv(kv[f"second_{i}_e1_minus"]),
                    _g2_from_csv(kv[f"second_{i}_e2_plus"]),
                    _g2_from_csv(kv[f"second_{i}_e2_minus"]),
                )
            )
        final_msg = dory.ScalarProductMessage(_g1_from_csv(kv["final_e1"]), _g2_from_csv(kv["final_e2"]))
        proof = dory.DoryProof(vmv, first_messages, second_messages, final_msg, nu, sigma)
        chi = [_gt_from_poly_csv(kv[f"chi_{k}"]) for k in range(sigma + 1)]
        delta_1l = [_gt_from_poly_csv(kv[f"delta_1l_{k}"]) for k in range(sigma + 1)]
        delta_1r = [_gt_from_poly_csv(kv[f"delta_1r_{k}"]) for k in range(sigma + 1)]
        delta_2l = [_gt_from_poly_csv(kv[f"delta_2l_{k}"]) for k in range(sigma + 1)]
        delta_2r = [_gt_from_poly_csv(kv[f"delta_2r_{k}"]) for k in range(sigma + 1)]
        setup = dory.DoryVerifierSetup(
            delta_1l,
            delta_1r,
            delta_2l,
            delta_2r,
            chi,
            _g1_from_csv(kv["g1_0"]),
            _g2_from_csv(kv["g2_0"]),
            _g1_from_csv(kv["h1"]),
            _g2_from_csv(kv["h2"]),
            _gt_from_poly_csv(kv["ht"]),
        )
        serde_blocks_len = int(kv["serde_blocks_len"])
        serde_blocks = [bytes.fromhex(kv[f"serde_block_{i}"]) for i in range(serde_blocks_len)]
        t = Blake2bTranscript.new(b"Jolt")
        with self.assertRaises(dory.DoryVerifyError):
            dory.verify(
                proof,
                setup,
                t,
                opening_point_be,
                evaluation + Fr.one(),
                commitment,
                dory_layout=kv["dory_layout"],
                log_T=int(kv["log_T"]),
                serde_blocks=serde_blocks,
            )

if __name__ == "__main__":  # unittest entrypoint
    unittest.main()

