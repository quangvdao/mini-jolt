import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from curve import G1, G2, GT, Fr, add, mul, multi_pairing, neg, pairing

from tests.oracle import parse_g1_csv, parse_g2_csv, run_rust_oracle


def g1_to_tuple(p):
    return None if p is None else (int(p[0]), int(p[1]))


def g2_to_tuple(p):
    if p is None:
        return None
    return (int(p[0].c[0]), int(p[0].c[1]), int(p[1].c[0]), int(p[1].c[1]))


def run_curve_oracle(cases):
    payload = "".join(f"{a} {b}\n" for a, b in cases)
    out = run_rust_oracle("curve", payload).strip()
    lines = [] if not out else out.splitlines()
    rows = []
    for line in lines:
        g1a, g1b, g1ab, g2a, g2b, g2ab, pair_rel, mpe_is_one = line.split("\t")
        rows.append((parse_g1_csv(g1a), parse_g1_csv(g1b), parse_g1_csv(g1ab), parse_g2_csv(g2a), parse_g2_csv(g2b), parse_g2_csv(g2ab), pair_rel == "1", mpe_is_one == "1"))
    return rows


class CrossLangCurveTests(unittest.TestCase):
    def test_curve_matches_rust(self):
        r = Fr.MODULUS
        rng = random.Random(11)
        cases = [(0, 0), (1, 1), (2, 3), (7, 9), (r - 1, r - 2)]
        for _ in range(16):
            cases.append((rng.randrange(0, r), rng.randrange(0, r)))
        rust_rows = run_curve_oracle(cases)
        self.assertEqual(len(rust_rows), len(cases))
        for (a, b), (rg1a, rg1b, rg1ab, rg2a, rg2b, rg2ab, r_pair_rel, r_mpe_is_one) in zip(cases, rust_rows):
            p_a, p_b = mul(G1, a), mul(G1, b)
            q_a, q_b = mul(G2, a), mul(G2, b)
            self.assertEqual(g1_to_tuple(p_a), rg1a)
            self.assertEqual(g1_to_tuple(p_b), rg1b)
            self.assertEqual(g1_to_tuple(add(p_a, p_b)), rg1ab)
            self.assertEqual(g2_to_tuple(q_a), rg2a)
            self.assertEqual(g2_to_tuple(q_b), rg2b)
            self.assertEqual(g2_to_tuple(add(q_a, q_b)), rg2ab)
            self.assertEqual(pairing(q_b, p_a) == pairing(G2, mul(G1, (a * b) % r)), r_pair_rel)
            self.assertEqual(multi_pairing([(q_b, p_a), (neg(q_b), p_a)]) == GT.one(), r_mpe_is_one)


if __name__ == "__main__":
    unittest.main()
