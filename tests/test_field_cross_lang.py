import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from field import Fq, Fr

from tests.oracle import limbs_csv_to_int, run_rust_oracle


class CrossLangFieldTests(unittest.TestCase):
    def check_field(self, field_name, cls):
        p = cls.MODULUS
        rng = random.Random(7)
        cases = [(0, 1, 0), (1, 1, 1), (p - 1, p - 1, 2), (42, p - 1, 17)]
        for _ in range(40):
            a = rng.randrange(0, p)
            b = rng.randrange(1, p)
            e = rng.randrange(0, 80)
            cases.append((a, b, e))
        payload = "".join(f"{a} {b} {e}\n" for a, b, e in cases)
        out = run_rust_oracle(field_name, payload).strip()
        lines = [] if not out else out.splitlines()
        rust_rows = [tuple(limbs_csv_to_int(part) for part in line.split("|")) for line in lines]
        self.assertEqual(len(rust_rows), len(cases))
        for (a, b, e), (r_add, r_sub, r_mul, r_div, r_pow) in zip(cases, rust_rows):
            x, y = cls(a), cls(b)
            self.assertEqual(int(x + y), r_add)
            self.assertEqual(int(x - y), r_sub)
            self.assertEqual(int(x * y), r_mul)
            self.assertEqual(int(x / y), r_div)
            self.assertEqual(int(x**e), r_pow)

    def test_fq_matches_rust(self):
        self.check_field("fq", Fq)

    def test_fr_matches_rust(self):
        self.check_field("fr", Fr)


if __name__ == "__main__":
    unittest.main()
