import pathlib  # locate repo root
import random  # deterministic PRNG for test cases
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow `import field`, `import polynomials`

from field import Fr  # BN254 Fr field elements
from polynomials import (
    CompressedUniPoly,  # compressed univariate for sumcheck verifier
    EqPlusOnePolynomial,  # eq+1 MLE for verifier utilities
    EqPolynomial,  # equality polynomial utilities
    IdentityPolynomial,  # identity polynomial over bit-vectors
    UniPoly,  # univariate polynomial (test helper)
    UnmapRamAddressPolynomial,  # RAM address mapping polynomial
)  # local polynomial module


class PolynomialTests(unittest.TestCase):  # Tests for verifier-minimal polynomial utilities.
    def test_compressed_unipoly_eval_from_hint_matches_decompress(self):  # eval_from_hint == decompress+evaluate.
        rng = random.Random(0)
        for deg in range(1, 6):
            for _ in range(20):
                coeffs = [Fr(rng.randrange(0, Fr.MODULUS)) for _ in range(deg + 1)]
                f = UniPoly(coeffs)
                hint = f.evaluate(0) + f.evaluate(1)
                coeffs_except_linear = [coeffs[0]] + coeffs[2:]
                c = CompressedUniPoly(coeffs_except_linear)
                x = Fr(rng.randrange(0, Fr.MODULUS))
                self.assertEqual(c.eval_from_hint(hint, x), f.evaluate(x))
                self.assertEqual(c.decompress(hint).evaluate(x), f.evaluate(x))

    def test_eq_polynomial_evals_big_endian_indexing(self):  # EqPolynomial.evals MSB-first index mapping.
        a = Fr(3)
        b = Fr(5)
        w = EqPolynomial.evals([a, b])
        self.assertEqual(len(w), 4)
        self.assertEqual(w[0], (Fr.one() - a) * (Fr.one() - b))  # 00
        self.assertEqual(w[1], (Fr.one() - a) * b)  # 01
        self.assertEqual(w[2], a * (Fr.one() - b))  # 10
        self.assertEqual(w[3], a * b)  # 11

    def test_eq_polynomial_mle_matches_table_lookup(self):  # mle(r, boolean_vertex) == evals(r)[index].
        r = [Fr(7), Fr(9), Fr(11)]
        table = EqPolynomial.evals(r)
        for idx in range(8):
            bits = [(idx >> (2 - i)) & 1 for i in range(3)]
            y = [Fr.one() if b else Fr.zero() for b in bits]
            self.assertEqual(EqPolynomial.mle(r, y), table[idx])

    def test_eq_plus_one_basic(self):  # eq+1 accepts y=x+1 for x != all-ones.
        l = 4
        for x_int in range(0, (1 << l) - 1):
            x_bits = [
                Fr.one() if (x_int >> (l - 1 - i)) & 1 else Fr.zero() for i in range(l)
            ]
            y_int = x_int + 1
            y_bits = [
                Fr.one() if (y_int >> (l - 1 - i)) & 1 else Fr.zero() for i in range(l)
            ]
            self.assertEqual(EqPlusOnePolynomial(x_bits).evaluate(y_bits), Fr.one())
        x_bits = [Fr.one() for _ in range(l)]
        y_bits = [Fr.zero() for _ in range(l)]
        self.assertEqual(EqPlusOnePolynomial(x_bits).evaluate(y_bits), Fr.zero())

    def test_identity_and_unmap_ram_address(self):  # Identity and UnmapRamAddress evaluation sanity.
        r = [Fr.zero(), Fr.one(), Fr.zero(), Fr.one()]  # 0101 = 5
        self.assertEqual(IdentityPolynomial(4).evaluate(r), Fr(5))
        self.assertEqual(
            UnmapRamAddressPolynomial(4, 0x80000000).evaluate(r),
            Fr(0x80000000 + 5 * 8),
        )


if __name__ == "__main__":
    unittest.main()
