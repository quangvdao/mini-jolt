import random
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from field import Fq, Fr


class FieldTests(unittest.TestCase):
    def check_field(self, cls):
        p = cls.MODULUS
        rng = random.Random(0)
        for _ in range(64):
            a = rng.randrange(0, p)
            b = rng.randrange(1, p)
            x, y = cls(a), cls(b)
            self.assertEqual(int(x + y), (a + b) % p)
            self.assertEqual(int(x - y), (a - b) % p)
            self.assertEqual(int(x * y), (a * b) % p)
            self.assertEqual(int(x / y), (a * pow(b, -1, p)) % p)
            self.assertEqual(int(x**7), pow(a, 7, p))
            self.assertEqual(cls.from_montgomery(x.v), x)
            if a:
                self.assertEqual(int(x * x.inv()), 1)
            else:
                with self.assertRaises(ZeroDivisionError):
                    x.inv()

        self.assertEqual(int(cls.zero()), 0)
        self.assertEqual(int(cls.one()), 1)

    def test_fq(self):
        self.check_field(Fq)

    def test_fr(self):
        self.check_field(Fr)


if __name__ == "__main__":
    unittest.main()
