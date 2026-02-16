import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from curve import G1, G2, GT, add, b, b2, b12, double, is_on_curve, mul, multi_pairing, neg, pairing, twist


class CurveTests(unittest.TestCase):
    def test_generators_on_curve(self):
        self.assertTrue(is_on_curve(G1, b))
        self.assertTrue(is_on_curve(G2, b2))
        self.assertTrue(is_on_curve(twist(G2), b12))

    def test_group_ops(self):
        self.assertEqual(mul(G1, 2), double(G1))
        self.assertEqual(mul(G2, 2), add(G2, G2))
        self.assertIsNone(add(G1, neg(G1)))
        self.assertIsNone(add(G2, neg(G2)))

    def test_pairing_and_multi_pairing(self):
        e = pairing(G2, G1)
        self.assertEqual(pairing(mul(G2, 2), G1), e**2)
        self.assertEqual(multi_pairing([(G2, G1), (neg(G2), G1)]), GT.one())


if __name__ == "__main__":
    unittest.main()
