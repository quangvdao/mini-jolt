import hashlib
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from field import Fr
from transcript import Blake2bTranscript


class TranscriptTests(unittest.TestCase):
    def test_init_matches_blake2b_padded_label(self):
        label = b"Jolt"
        expected = hashlib.blake2b(label + b"\x00" * (32 - len(label)), digest_size=32).digest()
        t = Blake2bTranscript.new(label)
        self.assertEqual(t.state, expected)
        self.assertEqual(t.n_rounds, 0)

    def test_label_constraints(self):
        with self.assertRaises(ValueError):
            Blake2bTranscript.new(b"a" * 33)
        t = Blake2bTranscript.new(b"ok")
        with self.assertRaises(ValueError):
            t.raw_append_label_with_len(b"a" * 25, 0)

    def test_round_count_append_and_challenge(self):
        t = Blake2bTranscript.new(b"Jolt")
        t.append_u64(b"lbl", 7)
        self.assertEqual(t.n_rounds, 2)
        t.append_bytes(b"b", b"\x01\x02\x03")
        self.assertEqual(t.n_rounds, 4)
        _ = t.challenge_u128()
        self.assertEqual(t.n_rounds, 5)

    def test_challenge_u128_and_scalar_share_stream(self):
        base = Blake2bTranscript.new(b"Jolt")
        t_bytes = base.copy()
        t_u128 = base.copy()
        t_fr = base.copy()

        b16 = t_bytes.challenge_bytes(16)
        expected_u = int.from_bytes(b16, "little")
        expected_fr = Fr(int.from_bytes(b16, "big"))

        u = t_u128.challenge_u128()
        fr = t_fr.challenge_scalar()
        self.assertEqual(u, expected_u)
        self.assertEqual(int(fr), int(expected_fr))
        self.assertEqual(t_u128.state, t_fr.state)
        self.assertEqual(t_u128.n_rounds, t_fr.n_rounds)


if __name__ == "__main__":
    unittest.main()

