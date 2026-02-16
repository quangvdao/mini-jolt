import pathlib  # path helpers
import random  # deterministic pseudo-random tests
import sys  # import path for repo-local modules
import unittest  # stdlib test runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from field import Fr  # BN254 scalar field
from lookup_tables import evaluate_mle  # table MLE implementation under test
from tests.oracle import limbs_csv_to_int, run_rust_oracle  # rust parity harness


class TestLookupTablesCrossLang(unittest.TestCase):
    def test_lookup_tables_against_rust_oracle(self):
        xlen = 64
        rng = random.Random(12345)

        # Table names mirror `lookup_tables.py` and Rust `LookupTables<64>` variants.
        tables_2x = [
            "RangeCheck",
            "RangeCheckAligned",
            "And",
            "Andn",
            "Or",
            "Xor",
            "Equal",
            "NotEqual",
            "UnsignedLessThan",
            "SignedLessThan",
            "UnsignedGreaterThanEqual",
            "SignedGreaterThanEqual",
            "LessThanEqual",
            "Movsign",
            "UpperWord",
            "ValidDiv0",
            "ValidUnsignedRemainder",
            "ValidSignedRemainder",
            "HalfwordAlignment",
            "WordAlignment",
            "LowerHalfWord",
            "SignExtendHalfWord",
            "Pow2",
            "Pow2W",
            "ShiftRightBitmask",
            "VirtualSRL",
            "VirtualSRA",
            "VirtualROTR",
            "VirtualROTRW",
            "VirtualChangeDivisor",
            "VirtualChangeDivisorW",
            "MulUNoOverflow",
            "VirtualXORROT32",
            "VirtualXORROT24",
            "VirtualXORROT16",
            "VirtualXORROT63",
            "VirtualXORROTW16",
            "VirtualXORROTW12",
            "VirtualXORROTW8",
            "VirtualXORROTW7",
        ]
        tables_x = ["VirtualRev8W"]

        # Keep this moderate: Rust oracle is a `cargo run` binary.
        samples_per_table = 12

        queries = []
        expected_ints = []
        meta = []  # (table, bits_len, sample_idx)

        def rand_bits(n):
            return [rng.getrandbits(1) for _ in range(n)]

        for table in tables_2x:
            for k in range(samples_per_table):
                bits = rand_bits(2 * xlen)
                r = [Fr(b) for b in bits]
                py = evaluate_mle(table, r, xlen=xlen)
                queries.append(f"{table} {','.join(str(b) for b in bits)}")
                expected_ints.append(int(py))
                meta.append((table, 2 * xlen, k))

        for table in tables_x:
            for k in range(samples_per_table):
                bits = rand_bits(xlen)
                r = [Fr(b) for b in bits]
                py = evaluate_mle(table, r, xlen=xlen)
                queries.append(f"{table} {','.join(str(b) for b in bits)}")
                expected_ints.append(int(py))
                meta.append((table, xlen, k))

        out = run_rust_oracle("lookup_table_mle_64", "\n".join(queries) + "\n").strip().splitlines()
        self.assertEqual(
            len(out),
            len(expected_ints),
            f"rust oracle returned {len(out)} lines, expected {len(expected_ints)}",
        )

        for i, line in enumerate(out):
            got = limbs_csv_to_int(line.strip())
            exp = expected_ints[i]
            if got != exp:
                table, bits_len, k = meta[i]
                self.fail(
                    f"lookup table mismatch: table={table} sample={k} bits_len={bits_len} "
                    f"expected={exp} got={got}"
                )


if __name__ == "__main__":
    unittest.main()

