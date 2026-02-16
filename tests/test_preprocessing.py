import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow importing local verifier modules

from jolt_preprocessing import ram_preprocess  # preprocessing under test
from rv64imac.constants import RAM_START_ADDRESS  # canonical base address


class PreprocessingTests(unittest.TestCase):  # Python-only tests for verifier preprocessing.
    def test_ram_preprocess_packs_words_like_rust(self):  # RAMPreprocessing packs memory_init bytes into u64 words.
        base = int(RAM_START_ADDRESS)
        memory_init = [(base + 0, 0x11), (base + 1, 0x22), (base + 2, 0x33), (base + 3, 0x44)]
        pp = ram_preprocess(memory_init)
        self.assertEqual(pp.min_bytecode_address, base)
        self.assertGreaterEqual(len(pp.bytecode_words), 1)
        self.assertEqual(pp.bytecode_words[0], 0x0000000044332211)


if __name__ == "__main__":  # unittest entrypoint
    unittest.main()

