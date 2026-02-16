import os  # env flags for heavy tests
import pathlib  # filesystem paths
import sys  # import local jolt-python modules
import unittest  # test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local mini-jolt/
sys.path.insert(0, str(ROOT))  # allow `import jolt_verifier`, etc
sys.path.insert(0, str(ROOT / "tests"))  # allow importing test-only Rust serde helpers

from jolt_verifier import verify_jolt  # end-to-end verifier entrypoint
from jolt_proof import JoltProof  # Rust `proof.bin` parsing + verifier-facing proof object
from rust_device_deserialize import parse_jolt_device_bytes  # Rust `program_io.bin` parsing (test-only)
from rv64imac.program import decode_program  # ELF -> expanded bytecode + memory init


class E2EVerifyFromRustTests(unittest.TestCase):  # E2E: Rust prover -> Python verifier.
    def _verify_guest_dir(self, guest_dir: pathlib.Path) -> None:
        proof_path = guest_dir / "proof.bin"
        io_path = guest_dir / "program_io.bin"
        pp_path = guest_dir / "verifier_preprocessing.bin"
        elf_path = guest_dir / "program.elf"
        if not guest_dir.exists():
            self.skipTest(f"guest dir not found: {guest_dir}")
        if not (proof_path.exists() and io_path.exists() and pp_path.exists() and elf_path.exists()):
            self.skipTest(f"missing Rust artifacts in {guest_dir} (need proof.bin, program_io.bin, verifier_preprocessing.bin, program.elf)")

        elf = elf_path.read_bytes()
        try:
            expanded, memory_init, _program_size = decode_program(elf)
        except NotImplementedError as exc:
            if "INLINE" in str(exc):
                self.skipTest("program uses INLINE opcodes (not yet supported)")
            raise

        program_io = parse_jolt_device_bytes(io_path.read_bytes())
        proof = JoltProof.from_rust_bytes(proof_path.read_bytes(), verifier_preprocessing_bin=pp_path.read_bytes())

        verify_jolt(expanded, program_io, proof, memory_init=memory_init)

    def test_verify_rust_proof_fib_fast(self):  # Verify a real Rust-produced proof (fib, fast).
        if os.environ.get("JOLT_PYTHON_SKIP_E2E") == "1":
            self.skipTest("JOLT_PYTHON_SKIP_E2E=1")
        guest_dir = os.environ.get("MINI_JOLT_FIB_GUEST_DIR")
        if not guest_dir:
            self.skipTest("set MINI_JOLT_FIB_GUEST_DIR to a Rust-generated guest artifacts directory")
        self._verify_guest_dir(pathlib.Path(guest_dir))

    def test_verify_rust_proof_btreemap_fast(self):  # Verify a real Rust-produced proof (btreemap, fast).
        if os.environ.get("JOLT_PYTHON_SKIP_E2E") == "1":
            self.skipTest("JOLT_PYTHON_SKIP_E2E=1")
        guest_dir = os.environ.get("MINI_JOLT_BTREE_GUEST_DIR")
        if not guest_dir:
            self.skipTest("set MINI_JOLT_BTREE_GUEST_DIR to a Rust-generated guest artifacts directory")
        self._verify_guest_dir(pathlib.Path(guest_dir))

    def test_verify_rust_proof_sha2_fast(self):  # Verify a real Rust-produced proof (sha2, fast).
        if os.environ.get("JOLT_PYTHON_SKIP_E2E") == "1":
            self.skipTest("JOLT_PYTHON_SKIP_E2E=1")
        guest_dir = os.environ.get("MINI_JOLT_SHA2_GUEST_DIR")
        if not guest_dir:
            self.skipTest("set MINI_JOLT_SHA2_GUEST_DIR to a Rust-generated guest artifacts directory")
        self._verify_guest_dir(pathlib.Path(guest_dir))

    def test_verify_rust_proof_sha3_fast(self):  # Verify a real Rust-produced proof (sha3, fast).
        if os.environ.get("JOLT_PYTHON_SKIP_E2E") == "1":
            self.skipTest("JOLT_PYTHON_SKIP_E2E=1")
        guest_dir = os.environ.get("MINI_JOLT_SHA3_GUEST_DIR")
        if not guest_dir:
            self.skipTest("set MINI_JOLT_SHA3_GUEST_DIR to a Rust-generated guest artifacts directory")
        self._verify_guest_dir(pathlib.Path(guest_dir))


if __name__ == "__main__":
    unittest.main()

