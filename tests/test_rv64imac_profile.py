import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow `import rv64imac.*`

from rv64imac.decode import decode_instruction  # word -> Instruction
from rv64imac.extensions import Extension, ISAProfile, ISAProfileError  # ISA profile gating
from rv64imac.inline_sequences import expand_program  # inline-sequence expansion
from rv64imac.isa import validate_isa_tables  # ISA metadata validation
from rv64imac.types import Xlen  # xlen tag


def _r_type(opcode: int, funct3: int, funct7: int, rd: int, rs1: int, rs2: int) -> int:  # encode an R-type u32
    return ((funct7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


def _i_type(opcode: int, funct3: int, rd: int, rs1: int, imm12: int) -> int:  # encode an I-type u32
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


class RV64IMACProfileTests(unittest.TestCase):  # ISAProfile gating + metadata tests
    def test_default_profile_smoke_decode_and_expand(self):  # Default behavior should work unchanged.
        addr = 0x8000_0000
        add = decode_instruction(_r_type(0x33, 0b000, 0b0000000, 1, 2, 3), addr, False)
        mul = decode_instruction(_r_type(0x33, 0b000, 0b0000001, 1, 2, 3), addr + 4, False)
        out1 = expand_program([add, mul], Xlen.Bit64)
        out2 = expand_program([add, mul], Xlen.Bit64, profile=ISAProfile.default())
        self.assertEqual([i.kind for i in out1], [i.kind for i in out2])

    def test_disable_m_rejects_mul_div_rem(self):  # Disabling M should fail fast on M kinds.
        p = ISAProfile.default().without(Extension.M)
        addr = 0x8000_0000
        with self.assertRaises(ISAProfileError):
            decode_instruction(_r_type(0x33, 0b000, 0b0000001, 1, 2, 3), addr, False, profile=p)  # MUL
        with self.assertRaises(ISAProfileError):
            decode_instruction(_r_type(0x33, 0b100, 0b0000001, 1, 2, 3), addr, False, profile=p)  # DIV
        with self.assertRaises(ISAProfileError):
            decode_instruction(_r_type(0x33, 0b110, 0b0000001, 1, 2, 3), addr, False, profile=p)  # REM

    def test_disable_c_rejects_compressed_decodes(self):  # Compressed flag requires C extension.
        p = ISAProfile.default().without(Extension.C)
        addr = 0x8000_0000
        addi = _i_type(0x13, 0b000, 1, 2, 1)  # ADDI x1,x2,1
        with self.assertRaises(ISAProfileError):
            decode_instruction(addi, addr, True, profile=p)

    def test_disable_zicsr_rejects_csr_kinds(self):  # CSR ops should be separately gateable.
        p = ISAProfile.default().without(Extension.Zicsr)
        addr = 0x8000_0000
        csrrs = _i_type(0x73, 0b010, 1, 2, 0x300)  # CSRRS x1, mstatus(0x300), x2
        with self.assertRaises(ISAProfileError):
            decode_instruction(csrrs, addr, False, profile=p)

    def test_validate_isa_tables(self):  # ISA metadata tables should be internally complete.
        validate_isa_tables()

    def test_optional_tinyrv_cross_check(self):  # Optional: cross-check a few opcodes vs riscv-opcodes-derived decoder.
        try:
            import tinyrv  # type: ignore[import-not-found]  # optional dependency
        except Exception:
            self.skipTest("tinyrv not installed; skipping external opcode-spec cross-check")

        addr = 0x8000_0000
        add_word = _r_type(0x33, 0b000, 0b0000000, 1, 2, 3)
        addi_word = _i_type(0x13, 0b000, 1, 2, 1)

        ours_add = decode_instruction(add_word, addr, False).kind
        ours_addi = decode_instruction(addi_word, addr + 4, False).kind

        their_add = getattr(tinyrv.decode(add_word), "name", None)
        their_addi = getattr(tinyrv.decode(addi_word), "name", None)
        if their_add is None or their_addi is None:
            self.skipTest("tinyrv.decode(..) did not return an object with .name")

        self.assertEqual(ours_add, str(their_add).upper())
        self.assertEqual(ours_addi, str(their_addi).upper())


if __name__ == "__main__":
    unittest.main()

