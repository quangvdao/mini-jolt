import pathlib  # locate repo root
import sys  # adjust import path for local modules
import unittest  # unit test framework

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local jolt-python/
sys.path.insert(0, str(ROOT))  # allow `import rv64imac.*`

from rv64imac.decode import decode_instruction  # word -> Instruction
from rv64imac.inline_sequences import expand_program  # inline-sequence expansion
from rv64imac.types import Xlen  # xlen tag


def _r_type(opcode: int, funct3: int, funct7: int, rd: int, rs1: int, rs2: int) -> int:  # encode an R-type u32
    return ((funct7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


def _i_type(opcode: int, funct3: int, rd: int, rs1: int, imm12: int) -> int:  # encode an I-type u32
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)


def _load(funct3: int, rd: int, rs1: int, imm12: int) -> int:  # load encoding (opcode 0x03)
    return _i_type(0x03, funct3, rd, rs1, imm12)


def _store(funct3: int, rs1: int, rs2: int, imm12: int) -> int:  # store encoding (opcode 0x23)
    imm = imm12 & 0xFFF
    return (((imm >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((imm & 0x1F) << 7) | 0x23


def _amo_word(funct3: int, funct5: int, rd: int, rs1: int, rs2: int) -> int:  # AMO encoding (opcode 0x2F)
    funct7 = (funct5 & 0x1F) << 2  # aq/rl=0
    return _r_type(0x2F, funct3, funct7, rd, rs1, rs2)


class RV64IMACInlineSequenceTests(unittest.TestCase):  # freeze key expansions for refactors
    def test_rv64_load_expansions(self):  # LB/LBU/LH/LHU/LW/LWU RV64 expansions
        addr = 0x8000_0000
        rs1, rs2, rd, imm = 2, 3, 1, 5

        lb = decode_instruction(_load(0b000, rd, rs1, imm), addr, False)
        lbu = decode_instruction(_load(0b100, rd, rs1, imm), addr, False)
        lh = decode_instruction(_load(0b001, rd, rs1, imm), addr, False)
        lhu = decode_instruction(_load(0b101, rd, rs1, imm), addr, False)
        lw = decode_instruction(_load(0b010, rd, rs1, imm), addr, False)
        lwu = decode_instruction(_load(0b110, rd, rs1, imm), addr, False)

        self.assertEqual(
            [i.kind for i in expand_program([lb], Xlen.Bit64)],
            ["ADDI", "ANDI", "LD", "XORI", "VirtualMULI", "VirtualPow2", "MUL", "VirtualSRAI"],
        )
        self.assertEqual(
            [i.kind for i in expand_program([lbu], Xlen.Bit64)],
            ["ADDI", "ANDI", "LD", "XORI", "VirtualMULI", "VirtualPow2", "MUL", "VirtualSRLI"],
        )
        self.assertEqual(
            [i.kind for i in expand_program([lh], Xlen.Bit64)],
            [
                "VirtualAssertHalfwordAlignment",
                "ADDI",
                "ANDI",
                "LD",
                "XORI",
                "VirtualMULI",
                "VirtualPow2",
                "MUL",
                "VirtualSRAI",
            ],
        )
        self.assertEqual(
            [i.kind for i in expand_program([lhu], Xlen.Bit64)],
            [
                "VirtualAssertHalfwordAlignment",
                "ADDI",
                "ANDI",
                "LD",
                "XORI",
                "VirtualMULI",
                "VirtualPow2",
                "MUL",
                "VirtualSRLI",
            ],
        )
        self.assertEqual(
            [i.kind for i in expand_program([lw], Xlen.Bit64)],
            [
                "VirtualAssertWordAlignment",
                "ADDI",
                "ANDI",
                "LD",
                "VirtualMULI",
                "VirtualShiftRightBitmask",
                "VirtualSRL",
                "VirtualSignExtendWord",
            ],
        )
        self.assertEqual(
            [i.kind for i in expand_program([lwu], Xlen.Bit64)],
            ["VirtualAssertWordAlignment", "ADDI", "ANDI", "LD", "XORI", "VirtualMULI", "VirtualPow2", "MUL", "VirtualSRLI"],
        )

    def test_rv64_store_expansions(self):  # SB/SH/SW RV64 expansions
        addr = 0x8000_0000
        rs1, rs2, imm = 2, 3, 5

        sb = decode_instruction(_store(0b000, rs1, rs2, imm), addr, False)
        sh = decode_instruction(_store(0b001, rs1, rs2, imm), addr, False)
        sw = decode_instruction(_store(0b010, rs1, rs2, imm), addr, False)

        self.assertEqual(
            [i.kind for i in expand_program([sb], Xlen.Bit64)],
            ["ADDI", "ANDI", "LD", "VirtualMULI", "LUI", "VirtualPow2", "MUL", "VirtualPow2", "MUL", "XOR", "AND", "XOR", "SD"],
        )
        self.assertEqual(
            [i.kind for i in expand_program([sh], Xlen.Bit64)],
            [
                "VirtualAssertHalfwordAlignment",
                "ADDI",
                "ANDI",
                "LD",
                "VirtualMULI",
                "LUI",
                "VirtualPow2",
                "MUL",
                "VirtualPow2",
                "MUL",
                "XOR",
                "AND",
                "XOR",
                "SD",
            ],
        )
        self.assertEqual(
            [i.kind for i in expand_program([sw], Xlen.Bit64)],
            [
                "VirtualAssertWordAlignment",
                "ADDI",
                "ANDI",
                "LD",
                "VirtualMULI",
                "ORI",
                "VirtualSRLI",
                "VirtualPow2",
                "MUL",
                "VirtualPow2",
                "MUL",
                "XOR",
                "AND",
                "XOR",
                "SD",
            ],
        )

    def test_advice_load_expansions(self):  # AdviceL* expansions
        from rv64imac.types import Instruction, NormalizedOperands  # local import keeps file light

        addr = 0x8000_0000
        rd = 1
        adv_lb = Instruction(kind="AdviceLB", address=addr, operands=NormalizedOperands(None, None, rd, 0))
        adv_lw = Instruction(kind="AdviceLW", address=addr, operands=NormalizedOperands(None, None, rd, 0))

        self.assertEqual([i.kind for i in expand_program([adv_lb], Xlen.Bit64)], ["VirtualAdviceLoad", "VirtualMULI", "VirtualSRAI"])
        self.assertEqual([i.kind for i in expand_program([adv_lw], Xlen.Bit32)], ["VirtualAdviceLoad"])

    def test_divu_remu_shapes(self):  # unsigned DIV/REM div0 nuance preserved
        addr = 0x8000_0000
        rd, rs1, rs2 = 1, 2, 3
        divu = decode_instruction(_r_type(0x33, 0b101, 0b0000001, rd, rs1, rs2), addr, False)
        remu = decode_instruction(_r_type(0x33, 0b111, 0b0000001, rd, rs1, rs2), addr, False)

        self.assertEqual(
            [i.kind for i in expand_program([divu], Xlen.Bit64)],
            ["VirtualAdvice", "VirtualAssertValidDiv0", "VirtualAssertMulUNoOverflow", "MUL", "VirtualAssertLTE", "SUB", "VirtualAssertValidUnsignedRemainder", "ADDI"],
        )
        self.assertEqual(
            [i.kind for i in expand_program([remu], Xlen.Bit64)],
            ["VirtualAdvice", "VirtualAssertMulUNoOverflow", "MUL", "VirtualAssertLTE", "SUB", "VirtualAssertValidUnsignedRemainder", "ADDI"],
        )

    def test_div_rem_output_regs(self):  # DIV vs REM differ only in which temp is returned
        addr = 0x8000_0000
        rd, rs1, rs2 = 1, 2, 3
        div = decode_instruction(_r_type(0x33, 0b100, 0b0000001, rd, rs1, rs2), addr, False)
        rem = decode_instruction(_r_type(0x33, 0b110, 0b0000001, rd, rs1, rs2), addr, False)
        self.assertEqual(expand_program([div], Xlen.Bit64)[-1].operands.rs1, 40)
        self.assertEqual(expand_program([rem], Xlen.Bit64)[-1].operands.rs1, 45)

    def test_divw_remw_divuw_remuw_outputs(self):  # W variants preserve output-register wiring
        addr = 0x8000_0000
        rd, rs1, rs2 = 1, 2, 3
        divw = decode_instruction(_r_type(0x3B, 0b100, 0b0000001, rd, rs1, rs2), addr, False)
        remw = decode_instruction(_r_type(0x3B, 0b110, 0b0000001, rd, rs1, rs2), addr, False)
        divuw = decode_instruction(_r_type(0x3B, 0b101, 0b0000001, rd, rs1, rs2), addr, False)
        remuw = decode_instruction(_r_type(0x3B, 0b111, 0b0000001, rd, rs1, rs2), addr, False)
        self.assertEqual(expand_program([divw], Xlen.Bit64)[-1].operands.rs1, 40)
        self.assertEqual(expand_program([remw], Xlen.Bit64)[-1].operands.rs1, 45)
        self.assertEqual(expand_program([divuw], Xlen.Bit64)[-1].kind, "VirtualAssertValidDiv0")
        self.assertEqual(expand_program([remuw], Xlen.Bit64)[-1].kind, "VirtualSignExtendWord")

    def test_amo_expansions(self):  # representative AMO expansions (sensitive to refactors)
        addr = 0x8000_0000
        rd, rs1, rs2 = 1, 2, 3

        amoaddw = decode_instruction(_amo_word(0b010, 0b00000, rd, rs1, rs2), addr, False)
        amomind = decode_instruction(_amo_word(0b011, 0b10000, rd, rs1, rs2), addr, False)
        amominuW = decode_instruction(_amo_word(0b010, 0b11000, rd, rs1, rs2), addr, False)

        self.assertEqual(
            [i.kind for i in expand_program([amoaddw], Xlen.Bit64)],
            [
                "VirtualAssertWordAlignment",
                "ANDI",
                "LD",
                "VirtualMULI",
                "VirtualShiftRightBitmask",
                "VirtualSRL",
                "ADD",
                "ORI",
                "VirtualSRLI",
                "VirtualPow2",
                "MUL",
                "VirtualPow2",
                "MUL",
                "XOR",
                "AND",
                "XOR",
                "ANDI",
                "SD",
                "VirtualSignExtendWord",
            ],
        )
        self.assertEqual([i.kind for i in expand_program([amomind], Xlen.Bit64)], ["LD", "SLT", "SUB", "MUL", "ADD", "SD", "ADDI"])
        self.assertEqual(
            [i.kind for i in expand_program([amominuW], Xlen.Bit64)],
            [
                "VirtualAssertWordAlignment",
                "ANDI",
                "LD",
                "VirtualMULI",
                "VirtualShiftRightBitmask",
                "VirtualSRL",
                "VirtualZeroExtendWord",
                "VirtualZeroExtendWord",
                "SLTU",
                "SUB",
                "MUL",
                "ADD",
                "ORI",
                "VirtualSRLI",
                "VirtualPow2",
                "MUL",
                "VirtualPow2",
                "MUL",
                "XOR",
                "AND",
                "XOR",
                "ANDI",
                "SD",
                "VirtualSignExtendWord",
            ],
        )

    def test_lr_sc_csr_shapes(self):  # freeze LR/SC + CSR expansion shapes
        from rv64imac.inline_sequences import VirtualRegisterAllocator, inline_sequence  # local import keeps file light

        addr = 0x8000_0000
        rd, rs1, rs2 = 1, 2, 3

        lrw = decode_instruction(_amo_word(0b010, 0b00010, rd, rs1, rs2), addr, False)
        scw = decode_instruction(_amo_word(0b010, 0b00011, rd, rs1, rs2), addr, False)
        lrd = decode_instruction(_amo_word(0b011, 0b00010, rd, rs1, rs2), addr, False)
        scd = decode_instruction(_amo_word(0b011, 0b00011, rd, rs1, rs2), addr, False)

        alloc = VirtualRegisterAllocator()
        self.assertEqual([i.kind for i in inline_sequence(lrw, alloc, Xlen.Bit64)], ["ADDI", "ADDI", "LW"])
        self.assertEqual(
            [i.kind for i in inline_sequence(scw, VirtualRegisterAllocator(), Xlen.Bit64)],
            [
                "VirtualAdvice",
                "ADDI",
                "VirtualAssertLTE",
                "SUB",
                "MUL",
                "VirtualAssertEQ",
                "ADDI",
                "LW",
                "SUB",
                "MUL",
                "ADD",
                "ADDI",
                "SW",
                "XORI",
                "ADDI",
                "ADDI",
            ],
        )
        self.assertEqual([i.kind for i in inline_sequence(lrd, VirtualRegisterAllocator(), Xlen.Bit64)], ["ADDI", "ADDI", "LD"])
        self.assertEqual(
            [i.kind for i in inline_sequence(scd, VirtualRegisterAllocator(), Xlen.Bit64)],
            [
                "VirtualAdvice",
                "ADDI",
                "VirtualAssertLTE",
                "SUB",
                "MUL",
                "VirtualAssertEQ",
                "LD",
                "SUB",
                "MUL",
                "ADD",
                "SD",
                "ADDI",
                "ADDI",
                "XORI",
            ],
        )

        csrrw = decode_instruction(_i_type(0x73, 0b001, rd, rs1, 0x300), addr, False)
        csrrs = decode_instruction(_i_type(0x73, 0b010, rd, rs1, 0x300), addr, False)
        self.assertEqual([i.kind for i in inline_sequence(csrrw, VirtualRegisterAllocator(), Xlen.Bit64)], ["ADDI", "ADDI"])
        self.assertEqual([i.kind for i in inline_sequence(csrrs, VirtualRegisterAllocator(), Xlen.Bit64)], ["ADDI", "OR"])


if __name__ == "__main__":
    unittest.main()

