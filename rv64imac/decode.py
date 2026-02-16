from __future__ import annotations  # keep type hints lightweight

from .extensions import ISAProfile, assert_kind_allowed  # extension/profile gating
from .types import Instruction, NormalizedOperands  # decoded instruction containers

class DecodeError(Exception):  # instruction decode error
    pass

CUSTOM_FUNCT3_KIND = {  # custom opcode funct3 -> kind
    0b000: "VirtualRev8W", 0b001: "VirtualAssertEQ", 0b010: "VirtualHostIO", 0b011: "AdviceLB",
    0b100: "AdviceLH", 0b101: "AdviceLW", 0b110: "AdviceLD", 0b111: "VirtualAdviceLen",
}

def _i32(x: int) -> int:  # cast u32->i32
    x &= 0xFFFF_FFFF
    return x - 0x1_0000_0000 if (x & 0x8000_0000) else x
def _u64_from_i32(x: int) -> int:  # sign-extend i32 to u64
    return _i32(x) & 0xFFFF_FFFF_FFFF_FFFF
def _imm_i12_i32(word: int) -> int:  # I-type imm as i32 bits
    return (0xFFFF_F800 if (word & 0x8000_0000) else 0) | ((word >> 20) & 0x7FF)

def _parse_r(word: int) -> NormalizedOperands:  # R-type operands
    return NormalizedOperands(rd=(word >> 7) & 0x1F, rs1=(word >> 15) & 0x1F, rs2=(word >> 20) & 0x1F, imm=0)

def _parse_i(word: int) -> NormalizedOperands:  # I-type operands
    return NormalizedOperands(
        rd=(word >> 7) & 0x1F,
        rs1=(word >> 15) & 0x1F,
        rs2=None,
        imm=_u64_from_i32(_imm_i12_i32(word)),
    )

def _parse_load(word: int) -> NormalizedOperands:  # load operands
    return NormalizedOperands(rd=(word >> 7) & 0x1F, rs1=(word >> 15) & 0x1F, rs2=None, imm=int(_i32(_imm_i12_i32(word))))

def _parse_s(word: int) -> NormalizedOperands:  # S-type operands
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    imm32 = (
        (0xFFFF_F000 if (word & 0x8000_0000) else 0)
        | ((word >> 20) & 0xFE0)
        | ((word >> 7) & 0x1F)
    )
    return NormalizedOperands(rd=None, rs1=rs1, rs2=rs2, imm=int(_i32(imm32)))

def _parse_b(word: int) -> NormalizedOperands:  # B-type operands
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    imm32 = (
        (0xFFFF_F000 if (word & 0x8000_0000) else 0)
        | ((word << 4) & 0x800)
        | ((word >> 20) & 0x7E0)
        | ((word >> 7) & 0x1E)
    )
    return NormalizedOperands(rd=None, rs1=rs1, rs2=rs2, imm=int(_i32(imm32)))

def _parse_u(word: int) -> NormalizedOperands:  # U-type operands
    rd = (word >> 7) & 0x1F
    imm32 = word & 0xFFFF_F000
    return NormalizedOperands(rd=rd, rs1=None, rs2=None, imm=_u64_from_i32(imm32))

def _parse_j(word: int) -> NormalizedOperands:  # J-type operands
    rd = (word >> 7) & 0x1F
    imm32 = (
        (0xFFF0_0000 if (word & 0x8000_0000) else 0)
        | (word & 0x000F_F000)
        | ((word & 0x0010_0000) >> 9)
        | ((word & 0x7FE0_0000) >> 20)
    )
    return NormalizedOperands(rd=rd, rs1=None, rs2=None, imm=_u64_from_i32(imm32))

def _parse_advice_load_i(word: int) -> NormalizedOperands:  # advice load operands
    rd = (word >> 7) & 0x1F
    imm = _u64_from_i32(word)
    return NormalizedOperands(rd=rd, rs1=None, rs2=None, imm=int(imm))

def _mk(kind: str, address: int, operands: NormalizedOperands, compressed: bool, profile: ISAProfile | None) -> Instruction:  # build instruction
    if profile is None:
        profile = ISAProfile.default()
    assert_kind_allowed(kind, profile, compressed=compressed)
    return Instruction(
        kind=kind,
        address=address,
        operands=operands,
        virtual_sequence_remaining=None,
        is_first_in_sequence=False,
        is_compressed=compressed,
        advice=0,
    )

def decode_instruction(word: int, address: int, compressed: bool, profile: ISAProfile | None = None) -> Instruction:  # decode 32-bit word
    word &= 0xFFFF_FFFF
    opcode = word & 0x7F

    if opcode == 0b0110111:
        return _mk("LUI", address, _parse_u(word), compressed, profile)
    if opcode == 0b0010111:
        return _mk("AUIPC", address, _parse_u(word), compressed, profile)
    if opcode == 0b1101111:
        return _mk("JAL", address, _parse_j(word), compressed, profile)
    if opcode == 0b1100111:
        funct3 = (word >> 12) & 0x7
        if funct3 != 0:
            raise DecodeError("Invalid funct3 for JALR")
        return _mk("JALR", address, _parse_i(word), compressed, profile)

    if opcode == 0b1100011:
        funct3 = (word >> 12) & 0x7
        kind = {
            0b000: "BEQ", 0b001: "BNE", 0b100: "BLT",
            0b101: "BGE", 0b110: "BLTU", 0b111: "BGEU",
        }.get(funct3)
        if kind is None:
            raise DecodeError("Invalid branch funct3")
        return _mk(kind, address, _parse_b(word), compressed, profile)

    if opcode == 0b0000011:
        funct3 = (word >> 12) & 0x7
        kind = {
            0b000: "LB", 0b001: "LH", 0b010: "LW", 0b011: "LD",
            0b100: "LBU", 0b101: "LHU", 0b110: "LWU",
        }.get(funct3)
        if kind is None:
            raise DecodeError("Invalid load funct3")
        # All loads use FormatLoad (signed imm) in tracer.
        return _mk(kind, address, _parse_load(word), compressed, profile)

    if opcode == 0b0100011:
        funct3 = (word >> 12) & 0x7
        kind = {0b000: "SB", 0b001: "SH", 0b010: "SW", 0b011: "SD"}.get(funct3)
        if kind is None:
            raise DecodeError("Invalid store funct3")
        return _mk(kind, address, _parse_s(word), compressed, profile)

    if opcode == 0b0010011:
        funct3 = (word >> 12) & 0x7
        funct6 = (word >> 26) & 0x3F
        if funct3 == 0b001:
            if funct6 == 0:
                return _mk("SLLI", address, _parse_i(word), compressed, profile)
            raise DecodeError("Invalid funct7 for SLLI")
        if funct3 == 0b101:
            if funct6 == 0b000000:
                return _mk("SRLI", address, _parse_i(word), compressed, profile)
            if funct6 == 0b010000:
                return _mk("SRAI", address, _parse_i(word), compressed, profile)
            raise DecodeError("Invalid ALU shift funct7")
        kind = {
            0b000: "ADDI", 0b010: "SLTI", 0b011: "SLTIU",
            0b100: "XORI", 0b110: "ORI", 0b111: "ANDI",
        }.get(funct3)
        if kind is None:
            raise DecodeError("Invalid I-type ALU funct3")
        return _mk(kind, address, _parse_i(word), compressed, profile)

    if opcode == 0b0011011:
        funct3 = (word >> 12) & 0x7
        funct7 = (word >> 25) & 0x7F
        kind = None
        if funct3 == 0b000:
            kind = "ADDIW"
        elif (funct3, funct7) == (0b001, 0b0000000):
            kind = "SLLIW"
        elif (funct3, funct7) == (0b101, 0b0000000):
            kind = "SRLIW"
        elif (funct3, funct7) == (0b101, 0b0100000):
            kind = "SRAIW"
        if kind is None:
            raise DecodeError("Invalid RV64I I-type arithmetic instruction")
        return _mk(kind, address, _parse_i(word), compressed, profile)

    if opcode == 0b0110011:
        funct3 = (word >> 12) & 0x7
        funct7 = (word >> 25) & 0x7F
        base = {
            (0b000, 0b0000000): "ADD", (0b000, 0b0100000): "SUB", (0b001, 0b0000000): "SLL",
            (0b010, 0b0000000): "SLT", (0b011, 0b0000000): "SLTU", (0b100, 0b0000000): "XOR",
            (0b101, 0b0000000): "SRL", (0b101, 0b0100000): "SRA", (0b110, 0b0000000): "OR",
            (0b111, 0b0000000): "AND", (0b111, 0b0100000): "ANDN",
            (0b000, 0b0000001): "MUL", (0b001, 0b0000001): "MULH", (0b010, 0b0000001): "MULHSU", (0b011, 0b0000001): "MULHU",
            (0b100, 0b0000001): "DIV", (0b101, 0b0000001): "DIVU", (0b110, 0b0000001): "REM", (0b111, 0b0000001): "REMU",
        }.get((funct3, funct7))
        if base is None:
            raise DecodeError("Invalid R-type arithmetic instruction")
        return _mk(base, address, _parse_r(word), compressed, profile)

    if opcode == 0b0111011:
        funct3 = (word >> 12) & 0x7
        funct7 = (word >> 25) & 0x7F
        kind = {
            (0b000, 0b0000000): "ADDW", (0b000, 0b0100000): "SUBW", (0b001, 0b0000000): "SLLW",
            (0b101, 0b0000000): "SRLW", (0b101, 0b0100000): "SRAW", (0b000, 0b0000001): "MULW",
            (0b100, 0b0000001): "DIVW", (0b101, 0b0000001): "DIVUW", (0b110, 0b0000001): "REMW", (0b111, 0b0000001): "REMUW",
        }.get((funct3, funct7))
        if kind is None:
            raise DecodeError("Invalid RV64I R-type arithmetic instruction")
        return _mk(kind, address, _parse_r(word), compressed, profile)

    if opcode == 0b0001111:
        return _mk("FENCE", address, NormalizedOperands(None, None, None, 0), compressed, profile)

    if opcode == 0b0101111:
        funct3 = (word >> 12) & 0x7
        funct5 = (word >> 27) & 0x1F
        kind = {
            (0b010, 0b00010): "LRW", (0b011, 0b00010): "LRD", (0b010, 0b00011): "SCW", (0b011, 0b00011): "SCD",
            (0b010, 0b00001): "AMOSWAPW", (0b011, 0b00001): "AMOSWAPD", (0b010, 0b00000): "AMOADDW", (0b011, 0b00000): "AMOADDD",
            (0b010, 0b01100): "AMOANDW", (0b011, 0b01100): "AMOANDD", (0b010, 0b01000): "AMOORW", (0b011, 0b01000): "AMOORD",
            (0b010, 0b00100): "AMOXORW", (0b011, 0b00100): "AMOXORD", (0b010, 0b10000): "AMOMINW", (0b011, 0b10000): "AMOMIND",
            (0b010, 0b10100): "AMOMAXW", (0b011, 0b10100): "AMOMAXD", (0b010, 0b11000): "AMOMINUW", (0b011, 0b11000): "AMOMINUD",
            (0b010, 0b11100): "AMOMAXUW", (0b011, 0b11100): "AMOMAXUD",
        }.get((funct3, funct5))
        if kind is None:
            raise DecodeError("Invalid atomic memory operation")
        return _mk(kind, address, _parse_r(word), compressed, profile)

    if opcode == 0b1110011:
        funct3 = (word >> 12) & 0x7
        sys0 = {0x0000_0073: "ECALL", 0x0010_0073: "EBREAK", 0x3020_0073: "MRET"}.get(word)
        if sys0 is not None:
            return _mk(sys0, address, _parse_i(word), compressed, profile)
        if funct3 == 1:
            return _mk("CSRRW", address, _parse_i(word), compressed, profile)
        if funct3 == 2:
            return _mk("CSRRS", address, _parse_i(word), compressed, profile)
        raise DecodeError("Unsupported SYSTEM instruction")

    if opcode == 0b0001011 or opcode == 0b0101011:
        return _mk("INLINE", address, _parse_r(word), compressed, profile)

    if opcode == 0b1011011:
        funct3 = (word >> 12) & 0x7
        kind = CUSTOM_FUNCT3_KIND.get(funct3)
        if kind is None:
            raise DecodeError("Invalid custom/virtual instruction")
        if kind.startswith("AdviceL"):
            return _mk(kind, address, _parse_advice_load_i(word), compressed, profile)
        if kind == "VirtualAdviceLen":
            return _mk(kind, address, _parse_i(word), compressed, profile)
        if kind == "VirtualAssertEQ":
            return _mk(kind, address, _parse_b(word), compressed, profile)
        return _mk(kind, address, _parse_i(word), compressed, profile)

    raise DecodeError("Unknown opcode")

def unimpl_instruction() -> Instruction:  # UNIMPL placeholder
    return Instruction(kind="UNIMPL", address=0, operands=NormalizedOperands(None, None, None, 0))

