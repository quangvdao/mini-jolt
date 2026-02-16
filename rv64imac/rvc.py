from __future__ import annotations  # keep type hints lightweight

from .types import Xlen  # ELF32 vs ELF64

def u32(x: int) -> int:  # mask to u32
    return x & 0xFFFF_FFFF

def enc_i(opcode: int, rd: int, rs1: int, funct3: int, imm12: int) -> int:  # encode I-type u32
    return u32(((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F))

def enc_s(opcode: int, rs1: int, rs2: int, funct3: int, imm12: int) -> int:  # encode S-type u32
    imm = imm12 & 0xFFF
    return u32((((imm >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | ((imm & 0x1F) << 7) | (opcode & 0x7F))

def _cj_offset(halfword: int) -> int:  # C.J / C.JAL signed offset
    return (
        (0xFFFFF000 if (halfword & 0x1000) else 0)
        | ((halfword >> 1) & 0x800)
        | ((halfword >> 7) & 0x10)
        | ((halfword >> 1) & 0x300)
        | ((halfword << 2) & 0x400)
        | ((halfword >> 1) & 0x40)
        | ((halfword << 1) & 0x80)
        | ((halfword >> 2) & 0xE)
        | ((halfword << 3) & 0x20)
    )

def enc_jal(rd: int, offset: int) -> int:  # encode JAL rd, offset
    imm = ((offset >> 1) & 0x80000) | ((offset << 8) & 0x7FE00) | ((offset >> 3) & 0x100) | ((offset >> 12) & 0xFF)
    return u32((imm << 12) | ((rd & 0x1F) << 7) | 0x6F)

def _cb_offset(halfword: int) -> int:  # C.BEQZ / C.BNEZ signed offset
    return (
        (0xFFFFFE00 if (halfword & 0x1000) else 0)
        | ((halfword >> 4) & 0x100)
        | ((halfword >> 7) & 0x18)
        | ((halfword << 1) & 0xC0)
        | ((halfword >> 2) & 0x6)
        | ((halfword << 3) & 0x20)
    )

def enc_b(rs1: int, rs2: int, funct3: int, offset: int) -> int:  # encode B-type u32
    imm2 = ((offset >> 6) & 0x40) | ((offset >> 5) & 0x3F)
    imm1 = (offset & 0x1E) | ((offset >> 11) & 0x1)
    return u32((imm2 << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) | (imm1 << 7) | 0x63)

def uncompress_instruction(halfword: int, xlen: Xlen) -> int:  # RVC halfword -> u32 word
    halfword &= 0xFFFF
    op = halfword & 0x3
    funct3 = (halfword >> 13) & 0x7

    if op == 0:
        if funct3 == 0:  # C.ADDI4SPN
            rd = (halfword >> 2) & 0x7  # [4:2]
            nzuimm = (
                ((halfword >> 7) & 0x30)  # [12:11] -> [5:4]
                | ((halfword >> 1) & 0x3C0)  # [10:7] -> [9:6]
                | ((halfword >> 4) & 0x4)  # [6] -> [2]
                | ((halfword >> 2) & 0x8)  # [5] -> [3]
            )
            if nzuimm != 0:
                return enc_i(0x13, rd + 8, 2, 0, nzuimm)
        elif funct3 == 1:  # C.FLD (32/64)
            rd = (halfword >> 2) & 0x7
            rs1 = (halfword >> 7) & 0x7
            offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xC0)
            return enc_i(0x07, rd + 8, rs1 + 8, 3, offset)
        elif funct3 == 2:  # C.LW
            rs1 = (halfword >> 7) & 0x7
            rd = (halfword >> 2) & 0x7
            offset = ((halfword >> 7) & 0x38) | ((halfword >> 4) & 0x4) | ((halfword << 1) & 0x40)
            return enc_i(0x03, rd + 8, rs1 + 8, 2, offset)
        elif funct3 == 3:  # C.LD (64)
            rs1 = (halfword >> 7) & 0x7
            rd = (halfword >> 2) & 0x7
            offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xC0)
            return enc_i(0x03, rd + 8, rs1 + 8, 3, offset)
        elif funct3 == 5:  # C.FSD
            rs1 = (halfword >> 7) & 0x7
            rs2 = (halfword >> 2) & 0x7
            offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xC0)
            return enc_s(0x27, rs1 + 8, rs2 + 8, 3, offset)
        elif funct3 == 6:  # C.SW
            rs1 = (halfword >> 7) & 0x7
            rs2 = (halfword >> 2) & 0x7
            offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0x40) | ((halfword >> 4) & 0x4)
            return enc_s(0x23, rs1 + 8, rs2 + 8, 2, offset)
        elif funct3 == 7:  # C.SD
            rs1 = (halfword >> 7) & 0x7
            rs2 = (halfword >> 2) & 0x7
            offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xC0)
            return enc_s(0x23, rs1 + 8, rs2 + 8, 3, offset)

    elif op == 1:
        if funct3 == 0:  # C.ADDI
            r = (halfword >> 7) & 0x1F
            imm = (0xFFFFFFC0 if (halfword & 0x1000) else 0) | ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
            if (r, imm) in ((0, 0), (0, imm), (r, 0)):
                return 0x13
            return enc_i(0x13, r, r, 0, imm)

        if funct3 == 1:  # C.JAL (RV32) / C.ADDIW (RV64)
            if xlen == Xlen.Bit32:
                return enc_jal(1, _cj_offset(halfword))

            if xlen == Xlen.Bit64:
                r = (halfword >> 7) & 0x1F
                imm = (0xFFFFFFC0 if (halfword & 0x1000) else 0) | ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
                if r == 0:
                    pass  # reserved
                elif imm == 0:
                    return enc_i(0x1B, r, r, 0, 0)
                else:
                    return enc_i(0x1B, r, r, 0, imm)

        if funct3 == 2:  # C.LI
            r = (halfword >> 7) & 0x1F
            imm = (0xFFFFFFC0 if (halfword & 0x1000) else 0) | ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
            if r != 0:
                return enc_i(0x13, r, 0, 0, imm)
            return 0x13

        if funct3 == 3:  # C.ADDI16SP / C.LUI
            r = (halfword >> 7) & 0x1F
            if r == 2:
                imm = (
                    (0xFFFFFC00 if (halfword & 0x1000) else 0)
                    | ((halfword >> 3) & 0x200)
                    | ((halfword >> 2) & 0x10)
                    | ((halfword << 1) & 0x40)
                    | ((halfword << 4) & 0x180)
                    | ((halfword << 3) & 0x20)
                )
                if imm != 0:
                    return enc_i(0x13, r, r, 0, imm)
            if r != 0 and r != 2:
                nzimm = (
                    (0xFFFC0000 if (halfword & 0x1000) else 0)
                    | ((halfword << 5) & 0x20000)
                    | ((halfword << 10) & 0x1F000)
                )
                if nzimm != 0:
                    return (nzimm | (r << 7) | 0x37) & 0xFFFFFFFF
            if r == 0:
                return 0x13

        if funct3 == 4:  # C.SRLI/C.SRAI/C.ANDI/C.SUB|XOR|OR|AND (+W variants)
            funct2 = (halfword >> 10) & 0x3
            if funct2 == 0:  # C.SRLI
                shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
                rs1 = (halfword >> 7) & 0x7
                return enc_i(0x13, rs1 + 8, rs1 + 8, 5, shamt)
            if funct2 == 1:  # C.SRAI
                shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
                rs1 = (halfword >> 7) & 0x7
                return enc_i(0x13, rs1 + 8, rs1 + 8, 5, (0x20 << 5) | shamt)
            if funct2 == 2:  # C.ANDI
                r = (halfword >> 7) & 0x7
                imm = (0xFFFFFFC0 if (halfword & 0x1000) else 0) | ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
                return enc_i(0x13, r + 8, r + 8, 7, imm)
            if funct2 == 3:  # C.SUB/C.XOR/C.OR/C.AND (+ C.SUBW/C.ADDW)
                funct1 = (halfword >> 12) & 1
                funct2_2 = (halfword >> 5) & 0x3
                rs1 = (halfword >> 7) & 0x7
                rs2 = (halfword >> 2) & 0x7
                if funct1 == 0:
                    if funct2_2 == 0:
                        return ((0x20 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | ((rs1 + 8) << 7) | 0x33) & 0xFFFFFFFF
                    if funct2_2 == 1:
                        return (((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (4 << 12) | ((rs1 + 8) << 7) | 0x33) & 0xFFFFFFFF
                    if funct2_2 == 2:
                        return (((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (6 << 12) | ((rs1 + 8) << 7) | 0x33) & 0xFFFFFFFF
                    if funct2_2 == 3:
                        return (((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (7 << 12) | ((rs1 + 8) << 7) | 0x33) & 0xFFFFFFFF
                if funct1 == 1:
                    if funct2_2 == 0:
                        return ((0x20 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | ((rs1 + 8) << 7) | 0x3B) & 0xFFFFFFFF
                    if funct2_2 == 1:
                        return (((rs2 + 8) << 20) | ((rs1 + 8) << 15) | ((rs1 + 8) << 7) | 0x3B) & 0xFFFFFFFF

        if funct3 == 5:  # C.J
            return enc_jal(0, _cj_offset(halfword))

        if funct3 == 6:  # C.BEQZ
            r = (halfword >> 7) & 0x7
            return enc_b(0, r + 8, 0, _cb_offset(halfword))

        if funct3 == 7:  # C.BNEZ
            r = (halfword >> 7) & 0x7
            return enc_b(0, r + 8, 1, _cb_offset(halfword))

    elif op == 2:
        if funct3 == 0:  # C.SLLI
            r = (halfword >> 7) & 0x1F
            shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1F)
            if r != 0:
                return enc_i(0x13, r, r, 1, shamt)

        if funct3 == 1:  # C.FLDSP
            rd = (halfword >> 7) & 0x1F
            offset = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x18) | ((halfword << 4) & 0x1C0)
            if rd != 0:
                return enc_i(0x07, rd, 2, 3, offset)

        if funct3 == 2:  # C.LWSP
            r = (halfword >> 7) & 0x1F
            offset = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1C) | ((halfword << 4) & 0xC0)
            if r != 0:
                return enc_i(0x03, r, 2, 2, offset)

        if funct3 == 3:  # C.LDSP
            rd = (halfword >> 7) & 0x1F
            offset = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x18) | ((halfword << 4) & 0x1C0)
            if rd != 0:
                return enc_i(0x03, rd, 2, 3, offset)

        if funct3 == 4:  # C.MV/C.JR/C.ADD/C.EBREAK/C.JALR
            funct1 = (halfword >> 12) & 1
            rs1 = (halfword >> 7) & 0x1F
            rs2 = (halfword >> 2) & 0x1F
            if funct1 == 0:
                if rs1 == 0 and rs2 == 0:
                    pass  # reserved
                elif rs2 == 0 and rs1 != 0:
                    return ((rs1 << 15) | 0x67) & 0xFFFFFFFF
                elif rs1 == 0 and rs2 != 0:
                    return 0x13
                else:
                    return ((rs2 << 20) | (rs1 << 7) | 0x33) & 0xFFFFFFFF
            else:
                if rs1 == 0 and rs2 == 0:
                    return 0x00100073
                if rs2 == 0 and rs1 != 0:
                    return ((rs1 << 15) | (1 << 7) | 0x67) & 0xFFFFFFFF
                if rs1 == 0 and rs2 != 0:
                    return 0x13
                return ((rs2 << 20) | (rs1 << 15) | (rs1 << 7) | 0x33) & 0xFFFFFFFF

        if funct3 == 5:  # C.FSDSP
            rs2 = (halfword >> 2) & 0x1F
            offset = ((halfword >> 7) & 0x38) | ((halfword >> 1) & 0x1C0)
            return enc_s(0x27, 2, rs2, 3, offset)

        if funct3 == 6:  # C.SWSP
            rs2 = (halfword >> 2) & 0x1F
            offset = ((halfword >> 7) & 0x3C) | ((halfword >> 1) & 0xC0)
            return enc_s(0x23, 2, rs2, 2, offset)

        if funct3 == 7:  # C.SDSP
            rs2 = (halfword >> 2) & 0x1F
            offset = ((halfword >> 7) & 0x38) | ((halfword >> 1) & 0x1C0)
            return enc_s(0x23, 2, rs2, 3, offset)

    return 0xFFFFFFFF
