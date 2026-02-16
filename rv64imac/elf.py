from __future__ import annotations  # allow forward refs + keep type hints lightweight

from dataclasses import dataclass  # simple, readable section container

from .constants import RAM_START_ADDRESS  # Rust-matching section filter + program_end baseline
from .decode import DecodeError, decode_instruction, unimpl_instruction  # decode words; UNIMPL on failure
from .rvc import uncompress_instruction  # Rust-matching RVC -> u32 mapping
from .types import Xlen  # ELF32 vs ELF64 -> xlen

class ElfError(Exception):  # ELF parse error
    pass

SHT_NOBITS = 8  # section has no file bytes (BSS)
SHF_ALLOC = 0x2  # section lives in memory
SHF_EXECINSTR = 0x4  # section is executable (Text)

@dataclass
class ElfSection:  # parsed section header + bytes
    addr: int
    size: int
    flags: int
    shtype: int
    data: bytes

    def is_text(self) -> bool:  # Rust SectionKind::Text heuristic
        return bool(self.flags & SHF_ALLOC) and bool(self.flags & SHF_EXECINSTR) and self.shtype != SHT_NOBITS and self.size > 0

def _u16(b: bytes, off: int) -> int:  # read le u16
    return int.from_bytes(b[off : off + 2], "little")
def _u32(b: bytes, off: int) -> int:  # read le u32
    return int.from_bytes(b[off : off + 4], "little")
def _u64(b: bytes, off: int) -> int:  # read le u64
    return int.from_bytes(b[off : off + 8], "little")

def _parse_sections(elf: bytes) -> tuple[list[ElfSection], Xlen]:  # parse section headers
    if len(elf) < 0x40:
        raise ElfError("ELF too small")
    if elf[0:4] != b"\x7fELF":
        raise ElfError("bad ELF magic")

    ei_class = elf[4]
    ei_data = elf[5]
    if ei_data != 1:
        raise ElfError("only little-endian ELF supported")

    if ei_class == 1:
        xlen = Xlen.Bit32
        e_shoff = _u32(elf, 0x20)
        e_shentsize = _u16(elf, 0x2E)
        e_shnum = _u16(elf, 0x30)
    elif ei_class == 2:
        xlen = Xlen.Bit64
        e_shoff = _u64(elf, 0x28)
        e_shentsize = _u16(elf, 0x3A)
        e_shnum = _u16(elf, 0x3C)
    else:
        raise ElfError("unknown ELF class")

    if e_shoff == 0 or e_shnum == 0:
        raise ElfError("missing section headers")
    if e_shentsize == 0:
        raise ElfError("bad e_shentsize")

    sh_end = e_shoff + e_shentsize * e_shnum
    if sh_end > len(elf):
        raise ElfError("section headers out of range")

    sections: list[ElfSection] = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if ei_class == 1:
            sh_type = _u32(elf, off + 0x04)
            sh_flags = _u32(elf, off + 0x08)
            sh_addr = _u32(elf, off + 0x0C)
            sh_offset = _u32(elf, off + 0x10)
            sh_size = _u32(elf, off + 0x14)
        else:
            sh_type = _u32(elf, off + 0x04)
            sh_flags = _u64(elf, off + 0x08)
            sh_addr = _u64(elf, off + 0x10)
            sh_offset = _u64(elf, off + 0x18)
            sh_size = _u64(elf, off + 0x20)

        if sh_addr < RAM_START_ADDRESS:
            continue
        if sh_type == SHT_NOBITS or sh_size == 0:
            data = b""
        else:
            end = sh_offset + sh_size
            if end > len(elf):
                raise ElfError("section data out of range")
            data = elf[sh_offset:end]
        sections.append(ElfSection(int(sh_addr), int(sh_size), int(sh_flags), int(sh_type), data))

    return sections, xlen

def decode_elf(elf: bytes) -> tuple[list, list[tuple[int, int]], int, Xlen]:  # Rust tracer::decode port
    sections, xlen = _parse_sections(elf)
    instructions = []
    memory_init: list[tuple[int, int]] = []
    program_end = RAM_START_ADDRESS

    for sec in sections:
        start = sec.addr
        end = sec.addr + sec.size
        program_end = max(program_end, end)
        raw = sec.data
        if sec.is_text():
            offset = 0
            while offset < len(raw):
                address = start + offset
                if offset + 1 >= len(raw):
                    break
                half = raw[offset] | (raw[offset + 1] << 8)
                if (half & 0b11) != 0b11:
                    if half == 0x0000:
                        offset += 2
                        continue
                    word = uncompress_instruction(half, xlen)
                    try:
                        inst = decode_instruction(word, address, True)
                    except DecodeError:
                        inst = unimpl_instruction()
                    instructions.append(inst)
                    offset += 2
                    continue
                if offset + 3 >= len(raw):
                    break
                word = int.from_bytes(raw[offset : offset + 4], "little")
                try:
                    inst = decode_instruction(word, address, False)
                except DecodeError:
                    inst = unimpl_instruction()
                instructions.append(inst)
                offset += 4
        for j, byte in enumerate(raw):
            memory_init.append((start + j, byte))

    return instructions, memory_init, int(program_end), xlen
