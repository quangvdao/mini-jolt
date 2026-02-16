from __future__ import annotations  # keep type hints lightweight

from .constants import RAM_START_ADDRESS  # program_size baseline
from .bytecode import BytecodePreprocessing  # bytecode padding + virtual PC mapping
from .elf import decode_elf  # ELF -> decoded instructions + memory_init + program_end + xlen
from .inline_sequences import expand_program  # full inline expansion spec

def decode_program(elf: bytes):  # Rust guest::program::decode port
    instructions, memory_init, program_end, xlen = decode_elf(elf)
    expanded = expand_program(instructions, xlen)
    for inst in expanded:
        if inst.kind == "INLINE":
            raise NotImplementedError("Guest inlines are out of scope for jolt-python; expanded bytecode contains INLINE")
    return expanded, memory_init, program_end - RAM_START_ADDRESS

def decode_preprocessed_program(elf: bytes):  # ELF -> expanded bytecode -> preprocessed bytecode
    expanded, memory_init, program_size = decode_program(elf)
    return BytecodePreprocessing.preprocess(expanded), memory_init, program_size

