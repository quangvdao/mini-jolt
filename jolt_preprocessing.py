from dataclasses import dataclass  # lightweight preprocessing container

from rv64imac.bytecode import BytecodePreprocessing  # bytecode padding + PC mapping
from zkvm_types import RAMPreprocessing  # verifier-facing RAM preprocessing type

BYTES_PER_INSTRUCTION = 4  # Rust common::constants::BYTES_PER_INSTRUCTION


def _next_multiple_of(x: int, m: int) -> int:  # ceil to multiple-of-m (m>0).
    x, m = int(x), int(m)
    if m <= 0:
        raise ValueError("m must be > 0")
    r = x % m
    return x if r == 0 else (x + (m - r))


def _pack_u64_le(byte_items):  # pack up to 8 (addr, byte) items into a u64 word.
    word = [0] * 8
    for addr, b in byte_items:
        word[int(addr) % 8] = int(b) & 0xFF
    return int.from_bytes(bytes(word), "little")


def ram_preprocess(memory_init):  # Rust: zkvm/ram/mod.rs RAMPreprocessing::preprocess.
    memory_init = list(memory_init)
    if memory_init:
        min_addr = min(int(a) for a, _ in memory_init)
        max_addr = max(int(a) for a, _ in memory_init) + (BYTES_PER_INSTRUCTION - 1)
    else:
        min_addr, max_addr = 0, BYTES_PER_INSTRUCTION - 1

    max_addr = _next_multiple_of(max_addr, 8)
    num_words = max_addr // 8 - (min_addr // 8) + 1
    bytecode_words = [0] * int(num_words)

    memory_init_sorted = sorted(memory_init, key=lambda t: int(t[0]))
    i = 0
    while i < len(memory_init_sorted):
        base_word = int(memory_init_sorted[i][0]) // 8
        chunk = []
        while i < len(memory_init_sorted) and (int(memory_init_sorted[i][0]) // 8) == base_word:
            chunk.append(memory_init_sorted[i])
            i += 1
        idx = int(base_word - (min_addr // 8))
        if 0 <= idx < len(bytecode_words):
            bytecode_words[idx] = _pack_u64_le(chunk)

    return RAMPreprocessing(min_bytecode_address=min_addr, bytecode_words=bytecode_words)


@dataclass(frozen=True)
class JoltPreprocessing:  # Shared verifier preprocessing (Rust: JoltSharedPreprocessing).
    ram: RAMPreprocessing  # packed initial program image words
    bytecode: BytecodePreprocessing  # padded bytecode + PC mapping

    @classmethod
    def preprocess(cls, expanded_bytecode, memory_init):  # Build shared preprocessing from expanded bytecode + ELF memory_init.
        return cls(ram=ram_preprocess(memory_init), bytecode=BytecodePreprocessing.preprocess(expanded_bytecode))
