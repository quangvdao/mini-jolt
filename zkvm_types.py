from dataclasses import dataclass  # lightweight struct-like containers

from rv64imac.constants import RAM_START_ADDRESS  # shared RAM base constant (0x8000_0000)

@dataclass(frozen=True)
class OneHotParams:  # Minimal subset of Rust OneHotParams needed by Stages 2–7.
    ram_k: int  # RAM domain size K (power of two)
    bytecode_k: int = 8  # Bytecode domain size K (power of two).
    log_k_chunk: int = 16  # One-hot chunk bit-length shared across families.
    lookups_ra_virtual_log_k_chunk: int = 16  # Instruction lookups virtual-RA chunk bit-length.

    @property
    def ram_d(self) -> int:  # Rust: ceil_div(log2(ram_k), log_k_chunk).
        log_ram_k = int(self.ram_k).bit_length() - 1
        return (log_ram_k + int(self.log_k_chunk) - 1) // int(self.log_k_chunk)

    @property
    def bytecode_d(self) -> int:  # Rust: ceil_div(log2(bytecode_k), log_k_chunk).
        log_bytecode_k = int(self.bytecode_k).bit_length() - 1
        return (log_bytecode_k + int(self.log_k_chunk) - 1) // int(self.log_k_chunk)

    @property
    def instruction_d(self) -> int:  # Rust: ceil_div(LOG_K_INSTRUCTION(=128), log_k_chunk).
        log_k_instruction = 128  # Rust: instruction_lookups::LOG_K = XLEN*2 = 128 for RV64.
        return (log_k_instruction + int(self.log_k_chunk) - 1) // int(self.log_k_chunk)

    def compute_r_address_chunks(self, r_address: list) -> list[list]:  # Rust: OneHotParams::compute_r_address_chunks.
        r_address = list(r_address)
        k = int(self.log_k_chunk)
        if k <= 0:
            raise ValueError("log_k_chunk must be positive")
        if len(r_address) % k != 0:
            pad = k - (len(r_address) % k)
            r_address = [0] * pad + r_address
        return [r_address[i : i + k] for i in range(0, len(r_address), k)]

@dataclass(frozen=True)
class ReadWriteConfig:  # Minimal subset of Rust ReadWriteConfig needed by Stage 2.
    ram_rw_phase1_num_rounds: int  # cycle vars bound in phase 1
    ram_rw_phase2_num_rounds: int  # address vars bound in phase 2
    registers_rw_phase1_num_rounds: int = 0  # cycle vars bound in phase 1 (registers)
    registers_rw_phase2_num_rounds: int = 0  # address vars bound in phase 2 (registers)

    def needs_single_advice_opening(self, log_T: int) -> bool:  # Rust: config.rs:95-102
        return int(self.ram_rw_phase1_num_rounds) == int(log_T)

@dataclass(frozen=True)
class MemoryLayout:  # Rust-shaped memory layout (common/src/jolt_device.rs) with safe defaults for tests.
    input_start: int  # byte address for public inputs region
    output_start: int  # byte address for public outputs region
    panic: int  # byte address of panic bit
    termination: int  # byte address of termination bit
    program_size: int = 0
    max_trusted_advice_size: int = 0
    trusted_advice_start: int = 0
    trusted_advice_end: int = 0
    max_untrusted_advice_size: int = 0
    untrusted_advice_start: int = 0
    untrusted_advice_end: int = 0
    max_input_size: int = 0
    max_output_size: int = 0
    input_end: int = 0
    output_end: int = 0
    stack_size: int = 0
    stack_end: int = 0
    heap_size: int = 0
    heap_end: int = 0
    io_end: int = 0

    def get_lowest_address(self):  # Rust: MemoryLayout::get_lowest_address (with test-friendly fallback).
        ta = int(self.trusted_advice_start)
        ua = int(self.untrusted_advice_start)
        if ta != 0 and ua != 0:
            return ta if ta < ua else ua
        cands = [
            int(self.input_start),
            int(self.output_start),
            int(self.panic),
            int(self.termination),
            ta,
            ua,
            int(RAM_START_ADDRESS),
        ]
        cands = [x for x in cands if x != 0]
        return min(cands) if cands else int(RAM_START_ADDRESS)

@dataclass(frozen=True)
class RAMPreprocessing:  # Minimal RAM preprocessing (bytecode words + base address).
    min_bytecode_address: int  # byte address
    bytecode_words: list[int]  # u64 words

@dataclass(frozen=True)
class JoltDevice:  # Minimal JoltDevice / program_io needed by Stage 2.
    memory_layout: MemoryLayout  # memory layout for remapping IO addresses
    inputs: bytes = b""  # public input bytes
    trusted_advice: bytes = b""  # trusted advice bytes (public for verifier preamble, but commitment is separate)
    untrusted_advice: bytes = b""  # untrusted advice bytes (public for verifier preamble, but commitment is separate)
    outputs: bytes = b""  # public output bytes
    panic_flag: bool = False  # panic bit (Rust: program_io.panic)
