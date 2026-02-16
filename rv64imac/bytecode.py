from __future__ import annotations  # keep type hints lightweight

from dataclasses import dataclass  # small stateful containers

from .constants import RAM_START_ADDRESS  # bytecode address base
from .types import Instruction, NormalizedOperands  # instruction containers

ALIGNMENT_FACTOR_BYTECODE = 2  # common/src/constants.rs ALIGNMENT_FACTOR_BYTECODE

def _noop() -> Instruction:  # canonical NoOp instruction
    return Instruction(kind="NoOp", address=0, operands=NormalizedOperands(None, None, None, 0))

def _next_pow2(n: int) -> int:  # ceil pow2 with n>0
    return 1 << (n - 1).bit_length()

@dataclass
class BytecodePCMapper:  # maps instruction memory address -> virtual PC
    indices: list[tuple[int, int] | None]

    @staticmethod
    def get_index(address: int) -> int:  # address -> indices slot
        if address < RAM_START_ADDRESS:
            raise ValueError("address must be >= RAM_START_ADDRESS")
        if address % ALIGNMENT_FACTOR_BYTECODE != 0:
            raise ValueError("address must be aligned for bytecode")
        return (address - RAM_START_ADDRESS) // ALIGNMENT_FACTOR_BYTECODE + 1

    @classmethod
    def new(cls, bytecode: list[Instruction]) -> "BytecodePCMapper":  # build address->pc index
        if not bytecode:
            return cls([None])
        last_addr = bytecode[-1].normalize().address
        indices = [None] * (cls.get_index(last_addr) + 1) if last_addr != 0 else [None]
        last_pc = 0
        indices[0] = (last_pc, 0)
        for inst in bytecode:
            inst = inst.normalize()
            if inst.address == 0:
                continue
            last_pc += 1
            idx = cls.get_index(inst.address)
            remaining = int(inst.virtual_sequence_remaining or 0)
            if indices[idx] is not None:
                _, max_seq = indices[idx]
                if remaining >= max_seq:
                    raise RuntimeError(f"Bytecode has non-decreasing inline sequences at index {idx}")
            else:
                indices[idx] = (last_pc, remaining)
        return cls(indices)

    def get_pc(self, address: int, virtual_sequence_remaining: int) -> int:  # mapped PC for (addr, remaining)
        idx = self.get_index(address)
        ent = self.indices[idx]
        if ent is None:
            raise KeyError("PC for address not found")
        base_pc, max_inline_seq = ent
        return base_pc + (max_inline_seq - int(virtual_sequence_remaining))

@dataclass
class BytecodePreprocessing:  # jolt-core BytecodePreprocessing
    code_size: int
    bytecode: list[Instruction]
    pc_map: BytecodePCMapper

    @classmethod
    def preprocess(cls, bytecode: list[Instruction]) -> "BytecodePreprocessing":  # pad + map
        bytecode = list(bytecode)
        bytecode.insert(0, _noop())
        pc_map = BytecodePCMapper.new(bytecode)
        code_size = max(_next_pow2(len(bytecode)), 2)
        if len(bytecode) < code_size:
            bytecode.extend([_noop()] * (code_size - len(bytecode)))
        return cls(code_size=code_size, bytecode=bytecode, pc_map=pc_map)

    def get_pc_for_instruction(self, inst: Instruction) -> int:  # mapped pc for a normalized instruction
        if inst.kind == "NoOp":
            return 0
        inst = inst.normalize()
        return self.pc_map.get_pc(inst.address, int(inst.virtual_sequence_remaining or 0))

