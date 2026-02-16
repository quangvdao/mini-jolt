from __future__ import annotations  # allow forward refs + keep type hints lightweight

from dataclasses import dataclass  # small immutable-ish containers
from enum import Enum  # Rust-like xlen enum
from typing import Optional  # Optional[u8] operands

class Xlen(Enum):  # register width
    Bit32 = 32
    Bit64 = 64

    @property
    def bits(self) -> int:  # width in bits
        return int(self.value)

@dataclass
class NormalizedOperands:  # decoded operands
    rs1: Optional[int]
    rs2: Optional[int]
    rd: Optional[int]
    imm: int

@dataclass
class Instruction:  # normalized instruction
    kind: str
    address: int
    operands: NormalizedOperands
    virtual_sequence_remaining: Optional[int] = None
    is_first_in_sequence: bool = False
    is_compressed: bool = False
    advice: int = 0

    def normalize(self) -> "Instruction":  # keep Rust-shaped API
        return self
