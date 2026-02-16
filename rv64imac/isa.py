from __future__ import annotations  # keep type hints lightweight

from .extensions import ISAProfile, assert_kind_allowed, extension_for_kind  # extension/profile gating + mapping
from .types import Instruction, Xlen  # core instruction container + xlen tag

CIRCUIT_FLAGS = [  # jolt-core/src/zkvm/instruction/mod.rs CircuitFlags order
    "AddOperands", "SubtractOperands", "MultiplyOperands", "Load", "Store", "Jump", "WriteLookupOutputToRD",
    "VirtualInstruction", "Assert", "DoNotUpdateUnexpandedPC", "Advice", "IsCompressed", "IsFirstInSequence", "IsLastInSequence",
]

INSTRUCTION_FLAGS = [  # jolt-core/src/zkvm/instruction/mod.rs InstructionFlags order
    "LeftOperandIsPC", "RightOperandIsImm", "LeftOperandIsRs1Value", "RightOperandIsRs2Value", "Branch", "IsNoop", "IsRdNotZero",
]

_CF = {name: i for i, name in enumerate(CIRCUIT_FLAGS)}  # name -> index
_IF = {name: i for i, name in enumerate(INSTRUCTION_FLAGS)}  # name -> index

def _source_kinds() -> set[str]:  # scan local sources for kind strings
    import re  # local import keeps module import light
    from pathlib import Path  # local import keeps module import light

    here = Path(__file__).resolve().parent
    src = (here / "decode.py").read_text(encoding="utf-8") + (here / "inline_sequences.py").read_text(encoding="utf-8")
    pat = r'\"(?:[A-Z]{2,}[A-Z0-9_]*|NoOp|UNIMPL|INLINE|Virtual[A-Za-z0-9_]*|Advice[A-Za-z0-9_]*)\"'
    return set(s.strip('"') for s in re.findall(pat, src)) | {"NoOp", "UNIMPL"}

_SOURCE_KINDS = _source_kinds()

_LOOKUP_TABLE_NON_NONE = (  # jolt-core/src/zkvm/instruction/*.rs InstructionLookup (only non-None entries)
    {k: "RangeCheck" for k in ("ADD", "ADDI", "AUIPC", "JAL", "LUI", "MUL", "SUB", "VirtualAdvice", "VirtualAdviceLen", "VirtualAdviceLoad", "VirtualMULI")}
    | {k: "And" for k in ("AND", "ANDI")}
    | {k: "Or" for k in ("OR", "ORI")}
    | {k: "Xor" for k in ("XOR", "XORI")}
    | {k: "SignedLessThan" for k in ("BLT", "SLT", "SLTI")}
    | {k: "UnsignedLessThan" for k in ("BLTU", "SLTIU", "SLTU")}
    | {"ANDN": "Andn", "BEQ": "Equal", "BNE": "NotEqual", "BGE": "SignedGreaterThanEqual", "BGEU": "UnsignedGreaterThanEqual", "JALR": "RangeCheckAligned", "MULHU": "UpperWord"}
    | {k: k for k in ("VirtualXORROT16", "VirtualXORROT24", "VirtualXORROT32", "VirtualXORROT63", "VirtualXORROTW7", "VirtualXORROTW8", "VirtualXORROTW12", "VirtualXORROTW16")}
    | {"VirtualAssertEQ": "Equal", "VirtualAssertLTE": "LessThanEqual", "VirtualAssertHalfwordAlignment": "HalfwordAlignment", "VirtualAssertWordAlignment": "WordAlignment"}
    | {"VirtualAssertMulUNoOverflow": "MulUNoOverflow", "VirtualAssertValidDiv0": "ValidDiv0", "VirtualAssertValidUnsignedRemainder": "ValidUnsignedRemainder"}
    | {"VirtualChangeDivisor": "VirtualChangeDivisor", "VirtualChangeDivisorW": "VirtualChangeDivisorW", "VirtualMovsign": "Movsign", "VirtualRev8W": "VirtualRev8W", "VirtualROTRI": "VirtualROTR", "VirtualROTRIW": "VirtualROTRW", "VirtualSignExtendWord": "SignExtendHalfWord", "VirtualShiftRightBitmask": "ShiftRightBitmask", "VirtualShiftRightBitmaskI": "ShiftRightBitmask"}
    | {"VirtualPow2": "Pow2", "VirtualPow2I": "Pow2", "VirtualPow2IW": "Pow2W", "VirtualPow2W": "Pow2W"}
    | {"VirtualSRA": "VirtualSRA", "VirtualSRAI": "VirtualSRA", "VirtualSRL": "VirtualSRL", "VirtualSRLI": "VirtualSRL", "VirtualZeroExtendWord": "LowerHalfWord"}
)

_LOOKUP_TABLE_BY_KIND = {k: None for k in _SOURCE_KINDS}  # kind -> lookup table name or None
_LOOKUP_TABLE_BY_KIND.update(_LOOKUP_TABLE_NON_NONE)

def _circuit_on(kind: str) -> tuple[str, ...]:  # static circuit flags per kind (meta flags handled separately)
    if kind == "LD":
        return ("Load",)
    if kind == "SD":
        return ("Store",)
    if kind in ("JAL", "JALR"):
        return ("AddOperands", "Jump")
    if kind in ("ADD", "ADDI", "AUIPC", "LUI"):
        return ("AddOperands", "WriteLookupOutputToRD")
    if kind == "SUB":
        return ("SubtractOperands", "WriteLookupOutputToRD")
    if kind in ("MUL", "MULHU", "VirtualMULI"):
        return ("MultiplyOperands", "WriteLookupOutputToRD")
    if kind.startswith("VirtualAdvice"):
        return ("Advice", "WriteLookupOutputToRD")
    if kind.startswith("VirtualAssert"):
        if kind in ("VirtualAssertHalfwordAlignment", "VirtualAssertWordAlignment"):
            return ("AddOperands", "Assert")
        if kind == "VirtualAssertMulUNoOverflow":
            return ("Assert", "MultiplyOperands")
        return ("Assert",)
    if kind.startswith(("VirtualPow2", "VirtualShiftRightBitmask")) or kind in ("VirtualSignExtendWord", "VirtualZeroExtendWord", "VirtualRev8W"):
        return ("AddOperands", "WriteLookupOutputToRD")
    if kind.startswith(("VirtualROTR", "VirtualXORROT")) or kind in (
        "AND",
        "ANDI",
        "ANDN",
        "OR",
        "ORI",
        "SLT",
        "SLTI",
        "SLTIU",
        "SLTU",
        "VirtualChangeDivisor",
        "VirtualChangeDivisorW",
        "VirtualMovsign",
        "VirtualSRA",
        "VirtualSRAI",
        "VirtualSRL",
        "VirtualSRLI",
        "XOR",
        "XORI",
    ):
        return ("WriteLookupOutputToRD",)
    return ()

def _instr_on(kind: str) -> tuple[str, ...]:  # static instruction flags per kind (rd!=0 handled separately)
    if kind in ("BEQ", "BGE", "BGEU", "BLT", "BLTU", "BNE"):
        return ("Branch", "LeftOperandIsRs1Value", "RightOperandIsRs2Value")
    if kind in ("AUIPC", "JAL"):
        return ("IsRdNotZero", "LeftOperandIsPC", "RightOperandIsImm")
    if kind == "JALR":
        return ("IsRdNotZero", "LeftOperandIsRs1Value", "RightOperandIsImm")
    if kind == "LUI":
        return ("IsRdNotZero", "RightOperandIsImm")
    if kind in ("LD", "ECALL", "EBREAK") or kind.startswith("VirtualAdvice") or kind == "VirtualHostIO":
        return ("IsRdNotZero",)
    if kind.startswith("VirtualAssert"):
        if kind in ("VirtualAssertHalfwordAlignment", "VirtualAssertWordAlignment"):
            return ("LeftOperandIsRs1Value", "RightOperandIsImm")
        return ("LeftOperandIsRs1Value", "RightOperandIsRs2Value")
    if kind in ("VirtualPow2I", "VirtualPow2IW", "VirtualShiftRightBitmaskI"):
        return ("IsRdNotZero", "RightOperandIsImm")
    if kind in ("VirtualPow2W", "VirtualRev8W", "VirtualSignExtendWord", "VirtualZeroExtendWord"):
        return ("IsRdNotZero", "LeftOperandIsRs1Value")
    if kind.startswith("VirtualXORROT") or kind in (
        "ADD",
        "AND",
        "ANDN",
        "MUL",
        "MULHU",
        "OR",
        "SLT",
        "SLTU",
        "SUB",
        "VirtualChangeDivisor",
        "VirtualChangeDivisorW",
        "VirtualSRA",
        "VirtualSRL",
        "XOR",
    ):
        return ("IsRdNotZero", "LeftOperandIsRs1Value", "RightOperandIsRs2Value")
    if kind.startswith("VirtualROTR") or kind.startswith(("VirtualMULI", "VirtualMovsign", "VirtualPow2", "VirtualSRAI", "VirtualSRLI", "VirtualShiftRightBitmask")) or kind in (
        "ADDI",
        "ANDI",
        "ORI",
        "SLTI",
        "SLTIU",
        "XORI",
    ):
        return ("IsRdNotZero", "LeftOperandIsRs1Value", "RightOperandIsImm")
    return ()

_META_MINIMAL = {  # kinds that only set IsFirstInSequence/IsCompressed in circuit flags
    "EBREAK",
    "ECALL",
    "FENCE",
    "VirtualHostIO",
}

def lookup_table(inst: Instruction, xlen: Xlen) -> str | None:  # per-instruction lookup table selector
    _ = xlen
    if inst.kind not in _LOOKUP_TABLE_BY_KIND:
        raise KeyError(f"lookup table not defined for instruction kind {inst.kind!r}")
    return _LOOKUP_TABLE_BY_KIND[inst.kind]

def circuit_flags(inst: Instruction) -> list[bool]:  # CircuitFlags bitvector
    flags = [False] * len(CIRCUIT_FLAGS)
    if inst.kind == "NoOp":
        flags[_CF["DoNotUpdateUnexpandedPC"]] = True
        return flags
    if inst.kind in _META_MINIMAL:
        flags[_CF["IsFirstInSequence"]] = bool(inst.is_first_in_sequence)
        flags[_CF["IsCompressed"]] = bool(inst.is_compressed)
    else:
        flags[_CF["VirtualInstruction"]] = inst.virtual_sequence_remaining is not None
        flags[_CF["DoNotUpdateUnexpandedPC"]] = (inst.virtual_sequence_remaining or 0) != 0
        flags[_CF["IsFirstInSequence"]] = bool(inst.is_first_in_sequence)
        flags[_CF["IsCompressed"]] = bool(inst.is_compressed)
        if inst.kind == "JALR":
            flags[_CF["IsLastInSequence"]] = inst.virtual_sequence_remaining == 0
    for name in _circuit_on(inst.kind):
        flags[_CF[name]] = True
    return flags

def instruction_flags(inst: Instruction) -> list[bool]:  # InstructionFlags bitvector
    flags = [False] * len(INSTRUCTION_FLAGS)
    if inst.kind == "NoOp":
        flags[_IF["IsNoop"]] = True
        return flags
    for name in _instr_on(inst.kind):
        if name == "IsRdNotZero":
            flags[_IF[name]] = (inst.operands.rd or 0) != 0
        else:
            flags[_IF[name]] = True
    return flags

def validate_isa_tables() -> None:  # validate kind coverage + extension mapping
    kinds = _source_kinds()

    missing = sorted(k for k in kinds if k not in _LOOKUP_TABLE_BY_KIND)
    if missing:
        raise AssertionError(f"missing ISA metadata for kinds: {missing}")

    for k in sorted(kinds):
        _ = extension_for_kind(k)

