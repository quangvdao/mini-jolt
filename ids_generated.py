"""Compact ID spec. Regenerate via `python3 jolt-python/scripts/gen_ids_generated.py`."""

from enum import StrEnum  # typed string identifiers

SUMCHECK_IDS = """
SpartanOuter SpartanProductVirtualization SpartanShift InstructionClaimReduction
InstructionInputVirtualization InstructionReadRaf InstructionRaVirtualization RamReadWriteChecking
RamRafEvaluation RamOutputCheck RamValEvaluation RamValFinalEvaluation RamRaClaimReduction
RamHammingBooleanity RamRaVirtualization RegistersClaimReduction RegistersReadWriteChecking
RegistersValEvaluation BytecodeReadRaf Booleanity AdviceClaimReductionCyclePhase
AdviceClaimReduction IncClaimReduction HammingWeightClaimReduction
""".split()  # Rust SumcheckId order

VIRTUAL_POLYS = """
PC UnexpandedPC NextPC NextUnexpandedPC NextIsNoop NextIsVirtual NextIsFirstInSequence
LeftLookupOperand RightLookupOperand LeftInstructionInput RightInstructionInput Product ShouldJump
ShouldBranch WritePCtoRD WriteLookupOutputToRD Rd Imm Rs1Value Rs2Value RdWriteValue Rs1Ra Rs2Ra
RdWa LookupOutput InstructionRaf InstructionRafFlag InstructionRa RegistersVal RamAddress RamRa
RamReadValue RamWriteValue RamVal RamValInit RamValFinal RamHammingWeight UnivariateSkip
OpFlags.AddOperands OpFlags.SubtractOperands OpFlags.MultiplyOperands OpFlags.Load OpFlags.Store
OpFlags.Jump OpFlags.WriteLookupOutputToRD OpFlags.VirtualInstruction OpFlags.Assert
OpFlags.DoNotUpdateUnexpandedPC OpFlags.Advice OpFlags.IsCompressed OpFlags.IsFirstInSequence
OpFlags.IsLastInSequence InstructionFlags.LeftOperandIsPC InstructionFlags.RightOperandIsImm
InstructionFlags.LeftOperandIsRs1Value InstructionFlags.RightOperandIsRs2Value
InstructionFlags.Branch InstructionFlags.IsNoop InstructionFlags.IsRdNotZero LookupTableFlag
""".split()  # Rust VirtualPolynomial (expanded OpFlags/InstructionFlags) + compat extras

COMMITTED_POLYS = """
RdInc RamInc InstructionRa BytecodeRa RamRa TrustedAdvice UntrustedAdvice
""".split()  # Rust CommittedPolynomial (family heads) + compat extras

LOOKUP_TABLES_64 = """
RangeCheck RangeCheckAligned And Andn Or Xor Equal SignedGreaterThanEqual UnsignedGreaterThanEqual
NotEqual SignedLessThan UnsignedLessThan Movsign UpperWord LessThanEqual ValidSignedRemainder
ValidUnsignedRemainder ValidDiv0 HalfwordAlignment WordAlignment LowerHalfWord SignExtendHalfWord
Pow2 Pow2W ShiftRightBitmask VirtualRev8W VirtualSRL VirtualSRA VirtualROTR VirtualROTRW
VirtualChangeDivisor VirtualChangeDivisorW MulUNoOverflow VirtualXORROT32 VirtualXORROT24
VirtualXORROT16 VirtualXORROT63 VirtualXORROTW16 VirtualXORROTW12 VirtualXORROTW8 VirtualXORROTW7
""".split()  # Rust LookupTables<64> order

SumcheckId = StrEnum(
    "SumcheckId", {n: n for n in SUMCHECK_IDS}
)  # typed sumcheck IDs
VirtualPolynomial = StrEnum(
    "VirtualPolynomial", {n.replace('.', '_'): n for n in VIRTUAL_POLYS}
)  # typed virtual polynomial IDs
CommittedPolynomial = StrEnum(
    "CommittedPolynomial", {n: n for n in COMMITTED_POLYS}
)  # typed committed poly IDs
