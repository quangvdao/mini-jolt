from __future__ import annotations  # keep type hints lightweight

from bisect import insort  # keep allocator pool sorted
from dataclasses import dataclass  # small stateful helpers

from .constants import (  # base regs + supported ZeroOS CSRs
    CSR_MCAUSE,
    CSR_MEPC,
    CSR_MSCRATCH,
    CSR_MSTATUS,
    CSR_MTVEC,
    CSR_MTVAL,
    RISCV_REGISTER_COUNT,
)
from .extensions import ISAProfile, assert_kind_allowed  # extension/profile gating
from .types import Instruction, NormalizedOperands, Xlen  # instruction containers + xlen

def u64(x: int) -> int:  # cast to u64
    return x & 0xFFFF_FFFF_FFFF_FFFF

@dataclass
class InstrReg:  # allocated instruction temp register
    allocator: "VirtualRegisterAllocator"  # owner allocator
    reg: int  # absolute register index
    freed: bool = False  # double-free guard

    def free(self):  # return reg to allocator
        if self.freed:
            return
        self.freed = True
        self.allocator.release(self.reg)

    def __int__(self) -> int:  # allow passing as int
        return int(self.reg)

@dataclass
class AllocFrame:  # RAII-ish allocator scope
    allocator: "VirtualRegisterAllocator"  # shared pool
    regs: list[InstrReg]  # allocated in this frame

    def alloc(self) -> InstrReg:  # allocate one temp reg
        r = self.allocator.allocate()
        self.regs.append(r)
        return r

    def __enter__(self) -> "AllocFrame":  # enter scope
        return self

    def __exit__(self, exc_type, exc, tb):  # free all regs
        for r in reversed(self.regs):
            r.free()
        return False

@dataclass
class VirtualRegisterAllocator:  # tracer virtual-register allocator
    available: list[int] | None = None

    def __post_init__(self):  # init sorted pool
        if self.available is None:
            self.available = list(range(RISCV_REGISTER_COUNT + 8, RISCV_REGISTER_COUNT + 15))

    def frame(self) -> AllocFrame:  # scoped allocator helper
        return AllocFrame(self, [])

    def allocate(self) -> InstrReg:  # allocate v40..v46
        if not self.available:
            raise RuntimeError("Failed to allocate virtual register for instruction: No registers left")
        r = self.available.pop(0)
        return InstrReg(self, r)

    def release(self, r: int):  # free register
        if self.available is None:
            self.available = []
        insort(self.available, int(r))

    def reservation_w_register(self) -> int:  # v32
        return RISCV_REGISTER_COUNT + 0

    def reservation_d_register(self) -> int:  # v33
        return RISCV_REGISTER_COUNT + 1

    def trap_handler_register(self) -> int:  # v34 (mtvec)
        return RISCV_REGISTER_COUNT + 2

    def mscratch_register(self) -> int:  # v35
        return RISCV_REGISTER_COUNT + 3

    def mepc_register(self) -> int:  # v36
        return RISCV_REGISTER_COUNT + 4

    def mcause_register(self) -> int:  # v37
        return RISCV_REGISTER_COUNT + 5

    def mtval_register(self) -> int:  # v38
        return RISCV_REGISTER_COUNT + 6

    def mstatus_register(self) -> int:  # v39
        return RISCV_REGISTER_COUNT + 7

def _csr_virtual_reg(allocator: VirtualRegisterAllocator, csr_addr: int) -> int | None:  # supported CSR addr -> virtual reg
    return {
        CSR_MSTATUS: allocator.mstatus_register(),
        CSR_MTVEC: allocator.trap_handler_register(),
        CSR_MSCRATCH: allocator.mscratch_register(),
        CSR_MEPC: allocator.mepc_register(),
        CSR_MCAUSE: allocator.mcause_register(),
        CSR_MTVAL: allocator.mtval_register(),
    }.get(csr_addr)

def _expand_lr(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, *, is_d: bool) -> list[Instruction]:  # LRW/LRD expansion
    if is_d and xlen != Xlen.Bit64:
        raise RuntimeError("LR.D is only available in RV64")
    v_reservation_w = allocator.reservation_w_register()
    v_reservation_d = allocator.reservation_d_register()
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    if is_d:
        asm.mov(v_reservation_d, inst.operands.rs1)
        asm.mov(v_reservation_w, 0)
        asm.emit_i("LD", inst.operands.rd, inst.operands.rs1, 0)
    else:
        asm.mov(v_reservation_w, inst.operands.rs1)
        asm.mov(v_reservation_d, 0)
        asm.emit_i("VirtualLW" if xlen == Xlen.Bit32 else "LW", inst.operands.rd, inst.operands.rs1, 0)
    return asm.finalize()

def _expand_scw(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # SCW expansion
    v_reservation = allocator.reservation_w_register()
    v_reservation_d = allocator.reservation_d_register()
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v_success, v_one, v_addr_diff = f.alloc(), f.alloc(), f.alloc()
        v_mem, v_diff = f.alloc(), f.alloc()
        asm.emit_j("VirtualAdvice", v_success, 0)
        asm.emit_i("ADDI", v_one, 0, 1)
        asm.emit_b("VirtualAssertLTE", v_success, v_one, 0)
        asm.emit_r("SUB", v_addr_diff, v_reservation, inst.operands.rs1)
        asm.emit_r("MUL", v_addr_diff, v_success, v_addr_diff)
        asm.emit_b("VirtualAssertEQ", v_addr_diff, 0, 0)
        if xlen == Xlen.Bit32:
            asm.emit_i("VirtualLW", v_mem, inst.operands.rs1, 0)
        else:
            asm.mov(v_reservation, v_success)
            asm.emit_i("LW", v_mem, inst.operands.rs1, 0)
        asm.emit_r("SUB", v_diff, inst.operands.rs2, v_mem)
        asm.emit_r("MUL", v_diff, v_diff, v_success if xlen == Xlen.Bit32 else v_reservation)
        asm.emit_r("ADD", v_diff, v_mem, v_diff)
        if xlen == Xlen.Bit32:
            asm.emit_s("VirtualSW", inst.operands.rs1, v_diff, 0)
            asm.mov(v_reservation, 0)
            asm.mov(v_reservation_d, 0)
            asm.emit_i("XORI", inst.operands.rd, v_success, 1)
        else:
            asm.mov(v_reservation_d, v_diff)
            asm.emit_s("SW", inst.operands.rs1, v_reservation_d, 0)
            asm.emit_i("XORI", inst.operands.rd, v_reservation, 1)
            asm.mov(v_reservation, 0)
            asm.mov(v_reservation_d, 0)
        return asm.finalize()

def _expand_scd(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # SCD expansion
    if xlen != Xlen.Bit64:
        raise RuntimeError("SC.D is only available in RV64")
    v_reservation = allocator.reservation_d_register()
    v_reservation_w = allocator.reservation_w_register()
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v_success, v_one, v_addr_diff = f.alloc(), f.alloc(), f.alloc()
        v_mem, v_diff = f.alloc(), f.alloc()
        asm.emit_j("VirtualAdvice", v_success, 0)
        asm.emit_i("ADDI", v_one, 0, 1)
        asm.emit_b("VirtualAssertLTE", v_success, v_one, 0)
        asm.emit_r("SUB", v_addr_diff, v_reservation, inst.operands.rs1)
        asm.emit_r("MUL", v_addr_diff, v_success, v_addr_diff)
        asm.emit_b("VirtualAssertEQ", v_addr_diff, 0, 0)
        asm.emit_i("LD", v_mem, inst.operands.rs1, 0)
        asm.emit_r("SUB", v_diff, inst.operands.rs2, v_mem)
        asm.emit_r("MUL", v_diff, v_diff, v_success)
        asm.emit_r("ADD", v_diff, v_mem, v_diff)
        asm.emit_s("SD", inst.operands.rs1, v_diff, 0)
        asm.mov(v_reservation, 0)
        asm.mov(v_reservation_w, 0)
        asm.emit_i("XORI", inst.operands.rd, v_success, 1)
        return asm.finalize()

_WORD_OP = {"ADDW": ("ADD", False), "SUBW": ("SUB", False), "MULW": ("MUL", False), "ADDIW": ("ADDI", True)}  # word ops

def _expand_word_op(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, op: str, is_imm: bool) -> list[Instruction]:  # ADDW/ADDIW/SUBW/MULW
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    if is_imm:
        asm.emit_i(op, inst.operands.rd, inst.operands.rs1, inst.operands.imm)
    else:
        asm.emit_r(op, inst.operands.rd, inst.operands.rs1, inst.operands.rs2)
    asm.signext_w(inst.operands.rd, inst.operands.rd)
    return asm.finalize()

def _expand_shiftw(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # *W shifts (RV64-only)
    with allocator.frame() as f:
        if xlen == Xlen.Bit32:
            raise RuntimeError(f"{inst.kind} is invalid in 32b mode")
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        if inst.kind in ("SRLW", "SRAW"):
            v_rs1 = f.alloc()
            v_bitmask = f.alloc()
            if inst.kind == "SRLW":
                asm.emit_i("SLLI", v_rs1, inst.operands.rs1, 32)
                asm.emit_i("ORI", v_bitmask, inst.operands.rs2, 32)
                asm.emit_i("VirtualShiftRightBitmask", v_bitmask, v_bitmask, 0)
                asm.emit_vshift_r("VirtualSRL", inst.operands.rd, v_rs1, v_bitmask)
            else:
                asm.signext_w(v_rs1, inst.operands.rs1)
                asm.emit_i("ANDI", v_bitmask, inst.operands.rs2, 0x1F)
                asm.emit_i("VirtualShiftRightBitmask", v_bitmask, v_bitmask, 0)
                asm.emit_vshift_r("VirtualSRA", inst.operands.rd, v_rs1, v_bitmask)
            asm.signext_w(inst.operands.rd, inst.operands.rd)
            return asm.finalize()
        if inst.kind in ("SRLIW", "SRAIW"):
            v_rs1 = f.alloc()
            if inst.kind == "SRLIW":
                asm.emit_i("SLLI", v_rs1, inst.operands.rs1, 32)
                shift = (inst.operands.imm & 0x1F) + 32
                bitmask = _sr_imm_bitmask(64, shift)
                asm.emit_vshift_i("VirtualSRLI", inst.operands.rd, v_rs1, bitmask)
            else:
                asm.signext_w(v_rs1, inst.operands.rs1)
                shift = inst.operands.imm & 0x1F
                # Rust tracer (SRAIW) computes the bitmask over a 64-bit lane after sign-extending rs1.
                bitmask = _sr_imm_bitmask(64, shift)
                asm.emit_vshift_i("VirtualSRAI", inst.operands.rd, v_rs1, bitmask)
            asm.signext_w(inst.operands.rd, inst.operands.rd)
            return asm.finalize()
        if inst.kind == "SLLW":
            v_pow2 = f.alloc()
            asm.emit_i("VirtualPow2W", v_pow2, inst.operands.rs2, 0)
            asm.emit_r("MUL", inst.operands.rd, inst.operands.rs1, v_pow2)
            asm.signext_w(inst.operands.rd, inst.operands.rd)
            return asm.finalize()
    if inst.kind == "SLLIW":
        if xlen == Xlen.Bit32:
            raise RuntimeError("SLLIW is invalid in 32b mode")
        shift = inst.operands.imm & 0x1F
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        asm.emit_i("VirtualMULI", inst.operands.rd, inst.operands.rs1, 1 << shift)
        asm.signext_w(inst.operands.rd, inst.operands.rd)
        return asm.finalize()
    raise RuntimeError(f"unexpected shift kind {inst.kind!r}")

def _expand_csr(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # CSRRW/CSRRS expansion
    _ = xlen
    csr_addr = inst.operands.imm & 0xFFF
    if csr_addr == 0:
        return []
    virtual_reg = _csr_virtual_reg(allocator, csr_addr)
    if virtual_reg is None:
        raise RuntimeError(f"{inst.kind}: Unsupported CSR 0x{csr_addr:03x}")
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    if inst.kind == "CSRRW":
        if inst.operands.rd == 0:
            asm.mov(virtual_reg, inst.operands.rs1)
            return asm.finalize()
        if inst.operands.rd == inst.operands.rs1:
            temp = allocator.allocate()
            asm.mov(temp, inst.operands.rs1)
            asm.mov(inst.operands.rd, virtual_reg)
            asm.mov(virtual_reg, temp)
            temp.free()
            return asm.finalize()
        asm.mov(inst.operands.rd, virtual_reg)
        asm.mov(virtual_reg, inst.operands.rs1)
        return asm.finalize()
    if inst.kind == "CSRRS":
        if inst.operands.rs1 == 0:
            asm.mov(inst.operands.rd, virtual_reg)
            return asm.finalize()
        if inst.operands.rd == 0:
            asm.emit_r("OR", virtual_reg, virtual_reg, inst.operands.rs1)
            return asm.finalize()
        if inst.operands.rd == inst.operands.rs1:
            temp = allocator.allocate()
            asm.mov(temp, inst.operands.rs1)
            asm.mov(inst.operands.rd, virtual_reg)
            asm.emit_r("OR", virtual_reg, virtual_reg, temp)
            temp.free()
            return asm.finalize()
        asm.mov(inst.operands.rd, virtual_reg)
        asm.emit_r("OR", virtual_reg, virtual_reg, inst.operands.rs1)
        return asm.finalize()
    raise RuntimeError(f"unexpected CSR kind {inst.kind!r}")

def _expand_amo_swap(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, *, is_d: bool) -> list[Instruction]:  # AMOSWAPW/AMOSWAPD expansion
    if is_d:
        if xlen != Xlen.Bit64:
            raise RuntimeError("AMOSWAPD is only available in RV64")
        with allocator.frame() as f:
            asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
            v_rd = f.alloc()
            asm.emit_i("LD", v_rd, inst.operands.rs1, 0)
            asm.emit_s("SD", inst.operands.rs1, inst.operands.rs2, 0)
            asm.mov(inst.operands.rd, v_rd)
            return asm.finalize()
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    if xlen == Xlen.Bit32:
        with allocator.frame() as f:
            v_rd = f.alloc()
            asm.emit_align("VirtualAssertWordAlignment", inst.operands.rs1, 0)
            asm.emit_i("VirtualLW", v_rd, inst.operands.rs1, 0)
            asm.emit_s("VirtualSW", inst.operands.rs1, inst.operands.rs2, 0)
            asm.mov(inst.operands.rd, v_rd)
            return asm.finalize()
    v_mask = allocator.allocate()
    v_dword = allocator.allocate()
    v_shift = allocator.allocate()
    v_rd = allocator.allocate()
    amo_pre64(asm, inst.operands.rs1, v_rd, v_dword, v_shift)
    amo_post64(asm, inst.operands.rs1, inst.operands.rs2, v_dword, v_shift, v_mask, inst.operands.rd, v_rd)
    v_mask.free()
    v_dword.free()
    v_shift.free()
    v_rd.free()
    return asm.finalize()

@dataclass
class InstrAssembler:  # build an inline sequence
    address: int
    is_compressed: bool
    xlen: Xlen
    allocator: VirtualRegisterAllocator
    seq: list[Instruction]

    @classmethod
    def new(cls, address: int, is_compressed: bool, xlen: Xlen, allocator: VirtualRegisterAllocator):  # constructor
        return cls(address, is_compressed, xlen, allocator, [])

    def finalize(self) -> list[Instruction]:  # set virtual sequence metadata
        if not self.seq:
            raise RuntimeError("sequence should not be empty")
        # NOTE: virtual-sequence metadata (remaining / first) must be assigned *after*
        # full recursive expansion, so that it matches the final flattened sequence.
        self.seq[-1].is_compressed = self.is_compressed
        return self.seq

    def _emit(self, kind: str, ops: NormalizedOperands):  # emit then recursively expand
        inst = Instruction(kind=kind, address=self.address, operands=ops, is_compressed=False)
        # Tracer semantics: emitted instructions are immediately expanded via their own
        # inline sequences, sharing the same allocator. This affects temp-register
        # allocation (v40..v46) across nested expansions and must match Rust bytecode.
        self.seq.extend(inline_sequence(inst, self.allocator, self.xlen))

    def emit_r(self, kind: str, rd: int | InstrReg, rs1: int | InstrReg, rs2: int | InstrReg):  # R-type emit
        self._emit(kind, NormalizedOperands(rs1=int(rs1), rs2=int(rs2), rd=int(rd), imm=0))

    def emit_i(self, kind: str, rd: int | InstrReg, rs1: int | InstrReg, imm: int):  # I-type emit
        self._emit(kind, NormalizedOperands(rs1=int(rs1), rs2=None, rd=int(rd), imm=int(imm)))

    def emit_s(self, kind: str, rs1: int | InstrReg, rs2: int | InstrReg, imm: int):  # S-type emit
        self._emit(kind, NormalizedOperands(rs1=int(rs1), rs2=int(rs2), rd=None, imm=int(imm)))

    def emit_u(self, kind: str, rd: int | InstrReg, imm: int):  # U-type emit
        self._emit(kind, NormalizedOperands(rs1=None, rs2=None, rd=int(rd), imm=int(imm)))

    def emit_b(self, kind: str, rs1: int | InstrReg, rs2: int | InstrReg, imm: int):  # B-type emit
        self._emit(kind, NormalizedOperands(rs1=int(rs1), rs2=int(rs2), rd=None, imm=int(imm)))

    def emit_j(self, kind: str, rd: int | InstrReg, imm: int):  # J-type emit
        self._emit(kind, NormalizedOperands(rs1=None, rs2=None, rd=int(rd), imm=int(imm)))

    def emit_align(self, kind: str, rs1: int | InstrReg, imm: int):  # assert-align emit (rs1, imm)
        self._emit(kind, NormalizedOperands(rs1=int(rs1), rs2=None, rd=None, imm=int(imm)))

    def emit_vshift_r(self, kind: str, rd: int | InstrReg, rs1: int | InstrReg, rs2: int | InstrReg):  # virtual right shift (R)
        self.emit_r(kind, rd, rs1, rs2)

    def emit_vshift_i(self, kind: str, rd: int | InstrReg, rs1: int | InstrReg, imm: int):  # virtual right shift (I)
        self.emit_i(kind, rd, rs1, imm)

    def mov(self, rd: int | InstrReg, rs1: int | InstrReg):  # rd <- rs1
        self.emit_i("ADDI", rd, rs1, 0)

    def signext_w(self, rd: int | InstrReg, rs1: int | InstrReg):  # sign-extend low 32b
        self.emit_i("VirtualSignExtendWord", rd, rs1, 0)

    def zeroext_w(self, rd: int | InstrReg, rs1: int | InstrReg):  # zero-extend low 32b
        self.emit_i("VirtualZeroExtendWord", rd, rs1, 0)

def _sr_imm_bitmask(n: int, shift: int) -> int:  # build VirtualSRLI/VirtualSRAI immediate bitmask
    ones = (1 << (n - shift)) - 1
    return (ones << shift) & 0xFFFF_FFFF_FFFF_FFFF

def amo_pre64(asm: InstrAssembler, rs1: int, v_rd: int | InstrReg, v_dword: int | InstrReg, v_shift: int | InstrReg):  # AMO.W prelude on RV64
    asm.emit_align("VirtualAssertWordAlignment", rs1, 0)
    asm.emit_i("ANDI", v_shift, rs1, u64(-8))
    asm.emit_i("LD", v_dword, v_shift, 0)
    asm.emit_i("SLLI", v_shift, rs1, 3)
    asm.emit_r("SRL", v_rd, v_dword, v_shift)

def amo_post64(asm: InstrAssembler, rs1: int, rs2: int | InstrReg, v_dword: int | InstrReg, v_shift: int | InstrReg, v_mask: int | InstrReg, rd: int, v_rd: int | InstrReg):  # AMO.W postlude on RV64
    asm.emit_i("ORI", v_mask, 0, u64(-1))
    asm.emit_i("SRLI", v_mask, v_mask, 32)
    asm.emit_r("SLL", v_mask, v_mask, v_shift)
    asm.emit_r("SLL", v_shift, rs2, v_shift)
    asm.emit_r("XOR", v_shift, v_dword, v_shift)
    asm.emit_r("AND", v_shift, v_shift, v_mask)
    asm.emit_r("XOR", v_dword, v_dword, v_shift)
    asm.emit_i("ANDI", v_mask, rs1, u64(-8))
    asm.emit_s("SD", v_mask, v_dword, 0)
    asm.emit_i("VirtualSignExtendWord", rd, v_rd, 0)

def amo_pre32(asm: InstrAssembler, rs1: int, v_rd: int | InstrReg):  # AMO.W prelude on RV32
    asm.emit_align("VirtualAssertWordAlignment", rs1, 0)
    asm.emit_i("VirtualLW", v_rd, rs1, 0)

def amo_post32(asm: InstrAssembler, rs2: int | InstrReg, rs1: int, rd: int, v_rd: int | InstrReg):  # AMO.W postlude on RV32
    asm.emit_s("VirtualSW", rs1, rs2, 0)
    asm.emit_i("ADDI", rd, v_rd, 0)

_AMO_BINOP_W = {"AMOADDW": "ADD", "AMOANDW": "AND", "AMOORW": "OR", "AMOXORW": "XOR"}  # AMO.W binops
_AMO_BINOP_D = {"AMOADDD": "ADD", "AMOANDD": "AND", "AMOORD": "OR", "AMOXORD": "XOR"}  # AMO.D binops
_AMO_CMP_D = {  # kind -> (slt_op, swap)
    "AMOMIND": ("SLT", False), "AMOMAXD": ("SLT", True),
    "AMOMINUD": ("SLTU", False), "AMOMAXUD": ("SLTU", True),
}
_AMO_CMP_W = {  # kind -> (slt_op, swap, extend_kind)
    "AMOMINW": ("SLT", False, "VirtualSignExtendWord"), "AMOMAXW": ("SLT", True, "VirtualSignExtendWord"),
    "AMOMINUW": ("SLTU", False, "VirtualZeroExtendWord"), "AMOMAXUW": ("SLTU", True, "VirtualZeroExtendWord"),
}

def _expand_amo_binop_w(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, op: str) -> list[Instruction]:  # AMO{ADD,AND,OR,XOR}W
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    v_rd = allocator.allocate()
    v_rs2 = allocator.allocate()
    if xlen == Xlen.Bit32:
        amo_pre32(asm, inst.operands.rs1, v_rd)
        asm.emit_r(op, v_rs2, v_rd, inst.operands.rs2)
        amo_post32(asm, v_rs2, inst.operands.rs1, inst.operands.rd, v_rd)
        v_rd.free()
        v_rs2.free()
        return asm.finalize()
    v_mask = allocator.allocate()
    v_dword = allocator.allocate()
    v_shift = allocator.allocate()
    amo_pre64(asm, inst.operands.rs1, v_rd, v_dword, v_shift)
    asm.emit_r(op, v_rs2, v_rd, inst.operands.rs2)
    amo_post64(asm, inst.operands.rs1, v_rs2, v_dword, v_shift, v_mask, inst.operands.rd, v_rd)
    v_mask.free()
    v_dword.free()
    v_shift.free()
    v_rd.free()
    v_rs2.free()
    return asm.finalize()

def _expand_amo_binop_d(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, op: str) -> list[Instruction]:  # AMO{ADD,AND,OR,XOR}D
    if xlen != Xlen.Bit64:
        raise RuntimeError(f"{inst.kind} is only available in RV64")
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v_rs2, v_rd = f.alloc(), f.alloc()
        asm.emit_i("LD", v_rd, inst.operands.rs1, 0)
        asm.emit_r(op, v_rs2, v_rd, inst.operands.rs2)
        asm.emit_s("SD", inst.operands.rs1, v_rs2, 0)
        asm.emit_i("ADDI", inst.operands.rd, v_rd, 0)
        return asm.finalize()

def _expand_amo_cmp_d(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, slt_op: str, swap: bool) -> list[Instruction]:  # AMOMIN/MAX{,U}D
    if xlen != Xlen.Bit64:
        raise RuntimeError(f"{inst.kind} is only available in RV64")
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v0, v1, v2 = f.alloc(), f.alloc(), f.alloc()
        asm.emit_i("LD", v0, inst.operands.rs1, 0)
        asm.emit_r(slt_op, v1, v0, inst.operands.rs2) if swap else asm.emit_r(slt_op, v1, inst.operands.rs2, v0)
        asm.emit_r("SUB", v2, inst.operands.rs2, v0)
        asm.emit_r("MUL", v2, v2, v1)
        asm.emit_r("ADD", v1, v0, v2)
        asm.emit_s("SD", inst.operands.rs1, v1, 0)
        asm.emit_i("ADDI", inst.operands.rd, v0, 0)
        return asm.finalize()

def _expand_amo_cmp_w(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, slt_op: str, swap: bool, extend: str) -> list[Instruction]:  # AMOMIN/MAX{,U}W
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    if xlen == Xlen.Bit32:
        v_rd = allocator.allocate()
        v_rs2 = allocator.allocate()
        v0 = allocator.allocate()
        v1 = allocator.allocate()
        amo_pre32(asm, inst.operands.rs1, v_rd)
        asm.emit_r(slt_op, v0, v_rd, inst.operands.rs2) if swap else asm.emit_r(slt_op, v0, inst.operands.rs2, v_rd)
        asm.emit_r("SUB", v1, inst.operands.rs2, v_rd)
        asm.emit_r("MUL", v1, v1, v0)
        asm.emit_r("ADD", v_rs2, v1, v_rd)
        amo_post32(asm, v_rs2, inst.operands.rs1, inst.operands.rd, v_rd)
        v_rd.free()
        v_rs2.free()
        v0.free()
        v1.free()
        return asm.finalize()
    v_rd = allocator.allocate()
    v_dword = allocator.allocate()
    v_shift = allocator.allocate()
    amo_pre64(asm, inst.operands.rs1, v_rd, v_dword, v_shift)
    v_rs2 = allocator.allocate()
    v0 = allocator.allocate()
    asm.emit_i(extend, v_rs2, inst.operands.rs2, 0)
    asm.emit_i(extend, v0, v_rd, 0)
    asm.emit_r(slt_op, v0, v0, v_rs2) if swap else asm.emit_r(slt_op, v0, v_rs2, v0)
    asm.emit_r("SUB", v_rs2, inst.operands.rs2, v_rd)
    asm.emit_r("MUL", v_rs2, v_rs2, v0)
    asm.emit_r("ADD", v_rs2, v_rs2, v_rd)
    amo_post64(asm, inst.operands.rs1, v_rs2, v_dword, v_shift, v0, inst.operands.rd, v_rd)
    v_rd.free()
    v_dword.free()
    v_shift.free()
    v_rs2.free()
    v0.free()
    return asm.finalize()

def _expand_divu_remu(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, want_remainder: bool) -> list[Instruction]:  # DIVU/REMU expansion
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v0 = f.alloc()
        v1 = None if want_remainder else f.alloc()
        asm.emit_j("VirtualAdvice", v0, 0)
        if not want_remainder:
            asm.emit_b("VirtualAssertValidDiv0", inst.operands.rs2, v0, 0)
        asm.emit_b("VirtualAssertMulUNoOverflow", v0, inst.operands.rs2, 0)
        if want_remainder:
            asm.emit_r("MUL", v0, v0, inst.operands.rs2)
            asm.emit_b("VirtualAssertLTE", v0, inst.operands.rs1, 0)
            asm.emit_r("SUB", v0, inst.operands.rs1, v0)
            asm.emit_b("VirtualAssertValidUnsignedRemainder", v0, inst.operands.rs2, 0)
            asm.emit_i("ADDI", inst.operands.rd, v0, 0)
        else:
            asm.emit_r("MUL", v1, v0, inst.operands.rs2)
            asm.emit_b("VirtualAssertLTE", v1, inst.operands.rs1, 0)
            asm.emit_r("SUB", v1, inst.operands.rs1, v1)
            asm.emit_b("VirtualAssertValidUnsignedRemainder", v1, inst.operands.rs2, 0)
            asm.emit_i("ADDI", inst.operands.rd, v0, 0)
        return asm.finalize()

def _expand_div_rem(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, want_remainder: bool) -> list[Instruction]:  # DIV/REM expansion
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        a0 = inst.operands.rs1
        a1 = inst.operands.rs2
        a2, a3 = f.alloc(), f.alloc()
        t0, t1 = f.alloc(), f.alloc()
        shmat = 31 if xlen == Xlen.Bit32 else 63
        asm.emit_j("VirtualAdvice", a2, 0)
        asm.emit_j("VirtualAdvice", a3, 0)
        asm.emit_b("VirtualAssertValidDiv0", a1, a2, 0)
        asm.emit_r("VirtualChangeDivisor", t0, a0, a1)
        asm.emit_r("MULH", t1, a2, t0)
        t2, t3 = f.alloc(), f.alloc()
        asm.emit_r("MUL", t2, a2, t0)
        asm.emit_i("SRAI", t3, t2, shmat)
        asm.emit_b("VirtualAssertEQ", t1, t3, 0)
        asm.emit_i("SRAI", t1, a0, shmat)
        asm.emit_r("XOR", t3, a3, t1)
        asm.emit_r("SUB", t3, t3, t1)
        asm.emit_r("ADD", t2, t2, t3)
        asm.emit_b("VirtualAssertEQ", t2, a0, 0)
        asm.emit_i("SRAI", t1, t0, shmat)
        if want_remainder:
            asm.emit_r("XOR", t2, t0, t1)
            asm.emit_r("SUB", t2, t2, t1)
            asm.emit_b("VirtualAssertValidUnsignedRemainder", a3, t2, 0)
            asm.emit_i("ADDI", inst.operands.rd, t3, 0)
        else:
            asm.emit_r("XOR", t3, t0, t1)
            asm.emit_r("SUB", t3, t3, t1)
            asm.emit_b("VirtualAssertValidUnsignedRemainder", a3, t3, 0)
            asm.emit_i("ADDI", inst.operands.rd, a2, 0)
        return asm.finalize()

def _expand_divuw_remuw(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, want_remainder: bool) -> list[Instruction]:  # DIVUW/REMUW expansion
    with allocator.frame() as f:
        if xlen == Xlen.Bit32:
            raise RuntimeError(f"{inst.kind} is invalid in 32b mode")
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        rs1, rs2 = f.alloc(), f.alloc()
        asm.emit_i("VirtualZeroExtendWord", rs1, inst.operands.rs1, 0)
        asm.emit_i("VirtualZeroExtendWord", rs2, inst.operands.rs2, 0)
        if want_remainder:
            asm.emit_j("VirtualAdvice", inst.operands.rd, 0)
            asm.emit_b("VirtualAssertMulUNoOverflow", inst.operands.rd, rs2, 0)
            asm.emit_r("MUL", inst.operands.rd, inst.operands.rd, rs2)
            asm.emit_b("VirtualAssertLTE", inst.operands.rd, rs1, 0)
            asm.emit_r("SUB", inst.operands.rd, rs1, inst.operands.rd)
            asm.emit_b("VirtualAssertValidUnsignedRemainder", inst.operands.rd, rs2, 0)
            asm.emit_i("VirtualSignExtendWord", inst.operands.rd, inst.operands.rd, 0)
            return asm.finalize()
        quo = f.alloc()
        asm.emit_j("VirtualAdvice", quo, 0)
        asm.emit_b("VirtualAssertMulUNoOverflow", quo, rs2, 0)
        asm.emit_r("MUL", inst.operands.rd, quo, rs2)
        asm.emit_b("VirtualAssertLTE", inst.operands.rd, rs1, 0)
        asm.emit_r("SUB", inst.operands.rd, rs1, inst.operands.rd)
        asm.emit_b("VirtualAssertValidUnsignedRemainder", inst.operands.rd, rs2, 0)
        asm.emit_i("VirtualSignExtendWord", inst.operands.rd, quo, 0)
        asm.emit_b("VirtualAssertValidDiv0", rs2, inst.operands.rd, 0)
        return asm.finalize()

def _expand_divw_remw(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, want_remainder: bool) -> list[Instruction]:  # DIVW/REMW expansion
    with allocator.frame() as f:
        if xlen == Xlen.Bit32:
            raise RuntimeError(f"{inst.kind} is invalid in 32b mode")
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        a0 = inst.operands.rs1
        a1 = inst.operands.rs2
        a2, a3 = f.alloc(), f.alloc()
        t0, t1 = f.alloc(), f.alloc()
        t2, t3, t4 = f.alloc(), f.alloc(), f.alloc()
        asm.emit_j("VirtualAdvice", a2, 0)
        asm.emit_j("VirtualAdvice", a3, 0)
        asm.emit_i("VirtualSignExtendWord", t4, a0, 0)
        asm.emit_i("VirtualSignExtendWord", t3, a1, 0)
        asm.emit_b("VirtualAssertValidDiv0", t3, a2, 0)
        asm.emit_r("VirtualChangeDivisorW", t0, t4, t3)
        asm.emit_i("VirtualSignExtendWord", t1, a2, 0)
        asm.emit_b("VirtualAssertEQ", t1, a2, 0)
        asm.emit_i("SRAI", t2, a3, 31)
        asm.emit_b("VirtualAssertEQ", t2, 0, 0)
        asm.emit_i("SRAI", t2, t4, 31)
        asm.emit_r("XOR", t3, a3, t2)
        asm.emit_r("SUB", t3, t3, t2)
        asm.emit_r("MUL", t1, a2, t0)
        asm.emit_r("ADD", t1, t1, t3)
        asm.emit_b("VirtualAssertEQ", t1, t4, 0)
        asm.emit_i("SRAI", t2, t0, 31)
        asm.emit_r("XOR", t1, t0, t2)
        asm.emit_r("SUB", t1, t1, t2)
        asm.emit_b("VirtualAssertValidUnsignedRemainder", a3, t1, 0)
        asm.emit_i("VirtualSignExtendWord", inst.operands.rd, t3 if want_remainder else a2, 0)
        return asm.finalize()

_SUBWORD_LOAD = {  # kind -> (nbytes, signed)
    "LB": (1, True), "LBU": (1, False), "LH": (2, True),
    "LHU": (2, False), "LW": (4, True), "LWU": (4, False),
}

_ADVICE_LOAD_NBYTES = {"AdviceLB": 1, "AdviceLH": 2, "AdviceLW": 4, "AdviceLD": 8}  # advice kind -> bytes
_SUBWORD_STORE_NBYTES = {"SB": 1, "SH": 2, "SW": 4}  # store kind -> bytes

def _expand_subword_load(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, nbytes: int, signed: bool) -> list[Instruction]:  # LB/LH/LW(LWU) expansion
    with allocator.frame() as f:
        if xlen == Xlen.Bit32:
            if nbytes == 4:
                if inst.kind == "LWU":
                    raise RuntimeError("LWU is invalid in 32b mode")
                asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
                asm.emit_i("VirtualLW", inst.operands.rd, inst.operands.rs1, u64(inst.operands.imm))
                return asm.finalize()
            asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
            v0 = f.alloc()
            if nbytes == 2:
                asm.emit_align("VirtualAssertHalfwordAlignment", inst.operands.rs1, inst.operands.imm)
            asm.emit_i("ADDI", v0, inst.operands.rs1, u64(inst.operands.imm))
            asm.emit_i("ANDI", inst.operands.rd, v0, u64(-4))
            asm.emit_i("VirtualLW", inst.operands.rd, inst.operands.rd, 0)
            asm.emit_i("XORI", v0, v0, 4 - nbytes)
            asm.emit_i("SLLI", v0, v0, 3)
            asm.emit_r("SLL", inst.operands.rd, inst.operands.rd, v0)
            asm.emit_i("SRAI" if signed else "SRLI", inst.operands.rd, inst.operands.rd, 32 - 8 * nbytes)
            return asm.finalize()

        if nbytes == 4 and signed:
            asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
            v0 = f.alloc()
            asm.emit_align("VirtualAssertWordAlignment", inst.operands.rs1, inst.operands.imm)
            asm.emit_i("ADDI", v0, inst.operands.rs1, u64(inst.operands.imm))
            asm.emit_i("ANDI", inst.operands.rd, v0, u64(-8))
            asm.emit_i("LD", inst.operands.rd, inst.operands.rd, 0)
            asm.emit_i("SLLI", v0, v0, 3)
            asm.emit_r("SRL", inst.operands.rd, inst.operands.rd, v0)
            asm.emit_i("VirtualSignExtendWord", inst.operands.rd, inst.operands.rd, 0)
            return asm.finalize()

        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v0 = f.alloc()
        if nbytes == 4:  # LWU
            asm.emit_align("VirtualAssertWordAlignment", inst.operands.rs1, inst.operands.imm)
        elif nbytes == 2:
            asm.emit_align("VirtualAssertHalfwordAlignment", inst.operands.rs1, inst.operands.imm)
        asm.emit_i("ADDI", v0, inst.operands.rs1, u64(inst.operands.imm))
        asm.emit_i("ANDI", inst.operands.rd, v0, u64(-8))
        asm.emit_i("LD", inst.operands.rd, inst.operands.rd, 0)
        asm.emit_i("XORI", v0, v0, 8 - nbytes)
        asm.emit_i("SLLI", v0, v0, 3)
        asm.emit_r("SLL", inst.operands.rd, inst.operands.rd, v0)
        asm.emit_i("SRAI" if signed else "SRLI", inst.operands.rd, inst.operands.rd, 64 - 8 * nbytes)
        return asm.finalize()

def _expand_advice_load(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, nbytes: int) -> list[Instruction]:  # AdviceLB/LH/LW/LD expansion
    if xlen == Xlen.Bit32 and nbytes == 8:
        raise RuntimeError("LD is not supported in 32-bit mode")
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    asm.emit_j("VirtualAdviceLoad", inst.operands.rd, nbytes)
    shift = xlen.bits - 8 * nbytes
    if shift:
        asm.emit_i("SLLI", inst.operands.rd, inst.operands.rd, shift)
        asm.emit_i("SRAI", inst.operands.rd, inst.operands.rd, shift)
    return asm.finalize()

def _expand_subword_store(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen, nbytes: int) -> list[Instruction]:  # SB/SH/SW expansion
    with allocator.frame() as f:
        if xlen == Xlen.Bit32:
            if nbytes == 4:
                asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
                asm.emit_s("VirtualSW", inst.operands.rs1, inst.operands.rs2, inst.operands.imm)
                return asm.finalize()
            asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
            v0, v1 = f.alloc(), f.alloc()
            v2, v3 = f.alloc(), f.alloc()
            if nbytes == 2:
                asm.emit_align("VirtualAssertHalfwordAlignment", inst.operands.rs1, inst.operands.imm)
            asm.emit_i("ADDI", v0, inst.operands.rs1, u64(inst.operands.imm))
            asm.emit_i("ANDI", v1, v0, u64(-4))
            asm.emit_i("VirtualLW", v2, v1, 0)
            asm.emit_i("SLLI", v3, v0, 3)
            asm.emit_u("LUI", v0, 0xFF if nbytes == 1 else 0xFFFF)
            asm.emit_r("SLL", v0, v0, v3)
            asm.emit_r("SLL", v3, inst.operands.rs2, v3)
            asm.emit_r("XOR", v3, v2, v3)
            asm.emit_r("AND", v3, v3, v0)
            asm.emit_r("XOR", v2, v2, v3)
            asm.emit_s("VirtualSW", v1, v2, 0)
            return asm.finalize()

        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v0, v1 = f.alloc(), f.alloc()
        v2, v3 = f.alloc(), f.alloc()
        if nbytes == 4:
            asm.emit_align("VirtualAssertWordAlignment", inst.operands.rs1, inst.operands.imm)
            asm.emit_i("ADDI", v0, inst.operands.rs1, u64(inst.operands.imm))
            asm.emit_i("ANDI", v1, v0, u64(-8))
            asm.emit_i("LD", v2, v1, 0)
            asm.emit_i("SLLI", v0, v0, 3)
            asm.emit_i("ORI", v3, 0, u64(-1))
            asm.emit_i("SRLI", v3, v3, 32)
            asm.emit_r("SLL", v3, v3, v0)
            asm.emit_r("SLL", v0, inst.operands.rs2, v0)
            asm.emit_r("XOR", v0, v2, v0)
            asm.emit_r("AND", v0, v0, v3)
            asm.emit_r("XOR", v2, v2, v0)
            asm.emit_s("SD", v1, v2, 0)
            return asm.finalize()
        if nbytes == 2:
            asm.emit_align("VirtualAssertHalfwordAlignment", inst.operands.rs1, inst.operands.imm)
        asm.emit_i("ADDI", v0, inst.operands.rs1, u64(inst.operands.imm))
        asm.emit_i("ANDI", v1, v0, u64(-8))
        asm.emit_i("LD", v2, v1, 0)
        asm.emit_i("SLLI", v3, v0, 3)
        asm.emit_u("LUI", v0, 0xFF if nbytes == 1 else 0xFFFF)
        asm.emit_r("SLL", v0, v0, v3)
        asm.emit_r("SLL", v3, inst.operands.rs2, v3)
        asm.emit_r("XOR", v3, v2, v3)
        asm.emit_r("AND", v3, v3, v0)
        asm.emit_r("XOR", v2, v2, v3)
        asm.emit_s("SD", v1, v2, 0)
        return asm.finalize()

def expand_instruction(inst: Instruction, xlen: Xlen, profile: ISAProfile | None = None) -> list[Instruction]:  # full recursive inline expansion
    if profile is None:
        profile = ISAProfile.default()
    assert_kind_allowed(inst.kind, profile, compressed=bool(inst.is_compressed))
    allocator = VirtualRegisterAllocator()
    # `InstrAssembler._emit` recursively expands emitted instructions, so a single
    # `inline_sequence` call returns a fully flattened sequence for this instruction.
    out = inline_sequence(inst, allocator, xlen)
    if len(out) == 1 and out[0] is inst:
        # No expansion: keep metadata unset (matches tracer bytecode for non-virtual instructions).
        return [inst]
    n = len(out)
    if n <= 0:
        raise RuntimeError("expanded instruction sequence must be non-empty")
    for i, x in enumerate(out):
        x.is_first_in_sequence = i == 0
        x.virtual_sequence_remaining = n - i - 1
        x.is_compressed = False
    out[-1].is_compressed = bool(inst.is_compressed)
    return out

def expand_program(instructions: list[Instruction], xlen: Xlen, profile: ISAProfile | None = None) -> list[Instruction]:  # expand a whole program
    if profile is None:
        profile = ISAProfile.default()
    out = []
    for inst in instructions:
        out.extend(expand_instruction(inst, xlen, profile))
    return out

def _expand_ecall(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # ecall.rs
    v_trap_handler_reg = allocator.trap_handler_register()
    vr_mepc = allocator.mepc_register()
    vr_mcause = allocator.mcause_register()
    vr_mtval = allocator.mtval_register()
    vr_mstatus = allocator.mstatus_register()
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    ecall_addr = allocator.allocate()
    asm.emit_u("AUIPC", ecall_addr, 0)
    asm.emit_i("ADDI", vr_mepc, ecall_addr, 0)
    ecall_addr.free()
    asm.emit_i("ADDI", vr_mcause, 0, 11)
    asm.emit_i("ADDI", vr_mtval, 0, 0)
    three = allocator.allocate()
    asm.emit_i("ADDI", three, 0, 3)
    asm.emit_i("SLLI", vr_mstatus, three, 11)
    three.free()
    asm.emit_i("JALR", 0, v_trap_handler_reg, 0)
    return asm.finalize()

def _expand_mret(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # mret.rs
    mepc_vr = allocator.mepc_register()
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    asm.emit_i("JALR", 0, mepc_vr, 0)
    return asm.finalize()

def _expand_mulh(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # mulh.rs
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v_sx, v_sy = f.alloc(), f.alloc()
        asm.emit_i("VirtualMovsign", v_sx, inst.operands.rs1, 0)
        asm.emit_i("VirtualMovsign", v_sy, inst.operands.rs2, 0)
        asm.emit_r("MUL", v_sx, v_sx, inst.operands.rs2)
        asm.emit_r("MUL", v_sy, v_sy, inst.operands.rs1)
        asm.emit_r("MULHU", inst.operands.rd, inst.operands.rs1, inst.operands.rs2)
        asm.emit_r("ADD", inst.operands.rd, inst.operands.rd, v_sx)
        asm.emit_r("ADD", inst.operands.rd, inst.operands.rd, v_sy)
        return asm.finalize()

def _expand_mulhsu(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # mulhsu.rs
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v0, v1 = f.alloc(), f.alloc()
        v2, v3 = f.alloc(), f.alloc()
        asm.emit_i("VirtualMovsign", v0, inst.operands.rs1, 0)
        asm.emit_i("ANDI", v1, v0, 1)
        asm.emit_r("XOR", v2, inst.operands.rs1, v0)
        asm.emit_r("ADD", v2, v2, v1)
        asm.emit_r("MULHU", v3, v2, inst.operands.rs2)
        asm.emit_r("MUL", v2, v2, inst.operands.rs2)
        asm.emit_r("XOR", v3, v3, v0)
        asm.emit_r("XOR", v2, v2, v0)
        asm.emit_r("ADD", v0, v2, v1)
        asm.emit_r("SLTU", v0, v0, v2)
        asm.emit_r("ADD", inst.operands.rd, v3, v0)
        return asm.finalize()

def _expand_srl_sra(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v_bitmask = f.alloc()
        asm.emit_i("VirtualShiftRightBitmask", v_bitmask, inst.operands.rs2, 0)
        asm.emit_vshift_r("VirtualSRL" if inst.kind == "SRL" else "VirtualSRA", inst.operands.rd, inst.operands.rs1, v_bitmask)
        return asm.finalize()

def _expand_srli_srai(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:
    shift_mask, n = (0x1F, 32) if xlen == Xlen.Bit32 else (0x3F, 64)
    shift = inst.operands.imm & shift_mask
    bitmask = _sr_imm_bitmask(n, shift)
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    asm.emit_vshift_i("VirtualSRLI" if inst.kind == "SRLI" else "VirtualSRAI", inst.operands.rd, inst.operands.rs1, bitmask)
    return asm.finalize()

def _expand_sll(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # sll.rs
    with allocator.frame() as f:
        asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
        v_pow2 = f.alloc()
        asm.emit_i("VirtualPow2", v_pow2, inst.operands.rs2, 0)
        asm.emit_r("MUL", inst.operands.rd, inst.operands.rs1, v_pow2)
        return asm.finalize()

def _expand_slli(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # slli.rs
    shift_mask = 0x1F if xlen == Xlen.Bit32 else 0x3F
    shift = inst.operands.imm & shift_mask
    asm = InstrAssembler.new(inst.address, inst.is_compressed, xlen, allocator)
    asm.emit_i("VirtualMULI", inst.operands.rd, inst.operands.rs1, 1 << shift)
    return asm.finalize()

_DISPATCH = (
    {"SCW": _expand_scw, "SCD": _expand_scd, "ECALL": _expand_ecall, "MRET": _expand_mret, "MULH": _expand_mulh, "MULHSU": _expand_mulhsu, "SLL": _expand_sll, "SLLI": _expand_slli}
    | {"LRW": (lambda inst, a, x: _expand_lr(inst, a, x, is_d=False)), "LRD": (lambda inst, a, x: _expand_lr(inst, a, x, is_d=True))}
    | {"AMOSWAPW": (lambda inst, a, x: _expand_amo_swap(inst, a, x, is_d=False)), "AMOSWAPD": (lambda inst, a, x: _expand_amo_swap(inst, a, x, is_d=True))}
    | {k: (lambda inst, a, x, op=op: _expand_amo_binop_w(inst, a, x, op)) for k, op in _AMO_BINOP_W.items()}
    | {k: (lambda inst, a, x, op=op: _expand_amo_binop_d(inst, a, x, op)) for k, op in _AMO_BINOP_D.items()}
    | {k: (lambda inst, a, x, p=p: _expand_amo_cmp_d(inst, a, x, *p)) for k, p in _AMO_CMP_D.items()}
    | {k: (lambda inst, a, x, p=p: _expand_amo_cmp_w(inst, a, x, *p)) for k, p in _AMO_CMP_W.items()}
    | {k: _expand_csr for k in ("CSRRW", "CSRRS")}
    | {k: (lambda inst, a, x, p=p: _expand_subword_load(inst, a, x, *p)) for k, p in _SUBWORD_LOAD.items()}
    | {k: (lambda inst, a, x, n=n: _expand_advice_load(inst, a, x, n)) for k, n in _ADVICE_LOAD_NBYTES.items()}
    | {k: (lambda inst, a, x, p=p: _expand_word_op(inst, a, x, *p)) for k, p in _WORD_OP.items()}
    | {"DIVU": (lambda inst, a, x: _expand_divu_remu(inst, a, x, want_remainder=False)), "REMU": (lambda inst, a, x: _expand_divu_remu(inst, a, x, want_remainder=True))}
    | {"DIV": (lambda inst, a, x: _expand_div_rem(inst, a, x, want_remainder=False)), "REM": (lambda inst, a, x: _expand_div_rem(inst, a, x, want_remainder=True))}
    | {"DIVUW": (lambda inst, a, x: _expand_divuw_remuw(inst, a, x, want_remainder=False)), "REMUW": (lambda inst, a, x: _expand_divuw_remuw(inst, a, x, want_remainder=True))}
    | {"DIVW": (lambda inst, a, x: _expand_divw_remw(inst, a, x, want_remainder=False)), "REMW": (lambda inst, a, x: _expand_divw_remw(inst, a, x, want_remainder=True))}
    | {k: (lambda inst, a, x, n=n: _expand_subword_store(inst, a, x, n)) for k, n in _SUBWORD_STORE_NBYTES.items()}
    | {k: _expand_srl_sra for k in ("SRL", "SRA")}
    | {k: _expand_srli_srai for k in ("SRLI", "SRAI")}
    | {k: _expand_shiftw for k in ("SRLW", "SRAW", "SRLIW", "SRAIW", "SLLW", "SLLIW")}
)

def inline_sequence(inst: Instruction, allocator: VirtualRegisterAllocator, xlen: Xlen) -> list[Instruction]:  # port tracer inline_sequence from C
    if inst.kind in ("NoOp", "UNIMPL"):  # elide
        return []
    if inst.kind == "INLINE":  # out of scope
        return [inst]
    h = _DISPATCH.get(inst.kind)
    return h(inst, allocator, xlen) if h else [inst]

