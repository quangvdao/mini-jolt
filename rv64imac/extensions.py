from __future__ import annotations  # keep type hints lightweight

from dataclasses import dataclass  # small immutable-ish profile container
from enum import Enum  # stable extension identifiers

class ISAProfileError(Exception):  # profile/extension enforcement error
    pass

class Extension(Enum):  # extension groups for ISA gating
    BaseI = "BaseI"  # RV64I base (incl. RV64-only ops we currently decode)
    M = "M"  # integer multiply/divide
    A = "A"  # atomics
    C = "C"  # compressed encodings (RVC)
    Zicsr = "Zicsr"  # CSR read/write instructions
    System = "System"  # ECALL/EBREAK/MRET (system/trap flow)
    JoltVirtual = "JoltVirtual"  # Jolt virtual opcodes + virtual-inline helpers
    JoltInline = "JoltInline"  # guest INLINE opcode (out-of-scope to expand)

@dataclass(frozen=True)
class ISAProfile:  # enabled extension set (defaults match current behavior)
    enabled: frozenset[Extension]

    @classmethod
    def default(cls) -> "ISAProfile":  # current behavior (all enabled)
        return cls(enabled=frozenset({Extension.BaseI, Extension.M, Extension.A, Extension.C, Extension.Zicsr, Extension.System, Extension.JoltVirtual, Extension.JoltInline}))

    def is_enabled(self, ext: Extension) -> bool:  # membership check
        return ext in self.enabled

    def without(self, *exts: Extension) -> "ISAProfile":  # disable a few extensions
        return ISAProfile(enabled=frozenset(e for e in self.enabled if e not in set(exts)))

_SYSTEM_KINDS = {"ECALL", "EBREAK", "MRET"}  # trap flow (kept enabled by default)

def extension_for_kind(kind: str) -> Extension:  # kind -> extension (single source of truth)
    if kind.startswith("Virtual") or kind.startswith("Advice"):
        return Extension.JoltVirtual
    if kind == "INLINE":
        return Extension.JoltInline
    if kind.startswith("CSR"):
        return Extension.Zicsr
    if kind in _SYSTEM_KINDS:
        return Extension.System
    if kind.startswith(("MUL", "DIV", "REM")):
        return Extension.M
    if kind.startswith(("AMO", "LR", "SC")):
        return Extension.A
    return Extension.BaseI

def assert_kind_allowed(kind: str, profile: ISAProfile, *, compressed: bool = False) -> None:  # gate a kind
    if profile is None:
        profile = ISAProfile.default()
    if compressed and not profile.is_enabled(Extension.C):
        raise ISAProfileError(f"compressed instruction requires extension {Extension.C.value}, but it is disabled")
    ext = extension_for_kind(kind)
    if not profile.is_enabled(ext):
        raise ISAProfileError(f"instruction kind {kind!r} requires extension {ext.value}, but it is disabled")
