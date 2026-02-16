# jolt-python

A minimal-dependency Jolt verifier for RV64IMAC in pure Python. The complete algorithm in ~2000 lines of readable code.

## Philosophy

This project takes direct inspiration from Karpathy's [microGPT](https://karpathy.ai/microgpt.html) — *"the most atomic way to train and inference a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."*

We apply the same ethos to Jolt:

- **Minimal dependencies.** Pure Python with a short, explicit allowlist. We re-derive finite field arithmetic, polynomial operations, elliptic curve math, and (most of) Fiat-Shamir from scratch, but we intentionally rely on a hardened hash implementation for transcript hashing.
- **Dependency allowlist (intentional):**
  - `hashlib.blake2b` (Python stdlib): security-critical hash function used by the Fiat-Shamir transcript. This avoids rolling complex cryptography by hand while keeping the project dependency-light.
- **Complete specification.** This is the full Jolt verifier: sumcheck, Twist/Shout lookups, Spartan (R1CS via sumcheck), Dory polynomial commitment verification, memory checking — everything needed to verify an RV64IMAC execution proof.
- **Readable, not fast.** Clarity over performance. Every function should be understandable by someone with basic math background and a willingness to learn. This is a *specification*, not a production implementation.
- **~2000 lines.** The entire verifier — RISC-V semantics, field arithmetic, polynomials, elliptic curves, commitments, transcript, and the Jolt verification logic — in roughly 2000 lines of Python, or less.

## What's in the 2000 lines?

A significant portion of the verifier is not crypto — it's defining **RISC-V semantics**. Jolt proves correct execution of RV64IMAC programs, and the verifier must know what correct execution *means*. Concretely:

**RISC-V ISA specification (~???)**
- ~80 instructions (RV64I base + M extension + A extension + C extension)
- Virtual instruction sequences: complex instructions (DIV, REM) are decomposed into sequences of simpler virtual instructions
- Per-instruction metadata: 14 circuit flags (Load, Store, Jump, AddOperands, ...) and 7 instruction flags (LeftOperandIsPC, Branch, ...)
- 39 lookup tables (RangeCheck, Equal, Xor, SignedLessThan, ...), each with an `evaluate_mle()` the verifier computes
- **Out of scope:** inlines (optimized cryptographic primitives like sha2, blake3, secp256k1). These are add-ons to the base Jolt verifier and not covered here.

**Cryptographic primitives (~???)**
- BN254 scalar field arithmetic (**implemented**)
- BN254 elliptic curve + pairing (**implemented**)
- Fiat-Shamir transcript (Blake2b default; Keccak optional) (**implemented**, Blake2b only)
- Verifier-minimal polynomial utilities (Eq/Eq+1/identity + compressed univariate) (**implemented**)
- Full multilinear polynomial evaluation (**TODO**; implement only what the verifier needs)

**Proof verification protocol (~???)**
- 19 uniform R1CS constraints encoding execution rules: RAM addressing, lookup operand formation, register writes, program counter updates
- Spartan: R1CS satisfaction via sumcheck + univariate skip
- Twist/Shout: lookup argument + memory checking
- Dory: batched polynomial commitment opening verification
- 8-stage verification pipeline tying it all together

(Line counts TBD as we build it out.)

## Temporary Python hygiene (WIP)

- Stay minimal, but keep code readable.
- Every `class` and every `def` should have an inline comment on the same line (for example: `def foo(...):  # does X`).
- Every `import`/`from ... import ...` should have an inline comment explaining why we need it (or remove it).
- Module-level constants should have an inline comment if there is room on the same line.
- Never have 2+ consecutive blank lines (max 1).
- Keep one blank line between each definition (defs should never be consecutive). Constants may be consecutive.
- End of file: leave exactly one blank line (not two).
- For very simple class constructors/factories, prefer one-line `classmethod(...)` assignment over a two-line `@classmethod` + `def` block.

## Temporary testing requirement (WIP)

- Every test should have two versions:
  - a Python-only version
  - a cross-check version against Rust behavior
- Use `tests/rust_oracle/src/main.rs` for Rust oracle logic.
- Prefer using the shared Python test harness in `tests/oracle.py` to invoke the Rust oracle and parse outputs (so we don't duplicate subprocess/parsing code across tests).
- It is fine to keep all Rust oracle test logic in that single `main.rs` for now.
- Later, we can split oracle code into multiple Rust files/modules for better modularity.

## Structure

```
jolt-python/
├── README.md          # you are here
├── field.py           # BN254 Fq/Fr arithmetic
├── curve.py           # BN254 elliptic curve, pairing
├── transcript.py      # Fiat-Shamir transcript (Blake2b; Rust-compatible)
├── polynomial.py      # verifier-minimal polynomials (Eq/Eq+1/identity + compressed univariate)
├── sumchecks.py       # (batched) sumcheck verifier template + concrete sumcheck instances (Rust-compatible transcript coupling)
├── lookup_tables.py   # Jolt lookup-table MLE evaluation (all tables used by our ISA metadata)
├── rv64imac/          # RV64IMAC bytecode expansion + ISA metadata (decode/expand/preprocess)
│   ├── __init__.py    # public API exports (decode_elf/decode_program/decode_preprocessed_program)
│   ├── isa.py         # lookup_table() + (circuit/instruction) flags (Rust-matched)
│   ├── decode.py      # 32-bit instruction decode (incl. custom/virtual + AMO kinds)
│   ├── rvc.py         # RVC decompression (C extension)
│   ├── elf.py         # ELF -> decoded instructions + memory init + program_end + xlen
│   ├── program.py     # ELF -> expanded bytecode (+ convenience preprocessed entrypoint)
│   ├── bytecode.py    # BytecodePreprocessing + BytecodePCMapper (virtual PC mapping + padding)
│   ├── inline_sequences.py # inline sequence expansion (incl. DIV/REM/LR/SC/AMO/CSR/ECALL/MRET/etc.)
│   ├── types.py       # Instruction / operands containers (Rust-shaped normalize() API)
│   └── constants.py   # RISC-V + Jolt constants mirrored from Rust
└── tests/
    ├── test_field.py  # finite field tests
    ├── test_curve.py  # curve + pairing tests
    ├── test_curve_cross_lang.py # python vs rust curve behavior
    ├── test_field_cross_lang.py # python vs rust field behavior
    ├── test_polynomial.py # polynomial tests
    ├── test_transcript.py # Fiat-Shamir transcript tests
    ├── test_transcript_cross_lang.py # python vs rust transcript behavior
    ├── test_sumcheck.py # sumcheck verifier tests
    ├── test_sumcheck_cross_lang.py # python vs rust sumcheck verifier behavior
    ├── oracle.py       # shared harness for invoking/parsing rust_oracle
    ├── rust_oracle/   # tiny arkworks field oracle binary
    └── test_verify.py # verify an actual Jolt proof
```

## Status

Work in progress. The goal is a Python verifier that can accept a serialized Jolt proof (produced by the Rust prover) and verify it correctly.

## Sumcheck instances (verifier side)

The file `sumchecks.py` is intended to eventually hold **all verifier-side sumcheck definitions** (one class per sumcheck instance), plus the shared batched verifier template.

Planned sumcheck IDs to mirror from Rust (see `jolt-core/src/poly/opening_proof.rs`):

- `SpartanOuter`
- `SpartanProductVirtualization`
- `SpartanShift`
- `InstructionClaimReduction`
- `InstructionInputVirtualization`
- `InstructionReadRaf`
- `InstructionRaVirtualization`
- `RamReadWriteChecking`
- `RamRafEvaluation`
- `RamOutputCheck`
- `RamValEvaluation`
- `RamValFinalEvaluation`
- `RamRaClaimReduction`
- `RamHammingBooleanity`
- `RamRaVirtualization`
- `RegistersClaimReduction`
- `RegistersReadWriteChecking`
- `RegistersValEvaluation`
- `BytecodeReadRaf`
- `Booleanity`
- `AdviceClaimReductionCyclePhase`
- `AdviceClaimReduction`
- `IncClaimReduction`
- `HammingWeightClaimReduction`

Current progress:

- **Implemented**: field arithmetic (`field.py`), curve + pairing (`curve.py`), transcript (`transcript.py`), verifier-minimal polynomial utilities (`polynomial.py`), batched sumcheck verifier template (`sumchecks.py`).
- **Cross-language parity tests**: field/curve/transcript/sumcheck are checked against `tests/rust_oracle` via `tests/oracle.py`.
- **Run tests**:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -q
```

## Target audience

- Cryptographers who want to understand Jolt without reading 50k lines of Rust
- Engineers evaluating Jolt who want a clear spec
- Students learning about SNARKs, sumcheck, and lookup arguments
- Anyone who believes the best documentation is working code
