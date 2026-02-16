# mini-jolt

A **pure-Python**, minimal-dependency **Jolt verifier** for **RV64IMAC**.

This repo is intentionally “spec-like”: it prioritizes readability and Rust parity over performance. As of the current tree, the repo is ~**10.7k LoC** total (see `wc -l`), with the largest single module being `sumchecks.py` (~2k LoC).

## Philosophy

This project takes direct inspiration from Karpathy's [microGPT](https://karpathy.ai/microgpt.html) — *"the most atomic way to train and inference a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."*

We apply the same ethos to Jolt:

- **Minimal dependencies**: stdlib-only. We re-derive finite field arithmetic, polynomial utilities, elliptic curve + pairing, and the verifier logic in plain Python.
- **Dependency allowlist**:
  - `hashlib.blake2b` (Python stdlib): the security-critical hash used for Fiat–Shamir transcript hashing.
- **Readable, not fast**: clarity over performance. Treat this as a specification / reference verifier, not the most optimized implementation.
- **Rust parity focus**: many components and tests are written to mirror Rust ordering/endianness/transcript coupling.

## What’s in the codebase?

Broadly:

- **BN254 primitives**: field arithmetic, elliptic curve group ops, and pairings.
- **Verifier-side polynomial utilities**: MLE equality polynomials, compressed univariates, Lagrange helpers, range masks, etc.
- **Jolt protocol verification**: staged sumchecks + reductions (Stages 1–7) and the PCS joint opening (Stage 8).
- **RV64IMAC decode/expansion**: ELF decode → instruction decode (incl. RVC) → Rust-matching inline-sequence expansion → bytecode preprocessing/PC mapping.
- **Rust wire-format compatibility**: minimal readers for Rust-produced `proof.bin` and `program_io.bin` (the latter is currently under `tests/`).

## Requirements

- **Python 3.11+** (uses `enum.StrEnum` and `X | Y` type syntax)
- No third-party Python dependencies

## Structure

```
mini-jolt/
├── .gitignore                # common Python/Rust ignore patterns
├── README.md                 # you are here
├── field.py                  # BN254 Fq/Fr arithmetic (Montgomery form)
├── curve.py                  # BN254 curve ops + pairing / multi-pairing
├── transcript.py             # Fiat–Shamir transcript (Blake2b; Rust-compatible)
├── polynomials.py            # verifier-minimal polynomials (Eq/Eq+1/identity/compressed univariate/etc.)
├── lookup_tables.py          # Jolt lookup-table MLE evaluation (`evaluate_mle`)
├── ids_generated.py          # canonical ID tables/enums (SumcheckId / VirtualPolynomial / CommittedPolynomial / tables)
├── openings.py               # typed opening IDs + verifier-side opening accumulator
├── r1cs.py                   # Spartan outer (uniform R1CS) constants + key evaluation
├── ram_io.py                 # RAM/IO MLE helpers (I/O blocks, initial RAM, advice scaling)
├── sumchecks.py              # Stage 1–7 verifier instances + batched sumcheck template (largest file)
├── dory.py                   # Dory PCS verification + RLC joint-opening combine
├── jolt_preprocessing.py     # verifier preprocessing (RAM word packing + bytecode preprocessing wrapper)
├── jolt_proof.py             # `JoltProof` container + Rust `proof.bin` deserializer (+ Dory verifier setup parsing)
├── jolt_verifier.py          # top-level verifier (`JoltVerifier`, `verify_jolt`) + stage orchestration helpers
├── zkvm_types.py             # Rust-shaped public types (JoltDevice/MemoryLayout/ReadWriteConfig/OneHotParams)
├── rv64imac/
│   ├── __init__.py           # public RV64IMAC API exports
│   ├── constants.py          # ISA + Jolt constants (RAM base, CSR IDs, reg counts)
│   ├── types.py              # Instruction/operands containers + `Xlen`
│   ├── extensions.py         # ISAProfile/Extension gating for decode/expansion
│   ├── rvc.py                # RVC decompression (C extension) -> u32 instruction words
│   ├── decode.py             # u32 instruction decode (incl. custom/virtual + INLINE opcode)
│   ├── isa.py                # per-kind lookup table + flag metadata (CircuitFlags / InstructionFlags)
│   ├── inline_sequences.py   # Rust-matching instruction expansion into virtual sequences
│   ├── elf.py                # ELF -> decoded instructions + memory init (+ xlen)
│   ├── bytecode.py           # BytecodePreprocessing + PC mapping + padding to pow2
│   └── program.py            # ELF -> expanded bytecode (+ preprocessed entrypoint)
├── scripts/
│   ├── README.md             # notes on reproducible extraction/codegen scripts (may assume a larger Rust tree)
│   └── gen_ids_generated.py  # generator that expects Rust sources (e.g. `jolt-core/...`) in a larger checkout
└── tests/
    ├── oracle.py                  # optional Rust oracle harness (skips if not provided)
    ├── rust_device_deserialize.py # test-only Rust `program_io.bin` reader
    └── test_*.py                  # unit tests + Rust-parity + optional E2E (see sections below)
```

## Status

- **Implemented**:
  - BN254 `Fq`/`Fr` arithmetic (`field.py`)
  - BN254 curve + pairing (`curve.py`)
  - Blake2b transcript (`transcript.py`)
  - Lookup-table MLEs (`lookup_tables.py`)
  - Verifier-side polynomial helpers (`polynomials.py`)
  - Staged verification pipeline for Stages **1–7** (`sumchecks.py` + orchestration in `jolt_verifier.py`)
  - Stage **8** Dory PCS joint-opening verification (`dory.py` + Stage-8 glue in `jolt_verifier.py`)
  - Rust proof parsing for `proof.bin` (arkworks `CanonicalSerialize`-style compressed) (`jolt_proof.py`)
- **Notable limitations / WIP**:
  - **Guest “INLINE” opcodes are treated as out-of-scope** during ELF→program decoding. `rv64imac.program.decode_program` raises if expanded bytecode still contains `INLINE`.
  - Several `scripts/` paths and docstrings still reference an older `jolt-python/` layout and may assume a larger Rust checkout (see `scripts/`).

## Quickstart: run the Python-only tests

```bash
python3 -m unittest discover -s tests -p "test_*.py" -q
```

By default, any test that requires Rust artifacts will **skip** if you haven’t provided them.

## Using the verifier (with Rust-produced artifacts)

If you have a Rust prover output directory containing `proof.bin`, `program_io.bin`, `verifier_preprocessing.bin`, and `program.elf`, you can verify it like this:

```python
import pathlib

from jolt_proof import JoltProof
from jolt_verifier import verify_jolt
from rv64imac.program import decode_program

# Note: `program_io.bin` parsing currently lives under tests/.
from tests.rust_device_deserialize import parse_jolt_device_bytes

guest_dir = pathlib.Path("/path/to/rust/artifacts_dir")
proof = JoltProof.from_rust_files(
    guest_dir / "proof.bin",
    verifier_preprocessing_path=guest_dir / "verifier_preprocessing.bin",
)
program_io = parse_jolt_device_bytes((guest_dir / "program_io.bin").read_bytes())
expanded, memory_init, _program_size = decode_program((guest_dir / "program.elf").read_bytes())

verify_jolt(expanded, program_io, proof, memory_init=memory_init)
```

## Cross-language parity tests (optional)

Many `*_cross_lang.py` tests call an **optional Rust oracle** via `cargo run` (see `tests/oracle.py`). Since this repo does **not** vendor `tests/rust_oracle/`, you have two options:

- **Vendor** a Rust oracle at `tests/rust_oracle/`, or
- Set `MINI_JOLT_RUST_ORACLE_MANIFEST=/path/to/rust_oracle/Cargo.toml`

If neither is set/present, oracle-backed tests will `SkipTest` with a clear message.

## End-to-end verification against Rust artifacts (optional)

The test `tests/test_e2e_verify_from_rust.py` verifies a **real Rust-produced proof** end-to-end, but it expects you to point it at a directory containing:

- `proof.bin`
- `program_io.bin`
- `verifier_preprocessing.bin`
- `program.elf`

Set one (or both) of:

- `MINI_JOLT_FIB_GUEST_DIR=/path/to/artifacts_dir`
- `MINI_JOLT_BTREE_GUEST_DIR=/path/to/artifacts_dir`

Then run:

```bash
python3 -m unittest tests.test_e2e_verify_from_rust -q
```

You can skip E2E with:

```bash
export JOLT_PYTHON_SKIP_E2E=1
```

## Developer notes (WIP conventions)

Some files were written with “minimal Python hygiene” conventions in mind (inline comments on defs/imports, avoid extra blank lines, etc.). Treat these as guidance for keeping the verifier readable.

## Target audience

- Cryptographers who want to understand Jolt without reading tens of thousands of lines of Rust
- Engineers evaluating Jolt who want a clear, executable verifier spec
- Students learning about sumcheck, lookups, and PCS verification
- Anyone who believes the best documentation is working code
