#!/usr/bin/env python3
"""
Regenerate `jolt-python/ids_generated.py` from Rust sources.

Design constraints:
- output stays compact + readable (minimal-LoC policy)
- deterministic ordering (mirrors Rust enum order)
"""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../jolt-python/
OUT = ROOT / "jolt-python" / "ids_generated.py"
MAX_LINE_LEN = 100  # strict line-length budget for ids_generated.py

RUST_SUMCHECK = ROOT / "jolt-core" / "src" / "poly" / "opening_proof.rs"
RUST_LOOKUP = ROOT / "jolt-core" / "src" / "zkvm" / "lookup_table" / "mod.rs"
RUST_WITNESS = ROOT / "jolt-core" / "src" / "zkvm" / "witness.rs"
RUST_INSTRUCTION = ROOT / "jolt-core" / "src" / "zkvm" / "instruction" / "mod.rs"


def _extract_enum_variants(src: str, enum_name: str) -> list[str]:
    m = re.search(rf"pub\s+enum\s+{re.escape(enum_name)}\s*\{{(.*?)\n\}}", src, re.S)
    if not m:
        raise RuntimeError(f"could not find enum {enum_name!r}")
    body = m.group(1)
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#["):
            continue
        if line.endswith(","):
            line = line[:-1].strip()
        if not line:
            continue
        # Match Variant or Variant(T)
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if mm:
            out.append(mm.group(1))
    if not out:
        raise RuntimeError(f"enum {enum_name!r} had no parsed variants")
    return out


def _extract_lookup_tables_64(src: str) -> list[str]:
    # Parse enum `LookupTables<const XLEN: usize>` and take the *variant names* in order.
    m = re.search(r"pub\s+enum\s+LookupTables<[^>]*>\s*\{(.*?)\n\}", src, re.S)
    if not m:
        raise RuntimeError("could not find LookupTables enum")
    body = m.group(1)
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith("#["):
            continue
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if mm:
            out.append(mm.group(1))
    if not out:
        raise RuntimeError("LookupTables enum had no parsed variants")
    return out


def _extract_enum_variant_heads(src: str, enum_name: str) -> list[str]:
    # Like `_extract_enum_variants`, but keeps variants with parameters (e.g. Foo(usize)) as `Foo`.
    m = re.search(rf"pub\s+enum\s+{re.escape(enum_name)}\s*\{{(.*?)\n\}}", src, re.S)
    if not m:
        raise RuntimeError(f"could not find enum {enum_name!r}")
    body = m.group(1)
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith("#["):
            continue
        if line.endswith(","):
            line = line[:-1].strip()
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if mm:
            out.append(mm.group(1))
    if not out:
        raise RuntimeError(f"enum {enum_name!r} had no parsed variants")
    return out


def _wrap_words(words: list[str], *, max_len: int) -> list[str]:
    if not words:
        raise ValueError("words must be non-empty")
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if not cur:
            cur = [w]
            cur_len = len(w)
            continue
        if cur_len + 1 + len(w) <= max_len:
            cur.append(w)
            cur_len += 1 + len(w)
        else:
            out.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    out.append(" ".join(cur))
    return out


def _format_split_block(name: str, items: list[str], *, comment: str = "") -> str:
    # Output:
    #   NAME = """
    #   a b c
    #   ...
    #   """.split()  # comment
    wrapped = _wrap_words(items, max_len=MAX_LINE_LEN)
    lines = [f'{name} = """'] + wrapped + ['""".split()']
    if comment:
        lines[-1] += f"  # {comment}"
    return "\n".join(lines)


def main() -> None:
    sumcheck_src = RUST_SUMCHECK.read_text(encoding="utf-8")
    lookup_src = RUST_LOOKUP.read_text(encoding="utf-8")
    witness_src = RUST_WITNESS.read_text(encoding="utf-8")
    instruction_src = RUST_INSTRUCTION.read_text(encoding="utf-8")

    sumchecks = _extract_enum_variants(sumcheck_src, "SumcheckId")
    lookup_tables_64 = _extract_lookup_tables_64(lookup_src)

    # Preserve the curated lists by reading the existing file (if present) and re-wrapping them
    # into the strict MAX_LINE_LEN format. This keeps edits local and reproducible.
    existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""

    def _extract_words_block(name: str) -> list[str] | None:
        # Match: NAME = """...""".split()
        pat = rf"^{re.escape(name)}\s*=\s*\"\"\"([\s\S]*?)\"\"\"\.split\(\)"
        m = re.search(pat, existing, re.M)
        if not m:
            return None
        return m.group(1).split()

    sumcheck_block = _format_split_block("SUMCHECK_IDS", sumchecks, comment="Rust SumcheckId order")
    lookup_block = _format_split_block("LOOKUP_TABLES_64", lookup_tables_64, comment="Rust LookupTables<64> order")

    # Extract committed/virtual polynomial enum heads from Rust.
    committed_heads = _extract_enum_variant_heads(witness_src, "CommittedPolynomial")
    virtual_heads = _extract_enum_variant_heads(witness_src, "VirtualPolynomial")
    circuit_flags = _extract_enum_variants(instruction_src, "CircuitFlags")
    instruction_flags = _extract_enum_variants(instruction_src, "InstructionFlags")

    # Expand OpFlags/InstructionFlags into per-flag dotted names (matches existing Python naming).
    virtual_words: list[str] = []
    for h in virtual_heads:
        if h == "OpFlags":
            virtual_words.extend([f"OpFlags.{x}" for x in circuit_flags])
            continue
        if h == "InstructionFlags":
            virtual_words.extend([f"InstructionFlags.{x}" for x in instruction_flags])
            continue
        virtual_words.append(h)

    # Keep existing curated words (if any) as a superset safety net during the transition,
    # but prefer Rust-extracted ordering.
    existing_virtual = _extract_words_block("VIRTUAL_POLYS") or []
    existing_committed = _extract_words_block("COMMITTED_POLYS") or []

    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    vpoly_words = _dedup_keep_order(virtual_words + existing_virtual) or ["UnivariateSkip"]
    cpoly_words = _dedup_keep_order(committed_heads + existing_committed) or ["RamInc", "RdInc"]

    vpoly_block = _format_split_block(
        "VIRTUAL_POLYS",
        vpoly_words,
        comment="Rust VirtualPolynomial (expanded OpFlags/InstructionFlags) + compat extras",
    )
    cpoly_block = _format_split_block(
        "COMMITTED_POLYS",
        cpoly_words,
        comment="Rust CommittedPolynomial (family heads) + compat extras",
    )

    out_text = "\n".join(
        [
            '"""Compact ID spec. Regenerate via `python3 jolt-python/scripts/gen_ids_generated.py`."""',
            "",
            "from enum import StrEnum  # typed string identifiers",
            "",
            sumcheck_block,
            "",
            vpoly_block,
            "",
            cpoly_block,
            "",
            lookup_block,
            "",
            "SumcheckId = StrEnum(",
            "    \"SumcheckId\", {n: n for n in SUMCHECK_IDS}",
            ")  # typed sumcheck IDs",
            "VirtualPolynomial = StrEnum(",
            "    \"VirtualPolynomial\", {n.replace('.', '_'): n for n in VIRTUAL_POLYS}",
            ")  # typed virtual polynomial IDs",
            "CommittedPolynomial = StrEnum(",
            "    \"CommittedPolynomial\", {n: n for n in COMMITTED_POLYS}",
            ")  # typed committed poly IDs",
        ]
    )

    for i, line in enumerate(out_text.splitlines(), 1):
        if len(line) > MAX_LINE_LEN:
            raise RuntimeError(
                f"generated ids_generated.py line {i} too long: {len(line)} > {MAX_LINE_LEN}: {line!r}"
            )
    OUT.write_text(out_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

