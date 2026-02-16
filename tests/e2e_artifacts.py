"""Centralized e2e artifact lookup for Rust-produced guest directories."""

import os  # env-var lookup
import pathlib  # filesystem paths
import unittest  # SkipTest

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local mini-jolt/

REQUIRED_FILES = ("proof.bin", "program_io.bin", "verifier_preprocessing.bin", "program.elf")

PROGRAM_ENV_VARS: dict[str, str] = {
    "fib": "MINI_JOLT_FIB_GUEST_DIR",
    "btreemap": "MINI_JOLT_BTREE_GUEST_DIR",
    "sha2": "MINI_JOLT_SHA2_GUEST_DIR",
    "sha3": "MINI_JOLT_SHA3_GUEST_DIR",
}


def _all_files_present(d: pathlib.Path) -> bool:
    return all((d / f).exists() for f in REQUIRED_FILES)


def ensure_guest_dir(program: str, *, fast: bool = True) -> pathlib.Path:
    """Return a Path to a directory containing Rust e2e artifacts for *program*.

    Resolution order:
      1. Env var  ``MINI_JOLT_{PROGRAM}_GUEST_DIR``  (if set and all 4 files exist).
      2. Cache dir  ``{repo}/target/jolt_python_e2e/{program}_{"fast"|"full"}/``.
      3. Raise ``unittest.SkipTest`` with actionable instructions.
    """
    env_var = PROGRAM_ENV_VARS.get(program)
    if env_var is None:
        raise ValueError(f"unknown program {program!r}; expected one of {sorted(PROGRAM_ENV_VARS)}")

    # 1. Env var
    env_val = os.environ.get(env_var)
    if env_val:
        p = pathlib.Path(env_val)
        if _all_files_present(p):
            return p

    # 2. Cache dir
    mode = "fast" if fast else "full"
    cache_dir = ROOT / "target" / "jolt_python_e2e" / f"{program}_{mode}"
    if _all_files_present(cache_dir):
        return cache_dir

    # 3. Skip with actionable message
    manifest = ROOT / "tests" / "rust_oracle" / "Cargo.toml"
    fast_flag = " --fast" if fast else ""
    cargo_cmd = (
        f"cargo run --manifest-path {manifest}"
        f" --features e2e"
        f" -- --program {program}{fast_flag}"
        f" --out-dir {cache_dir}"
    )
    raise unittest.SkipTest(
        f"e2e artifacts for {program!r} ({mode}) not found.\n"
        f"  Option A: set {env_var}=/path/to/artifacts_dir\n"
        f"  Option B: generate them with:\n"
        f"    {cargo_cmd}\n"
        f"  Note: the Jolt workspace root must be accessible for cargo to resolve deps."
    )
