import pathlib  # path helpers
import shutil  # find cargo
import subprocess  # run rust oracle
import textwrap  # nice error messages
import unittest  # SkipTest


ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo-local mini-jolt/
ORACLE_MANIFEST = ROOT / "tests" / "rust_oracle" / "Cargo.toml"  # optional rust oracle manifest


def run_rust_oracle(mode, stdin_text):  # Run `tests/rust_oracle` with given mode.
    if not ORACLE_MANIFEST.exists():
        raise unittest.SkipTest("rust oracle not present (tests/rust_oracle).")
    if shutil.which("cargo") is None:
        raise unittest.SkipTest("cargo not found; skipping rust oracle tests.")
    p = subprocess.run(
        ["cargo", "run", "--quiet", "--manifest-path", str(ORACLE_MANIFEST), "--", mode],
        input=stdin_text,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            textwrap.dedent(
                f"""
                Rust oracle failed (mode={mode!r}, exit={p.returncode}).

                --- stdout ---
                {p.stdout}
                --- stderr ---
                {p.stderr}
                """
            ).strip()
        )
    return p.stdout  # Return full stdout (caller can splitlines/strip as needed).


def limbs_csv_to_int(csv):  # Convert arkworks BigInt limbs CSV (LE u64 limbs) into int.
    return sum(int(w) << (64 * i) for i, w in enumerate(csv.split(",")))


def parse_g1_csv(s):  # Parse Rust oracle G1 CSV ("inf" or "x:y") into Python tuple/None.
    if s == "inf":
        return None
    x, y = s.split(":")
    return (limbs_csv_to_int(x), limbs_csv_to_int(y))


def parse_g2_csv(s):  # Parse Rust oracle G2 CSV ("inf" or "x0,x1/y0,y1" nested CSVs).
    if s == "inf":
        return None
    x, y = s.split(":")
    x0, x1 = x.split("/")
    y0, y1 = y.split("/")
    return (limbs_csv_to_int(x0), limbs_csv_to_int(x1), limbs_csv_to_int(y0), limbs_csv_to_int(y1))

