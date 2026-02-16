import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from field import Fr
from tests.oracle import run_rust_oracle
from transcript import Blake2bTranscript


def run_python_transcript_script(script_text):
    t = None
    out_lines = []
    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        op = parts[0]
        if op == "new":
            t = Blake2bTranscript.new(parts[1].encode())
        elif op == "append_u64":
            t.append_u64(parts[1].encode(), int(parts[2]))
        elif op == "append_bytes":
            t.append_bytes(parts[1].encode(), bytes.fromhex(parts[2]))
        elif op == "append_scalar_fr":
            t.append_scalar(parts[1].encode(), Fr(int(parts[2])))
        elif op == "append_scalars_fr":
            label = parts[1].encode()
            n = int(parts[2])
            xs = [Fr(int(x)) for x in parts[3 : 3 + n]]
            t.append_scalars(label, xs)
        elif op == "append_serializable_uncompressed":
            t.append_serializable_bytes_uncompressed(parts[1].encode(), bytes.fromhex(parts[2]))
        elif op == "append_serializable_bytes_reversed":
            t.append_serializable_bytes_reversed(parts[1].encode(), bytes.fromhex(parts[2]))
        elif op == "challenge_u128":
            x = t.challenge_u128()
            out_lines.append(f"challenge_u128={x}")
        elif op == "challenge_scalar_fr":
            x = t.challenge_scalar()
            out_lines.append(f"challenge_fr={int(x)}")
        elif op == "challenge_vector_fr":
            n = int(parts[1])
            xs = t.challenge_vector(n)
            out_lines.append("challenge_vec_fr=" + ",".join(str(int(x)) for x in xs))
        elif op == "challenge_scalar_powers_fr":
            n = int(parts[1])
            xs = t.challenge_scalar_powers(n)
            out_lines.append("challenge_pows_fr=" + ",".join(str(int(x)) for x in xs))
        else:
            raise ValueError(f"unknown op: {op}")
        out_lines.append("state=" + t.state_hex())
    return "\n".join(out_lines).strip()


class TranscriptCrossLangTests(unittest.TestCase):
    def test_transcript_matches_rust_oracle(self):
        label32 = "L" * 32
        label24 = "l" * 24
        script = "\n".join(
            [
                "new Jolt",
                f"append_u64 {label32} 7",
                f"append_bytes {label24} 00ff10",
                f"append_scalar_fr {label32} 42",
                f"append_scalars_fr {label24} 3 1 2 3",
                f"append_serializable_uncompressed {label24} deadbeef",
                "challenge_u128",
                "challenge_scalar_fr",
                "challenge_vector_fr 4",
                "challenge_scalar_powers_fr 5",
            ]
        )
        rust_out = run_rust_oracle("transcript_blake2b", script).strip()
        py_out = run_python_transcript_script(script).strip()
        self.assertEqual(py_out, rust_out)


if __name__ == "__main__":
    unittest.main()

