import hashlib  # blake2b hash primitive

class Blake2bTranscript:  # Fiat-Shamir transcript matching Rust `Blake2bTranscript`.
    def __init__(self, label):  # Initialize transcript state from domain label.
        label_b = label.encode() if isinstance(label, str) else bytes(label)
        if len(label_b) > 32:
            raise ValueError("label must be <= 32 bytes")
        label_padded = label_b + b"\x00" * (32 - len(label_b))
        self.state = hashlib.blake2b(label_padded, digest_size=32).digest()
        self.n_rounds = 0

    new = classmethod(lambda cls, label: cls(label))  # Rust-style constructor alias.

    def copy(self):  # Return a cheap clone for tests/debugging.
        t = object.__new__(type(self))
        t.state = self.state
        t.n_rounds = self.n_rounds
        return t

    def state_hex(self):  # Return current 32-byte state as lowercase hex.
        return self.state.hex()

    @staticmethod
    def _label_word(label_b):  # Encode label as 32-byte right-padded word.
        if len(label_b) > 32:
            raise ValueError("label must be <= 32 bytes")
        return label_b + b"\x00" * (32 - len(label_b))

    @staticmethod
    def _label_with_len_word(label_b, n):  # Encode 24-byte label + u64(be) length in 32 bytes.
        if len(label_b) > 24:
            raise ValueError("label must be <= 24 bytes for length-prefixed methods")
        return label_b + b"\x00" * (24 - len(label_b)) + int(n).to_bytes(8, "big")

    def _round_tag(self):  # Encode the 32-byte round tag (zero28 || be_u32(n_rounds)).
        return b"\x00" * 28 + int(self.n_rounds).to_bytes(4, "big")

    def _absorb(self, payload):  # Update state := H(state || round_tag || payload), increment round.
        h = hashlib.blake2b(digest_size=32)
        h.update(self.state)
        h.update(self._round_tag())
        h.update(payload)
        self.state = h.digest()
        self.n_rounds += 1

    def _challenge_block32(self):  # Draw 32 bytes: rand := H(state || round_tag), then state := rand.
        h = hashlib.blake2b(digest_size=32)
        h.update(self.state)
        h.update(self._round_tag())
        rand = h.digest()
        self.state = rand
        self.n_rounds += 1
        return rand

    def raw_append_bytes(self, payload):  # Append raw bytes (one absorb).
        self._absorb(bytes(payload))

    def raw_append_label(self, label):  # Append fixed-size label word (one absorb).
        label_b = label.encode() if isinstance(label, str) else bytes(label)
        self._absorb(self._label_word(label_b))

    def raw_append_label_with_len(self, label, n):  # Append packed label+len prefix (one absorb).
        label_b = label.encode() if isinstance(label, str) else bytes(label)
        self._absorb(self._label_with_len_word(label_b, n))

    def raw_append_u64(self, x):  # Append u64 as EVM uint256 (one absorb).
        self._absorb(b"\x00" * 24 + int(x).to_bytes(8, "big"))

    def raw_append_scalar_fr(self, fr):  # Append BN254 Fr scalar bytes (one absorb).
        from field import Fr  # local import to keep module dependency minimal

        if not isinstance(fr, Fr):
            fr = Fr(fr)
        self._absorb(int(fr).to_bytes(32, "big"))

    def append_bytes(self, label, data):  # Append labeled bytes with length prefix (two absorbs).
        data_b = bytes(data)
        self.raw_append_label_with_len(label, len(data_b))
        self.raw_append_bytes(data_b)

    def append_u64(self, label, x):  # Append labeled u64 (two absorbs).
        self.raw_append_label(label)
        self.raw_append_u64(x)

    def append_scalar(self, label, fr):  # Append labeled Fr scalar (two absorbs).
        self.raw_append_label(label)
        self.raw_append_scalar_fr(fr)

    def append_scalars(self, label, frs):  # Append labeled list of Fr scalars (1 + N absorbs).
        frs = list(frs)
        self.raw_append_label_with_len(label, len(frs))
        for fr in frs:
            self.raw_append_scalar_fr(fr)

    def append_serializable_bytes_uncompressed(self, label, blob):  # Append Rust-serialized bytes (reverse inside).
        b = bytes(blob)
        self.raw_append_label_with_len(label, len(b))
        self.raw_append_bytes(b[::-1])

    def append_serializable(self, label, blob):  # Alias for append_serializable_bytes_uncompressed.
        self.append_serializable_bytes_uncompressed(label, blob)

    def append_serializable_bytes_reversed(self, label, blob):  # Append bytes already reversed (no extra reverse).
        self.append_bytes(label, blob)

    def challenge_bytes(self, n):  # Draw n bytes using ceil(n/32) blocks (Rust-compatible).
        n = int(n)
        out = bytearray(n)
        remaining = n
        start = 0
        while remaining > 32:
            out[start : start + 32] = self._challenge_block32()
            start += 32
            remaining -= 32
        full = self._challenge_block32()
        out[start : start + remaining] = full[:remaining]
        return bytes(out)

    def challenge_u128(self):  # Draw a u128 as Rust does (16 bytes, reverse, be->int).
        b = self.challenge_bytes(16)[::-1]
        return int.from_bytes(b, "big")

    def challenge_scalar_128_bits(self):  # Draw a BN254 Fr from 16 bytes (Rust-compatible mapping).
        from field import Fr  # local import to keep module dependency minimal

        b = self.challenge_bytes(16)
        x = int.from_bytes(b, "big")  # equivalent to Rust reverse + from_le_bytes_mod_order
        return Fr(x)

    def challenge_scalar(self):  # Draw a BN254 Fr scalar challenge.
        return self.challenge_scalar_128_bits()

    def challenge_scalar_optimized(self):  # Draw an optimized challenge (Rust: F::Challenge from u128).
        from field import Fr  # local import to keep module dependency minimal

        # Rust `MontU128Challenge<Fr>`:
        # - only low 125 bits are used (top 3 bits masked off)
        # - stored directly in Montgomery form as a 4-limb BigInt([0, 0, low, high]).
        # For BN254 Fr this value is < modulus, so it's a valid Montgomery residue.
        x = int(self.challenge_u128()) & ((1 << 125) - 1)
        low = x & ((1 << 64) - 1)
        high = x >> 64
        mont = (low << 128) | (high << 192)
        return Fr.from_montgomery(mont)

    def challenge_vector(self, n):  # Draw a vector of n Fr challenges.
        return [self.challenge_scalar() for _ in range(int(n))]

    def challenge_vector_optimized(self, n):  # Draw a vector of optimized Fr challenges.
        return [self.challenge_scalar_optimized() for _ in range(int(n))]

    def challenge_scalar_powers(self, n):  # Draw q then return [1, q, q^2, ...].
        from field import Fr  # local import to keep module dependency minimal

        n = int(n)
        q = self.challenge_scalar()
        out = [Fr.one() for _ in range(n)]
        for i in range(1, n):
            out[i] = out[i - 1] * q
        return out

