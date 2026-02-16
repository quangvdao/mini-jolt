from __future__ import annotations  # keep type hints lightweight

from field import Fr  # BN254 scalar field for MLE evaluation

def _eq_bits(v: int, r: list[Fr]) -> Fr:  # multilinear equality poly at point r
    acc = Fr.one()
    for i in range(len(r)):
        bit = (v >> (len(r) - 1 - i)) & 1
        acc *= r[i] if bit else (Fr.one() - r[i])
    return acc

def _check_len(r: list[Fr], n: int):  # debug assert on r length
    if len(r) != n:
        raise ValueError(f"expected r of length {n}, got {len(r)}")

def _paired_sum(r, xlen, f):  # Σ_i 2^(xlen-1-i) * f(r[2i], r[2i+1]) — common pattern for bitwise ops.
    _check_len(r, 2 * xlen)
    acc = Fr.zero()
    for i in range(xlen):
        acc += Fr(1 << (xlen - 1 - i)) * f(r[2 * i], r[2 * i + 1])
    return acc

def _lt_eq_scan(r, xlen):  # Scan paired bits, return (lt, eq) for unsigned less-than / equality.
    _check_len(r, 2 * xlen)
    lt, eq = Fr.zero(), Fr.one()
    for i in range(xlen):
        x_i, y_i = r[2 * i], r[2 * i + 1]
        lt += (Fr.one() - x_i) * y_i * eq
        eq *= x_i * y_i + (Fr.one() - x_i) * (Fr.one() - y_i)
    return lt, eq

def evaluate_mle(table: str | None, r: list[Fr], xlen: int = 64) -> Fr:  # JoltLookupTable::evaluate_mle port
    if table is None:
        return Fr.zero()
    if table == "RangeCheck":  # identity/value table: interpret second operand bits as an XLEN-bit integer
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        for i in range(xlen):
            shift = xlen - 1 - i
            acc += Fr(1 << shift) * r[xlen + i]
        return acc
    if table == "RangeCheckAligned":  # like RangeCheck but forces LSB=0 (even/aligned value)
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        for i in range(xlen - 1):
            shift = xlen - 1 - i
            acc += Fr(1 << shift) * r[xlen + i]
        return acc
    if table == "And":  # bitwise AND (x & y)
        return _paired_sum(r, xlen, lambda x, y: x * y)
    if table == "Andn":  # bitwise AND-NOT (x & ~y)
        return _paired_sum(r, xlen, lambda x, y: x * (Fr.one() - y))
    if table == "Or":  # bitwise OR (x | y)
        return _paired_sum(r, xlen, lambda x, y: x + y - x * y)
    if table == "Xor":  # bitwise XOR (x ^ y)
        return _paired_sum(r, xlen, lambda x, y: (Fr.one() - x) * y + x * (Fr.one() - y))
    if table == "Equal":  # boolean equality predicate EQ(x, y) (product of per-bit equalities)
        if len(r) % 2 != 0:
            raise ValueError("r must have even length")
        acc = Fr.one()
        for i in range(0, len(r), 2):
            x_i = r[i]
            y_i = r[i + 1]
            acc *= x_i * y_i + (Fr.one() - x_i) * (Fr.one() - y_i)
        return acc
    if table == "NotEqual":  # boolean inequality predicate NEQ(x, y) = 1 - EQ(x, y)
        return Fr.one() - evaluate_mle("Equal", r, xlen=xlen)
    if table == "UnsignedLessThan":  # boolean unsigned less-than LTU(x, y)
        return _lt_eq_scan(r, xlen)[0]
    if table == "SignedLessThan":  # boolean signed less-than LTS(x, y) (sign-bit adjustment)
        lt, _ = _lt_eq_scan(r, xlen)
        return r[0] - r[1] + lt
    if table == "UnsignedGreaterThanEqual":  # boolean unsigned greater-or-equal predicate GTEU(x, y) = 1 - LTU(x, y)
        return Fr.one() - evaluate_mle("UnsignedLessThan", r, xlen=xlen)
    if table == "SignedGreaterThanEqual":  # boolean signed greater-or-equal predicate GTES(x, y) = 1 - LTS(x, y)
        return Fr.one() - evaluate_mle("SignedLessThan", r, xlen=xlen)
    if table == "LessThanEqual":  # boolean unsigned less-or-equal LTEU(x, y) = LTU + EQ
        lt, eq = _lt_eq_scan(r, xlen)
        return lt + eq
    if table == "Movsign":  # sign-mask helper: output is 0 if sign=0 else (2^XLEN - 1) (all ones)
        _check_len(r, 2 * xlen)
        ones = (1 << xlen) - 1
        return r[0] * Fr(ones)
    if table == "UpperWord":  # extract the "upper word" (first XLEN bits of the concatenated input)
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        for i in range(xlen):
            acc += Fr(1 << (xlen - 1 - i)) * r[i]
        return acc
    if table == "ValidDiv0":  # DIV-by-zero validity: either divisor!=0, or (divisor==0 and quotient==2^XLEN-1)
        _check_len(r, 2 * xlen)
        divisor_is_zero = Fr.one()
        is_valid_div_by_zero = Fr.one()
        for i in range(xlen):
            x_i = r[2 * i]
            y_i = r[2 * i + 1]
            divisor_is_zero *= Fr.one() - x_i
            is_valid_div_by_zero *= (Fr.one() - x_i) * y_i
        return Fr.one() - divisor_is_zero + is_valid_div_by_zero
    if table == "ValidUnsignedRemainder":  # REMU validity: divisor==0 OR (remainder < divisor)
        _check_len(r, 2 * xlen)
        lt, _ = _lt_eq_scan(r, xlen)
        divisor_is_zero = Fr.one()
        for i in range(xlen):
            divisor_is_zero *= Fr.one() - r[2 * i + 1]
        return lt + divisor_is_zero
    if table == "ValidSignedRemainder":  # REM validity: divisor==0 OR remainder==0 OR (|rem|<|div| and sign(rem)==sign(div))
        _check_len(r, 2 * xlen)
        x_sign = r[0]
        y_sign = r[1]
        remainder_is_zero = Fr.one() - r[0]
        divisor_is_zero = Fr.one() - r[1]
        positive_eq = (Fr.one() - x_sign) * (Fr.one() - y_sign)
        positive_lt = (Fr.one() - x_sign) * (Fr.one() - y_sign)
        negative_eq = x_sign * y_sign
        negative_gt = x_sign * y_sign
        for i in range(1, xlen):
            x_i = r[2 * i]
            y_i = r[2 * i + 1]
            if i == 1:
                positive_lt *= (Fr.one() - x_i) * y_i
                negative_gt *= x_i * (Fr.one() - y_i)
            else:
                positive_lt += positive_eq * (Fr.one() - x_i) * y_i
                negative_gt += negative_eq * x_i * (Fr.one() - y_i)
            eq_i = x_i * y_i + (Fr.one() - x_i) * (Fr.one() - y_i)
            positive_eq *= eq_i
            negative_eq *= eq_i
            remainder_is_zero *= Fr.one() - x_i
            divisor_is_zero *= Fr.one() - y_i
        return positive_lt + negative_gt + y_sign * remainder_is_zero + divisor_is_zero
    if table == "HalfwordAlignment":  # alignment predicate: 1 iff address % 2 == 0 (LSB is 0)
        lsb = r[-1]
        return Fr.one() - lsb
    if table == "WordAlignment":  # alignment predicate: 1 iff address % 4 == 0 (two LSBs are 0)
        lsb0 = r[-1]
        lsb1 = r[-2]
        return (Fr.one() - lsb0) * (Fr.one() - lsb1)
    if table == "LowerHalfWord":  # extract lower half-word from second operand (e.g. low 32 bits when XLEN=64)
        _check_len(r, 2 * xlen)
        half = xlen // 2
        acc = Fr.zero()
        for i in range(half):
            acc += Fr(1 << (half - 1 - i)) * r[xlen + half + i]
        return acc
    if table == "SignExtendHalfWord":  # sign-extend the lower half-word (e.g. 32->64) from second operand bits
        _check_len(r, 2 * xlen)
        half = xlen // 2
        lower = Fr.zero()
        for i in range(half):
            lower += Fr(1 << (half - 1 - i)) * r[xlen + half + i]
        sign_bit = r[xlen + half]
        upper = Fr.zero()
        for i in range(half):
            upper += Fr(1 << (half - 1 - i)) * sign_bit
        return lower + upper * Fr(1 << half)
    if table == "Pow2":  # power-of-two selector: output is 2^(shift mod XLEN) from the last log2(XLEN) bits
        _check_len(r, 2 * xlen)
        log_w = (xlen.bit_length() - 1)
        acc = Fr.one()
        for i in range(log_w):
            acc *= Fr.one() + Fr((1 << (1 << i)) - 1) * r[len(r) - i - 1]
        return acc
    if table == "Pow2W":  # power-of-two selector mod 32: output is 2^(shift mod 32) from the last 5 bits
        _check_len(r, 2 * xlen)
        acc = Fr.one()
        for i in range(5):
            acc *= Fr.one() + Fr((1 << (1 << i)) - 1) * r[len(r) - i - 1]
        return acc
    if table == "ShiftRightBitmask":  # maps shift s to mask ((1<<(XLEN-s))-1)<<s (ones above the shift boundary)
        _check_len(r, 2 * xlen)
        log_w = (xlen.bit_length() - 1)
        rr = r[len(r) - log_w :]
        acc = Fr.zero()
        for s in range(xlen):
            bitmask = ((1 << (xlen - s)) - 1) << s
            acc += Fr(bitmask) * _eq_bits(s, rr)
        return acc
    if table == "VirtualSRL":  # variable logical right-shift helper (bitmask-driven shift encoding)
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        for i in range(xlen):
            x_i = r[2 * i]
            y_i = r[2 * i + 1]
            acc *= Fr.one() + y_i
            acc += x_i * y_i
        return acc
    if table == "VirtualSRA":  # variable arithmetic right-shift helper (VirtualSRL + sign-extension term)
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        sign_extension = Fr.zero()
        for i in range(xlen):
            x_i = r[2 * i]
            y_i = r[2 * i + 1]
            acc *= Fr.one() + y_i
            acc += x_i * y_i
            if i != 0:
                sign_extension += Fr(1 << i) * (Fr.one() - y_i)
        return acc + r[0] * sign_extension
    if table == "VirtualROTR":  # rotate-right helper (combines right/left shift contributions under a bitmask)
        if len(r) % 2 != 0 or len(r) // 2 != xlen:
            raise ValueError("r must have length 2 * XLEN")
        prod_one_plus_y = Fr.one()
        first_sum = Fr.zero()
        second_sum = Fr.zero()
        for i in range(xlen):
            r_x = r[2 * i]
            r_y = r[2 * i + 1]
            first_sum *= Fr.one() + r_y
            first_sum += r_x * r_y
            second_sum += r_x * (Fr.one() - r_y) * prod_one_plus_y * Fr(1 << (xlen - 1 - i))
            prod_one_plus_y *= Fr.one() + r_y
        return first_sum + second_sum
    if table == "VirtualROTRW":  # rotate-right helper for 32-bit word (W) semantics (operates on low half)
        if len(r) % 2 != 0 or len(r) // 2 != xlen:
            raise ValueError("r must have length 2 * XLEN")
        prod_one_plus_y = Fr.one()
        first_sum = Fr.zero()
        second_sum = Fr.zero()
        for i in range(xlen // 2, xlen):
            r_x = r[2 * i]
            r_y = r[2 * i + 1]
            first_sum *= Fr.one() + r_y
            first_sum += r_x * r_y
            second_sum += r_x * (Fr.one() - r_y) * prod_one_plus_y * Fr(1 << (xlen - 1 - i))
            prod_one_plus_y *= Fr.one() + r_y
        return first_sum + second_sum
    if table == "VirtualRev8W":  # rev8w: reverse bytes within each 32-bit word (abcd:efgh -> dcba:hgfe)
        if len(r) < xlen:
            raise ValueError("r must have length >= XLEN")
        r = r[len(r) - xlen :]  # Rust: uses the last XLEN coordinates.
        _check_len(r, xlen)
        bits = list(reversed(r))
        def byte_at(j: int) -> Fr:
            acc = Fr.zero()
            for i in range(8):
                acc += bits[8 * j + i] * (1 << i)
            return acc
        a, b, c, d, e, f, g, h = [byte_at(j) for j in range(8)]
        out = [d, c, b, a, h, g, f, e]
        acc = Fr.zero()
        for i, bb in enumerate(out):
            acc += bb * (1 << (i * 8))
        return acc
    if table == "VirtualChangeDivisor":  # DIV overflow helper: adjust divisor in the MIN / -1 special-case
        _check_len(r, 2 * xlen)
        divisor_value = Fr.zero()
        for i in range(xlen):
            bit_value = r[2 * i + 1]
            shift = xlen - 1 - i
            divisor_value += Fr(1 << shift) * bit_value
        x_product = r[0]
        for i in range(1, xlen):
            x_product *= Fr.one() - r[2 * i]
        y_product = Fr.one()
        for i in range(xlen):
            y_product *= r[2 * i + 1]
        adjustment = Fr(2) - Fr(1 << xlen)
        return divisor_value + x_product * y_product * adjustment
    if table == "VirtualChangeDivisorW":  # DIVW overflow helper: adjust divisor under 32-bit word semantics + sign extension
        _check_len(r, 2 * xlen)
        sign_bit = r[xlen + 1]
        divisor_value = Fr.zero()
        for i in range(xlen // 2, xlen):
            bit_value = r[2 * i + 1]
            shift = xlen - 1 - i
            divisor_value += Fr(1 << shift) * bit_value
        x_product = r[xlen]
        for i in range(xlen // 2 + 1, xlen):
            x_product *= Fr.one() - r[2 * i]
        y_product = Fr.one()
        for i in range(xlen // 2, xlen):
            y_product *= r[2 * i + 1]
        sign_extension = Fr((1 << xlen) - (1 << (xlen // 2))) * sign_bit
        adjustment = Fr(2) - Fr(1 << xlen)
        return divisor_value + adjustment * x_product * y_product + sign_extension
    if table == "MulUNoOverflow":  # MULU overflow predicate: 1 iff upper XLEN bits of the 2*XLEN-bit product are all zero
        _check_len(r, 2 * xlen)
        acc = Fr.one()
        for i in range(xlen):
            acc *= Fr.one() - r[i]
        return acc
    if table.startswith("VirtualXORROTW"):  # XOR then rotate-right by ROT, but only on the lower half-word (W/32-bit semantics)
        rot = int(table.removeprefix("VirtualXORROTW"))
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        for idx in range(xlen // 2, xlen):
            r_x = r[2 * idx]
            r_y = r[2 * idx + 1]
            xor_bit = (Fr.one() - r_x) * r_y + r_x * (Fr.one() - r_y)
            position = idx - (xlen // 2)
            rotated_position = (position + rot) % (xlen // 2)
            rotated_position = (xlen // 2) - 1 - rotated_position
            acc += Fr(1 << rotated_position) * xor_bit
        return acc
    if table.startswith("VirtualXORROT"):  # XOR then rotate-right by a fixed amount (ROT), full XLEN-bit word
        rot = int(table.removeprefix("VirtualXORROT"))
        _check_len(r, 2 * xlen)
        acc = Fr.zero()
        for i in range(xlen):
            x_i = r[2 * i]
            y_i = r[2 * i + 1]
            rotated_position = (i + rot) % xlen
            bit_position = xlen - 1 - rotated_position
            acc += Fr(1 << bit_position) * ((Fr.one() - x_i) * y_i + x_i * (Fr.one() - y_i))
        return acc
    raise KeyError(f"unknown lookup table {table!r}")
