from field import Fq, Fr  # BN254 base/scalar fields (Fq, Fr)

MODULUS, CURVE_ORDER = Fq.MODULUS, Fr.MODULUS  # base field modulus, scalar field order
ATE_LOOP_COUNT, LOG_ATE_LOOP_COUNT = 29793968203157093288, 63  # BN254 6x+2 (kept for reference/tests)

def _pad(xs, n):  # Right-pad coefficient list to size n.
    return list(xs) + [0] * (n - len(xs))

def _z(n):  # Allocate n zero Fq coefficients.
    return [Fq(0)] * n

def _trim(p):  # Remove trailing zeros from polynomial coefficients.
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p

def _deg(p):  # Return polynomial degree under trailing-zero normalization.
    p = _trim(p)
    i = len(p) - 1
    while i > 0 and p[i] == 0:
        i -= 1
    return i

def _poly_div(a, b):  # Polynomial long division over Fq.
    a, b = _trim(a), _trim(b)
    q = _z(max(len(a) - len(b) + 1, 0))
    while len(a) >= len(b):
        c, k = a[-1] / b[-1], len(a) - len(b)
        q[k] = c
        for i in range(len(b)):
            a[k + i] = a[k + i] - c * b[i]
        a = _trim(a)
    return q

class FqP:  # Polynomial-extension field element over Fq.
    DEG, MOD = 0, ()

    def __init__(self, coeffs):  # Normalize and store extension coefficients.
        cs = coeffs.c if hasattr(coeffs, "c") else coeffs
        self.c = tuple(Fq(int(x)) if isinstance(x, Fq) else Fq(x) for x in _pad(cs, type(self).DEG)[: type(self).DEG])

    one = classmethod(lambda cls: cls([1] + [0] * (cls.DEG - 1)))  # Multiplicative identity.

    zero = classmethod(lambda cls: cls([0] * cls.DEG))  # Additive identity.

    def __add__(self, o):  # Coefficient-wise extension-field addition.
        return type(self)([a + b for a, b in zip(self.c, self._c(o).c)])

    def __sub__(self, o):  # Coefficient-wise extension-field subtraction.
        return type(self)([a - b for a, b in zip(self.c, self._c(o).c)])

    def __neg__(self):  # Additive inverse in extension field.
        return type(self)([-x for x in self.c])

    def __eq__(self, o):  # Equality over same extension type.
        return isinstance(o, type(self)) and self.c == o.c

    def _c(self, o):  # Coerce scalar/same-type into this extension type.
        if isinstance(o, type(self)):
            return o
        if isinstance(o, (int, Fq)):
            return type(self)([o] + [0] * (type(self).DEG - 1))
        raise TypeError(f"expected {type(self).__name__}, int or Fq")

    def __mul__(self, o):  # Multiply in quotient ring Fq[x]/(modulus).
        o = self._c(o) if not isinstance(o, (int, Fq)) else o
        if isinstance(o, (int, Fq)):
            return type(self)([x * o for x in self.c])
        n, m, b = type(self).DEG, [Fq(x) for x in type(self).MOD], _z(2 * type(self).DEG - 1)
        for i in range(n):
            for j in range(n):
                b[i + j] = b[i + j] + self.c[i] * o.c[j]
        while len(b) > n:
            top, k = b.pop(), len(b) - n
            for i in range(n):
                b[k + i] = b[k + i] - top * m[i]
        return type(self)(b)

    def __rmul__(self, o):  # Support scalar * extension-element.
        return self * o

    def __pow__(self, e):  # Square-and-multiply exponentiation.
        if e < 0:
            return (self.inv()) ** (-e)
        out, t = type(self).one(), self
        while e:
            if e & 1:
                out = out * t
            t, e = t * t, e >> 1
        return out

    def inv(self):  # Invert via extended Euclid on polynomials.
        n = type(self).DEG
        lm, hm = [Fq(1)] + _z(n), _z(n + 1)
        low, high = list(self.c) + [Fq(0)], [Fq(x) for x in type(self).MOD] + [Fq(1)]
        while _deg(low):
            r0 = _poly_div(high, low)
            r = r0 + _z(n + 1 - len(r0))
            nm, new = hm[:], high[:]
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    nm[i + j], new[i + j] = nm[i + j] - lm[i] * r[j], new[i + j] - low[i] * r[j]
            lm, low, hm, high = nm, new, lm, low
        return type(self)(lm[:n]) * (Fq(1) / low[0])

    def __truediv__(self, o):  # Division as multiply by inverse.
        return self * self._c(o).inv()

    def __repr__(self):  # Compact debug string with canonical coefficients.
        return f"{type(self).__name__}({[int(x) for x in self.c]})"

class Fq2(FqP):  # Quadratic extension field over Fq, for G2.
    DEG, MOD = 2, (1, 0)

class Fq12(FqP):  # Degree-12 extension field over Fq, for GT.
    DEG, MOD = 12, (82, 0, 0, 0, 0, 0, -18, 0, 0, 0, 0, 0)

GT = Fq12  # target group field
b, b2, b12 = Fq(3), Fq2([3, 0]) / Fq2([9, 1]), Fq12([3] + [0] * 11)  # curve params in Fq/Fq2/Fq12
G1 = (Fq(1), Fq(2))  # G1 generator
G2 = (
    Fq2([10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]),
    Fq2([8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]),
)
w = Fq12([0, 1] + [0] * 10)  # Fq12 non-residue element for twist embedding

def is_on_curve(P, B):  # Check y^2 = x^3 + B or allow point-at-infinity.
    return P is None or (P[1] * P[1] - P[0] * P[0] * P[0]) == B

def neg(P):  # Negate elliptic-curve point.
    return None if P is None else (P[0], -P[1])

def double(P):  # Elliptic-curve point doubling.
    if P is None:
        return None
    x, y = P
    m = (x * x * 3) / (y * 2)
    nx = m * m - x - x
    return (nx, -m * nx + m * x - y)

def add(P, Q):  # Elliptic-curve point addition.
    if P is None or Q is None:
        return P if Q is None else Q
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and y1 == y2:
        return double(P)
    if x1 == x2:
        return None
    m = (y2 - y1) / (x2 - x1)
    nx = m * m - x1 - x2
    return (nx, -m * nx + m * x1 - y1)

def mul(P, n):  # Double-and-add scalar multiplication.
    if n == 0 or P is None:
        return None
    if n < 0:
        return mul(neg(P), -n)
    out, a = None, P
    while n:
        if n & 1:
            out = add(out, a)
        a, n = double(a), n >> 1
    return out

def twist(P):  # Map G2 point from Fq2 to Fq12 twist representation.
    if P is None:
        return None
    x, y = P
    xc, yc = (x.c[0] - x.c[1] * 9, x.c[1]), (y.c[0] - y.c[1] * 9, y.c[1])
    nx = Fq12([int(xc[0])] + [0] * 5 + [int(xc[1])] + [0] * 5)
    ny = Fq12([int(yc[0])] + [0] * 5 + [int(yc[1])] + [0] * 5)
    return (nx * (w ** 2), ny * (w ** 3))

def cast_g1_to_fq12(P):  # Embed G1 point into Fq12 coefficients.
    return None if P is None else (Fq12([int(P[0])] + [0] * 11), Fq12([int(P[1])] + [0] * 11))

def linefunc(P1, P2, T):  # Evaluate tangent/chord line through P1,P2 at T.
    x1, y1 = P1
    x2, y2 = P2
    xt, yt = T
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
        return m * (xt - x1) - (yt - y1)
    if y1 == y2:
        m = (x1 * x1 * 3) / (y1 * 2)
        return m * (xt - x1) - (yt - y1)
    return xt - x1

BN_X = 4965661367192848881  # Arkworks BN254 curve parameter X (positive).
FINAL_EXP = (MODULUS ** 12 - 1) // CURVE_ORDER  # Canonical final exponent (reduced Tate pairing).
FINAL_EXP_K = (2 * BN_X * (6 * BN_X * BN_X + 3 * BN_X + 1)) % CURVE_ORDER  # Arkworks pairing equals canonical pairing raised to this factor.

def miller_loop(Q12, P12):  # Standard BN254 Miller loop over twisted G2 (py_pairing-style).
    if Q12 is None or P12 is None:
        return Fq12.one()
    R, f = Q12, Fq12.one()
    for i in range(LOG_ATE_LOOP_COUNT, -1, -1):
        f, R = f * f * linefunc(R, R, P12), double(R)
        if ATE_LOOP_COUNT & (1 << i):
            f, R = f * linefunc(R, Q12, P12), add(R, Q12)
    q1 = (Q12[0] ** MODULUS, Q12[1] ** MODULUS)
    nq2 = (q1[0] ** MODULUS, -(q1[1] ** MODULUS))
    f = f * linefunc(R, q1, P12)
    R = add(R, q1)
    f = f * linefunc(R, nq2, P12)
    return f

def final_exponentiate(x):  # Canonical final exponentiation into μ_r ⊂ Fq12*.
    return GT(x ** FINAL_EXP)

def _validate_pair(Q, P, i=None):  # Validate one G2/G1 input tuple for pairing.
    tag = f"pair[{i}] " if i is not None else ""
    if not is_on_curve(Q, b2):
        raise ValueError(f"{tag}Q not on G2")
    if not is_on_curve(P, b):
        raise ValueError(f"{tag}P not on G1")

def pairing(Q, P):  # Single-pair convenience wrapper over multi_pairing.
    return multi_pairing([(Q, P)])

def multi_pairing(pairs):  # Product of pairings with one final exponentiation.
    f = Fq12.one()
    for i, (Q, P) in enumerate(pairs):
        _validate_pair(Q, P, i)
        f = f * miller_loop(twist(Q), cast_g1_to_fq12(P))
    # Arkworks BN254 pairing output differs from the canonical reduced pairing by a fixed exponentiation in μ_r.
    return final_exponentiate(f) ** FINAL_EXP_K
