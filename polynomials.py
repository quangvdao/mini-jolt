import math  # factorials for barycentric Lagrange weights

from field import Fr  # BN254 scalar field

def log2_pow2(n):  # Compute log2(n) for n a power of two.
    n = int(n)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("expected power-of-two n")
    return n.bit_length() - 1

class UniPoly:  # Univariate polynomial with coefficients in Fr (for tests/debug).
    def __init__(self, coeffs):  # Store coefficients in ascending order (c0, c1, ...).
        self.coeffs = [c if isinstance(c, Fr) else Fr(c) for c in coeffs]

    def degree(self):  # Degree of the polynomial.
        return max(0, len(self.coeffs) - 1)

    def evaluate(self, x):  # Evaluate by Horner.
        x = x if isinstance(x, Fr) else Fr(x)
        out = Fr.zero()
        for c in reversed(self.coeffs):
            out = out * x + c
        return out

    def check_sum_evals_symmetric_domain(self, n, claim, *, expected_num_coeffs=None):  # Rust `UniPoly::check_sum_evals` (simplified).
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if expected_num_coeffs is not None and len(self.coeffs) != int(expected_num_coeffs):
            raise ValueError("unexpected number of coefficients")
        claim = claim if isinstance(claim, Fr) else Fr(claim)
        start = -((n - 1) // 2)
        s = Fr.zero()
        for i in range(n):
            s += self.evaluate(start + i)
        return s == claim

class CompressedUniPoly:  # Rust-style compressed univariate poly (missing linear term).
    def __init__(self, coeffs_except_linear_term):  # Store [c0, c2, c3, ...].
        self.coeffs_except_linear_term = [
            c if isinstance(c, Fr) else Fr(c) for c in coeffs_except_linear_term
        ]

    def degree(self):  # Degree bound in Rust equals len(coeffs_except_linear_term).
        return len(self.coeffs_except_linear_term)

    def _recover_linear_term(self, hint):  # Recover c1 from hint = f(0) + f(1).
        hint = hint if isinstance(hint, Fr) else Fr(hint)
        c0 = self.coeffs_except_linear_term[0]
        linear = hint - c0 - c0
        for c in self.coeffs_except_linear_term[1:]:
            linear -= c
        return linear

    def decompress(self, hint):  # Decompress into UniPoly using provided hint (for tests).
        c1 = self._recover_linear_term(hint)
        coeffs = [self.coeffs_except_linear_term[0], c1] + self.coeffs_except_linear_term[1:]
        return UniPoly(coeffs)

    def eval_from_hint(self, hint, x):  # Verifier evaluation without checking f(0)+f(1)=hint.
        x = x if isinstance(x, Fr) else Fr(x)
        c1 = self._recover_linear_term(hint)
        running_point = x
        running_sum = self.coeffs_except_linear_term[0] + x * c1
        for c in self.coeffs_except_linear_term[1:]:
            running_point = running_point * x
            running_sum += c * running_point
        return running_sum

class EqPolynomial:  # Equality polynomial utilities (verifier-only).
    @staticmethod
    def mle(x, y):  # Compute ∏ (x_i*y_i + (1-x_i)*(1-y_i)).
        if len(x) != len(y):
            raise ValueError("mle requires equal-length vectors")
        out = Fr.one()
        for xi, yi in zip(x, y):
            xi = xi if isinstance(xi, Fr) else Fr(xi)
            yi = yi if isinstance(yi, Fr) else Fr(yi)
            out *= xi * yi + (Fr.one() - xi) * (Fr.one() - yi)
        return out

    @staticmethod
    def mle_endian(x, y, same_endian=True):  # Like Rust `mle_endian` (reverse if endianness differs).
        if len(x) != len(y):
            raise ValueError("mle_endian requires equal-length vectors")
        y_it = y if same_endian else list(reversed(y))
        return EqPolynomial.mle(x, y_it)

    @staticmethod
    def evals(r, scaling_factor=None):  # Compute table { eq(r, b) : b∈{0,1}^n } in big-endian order.
        r = [x if isinstance(x, Fr) else Fr(x) for x in r]
        scale = (
            Fr.one()
            if scaling_factor is None
            else (scaling_factor if isinstance(scaling_factor, Fr) else Fr(scaling_factor))
        )
        evals = [scale]
        for x in r:
            nxt = [Fr.zero()] * (2 * len(evals))
            for i, s in enumerate(evals):
                nxt[2 * i] = s - s * x
                nxt[2 * i + 1] = s * x
            evals = nxt
        return evals

class LagrangePolynomial:  # Lagrange polynomials over symmetric consecutive-integer grids (verifier-only).
    @staticmethod
    def _nodes_i64(n):  # Return symmetric node integers [start, start+1, ..., start+n-1].
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        start = -((n - 1) // 2)
        return [start + i for i in range(n)]

    @staticmethod
    def _weights(n):  # Return barycentric weights w_i = (-1)^(n-1-i) / (i! (n-1-i)!).
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if n > 20:
            raise ValueError("intended for small n (<= 20)")
        out = []
        for i in range(n):
            sign = -1 if ((n - 1 - i) & 1) else 1
            denom = math.factorial(i) * math.factorial(n - 1 - i)
            out.append(Fr(sign) / Fr(denom))
        return out

    @staticmethod
    def evals(r, n):  # Return [L_i(r)] for symmetric grid of size n (Rust `LagrangePolynomial::evals`).
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if n > 20:
            raise ValueError("intended for small n (<= 20)")
        r = r if isinstance(r, Fr) else Fr(r)
        nodes = LagrangePolynomial._nodes_i64(n)
        dists = [r - Fr(xi) for xi in nodes]
        for i, d in enumerate(dists):
            if d == Fr.zero():
                out = [Fr.zero()] * n
                out[i] = Fr.one()
                return out
        weights = LagrangePolynomial._weights(n)
        terms = [w / d for (w, d) in zip(weights, dists)]
        s = Fr.zero()
        for t in terms:
            s += t
        inv_s = s.inv()
        return [t * inv_s for t in terms]

    @staticmethod
    def lagrange_kernel(x, y, n):  # Rust `LagrangePolynomial::lagrange_kernel` for symmetric grid.
        lx = LagrangePolynomial.evals(x, n)
        ly = LagrangePolynomial.evals(y, n)
        out = Fr.zero()
        for a, b in zip(lx, ly):
            out += a * b
        return out

class EqPlusOnePolynomial:  # MLE for eq+1 (verifier-only).
    def __init__(self, x):  # Store x in big-endian bit order.
        self.x = [v if isinstance(v, Fr) else Fr(v) for v in x]

    def evaluate(self, y):  # Evaluate eq+1(x, y) (assumes x,y big-endian).
        y = [v if isinstance(v, Fr) else Fr(v) for v in y]
        l = len(self.x)
        if len(y) != l:
            raise ValueError("eq+1 requires equal-length vectors")
        one = Fr.one()
        total = Fr.zero()
        for k in range(l):
            lower = Fr.one()
            for i in range(k):
                lower *= self.x[l - 1 - i] * (one - y[l - 1 - i])
            kth = (one - self.x[l - 1 - k]) * y[l - 1 - k]
            higher = Fr.one()
            for i in range(k + 1, l):
                xi = self.x[l - 1 - i]
                yi = y[l - 1 - i]
                higher *= xi * yi + (one - xi) * (one - yi)
            total += lower * kth * higher
        return total

class IdentityPolynomial:  # Identity polynomial over bits (verifier-only).
    def __init__(self, num_vars):  # Store number of variables.
        self.num_vars = int(num_vars)

    def evaluate(self, r):  # Compute Σ r[i] * 2^(n-1-i) (big-endian bit order).
        r = [x if isinstance(x, Fr) else Fr(x) for x in r]
        if len(r) != self.num_vars:
            raise ValueError("identity evaluate: wrong input length")
        out = Fr.zero()
        for i, x in enumerate(r):
            out += x * (1 << (self.num_vars - 1 - i))
        return out

class OperandPolynomial:  # Uninterleaved operand polynomial (verifier-only; Rust `OperandPolynomial`).
    LEFT = "left"  # Use even indices r[2*i] (Rust: OperandSide::Left path).
    RIGHT = "right"  # Use odd indices r[2*i+1] (Rust: OperandSide::Right path).

    def __init__(self, num_vars, side):  # Store num_vars + side selector (left/right).
        self.num_vars = int(num_vars)
        if (self.num_vars & 1) != 0:
            raise ValueError("OperandPolynomial requires even num_vars")
        side = str(side)
        if side not in (self.LEFT, self.RIGHT):
            raise ValueError("OperandPolynomial side must be 'left' or 'right'")
        self.side = side

    def evaluate(self, r):  # Evaluate at r: pack either even or odd bits into an integer.
        r = [x if isinstance(x, Fr) else Fr(x) for x in r]
        if len(r) != self.num_vars:
            raise ValueError("operand evaluate: wrong input length")
        half = self.num_vars // 2
        out = Fr.zero()
        for i in range(half):
            idx = 2 * i if self.side == self.LEFT else (2 * i + 1)
            out += r[idx] * (1 << (half - 1 - i))
        return out

class UnmapRamAddressPolynomial:  # Polynomial mapping k -> k*8 + start_address.
    def __init__(self, num_vars, start_address):  # Store mapping parameters.
        self.int_poly = IdentityPolynomial(int(num_vars))
        self.start_address = int(start_address)

    def evaluate(self, r):  # Evaluate mapped address at point r.
        return self.int_poly.evaluate(r) * 8 + self.start_address

class RangeMaskPolynomial:  # Range mask polynomial (verifier-only), matching Rust semantics.
    def __init__(self, range_start, range_end):  # Store [start, end) bounds as u128-ish ints.
        self.range_start = int(range_start)
        self.range_end = int(range_end)

    @staticmethod
    def _lt_mle(r, bound):  # Evaluate LT(r, bound) where bound is an integer.
        r = [x if isinstance(x, Fr) else Fr(x) for x in list(r)]
        bits = [(int(bound) >> (len(r) - 1 - i)) & 1 for i in range(len(r))]
        lt = Fr.zero()
        eq = Fr.one()
        one = Fr.one()
        for r_i, b in zip(r, bits):
            if b == 1:
                lt += eq * (one - r_i)
                eq *= r_i
            else:
                eq *= one - r_i
        return lt

    def evaluate_mle(self, r):  # Evaluate mask(r) = LT(r, end) - LT(r, start).
        return self._lt_mle(r, self.range_end) - self._lt_mle(r, self.range_start)
