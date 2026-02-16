class MontgomeryField:  # Prime field element in Montgomery representation.
    R = 1 << 256  # Montgomery radix.
    MASK = R - 1  # Low-256-bit mask.

    def __init_subclass__(cls):  # Precompute Montgomery constants for each subclass.
        if "MODULUS" not in cls.__dict__:
            return
        p = cls.MODULUS
        if p % 2 == 0 or p >= cls.R:
            raise ValueError("MODULUS must be odd and < 2^256")
        cls.NP = (-pow(p, -1, cls.R)) & cls.MASK
        cls.R1 = cls.R % p
        cls.R2 = (cls.R1 * cls.R1) % p

    def __init__(self, x=0, mont=False):  # Build element from canonical int or raw Montgomery value.
        p = type(self).MODULUS
        self.v = x % p if mont else type(self)._red((x % p) * type(self).R2)

    @classmethod
    def _red(cls, t):  # Montgomery reduction: t * R^-1 mod MODULUS.
        m = ((t & cls.MASK) * cls.NP) & cls.MASK
        u = (t + m * cls.MODULUS) >> 256
        return u - cls.MODULUS if u >= cls.MODULUS else u

    zero = classmethod(lambda cls: cls(0, mont=True))  # Additive identity in Montgomery form.

    one = classmethod(lambda cls: cls(cls.R1, mont=True))  # Multiplicative identity in Montgomery form.

    from_montgomery = classmethod(lambda cls, x: cls(x, mont=True))  # Wrap raw Montgomery residue.

    def to_int(self): return type(self)._red(self.v)  # Convert to canonical integer form.

    def inv(self):  # Multiplicative inverse in the same field.
        if self.v == 0: raise ZeroDivisionError("cannot invert zero")
        return type(self)(pow(self.to_int(), -1, type(self).MODULUS))

    def _c(self, other):  # Coerce int/same-type operand into field element.
        cls = type(self)
        if isinstance(other, cls):
            return other
        if isinstance(other, int):
            return cls(other)
        raise TypeError(f"expected {cls.__name__} or int")

    def __add__(self, other):  # Field addition modulo MODULUS.
        v = self.v + self._c(other).v
        return type(self)(v - type(self).MODULUS if v >= type(self).MODULUS else v, mont=True)

    def __sub__(self, other):  # Field subtraction modulo MODULUS.
        v = self.v - self._c(other).v
        return type(self)(v + type(self).MODULUS if v < 0 else v, mont=True)

    def __mul__(self, other):  # Field multiplication via Montgomery reduction.
        return type(self)(type(self)._red(self.v * self._c(other).v), mont=True)

    def __pow__(self, e):  # Exponentiation with modular power semantics.
        return (self.inv()) ** (-e) if e < 0 else type(self)(pow(self.to_int(), e, type(self).MODULUS))

    def __truediv__(self, other):  # Division as multiply by inverse.
        return self * self._c(other).inv()

    def __neg__(self):  # Additive inverse modulo MODULUS.
        return self if self.v == 0 else type(self)(type(self).MODULUS - self.v, mont=True)

    def __eq__(self, other):  # Equality with field elements or canonical ints.
        if isinstance(other, type(self)):
            return self.v == other.v
        return self.to_int() == (other % type(self).MODULUS) if isinstance(other, int) else False

    def __int__(self): return self.to_int()  # int(...) exposes canonical integer.

    def __repr__(self): return f"{type(self).__name__}({self.to_int()})"  # Debug-friendly printable form.

class Fq(MontgomeryField):  # BN254 base field.
    MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583  # BN254 Fq modulus

class Fr(MontgomeryField):  # BN254 scalar field.
    MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617  # BN254 Fr modulus
