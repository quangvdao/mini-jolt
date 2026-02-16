from field import Fr  # BN254 scalar field for verifier computations
from polynomials import LagrangePolynomial, log2_pow2  # barycentric Lagrange basis + shared utility
from openings import VirtualPolynomial  # typed polynomial identifiers

NUM_R1CS_CONSTRAINTS = 19  # Uniform R1CS constraints count (Rust: R1CSConstraintLabel::COUNT).
OUTER_UNIVARIATE_SKIP_DEGREE = (NUM_R1CS_CONSTRAINTS - 1) // 2  # Rust: (NUM_R1CS_CONSTRAINTS - 1) / 2.
OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE = OUTER_UNIVARIATE_SKIP_DEGREE + 1  # Rust: degree + 1.
OUTER_FIRST_ROUND_POLY_NUM_COEFFS = 3 * OUTER_UNIVARIATE_SKIP_DEGREE + 1  # Rust: 3*degree + 1.
OUTER_FIRST_ROUND_POLY_DEGREE_BOUND = OUTER_FIRST_ROUND_POLY_NUM_COEFFS - 1  # Rust: num_coeffs - 1.

ALL_R1CS_INPUTS = [  # Canonical ordering of SpartanOuter inputs (Rust: ALL_R1CS_INPUTS).
    VirtualPolynomial.LeftInstructionInput,
    VirtualPolynomial.RightInstructionInput,
    VirtualPolynomial.Product,
    VirtualPolynomial.WriteLookupOutputToRD,
    VirtualPolynomial.WritePCtoRD,
    VirtualPolynomial.ShouldBranch,
    VirtualPolynomial.PC,
    VirtualPolynomial.UnexpandedPC,
    VirtualPolynomial.Imm,
    VirtualPolynomial.RamAddress,
    VirtualPolynomial.Rs1Value,
    VirtualPolynomial.Rs2Value,
    VirtualPolynomial.RdWriteValue,
    VirtualPolynomial.RamReadValue,
    VirtualPolynomial.RamWriteValue,
    VirtualPolynomial.LeftLookupOperand,
    VirtualPolynomial.RightLookupOperand,
    VirtualPolynomial.NextUnexpandedPC,
    VirtualPolynomial.NextPC,
    VirtualPolynomial.NextIsVirtual,
    VirtualPolynomial.NextIsFirstInSequence,
    VirtualPolynomial.LookupOutput,
    VirtualPolynomial.ShouldJump,
    VirtualPolynomial.OpFlags_AddOperands,
    VirtualPolynomial.OpFlags_SubtractOperands,
    VirtualPolynomial.OpFlags_MultiplyOperands,
    VirtualPolynomial.OpFlags_Load,
    VirtualPolynomial.OpFlags_Store,
    VirtualPolynomial.OpFlags_Jump,
    VirtualPolynomial.OpFlags_WriteLookupOutputToRD,
    VirtualPolynomial.OpFlags_VirtualInstruction,
    VirtualPolynomial.OpFlags_Assert,
    VirtualPolynomial.OpFlags_DoNotUpdateUnexpandedPC,
    VirtualPolynomial.OpFlags_Advice,
    VirtualPolynomial.OpFlags_IsCompressed,
    VirtualPolynomial.OpFlags_IsFirstInSequence,
    VirtualPolynomial.OpFlags_IsLastInSequence,
]

NUM_R1CS_INPUTS = len(ALL_R1CS_INPUTS)  # Must be 37 for SpartanOuter.
_IDX = {str(name): i for i, name in enumerate(ALL_R1CS_INPUTS)}  # Name -> index mapping.

class LC:  # Linear combination Σ coeff_i * z[idx_i] + const * z[const_col].
    def __init__(self, terms=None, const=0):  # Store (index, coeff_int) terms + const term.
        self.terms = list(terms or [])
        self.const = int(const)

    @staticmethod
    def term(name, coeff=1, *, const=0):  # Convenience constructor for a single named input term.
        return LC([(_IDX[str(name)], int(coeff))], const=int(const))

    def dot(self, z, const_col):  # Evaluate this LC on z with explicit const column index.
        out = Fr.zero()
        for idx, coeff in self.terms:
            out += Fr(int(coeff)) * z[int(idx)]
        if self.const:
            out += Fr(self.const) * z[int(const_col)]
        return out

class R1CSConstraint:  # A single uniform constraint row (A,B), proving A(z)*B(z)=0.
    def __init__(self, a_lc, b_lc):  # Store LCs for A and B.
        self.a = a_lc
        self.b = b_lc

def _canon_name(name):  # Normalize short names like "Load" -> "OpFlags.Load".
    name = str(name)
    if name in _IDX:
        return name
    op = f"OpFlags.{name}"
    if op in _IDX:
        return op
    raise KeyError(f"unknown R1CS input name: {name!r}")

def lc(expr):  # Parse a tiny LC expression: sums of NAME / k*NAME / ints, with + and -.
    s = str(expr).replace(" ", "")
    if not s:
        return LC([], 0)
    if s[0] not in "+-":
        s = "+" + s
    parts = []
    start = 0
    for i in range(1, len(s)):
        if s[i] in "+-":
            parts.append(s[start:i])
            start = i
    parts.append(s[start:])
    terms = []
    const = 0
    for part in parts:
        sign = -1 if part[0] == "-" else 1
        tok = part[1:]
        if not tok:
            continue
        if "*" in tok:
            a, b = tok.split("*", 1)
            coeff = int(a, 0) * sign
            name = _canon_name(b)
            terms.append((_IDX[name], coeff))
            continue
        # constant
        if tok[0].isdigit() or tok.startswith("0x"):
            const += sign * int(tok, 0)
            continue
        # bare name
        name = _canon_name(tok)
        terms.append((_IDX[name], sign))
    return LC(terms, const=const)

def r1cs_eq_conditional(label, cond_expr, left_expr, right_expr):  # Rust-like macro: condition * (left-right) == 0.
    left = lc(left_expr)
    right = lc(right_expr)
    b_terms = list(left.terms) + [(i, -c) for (i, c) in right.terms]
    b_const = left.const - right.const
    return (str(label), R1CSConstraint(lc(cond_expr), LC(b_terms, const=b_const)))

_FIRST_GROUP_SPEC = [  # Rust: R1CS_CONSTRAINTS_FIRST_GROUP_LABELS (order preserved).
    ("RamAddrEqZeroIfNotLoadStore", "1 - Load - Store", "RamAddress", "0"),
    ("RamReadEqRamWriteIfLoad", "Load", "RamReadValue", "RamWriteValue"),
    ("RamReadEqRdWriteIfLoad", "Load", "RamReadValue", "RdWriteValue"),
    ("Rs2EqRamWriteIfStore", "Store", "Rs2Value", "RamWriteValue"),
    ("LeftLookupZeroUnlessAddSubMul", "AddOperands + SubtractOperands + MultiplyOperands", "LeftLookupOperand", "0"),
    ("LeftLookupEqLeftInputOtherwise", "1 - AddOperands - SubtractOperands - MultiplyOperands", "LeftLookupOperand", "LeftInstructionInput"),
    ("AssertLookupOne", "Assert", "LookupOutput", "1"),
    ("NextUnexpPCEqLookupIfShouldJump", "ShouldJump", "NextUnexpandedPC", "LookupOutput"),
    ("NextPCEqPCPlusOneIfInline", "VirtualInstruction - IsLastInSequence", "NextPC", "PC + 1"),
    ("MustStartSequenceFromBeginning", "NextIsVirtual - NextIsFirstInSequence", "1", "DoNotUpdateUnexpandedPC"),
]

_SECOND_GROUP_SPEC = [  # Rust: complement of first group within `R1CS_CONSTRAINTS` table, preserving order.
    ("RamAddrEqRs1PlusImmIfLoadStore", "Load + Store", "RamAddress", "Rs1Value + Imm"),
    ("RightLookupAdd", "AddOperands", "RightLookupOperand", "LeftInstructionInput + RightInstructionInput"),
    ("RightLookupSub", "SubtractOperands", "RightLookupOperand", "LeftInstructionInput - RightInstructionInput + 0x10000000000000000"),
    ("RightLookupEqProductIfMul", "MultiplyOperands", "RightLookupOperand", "Product"),
    ("RightLookupEqRightInputOtherwise", "1 - AddOperands - SubtractOperands - MultiplyOperands - Advice", "RightLookupOperand", "RightInstructionInput"),
    ("RdWriteEqLookupIfWriteLookupToRd", "WriteLookupOutputToRD", "RdWriteValue", "LookupOutput"),
    ("RdWriteEqPCPlusConstIfWritePCtoRD", "WritePCtoRD", "RdWriteValue", "UnexpandedPC + 4 - 2*IsCompressed"),
    ("NextUnexpPCEqPCPlusImmIfShouldBranch", "ShouldBranch", "NextUnexpandedPC", "UnexpandedPC + Imm"),
    ("NextUnexpPCUpdateOtherwise", "1 - ShouldBranch - Jump", "NextUnexpandedPC", "UnexpandedPC + 4 - 4*DoNotUpdateUnexpandedPC - 2*IsCompressed"),
]

R1CS_CONSTRAINTS_FIRST_GROUP = [c for (_lbl, c) in (r1cs_eq_conditional(*t) for t in _FIRST_GROUP_SPEC)]
R1CS_CONSTRAINTS_SECOND_GROUP = [c for (_lbl, c) in (r1cs_eq_conditional(*t) for t in _SECOND_GROUP_SPEC)]

class UniformSpartanKey:  # Minimal verifier key for Spartan outer inner-sum-product.
    def __init__(self, num_steps):  # Store padded trace length (power of two).
        self.num_steps = int(num_steps)
        _ = log2_pow2(self.num_steps)

    def num_cycle_vars(self):  # Number of bits to represent cycles (Rust: num_steps.log2()).
        return log2_pow2(self.num_steps)

    def num_rows_bits(self):  # Rust: num_cycle_vars + 2.
        return self.num_cycle_vars() + 2

    def evaluate_inner_sum_product_at_point(self, rx_constr, r1cs_input_evals):  # Rust: key.rs:71-121.
        rx_constr = list(rx_constr)
        if len(rx_constr) < 2:
            raise ValueError("rx_constr must have len>=2")
        r_stream = rx_constr[0]
        r0 = rx_constr[1]
        w = LagrangePolynomial.evals(r0, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE)
        z = list(r1cs_input_evals) + [Fr.one()]  # trailing const column
        const_col = NUM_R1CS_INPUTS

        az_g0 = Fr.zero()
        bz_g0 = Fr.zero()
        for i, row in enumerate(R1CS_CONSTRAINTS_FIRST_GROUP):
            az_g0 += w[i] * row.a.dot(z, const_col)
            bz_g0 += w[i] * row.b.dot(z, const_col)

        az_g1 = Fr.zero()
        bz_g1 = Fr.zero()
        g2_len = min(len(R1CS_CONSTRAINTS_SECOND_GROUP), OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE)
        for i in range(g2_len):
            row = R1CS_CONSTRAINTS_SECOND_GROUP[i]
            az_g1 += w[i] * row.a.dot(z, const_col)
            bz_g1 += w[i] * row.b.dot(z, const_col)

        az_final = az_g0 + r_stream * (az_g1 - az_g0)
        bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
        return az_final * bz_final
