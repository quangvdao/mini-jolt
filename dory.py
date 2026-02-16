from field import Fr  # BN254 scalar field for challenges/scalars
from curve import add, mul, multi_pairing  # group ops + pairing product for verification

class DoryVerifyError(Exception):  # Raised when Dory PCS verification fails.
    pass

class VMVMessage:  # VMV message (c, d2, e1) for Dory evaluation proof.
    def __init__(self, c, d2, e1):  # Store GT/GT/G1 elements.
        self.c = c
        self.d2 = d2
        self.e1 = e1

class FirstReduceMessage:  # First reduce message (d1/d2 halves + e1_beta/e2_beta).
    def __init__(self, d1_left, d1_right, d2_left, d2_right, e1_beta, e2_beta):  # Store GT/GT/GT/GT/G1/G2.
        self.d1_left = d1_left
        self.d1_right = d1_right
        self.d2_left = d2_left
        self.d2_right = d2_right
        self.e1_beta = e1_beta
        self.e2_beta = e2_beta

class SecondReduceMessage:  # Second reduce message (c cross terms + e cross terms).
    def __init__(self, c_plus, c_minus, e1_plus, e1_minus, e2_plus, e2_minus):  # Store GT/GT/G1/G1/G2/G2.
        self.c_plus = c_plus
        self.c_minus = c_minus
        self.e1_plus = e1_plus
        self.e1_minus = e1_minus
        self.e2_plus = e2_plus
        self.e2_minus = e2_minus

class ScalarProductMessage:  # Final scalar product message (e1, e2).
    def __init__(self, e1, e2):  # Store G1 and G2 elements.
        self.e1 = e1
        self.e2 = e2

class DoryProof:  # Full Dory evaluation proof (matches dory-pcs v0.2.0 structure).
    def __init__(self, vmv_message, first_messages, second_messages, final_message, nu, sigma):  # Store proof fields.
        self.vmv_message = vmv_message
        self.first_messages = list(first_messages)
        self.second_messages = list(second_messages)
        self.final_message = final_message
        self.nu = int(nu)
        self.sigma = int(sigma)

class DoryVerifierSetup:  # Verifier setup for Dory PCS (precomputed deltas/chis + generators).
    def __init__(self, delta_1l, delta_1r, delta_2l, delta_2r, chi, g1_0, g2_0, h1, h2, ht):  # Store setup fields.
        self.delta_1l = list(delta_1l)
        self.delta_1r = list(delta_1r)
        self.delta_2l = list(delta_2l)
        self.delta_2r = list(delta_2r)
        self.chi = list(chi)
        self.g1_0 = g1_0
        self.g2_0 = g2_0
        self.h1 = h1
        self.h2 = h2
        self.ht = ht

class _DoryVerifierState:  # Internal accumulator state for Dory reduce-and-fold verification.
    def __init__(self, setup, vmv_message, commitment, evaluation, point_dory):  # Initialize state from instance inputs.
        self.setup = setup
        self.c = vmv_message.c
        self.d1 = commitment
        self.d2 = vmv_message.d2
        self.e1 = vmv_message.e1
        self.e2 = mul(setup.h2, int(evaluation))
        self.e1_init = vmv_message.e1
        self.d2_init = vmv_message.d2
        self.s1_acc = Fr.one()
        self.s2_acc = Fr.one()
        self.s1_coords = []
        self.s2_coords = []
        self.num_rounds = 0

    @classmethod
    def new(cls, setup, vmv_message, commitment, evaluation, point_dory, nu, sigma):  # Construct state using dory-pcs coordinate rules.
        st = cls(setup, vmv_message, commitment, evaluation, point_dory)
        nu = int(nu)
        sigma = int(sigma)
        st.num_rounds = sigma
        st.s1_coords = list(point_dory[:sigma])
        st.s2_coords = [Fr.zero() for _ in range(sigma)]
        row_coords = list(point_dory[sigma : sigma + nu])
        for i, x in enumerate(row_coords):
            st.s2_coords[i] = x
        return st

    def process_round(self, first_msg, second_msg, alpha, beta):  # Update accumulators for one reduce-and-fold round.
        if self.num_rounds <= 0:
            raise DoryVerifyError("no rounds remaining")
        alpha = alpha if isinstance(alpha, Fr) else Fr(alpha)
        beta = beta if isinstance(beta, Fr) else Fr(beta)
        alpha_inv = alpha.inv()
        beta_inv = beta.inv()
        k = self.num_rounds
        chi_k = self.setup.chi[k]
        self.c = self.c * chi_k
        self.c = self.c * (self.d2 ** int(beta))
        self.c = self.c * (self.d1 ** int(beta_inv))
        self.c = self.c * (second_msg.c_plus ** int(alpha))
        self.c = self.c * (second_msg.c_minus ** int(alpha_inv))
        delta_1l = self.setup.delta_1l[k]
        delta_1r = self.setup.delta_1r[k]
        alpha_beta = alpha * beta
        self.d1 = (first_msg.d1_left ** int(alpha)) * first_msg.d1_right
        self.d1 = self.d1 * (delta_1l ** int(alpha_beta))
        self.d1 = self.d1 * (delta_1r ** int(beta))
        delta_2l = self.setup.delta_2l[k]
        delta_2r = self.setup.delta_2r[k]
        alpha_inv_beta_inv = alpha_inv * beta_inv
        self.d2 = (first_msg.d2_left ** int(alpha_inv)) * first_msg.d2_right
        self.d2 = self.d2 * (delta_2l ** int(alpha_inv_beta_inv))
        self.d2 = self.d2 * (delta_2r ** int(beta_inv))
        self.e1 = add(self.e1, mul(first_msg.e1_beta, int(beta)))
        self.e1 = add(self.e1, mul(second_msg.e1_plus, int(alpha)))
        self.e1 = add(self.e1, mul(second_msg.e1_minus, int(alpha_inv)))
        self.e2 = add(self.e2, mul(first_msg.e2_beta, int(beta_inv)))
        self.e2 = add(self.e2, mul(second_msg.e2_plus, int(alpha)))
        self.e2 = add(self.e2, mul(second_msg.e2_minus, int(alpha_inv)))
        idx = self.num_rounds - 1
        y_t = self.s1_coords[idx]
        x_t = self.s2_coords[idx]
        one = Fr.one()
        s1_term = alpha * (one - y_t) + y_t
        s2_term = alpha_inv * (one - x_t) + x_t
        self.s1_acc = self.s1_acc * s1_term
        self.s2_acc = self.s2_acc * s2_term
        self.num_rounds -= 1

    def verify_final(self, final_msg, gamma, d):  # Perform the final 3-pairing check (includes deferred VMV constraint).
        if self.num_rounds != 0:
            raise DoryVerifyError("final verification requires num_rounds == 0")
        gamma = gamma if isinstance(gamma, Fr) else Fr(gamma)
        d = d if isinstance(d, Fr) else Fr(d)
        gamma_inv = gamma.inv()
        d_inv = d.inv()
        d_sq = d * d
        s_product = self.s1_acc * self.s2_acc
        rhs = self.c * (self.setup.ht ** int(s_product))
        rhs = rhs * self.setup.chi[0]
        rhs = rhs * (self.d2 ** int(d))
        rhs = rhs * (self.d1 ** int(d_inv))
        rhs = rhs * (self.d2_init ** int(d_sq))
        p1_g1 = add(final_msg.e1, mul(self.setup.g1_0, int(d)))
        p1_g2 = add(final_msg.e2, mul(self.setup.g2_0, int(d_inv)))
        d_inv_s1 = d_inv * self.s1_acc
        g2_term = add(self.e2, mul(self.setup.g2_0, int(d_inv_s1)))
        p2_g1 = self.setup.h1
        p2_g2 = mul(g2_term, int(-gamma))
        d_s2 = d * self.s2_acc
        g1_term = add(self.e1, mul(self.setup.g1_0, int(d_s2)))
        p3_g1 = add(mul(g1_term, int(-gamma_inv)), mul(self.e1_init, int(d_sq)))
        p3_g2 = self.setup.h2
        lhs = multi_pairing([(p1_g2, p1_g1), (p2_g2, p2_g1), (p3_g2, p3_g1)])
        if lhs != rhs:
            raise DoryVerifyError("invalid opening proof")

def verify(proof, setup, transcript, opening_point_be, evaluation, commitment, *, dory_layout="CycleMajor", log_T=None, serde_blocks=None):  # Verify a Dory opening proof (dory-pcs v0.2.0).
    if serde_blocks is None:
        raise ValueError("serde_blocks is required for transcript coupling")
    nu = int(proof.nu)
    sigma = int(proof.sigma)
    if len(proof.first_messages) != sigma or len(proof.second_messages) != sigma:
        raise DoryVerifyError("invalid proof length for sigma")
    evaluation = evaluation if isinstance(evaluation, Fr) else Fr(evaluation)
    opening_point_be = [x if isinstance(x, Fr) else Fr(x) for x in list(opening_point_be)]
    if dory_layout == "AddressMajor":
        if log_T is None:
            raise ValueError("log_T is required for AddressMajor point reordering")
        log_T = int(log_T)
        log_K = len(opening_point_be) - log_T
        if log_K < 0:
            raise ValueError("invalid log_T for opening_point_be length")
        r_address = opening_point_be[:log_K]
        r_cycle = opening_point_be[log_K:]
        reordered = list(r_cycle) + list(r_address)
    else:
        reordered = opening_point_be
    point_dory = list(reversed(reordered))
    if len(point_dory) != nu + sigma:
        raise DoryVerifyError("opening point dimension mismatch")
    expected_blocks = 5 + 12 * sigma
    if len(serde_blocks) != expected_blocks:
        raise DoryVerifyError("serde_blocks length mismatch")
    idx = 0
    transcript.append_bytes(b"dory_serde", serde_blocks[idx])
    idx += 1
    transcript.append_bytes(b"dory_serde", serde_blocks[idx])
    idx += 1
    transcript.append_bytes(b"dory_serde", serde_blocks[idx])
    idx += 1
    state = _DoryVerifierState.new(setup, proof.vmv_message, commitment, evaluation, point_dory, nu, sigma)
    for r in range(sigma):
        for _ in range(6):
            transcript.append_bytes(b"dory_serde", serde_blocks[idx])
            idx += 1
        beta = transcript.challenge_scalar()
        for _ in range(6):
            transcript.append_bytes(b"dory_serde", serde_blocks[idx])
            idx += 1
        alpha = transcript.challenge_scalar()
        state.process_round(proof.first_messages[r], proof.second_messages[r], alpha, beta)
    gamma = transcript.challenge_scalar()
    transcript.append_bytes(b"dory_serde", serde_blocks[idx])
    idx += 1
    transcript.append_bytes(b"dory_serde", serde_blocks[idx])
    idx += 1
    d = transcript.challenge_scalar()
    if idx != expected_blocks:
        raise DoryVerifyError("internal serde_blocks index mismatch")
    state.verify_final(proof.final_message, gamma, d)

def rlc_combine_claims_and_commitments(claims, commitments, transcript, label=b"rlc_claims"):  # Rust-style RLC combine at one opening point.
    claims = [x if isinstance(x, Fr) else Fr(x) for x in list(claims)]
    commitments = list(commitments)
    if len(claims) != len(commitments):
        raise ValueError("claims and commitments must have the same length")
    if not claims:
        raise ValueError("claims and commitments must be non-empty")
    transcript.append_scalars(label, claims)
    gamma_pows = transcript.challenge_scalar_powers(len(claims))
    joint_claim = Fr.zero()
    joint_commitment = None
    for c_i, C_i, coeff in zip(claims, commitments, gamma_pows):
        joint_claim += c_i * coeff
        term = C_i ** int(coeff)
        joint_commitment = term if joint_commitment is None else (joint_commitment * term)
    return joint_claim, joint_commitment

def verify_rlc_joint_opening(proof, setup, transcript, opening_point_be, claims, commitments, *, dory_layout="CycleMajor", log_T=None, serde_blocks=None, rlc_label=b"rlc_claims"):  # Combine (claim, commitment) pairs then verify one Dory opening.
    joint_claim, joint_commitment = rlc_combine_claims_and_commitments(claims, commitments, transcript, rlc_label)
    return verify(
        proof,
        setup,
        transcript,
        opening_point_be,
        joint_claim,
        joint_commitment,
        dory_layout=dory_layout,
        log_T=log_T,
        serde_blocks=serde_blocks,
    )

