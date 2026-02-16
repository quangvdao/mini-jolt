from field import Fr  # BN254 scalar field
from polynomials import EqPolynomial  # equality polynomial helper
from rv64imac.constants import RAM_START_ADDRESS  # shared RAM base constant
from openings import OpeningPoint, SumcheckId, VirtualPolynomial  # OpeningPoint + typed IDs

def remap_address(address, memory_layout):  # Mirror Rust `ram::remap_address` (verifier-side).
    address = int(address)
    if address == 0:
        return None
    lowest = int(memory_layout.get_lowest_address())
    if address < lowest:
        raise ValueError("unexpected address below lowest_address")
    return (address - lowest) // 8

def _u64_words_le(data):  # Pack bytes into u64 words (little-endian) as Rust does.
    data = bytes(data)
    out = []
    for i in range(0, len(data), 8):
        chunk = data[i : i + 8]
        word = int.from_bytes(chunk + b"\x00" * (8 - len(chunk)), "little")
        out.append(word)
    return out

def _eq_at_index(r, idx, num_vars):  # Compute eq(r, bits(idx)) with big-endian bit order.
    bits = [(idx >> (num_vars - 1 - i)) & 1 for i in range(num_vars)]
    return EqPolynomial.mle(r, [Fr(b) for b in bits])

def sparse_eval_u64_block(start_index, values_u64, r):  # Naive sparse MLE evaluation over shifted block.
    values_u64 = list(values_u64)
    if not values_u64:
        return Fr.zero()
    r = list(r)
    num_vars = len(r)
    acc = Fr.zero()
    for j, v in enumerate(values_u64):
        if int(v) == 0:
            continue
        acc += Fr(int(v)) * _eq_at_index(r, int(start_index) + int(j), num_vars)
    return acc

def eval_io_mle(program_io, r_address):  # Mirror Rust `ram::eval_io_mle` (readable, slow).
    r_address = [x if isinstance(x, Fr) else Fr(x) for x in list(r_address)]
    range_end_words = int(remap_address(RAM_START_ADDRESS, program_io.memory_layout))
    if range_end_words <= 1:
        io_len_words = 1
    else:
        io_len_words = 1 << ((range_end_words - 1).bit_length())  # next_power_of_two(range_end_words)
    num_io_vars = (io_len_words.bit_length() - 1) if io_len_words > 1 else 0
    if num_io_vars == 0:
        r_hi, r_lo = r_address, []
    else:
        r_hi, r_lo = r_address[: -num_io_vars], r_address[-num_io_vars:]
    hi_scale = Fr.one()
    for r_i in r_hi:
        hi_scale *= Fr.one() - r_i
    acc = Fr.zero()
    if program_io.inputs:
        input_start = int(remap_address(program_io.memory_layout.input_start, program_io.memory_layout))
        acc += sparse_eval_u64_block(input_start, _u64_words_le(program_io.inputs), r_lo)
    if program_io.outputs:
        output_start = int(remap_address(program_io.memory_layout.output_start, program_io.memory_layout))
        acc += sparse_eval_u64_block(output_start, _u64_words_le(program_io.outputs), r_lo)
    panic_idx = int(remap_address(program_io.memory_layout.panic, program_io.memory_layout))
    acc += sparse_eval_u64_block(panic_idx, [1 if program_io.panic_flag else 0], r_lo)
    if not program_io.panic_flag:
        term_idx = int(remap_address(program_io.memory_layout.termination, program_io.memory_layout))
        acc += sparse_eval_u64_block(term_idx, [1], r_lo)
    return hi_scale * acc

def calculate_advice_memory_evaluation(advice_opening, advice_num_vars, advice_start, memory_layout, r_address, total_memory_vars):  # Rust: ram/mod.rs:403-437.
    if advice_opening is None:
        return Fr.zero()
    _point, eval_ = advice_opening
    num_missing_vars = int(total_memory_vars) - int(advice_num_vars)
    index = remap_address(int(advice_start), memory_layout)
    if index is None:
        raise ValueError("unexpected advice_start remaps to None")
    index = int(index)
    scaling_factor = Fr.one()
    index_binary = [((index >> i) & 1) == 1 for i in reversed(range(int(total_memory_vars)))]
    selector_bits = index_binary[:num_missing_vars]
    r_address = [x if isinstance(x, Fr) else Fr(x) for x in list(r_address)]
    for i, bit in enumerate(selector_bits):
        scaling_factor *= r_address[i] if bit else (Fr.one() - r_address[i])
    return eval_ * scaling_factor

def eval_initial_ram_mle(ram_preprocessing, program_io, r_address):  # Rust: ram/mod.rs:492-528.
    r_address = [x if isinstance(x, Fr) else Fr(x) for x in list(r_address)]
    bytecode_start = remap_address(int(ram_preprocessing.min_bytecode_address), program_io.memory_layout)
    if bytecode_start is None:
        raise ValueError("bytecode_start remaps to None")
    acc = sparse_eval_u64_block(int(bytecode_start), list(ram_preprocessing.bytecode_words), r_address)
    if program_io.inputs:
        input_start = remap_address(program_io.memory_layout.input_start, program_io.memory_layout)
        if input_start is None:
            raise ValueError("input_start remaps to None")
        acc += sparse_eval_u64_block(int(input_start), _u64_words_le(program_io.inputs), r_address)
    return acc

def verifier_accumulate_advice(ram_K, program_io, has_untrusted_advice_commitment, has_trusted_advice_commitment, opening_accumulator, transcript, single_opening):  # Rust: ram/mod.rs:276-352.
    total_vars = int(ram_K).bit_length() - 1

    r_rw = opening_accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial.RamVal,
        SumcheckId.RamReadWriteChecking,
    )[0]
    r_address_rw = OpeningPoint(r_rw.r[:total_vars], "big")

    def _advice_num_vars(max_size_bytes):
        n_words = max(1, int(max_size_bytes) // 8)
        pow2 = 1 << (n_words - 1).bit_length()
        return (pow2.bit_length() - 1) if pow2 > 1 else 0

    def _compute_advice_point(r_address, max_advice_size):
        advice_vars = _advice_num_vars(max_advice_size)
        if advice_vars == 0:
            return OpeningPoint([], "big")
        return OpeningPoint(list(r_address.r[total_vars - advice_vars :]), "big")

    if has_untrusted_advice_commitment:
        max_size = int(program_io.memory_layout.max_untrusted_advice_size)
        point_rw = _compute_advice_point(r_address_rw, max_size)
        opening_accumulator.append_untrusted_advice(transcript, SumcheckId.RamValEvaluation, point_rw)
        if not single_opening:
            r_raf = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial.RamValFinal,
                SumcheckId.RamOutputCheck,
            )[0]
            point_raf = _compute_advice_point(r_raf, max_size)
            opening_accumulator.append_untrusted_advice(transcript, SumcheckId.RamValFinalEvaluation, point_raf)

    if has_trusted_advice_commitment:
        max_size = int(program_io.memory_layout.max_trusted_advice_size)
        point_rw = _compute_advice_point(r_address_rw, max_size)
        opening_accumulator.append_trusted_advice(transcript, SumcheckId.RamValEvaluation, point_rw)
        if not single_opening:
            r_raf = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial.RamValFinal,
                SumcheckId.RamOutputCheck,
            )[0]
            point_raf = _compute_advice_point(r_raf, max_size)
            opening_accumulator.append_trusted_advice(transcript, SumcheckId.RamValFinalEvaluation, point_raf)
