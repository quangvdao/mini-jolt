from jolt_proof import RustDeserializeError, _Reader  # reuse the proof reader primitives
from zkvm_types import JoltDevice, MemoryLayout


def parse_memory_layout_bytes(r: _Reader) -> MemoryLayout:  # Parse Rust `MemoryLayout` (CanonicalDeserialize compressed).
    # Field order matches `common/src/jolt_device.rs`.
    program_size = r.u64()
    max_trusted_advice_size = r.u64()
    trusted_advice_start = r.u64()
    trusted_advice_end = r.u64()
    max_untrusted_advice_size = r.u64()
    untrusted_advice_start = r.u64()
    untrusted_advice_end = r.u64()
    max_input_size = r.u64()
    max_output_size = r.u64()
    input_start = r.u64()
    input_end = r.u64()
    output_start = r.u64()
    output_end = r.u64()
    stack_size = r.u64()
    stack_end = r.u64()
    heap_size = r.u64()
    heap_end = r.u64()
    panic = r.u64()
    termination = r.u64()
    io_end = r.u64()
    return MemoryLayout(
        input_start=int(input_start),
        output_start=int(output_start),
        panic=int(panic),
        termination=int(termination),
        program_size=int(program_size),
        max_trusted_advice_size=int(max_trusted_advice_size),
        trusted_advice_start=int(trusted_advice_start),
        trusted_advice_end=int(trusted_advice_end),
        max_untrusted_advice_size=int(max_untrusted_advice_size),
        untrusted_advice_start=int(untrusted_advice_start),
        untrusted_advice_end=int(untrusted_advice_end),
        max_input_size=int(max_input_size),
        max_output_size=int(max_output_size),
        input_end=int(input_end),
        output_end=int(output_end),
        stack_size=int(stack_size),
        stack_end=int(stack_end),
        heap_size=int(heap_size),
        heap_end=int(heap_end),
        io_end=int(io_end),
    )


def parse_jolt_device_bytes(data: bytes) -> JoltDevice:  # Parse Rust `common::JoltDevice` (CanonicalDeserialize compressed).
    r = _Reader(data)
    inputs = bytes(r.vec(lambda: r.u8()))
    trusted = bytes(r.vec(lambda: r.u8()))
    untrusted = bytes(r.vec(lambda: r.u8()))
    outputs = bytes(r.vec(lambda: r.u8()))
    panic = r.bool()
    ml = parse_memory_layout_bytes(r)
    if r.remaining() != 0:
        raise RustDeserializeError("trailing bytes in JoltDevice")
    return JoltDevice(
        memory_layout=ml,
        inputs=inputs,
        trusted_advice=trusted,
        untrusted_advice=untrusted,
        outputs=outputs,
        panic_flag=panic,
    )

