RAM_START_ADDRESS = 0x8000_0000  # Rust common::constants::RAM_START_ADDRESS
ALIGNMENT_FACTOR_BYTECODE = 2  # Rust common::constants::ALIGNMENT_FACTOR_BYTECODE

RISCV_REGISTER_COUNT = 32  # architectural registers x0..x31
VIRTUAL_REGISTER_COUNT = 96  # virtual registers v32..v127
VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT = 8  # reserved virtual regs v32..v39
REGISTER_COUNT = RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT  # must be a power of 2 (Rust: common::constants::REGISTER_COUNT)

CSR_MSTATUS = 0x300  # machine status CSR
CSR_MTVEC = 0x305  # machine trap-vector base CSR
CSR_MSCRATCH = 0x340  # machine scratch CSR
CSR_MEPC = 0x341  # machine exception PC CSR
CSR_MCAUSE = 0x342  # machine trap cause CSR
CSR_MTVAL = 0x343  # machine trap value CSR

