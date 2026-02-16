use clap::{Parser, ValueEnum};
use jolt_core::host;
use jolt_core::zkvm::Serializable;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
use jolt_core::zkvm::{RV64IMACProver, RV64IMACVerifier};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Program {
    Fib,
    Sha2,
    Sha3,
    Btreemap,
    Sha2Chain,
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(long, value_enum)]
    program: Option<Program>,
    #[clap(long)]
    all: bool,
    #[clap(long)]
    fast: bool,
    #[clap(long, default_value = "target/jolt_python_e2e")]
    out_dir: String,
}

pub fn main_from_cli() {
    // `rust_oracle` uses a manual mode dispatch (`field_oracle <mode> ...`).
    // When invoked as `field_oracle e2e_verify --program fib ...`, clap would see
    // the extra `e2e_verify` positional and error. Strip the first two argv entries.
    let argv = std::iter::once("e2e_verify".to_string()).chain(std::env::args().skip(2));
    let args = Args::parse_from(argv);
    let progs: Vec<Program> = if args.all || args.program.is_none() {
        vec![
            Program::Fib,
            Program::Sha2,
            Program::Sha3,
            Program::Btreemap,
            Program::Sha2Chain,
        ]
    } else {
        vec![args.program.unwrap()]
    };
    for p in progs {
        run_one(p, args.fast, &args.out_dir);
    }
}

fn run_one(p: Program, fast: bool, out_dir: &str) {
    // Force-link inline registrations for SHA2/SHA3 examples.
    #[allow(unused_imports)]
    {
        use jolt_inlines_keccak256 as _;
        use jolt_inlines_sha2 as _;
    }

    let (guest, input) = match p {
        Program::Fib => (
            "fibonacci-guest",
            postcard::to_stdvec(&(if fast { 10000u32 } else { 400000u32 })).unwrap(),
        ),
        Program::Sha2 => (
            "sha2-guest",
            postcard::to_stdvec(&vec![5u8; if fast { 256 } else { 2048 }]).unwrap(),
        ),
        Program::Sha3 => (
            "sha3-guest",
            postcard::to_stdvec(&vec![5u8; if fast { 256 } else { 2048 }]).unwrap(),
        ),
        Program::Btreemap => (
            "btreemap-guest",
            postcard::to_stdvec(&(if fast { 10u32 } else { 50u32 })).unwrap(),
        ),
        Program::Sha2Chain => {
            let mut inputs = vec![];
            inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
            inputs.append(&mut postcard::to_stdvec(&(if fast { 50u32 } else { 5000u32 })).unwrap());
            ("sha2-chain-guest", inputs)
        }
    };

    eprintln!("[rust_oracle:e2e_verify] proving {guest} (fast={fast})");

    let mut program = host::Program::new(guest);
    let (bytecode, init_memory_state, _program_size) = program.decode();
    let (_lazy_trace, trace, _profile, program_io) = program.trace(&input, &[], &[]);
    let padded_trace_len = (trace.len() + 1).next_power_of_two();
    drop(trace);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        program_io.memory_layout.clone(),
        init_memory_state,
        padded_trace_len,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let elf_opt = program.get_elf_contents();
    let elf = elf_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf,
        &input,
        &[],
        &[],
        None,
        None,
        None,
    );
    let program_io = prover.program_io.clone();
    let (proof, _dbg) = prover.prove();
    let proof_bytes = proof.serialize_to_bytes().expect("serialize proof");

    let verifier_preprocessing = jolt_core::zkvm::verifier::JoltVerifierPreprocessing::<
        ark_bn254::Fr,
        jolt_core::poly::commitment::dory::DoryCommitmentScheme,
    >::from(&prover_preprocessing);
    let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io.clone(), None, None)
        .expect("Failed to create verifier");
    verifier.verify().expect("Rust verifier rejected proof");

    let mut dir = PathBuf::from(out_dir);
    dir.push(guest);
    fs::create_dir_all(&dir).expect("create out dir");

    fs::write(dir.join("program.elf"), elf).expect("write elf");
    fs::write(dir.join("input.postcard"), &input).expect("write input");
    fs::write(dir.join("proof.bin"), &proof_bytes).expect("write proof");
    fs::write(
        dir.join("program_io.bin"),
        program_io.serialize_to_bytes().expect("serialize program_io"),
    )
    .expect("write program_io");
    fs::write(
        dir.join("verifier_preprocessing.bin"),
        verifier_preprocessing
            .serialize_to_bytes()
            .expect("serialize verifier preprocessing"),
    )
    .expect("write verifier preprocessing");

    eprintln!("[rust_oracle:e2e_verify] wrote artifacts to {}", dir.display());
}

