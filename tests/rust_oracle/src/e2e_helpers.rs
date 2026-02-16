use crate::hex_utils::bytes_to_hex;
use crate::transcript::Blake2b256;

pub(crate) fn run_e2e_transcript_pre_stage1(mut args: std::env::Args) {
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript;
    use jolt_core::zkvm::Serializable;
    use jolt_core::zkvm::RV64IMACProof;
    use jolt_core::zkvm::fiat_shamir_preamble;
    use common::jolt_device::JoltDevice;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());
    let proof_bytes = std::fs::read(std::path::Path::new(&dir).join("proof.bin"))
        .expect("read proof.bin");
    let program_io_bytes = std::fs::read(std::path::Path::new(&dir).join("program_io.bin"))
        .expect("read program_io.bin");
    let mut program_io = JoltDevice::deserialize_from_bytes(&program_io_bytes).expect("parse program_io");
    let proof = RV64IMACProof::deserialize_from_bytes(&proof_bytes).expect("parse proof");

    // Mirror Rust verifier: truncate trailing zeros on outputs.
    program_io.outputs.truncate(
        program_io
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    let mut t = Blake2bTranscript::new(b"Jolt");
    fiat_shamir_preamble(&program_io, proof.ram_K, proof.trace_length, &mut t);

    for c in &proof.commitments {
        t.append_serializable(b"commitment", c);
    }
    if let Some(ref c) = proof.untrusted_advice_commitment {
        t.append_serializable(b"untrusted_advice", c);
    }

    let final_state = bytes_to_hex(&t.state);
    println!("transcript_pre_stage1_state={final_state}");
}

pub(crate) fn run_e2e_stage1_debug(mut args: std::env::Args) {
    use common::jolt_device::JoltDevice;
    use jolt_core::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript;
    use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
    use jolt_core::zkvm::spartan::outer::OuterRemainingSumcheckVerifier;
    use jolt_core::zkvm::spartan::verify_stage1_uni_skip;
    use jolt_core::zkvm::fiat_shamir_preamble;
    use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
    use jolt_core::zkvm::Serializable;
    use jolt_core::poly::opening_proof::{OpeningPoint, VerifierOpeningAccumulator};
    use jolt_core::utils::math::Math;

    type F = ark_bn254::Fr;
    type PCS = jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    type PT = jolt_core::transcripts::Blake2bTranscript;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());
    let proof_bytes = std::fs::read(std::path::Path::new(&dir).join("proof.bin"))
        .expect("read proof.bin");
    let program_io_bytes = std::fs::read(std::path::Path::new(&dir).join("program_io.bin"))
        .expect("read program_io.bin");

    let mut program_io = JoltDevice::deserialize_from_bytes(&program_io_bytes).expect("parse program_io");
    let proof: CoreJoltProof<F, PCS, PT> =
        CoreJoltProof::deserialize_from_bytes(&proof_bytes).expect("parse proof");

    // Mirror verifier: truncate trailing zeros on outputs.
    program_io.outputs.truncate(
        program_io
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new(proof.trace_length.log_2());
    for (key, (_pt, claim)) in &proof.opening_claims.0 {
        opening_accumulator
            .openings
            .insert(*key, (OpeningPoint::default(), *claim));
    }

    let mut transcript = Blake2bTranscript::new(b"Jolt");
    fiat_shamir_preamble(&program_io, proof.ram_K, proof.trace_length, &mut transcript);
    for commitment in &proof.commitments {
        transcript.append_serializable(b"commitment", commitment);
    }
    if let Some(ref c) = proof.untrusted_advice_commitment {
        transcript.append_serializable(b"untrusted_advice", c);
    }

    let key = UniformSpartanKey::<F>::new(proof.trace_length.next_power_of_two());
    let uni_skip_params = verify_stage1_uni_skip(
        &proof.stage1_uni_skip_first_round_proof,
        &key,
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage1 uniskip verify");

    let outer_remaining =
        OuterRemainingSumcheckVerifier::<F>::new(key, proof.trace_length, uni_skip_params, &opening_accumulator);

    // Manual BatchedSumcheck::verify, but also print output vs expected.
    let input_claim = <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<
        F,
        Blake2bTranscript,
    >>::input_claim(&outer_remaining, &opening_accumulator);
    transcript.append_scalar(b"sumcheck_claim", &input_claim);
    let batching_coeffs: Vec<F> = transcript.challenge_vector(1);
    let coeff = batching_coeffs[0];
    let claim = input_claim * coeff;
    let (output_claim, r_sumcheck) = proof
        .stage1_sumcheck_proof
        .verify(
            claim,
            <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::num_rounds(&outer_remaining),
            <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::degree(&outer_remaining),
            &mut transcript,
        )
        .expect("sumcheck proof verify");

    let nr = <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::num_rounds(&outer_remaining);
    let r_slice = &r_sumcheck[..nr];
    <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::cache_openings(
        &outer_remaining,
        &mut opening_accumulator,
        &mut transcript,
        r_slice,
    );
    let expected =
        <OuterRemainingSumcheckVerifier<F> as SumcheckInstanceVerifier<F, Blake2bTranscript>>::expected_output_claim(
            &outer_remaining,
            &opening_accumulator,
            r_slice,
        ) * coeff;

    println!("stage1_output_claim={output_claim}");
    println!("stage1_expected_claim={expected}");
    println!("stage1_claim_diff={}", output_claim - expected);
    println!("stage1_r_len={}", r_sumcheck.len());
    println!("stage1_transcript_state_end={}", bytes_to_hex(&transcript.state));
}

pub(crate) fn run_e2e_stage6_breakdown(mut args: std::env::Args) {
    use ark_ff::Zero;
    use common::jolt_device::JoltDevice;
    use jolt_core::field::JoltField;
    use jolt_core::poly::opening_proof::{OpeningPoint, VerifierOpeningAccumulator};
    use jolt_core::subprotocols::booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier};
    use jolt_core::subprotocols::sumcheck::BatchedSumcheck;
    use jolt_core::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
    use jolt_core::transcripts::Blake2bTranscript;
    use jolt_core::transcripts::Transcript;
    use jolt_core::utils::math::Math;
    use jolt_core::zkvm::claim_reductions::{
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        RamRaClaimReductionSumcheckVerifier,
    };
    use jolt_core::zkvm::config::OneHotParams;
    use jolt_core::zkvm::fiat_shamir_preamble;
    use jolt_core::zkvm::instruction_lookups::ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier;
    use jolt_core::zkvm::instruction_lookups::read_raf_checking::InstructionReadRafSumcheckVerifier;
    use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
    use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
    use jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheckVerifier;
    use jolt_core::zkvm::ram::output_check::OutputSumcheckVerifier;
    use jolt_core::zkvm::ram::ra_virtual::RamRaVirtualSumcheckVerifier;
    use jolt_core::zkvm::ram::raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier;
    use jolt_core::zkvm::ram::read_write_checking::RamReadWriteCheckingVerifier;
    use jolt_core::zkvm::ram::val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier;
    use jolt_core::zkvm::ram::val_final::ValFinalSumcheckVerifier;
    use jolt_core::zkvm::ram::verifier_accumulate_advice;
    use jolt_core::zkvm::registers::read_write_checking::RegistersReadWriteCheckingVerifier;
    use jolt_core::zkvm::registers::val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier;
    use jolt_core::zkvm::spartan::outer::OuterRemainingSumcheckVerifier;
    use jolt_core::zkvm::spartan::product::ProductVirtualRemainderVerifier;
    use jolt_core::zkvm::spartan::shift::ShiftSumcheckVerifier;
    use jolt_core::zkvm::spartan::instruction_input::InstructionInputSumcheckVerifier;
    use jolt_core::zkvm::spartan::{verify_stage1_uni_skip, verify_stage2_uni_skip};
    use jolt_core::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier;
    use jolt_core::zkvm::Serializable;
    use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;

    type F = ark_bn254::Fr;
    type PCS = jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    type PT = jolt_core::transcripts::Blake2bTranscript;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());

    let proof_bytes = std::fs::read(std::path::Path::new(&dir).join("proof.bin"))
        .expect("read proof.bin");
    let program_io_bytes = std::fs::read(std::path::Path::new(&dir).join("program_io.bin"))
        .expect("read program_io.bin");
    let preprocessing_bytes =
        std::fs::read(std::path::Path::new(&dir).join("verifier_preprocessing.bin"))
            .expect("read verifier_preprocessing.bin");

    let mut program_io =
        JoltDevice::deserialize_from_bytes(&program_io_bytes).expect("parse program_io");
    let proof: CoreJoltProof<F, PCS, PT> =
        CoreJoltProof::deserialize_from_bytes(&proof_bytes).expect("parse proof");
    let preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::deserialize_from_bytes(&preprocessing_bytes)
            .expect("parse verifier_preprocessing");

    // Mirror verifier: truncate trailing zeros on outputs.
    program_io.outputs.truncate(
        program_io
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    let one_hot_params =
        OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

    let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new(proof.trace_length.log_2());
    for (key, (_pt, claim)) in &proof.opening_claims.0 {
        opening_accumulator
            .openings
            .insert(*key, (OpeningPoint::default(), *claim));
    }

    let mut transcript = Blake2bTranscript::new(b"Jolt");
    fiat_shamir_preamble(&program_io, proof.ram_K, proof.trace_length, &mut transcript);
    for commitment in &proof.commitments {
        transcript.append_serializable(b"commitment", commitment);
    }
    if let Some(ref c) = proof.untrusted_advice_commitment {
        transcript.append_serializable(b"untrusted_advice", c);
    }

    // Stage 1
    let spartan_key = UniformSpartanKey::<F>::new(proof.trace_length.next_power_of_two());
    let stage1_params = verify_stage1_uni_skip(
        &proof.stage1_uni_skip_first_round_proof,
        &spartan_key,
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage1 uniskip verify");
    let outer_remaining = OuterRemainingSumcheckVerifier::<F>::new(
        spartan_key,
        proof.trace_length,
        stage1_params,
        &opening_accumulator,
    );
    BatchedSumcheck::verify(
        &proof.stage1_sumcheck_proof,
        vec![&outer_remaining],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage1 sumcheck");

    // Stage 2
    let stage2_params = verify_stage2_uni_skip(
        &proof.stage2_uni_skip_first_round_proof,
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage2 uniskip verify");
    let ram_rw = RamReadWriteCheckingVerifier::new(
        &opening_accumulator,
        &mut transcript,
        &one_hot_params,
        proof.trace_length,
        &proof.rw_config,
    );
    let spartan_product_remainder = ProductVirtualRemainderVerifier::new(
        proof.trace_length,
        stage2_params,
        &opening_accumulator,
    );
    let lookups_claim_reduction =
        InstructionLookupsClaimReductionSumcheckVerifier::new(
            proof.trace_length,
            &opening_accumulator,
            &mut transcript,
        );
    let ram_raf_eval = RamRafEvaluationSumcheckVerifier::new(
        &program_io.memory_layout,
        &one_hot_params,
        &opening_accumulator,
    );
    let ram_output_check = OutputSumcheckVerifier::new(proof.ram_K, &program_io, &mut transcript);
    BatchedSumcheck::verify(
        &proof.stage2_sumcheck_proof,
        vec![
            &ram_rw,
            &spartan_product_remainder,
            &lookups_claim_reduction,
            &ram_raf_eval,
            &ram_output_check,
        ],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage2 sumcheck");

    // Stage 3
    let spartan_shift = ShiftSumcheckVerifier::new(
        proof.trace_length.log_2(),
        &opening_accumulator,
        &mut transcript,
    );
    let spartan_instruction_input =
        InstructionInputSumcheckVerifier::new(&opening_accumulator, &mut transcript);
    let regs_claim_reduction =
        jolt_core::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier::new(
            proof.trace_length,
            &opening_accumulator,
            &mut transcript,
        );
    BatchedSumcheck::verify(
        &proof.stage3_sumcheck_proof,
        vec![&spartan_shift, &spartan_instruction_input, &regs_claim_reduction],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage3 sumcheck");

    // Stage 4
    verifier_accumulate_advice::<F>(
        proof.ram_K,
        &program_io,
        proof.untrusted_advice_commitment.is_some(),
        false,
        &mut opening_accumulator,
        &mut transcript,
        proof.rw_config.needs_single_advice_opening(proof.trace_length.log_2()),
    );
    let regs_rw = RegistersReadWriteCheckingVerifier::new(
        proof.trace_length,
        &opening_accumulator,
        &mut transcript,
        &proof.rw_config,
    );
    let ram_val_eval = RamValEvaluationSumcheckVerifier::new(
        &preprocessing.shared.ram,
        &program_io,
        proof.trace_length,
        proof.ram_K,
        &opening_accumulator,
    );
    let ram_val_final = ValFinalSumcheckVerifier::new(
        &preprocessing.shared.ram,
        &program_io,
        proof.trace_length,
        proof.ram_K,
        &opening_accumulator,
        &proof.rw_config,
    );
    BatchedSumcheck::verify(
        &proof.stage4_sumcheck_proof,
        vec![&regs_rw, &ram_val_eval, &ram_val_final],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage4 sumcheck");

    // Stage 5
    let n_cycle_vars = proof.trace_length.log_2();
    let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
        n_cycle_vars,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
        proof.trace_length,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let regs_val_eval = RegistersValEvaluationSumcheckVerifier::new(&opening_accumulator);
    BatchedSumcheck::verify(
        &proof.stage5_sumcheck_proof,
        vec![&lookups_read_raf, &ram_ra_reduction, &regs_val_eval],
        &mut opening_accumulator,
        &mut transcript,
    )
    .expect("stage5 sumcheck");

    // Stage 6 breakdown (manual batched sumcheck).
    let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
        &preprocessing.shared.bytecode,
        n_cycle_vars,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let ram_hamming_booleanity = HammingBooleanitySumcheckVerifier::new(&opening_accumulator);
    let booleanity_params =
        BooleanitySumcheckParams::new(n_cycle_vars, &one_hot_params, &opening_accumulator, &mut transcript);
    let booleanity = BooleanitySumcheckVerifier::new(booleanity_params);
    let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
        proof.trace_length,
        &one_hot_params,
        &opening_accumulator,
        &mut transcript,
    );
    let lookups_ra_virtual =
        LookupsRaSumcheckVerifier::new(&one_hot_params, &opening_accumulator, &mut transcript);
    let inc_reduction =
        IncClaimReductionSumcheckVerifier::new(proof.trace_length, &opening_accumulator, &mut transcript);

    let instances: Vec<&dyn SumcheckInstanceVerifier<F, Blake2bTranscript>> = vec![
        &bytecode_read_raf,
        &booleanity,
        &ram_hamming_booleanity,
        &ram_ra_virtual,
        &lookups_ra_virtual,
        &inc_reduction,
    ];

    println!("stage6_instances={}", instances.len());
    for (i, inst) in instances.iter().enumerate() {
        let ic = inst.input_claim(&opening_accumulator);
        println!("stage6_input_claim_{i}={ic}");
        transcript.append_scalar(b"sumcheck_claim", &ic);
    }
    let batch_coeffs: Vec<F> = transcript.challenge_vector(instances.len());
    for (i, c) in batch_coeffs.iter().enumerate() {
        println!("stage6_batch_coeff_{i}={c}");
    }

    let max_degree = instances.iter().map(|s| s.degree()).max().unwrap();
    let max_rounds = instances.iter().map(|s| s.num_rounds()).max().unwrap();
    let claim: F = instances
        .iter()
        .zip(batch_coeffs.iter())
        .map(|(s, coeff)| s.input_claim(&opening_accumulator).mul_pow_2(max_rounds - s.num_rounds()) * coeff)
        .sum();
    println!("stage6_batched_input_claim={claim}");

    let (output_claim, r_sumcheck) = proof
        .stage6_sumcheck_proof
        .verify(claim, max_rounds, max_degree, &mut transcript)
        .expect("stage6 proof.verify");
    println!("stage6_output_claim={output_claim}");
    println!("stage6_r_len={}", r_sumcheck.len());

    let mut expected_total = F::zero();
    for (i, (s, coeff)) in instances.iter().zip(batch_coeffs.iter()).enumerate() {
        let offset = s.round_offset(max_rounds);
        let r_slice = &r_sumcheck[offset..offset + s.num_rounds()];
        s.cache_openings(&mut opening_accumulator, &mut transcript, r_slice);
        let exp = s.expected_output_claim(&opening_accumulator, r_slice);
        println!("stage6_expected_output_{i}={exp}");
        expected_total += exp * coeff;
    }

    println!("stage6_expected_total={expected_total}");
    println!("stage6_claim_diff={}", output_claim - expected_total);
    println!("stage6_transcript_state_end={}", bytes_to_hex(&transcript.state));
    println!("stage6_note=diff should be 0 in Rust");
}

pub(crate) fn run_e2e_bytecode_fingerprint(mut args: std::env::Args) {
    use blake2::Digest;
    use jolt_core::zkvm::instruction::{Flags, InstructionLookup, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
    use jolt_core::zkvm::lookup_table::LookupTables;
    use jolt_core::zkvm::Serializable;
    use jolt_core::zkvm::verifier::JoltVerifierPreprocessing;

    type F = ark_bn254::Fr;
    type PCS = jolt_core::poly::commitment::dory::DoryCommitmentScheme;

    let dir = args
        .next()
        .unwrap_or_else(|| "jolt-python/tests/rust_oracle/target/jolt_python_e2e/fibonacci-guest".to_string());
    let mut idx_opt: Option<usize> = None;
    let mut chunk_opt: Option<usize> = None;
    if let Some(s) = args.next() {
        if let Some(rest) = s.strip_prefix("chunk=") {
            chunk_opt = rest.parse::<usize>().ok();
        } else {
            idx_opt = s.parse::<usize>().ok();
        }
    }

    let preprocessing_bytes =
        std::fs::read(std::path::Path::new(&dir).join("verifier_preprocessing.bin"))
            .expect("read verifier_preprocessing.bin");
    let preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::deserialize_from_bytes(&preprocessing_bytes)
            .expect("parse verifier_preprocessing");

    let bytecode = &preprocessing.shared.bytecode.bytecode;
    println!("bytecode_len={}", bytecode.len());
    println!("bytecode_code_size={}", preprocessing.shared.bytecode.code_size);

    let mut h = Blake2b256::new();
    h.update((bytecode.len() as u64).to_le_bytes());

    for instr in bytecode.iter() {
        let n = instr.normalize();
        let cf = instr.circuit_flags();
        let inf = instr.instruction_flags();
        debug_assert_eq!(cf.len(), NUM_CIRCUIT_FLAGS);
        debug_assert_eq!(inf.len(), NUM_INSTRUCTION_FLAGS);

        let mut cf_mask: u16 = 0;
        for (i, b) in cf.iter().enumerate() {
            if *b {
                cf_mask |= 1u16 << i;
            }
        }
        let mut inf_mask: u8 = 0;
        for (i, b) in inf.iter().enumerate() {
            if *b {
                inf_mask |= 1u8 << i;
            }
        }

        let lt_idx: i16 = instr
            .lookup_table()
            .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
            .unwrap_or(-1);

        let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
        let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
        let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);

        h.update((n.address as u64).to_le_bytes());
        h.update(rd.to_le_bytes());
        h.update(rs1.to_le_bytes());
        h.update(rs2.to_le_bytes());
        h.update((n.operands.imm as i128).to_le_bytes());
        h.update(cf_mask.to_le_bytes());
        h.update([inf_mask]);
        h.update(lt_idx.to_le_bytes());
    }

    let digest = h.finalize();
    println!("bytecode_fingerprint={}", bytes_to_hex(&digest[..]));

    let chunk_size: usize = 256;
    println!("bytecode_chunk_size={chunk_size}");
    for (chunk_idx, chunk) in bytecode.chunks(chunk_size).enumerate() {
        let mut hc = Blake2b256::new();
        hc.update((chunk_idx as u64).to_le_bytes());
        hc.update((chunk.len() as u64).to_le_bytes());
        for instr in chunk.iter() {
            let n = instr.normalize();
            let cf = instr.circuit_flags();
            let inf = instr.instruction_flags();
            let mut cf_mask: u16 = 0;
            for (i, b) in cf.iter().enumerate() {
                if *b {
                    cf_mask |= 1u16 << i;
                }
            }
            let mut inf_mask: u8 = 0;
            for (i, b) in inf.iter().enumerate() {
                if *b {
                    inf_mask |= 1u8 << i;
                }
            }
            let lt_idx: i16 = instr
                .lookup_table()
                .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
                .unwrap_or(-1);
            let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
            let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
            let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);

            hc.update((n.address as u64).to_le_bytes());
            hc.update(rd.to_le_bytes());
            hc.update(rs1.to_le_bytes());
            hc.update(rs2.to_le_bytes());
            hc.update((n.operands.imm as i128).to_le_bytes());
            hc.update(cf_mask.to_le_bytes());
            hc.update([inf_mask]);
            hc.update(lt_idx.to_le_bytes());
        }
        let cd = hc.finalize();
        println!("bytecode_chunk_{chunk_idx}={}", bytes_to_hex(&cd[..]));
    }

    if let Some(chunk_idx) = chunk_opt {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(bytecode.len());
        println!("bytecode_dump_chunk={chunk_idx}");
        println!("bytecode_dump_range={start}..{end}");
        for i in start..end {
            let instr = &bytecode[i];
            let n = instr.normalize();
            let cf = instr.circuit_flags();
            let inf = instr.instruction_flags();
            let mut cf_mask: u16 = 0;
            for (j, b) in cf.iter().enumerate() {
                if *b {
                    cf_mask |= 1u16 << j;
                }
            }
            let mut inf_mask: u8 = 0;
            for (j, b) in inf.iter().enumerate() {
                if *b {
                    inf_mask |= 1u8 << j;
                }
            }
            let lt_idx: i16 = instr
                .lookup_table()
                .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
                .unwrap_or(-1);
            let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
            let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
            let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);
            println!(
                "bytecode_dump_entry i={i} addr={} rd={rd} rs1={rs1} rs2={rs2} imm={} cf={cf_mask} inf={inf_mask} lt={lt_idx}",
                n.address,
                n.operands.imm
            );
        }
        return;
    }

    if let Some(i) = idx_opt {
        let instr = bytecode.get(i).expect("index in range");
        let n = instr.normalize();
        let cf = instr.circuit_flags();
        let inf = instr.instruction_flags();
        let lt_idx: i16 = instr
            .lookup_table()
            .map(|t: LookupTables<64>| LookupTables::<64>::enum_index(&t) as i16)
            .unwrap_or(-1);
        let rd: i16 = n.operands.rd.map(|x| x as i16).unwrap_or(-1);
        let rs1: i16 = n.operands.rs1.map(|x| x as i16).unwrap_or(-1);
        let rs2: i16 = n.operands.rs2.map(|x| x as i16).unwrap_or(-1);

        let mut cf_mask: u16 = 0;
        for (j, b) in cf.iter().enumerate() {
            if *b {
                cf_mask |= 1u16 << j;
            }
        }
        let mut inf_mask: u8 = 0;
        for (j, b) in inf.iter().enumerate() {
            if *b {
                inf_mask |= 1u8 << j;
            }
        }

        println!("bytecode_index={i}");
        println!("bytecode_addr={}", n.address);
        println!("bytecode_rd={rd}");
        println!("bytecode_rs1={rs1}");
        println!("bytecode_rs2={rs2}");
        println!("bytecode_imm={}", n.operands.imm);
        println!("bytecode_cf_mask={cf_mask}");
        println!("bytecode_inf_mask={inf_mask}");
        println!("bytecode_lookup_table_index={lt_idx}");
    }
}
