This folder contains **reproducible** extraction/codegen scripts for the Python verifier.

## IDs / enum spec

- **Generate**: `jolt-python/ids_generated.py`
- **Source of truth** (Rust):
  - `jolt-core/src/poly/opening_proof.rs` (`SumcheckId`)
  - `jolt-core/src/zkvm/lookup_table/mod.rs` (`LookupTables<64>` order)

Run:

```bash
python3 jolt-python/scripts/gen_ids_generated.py
```

