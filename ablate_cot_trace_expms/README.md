# CoT trace & pipeline ablations

Controlled experiments for reviewer-style confound checks on the CEBRA + SDS pipeline.

| Subfolder | Reviewer item | Description |
|-----------|---------------|-------------|
| [`time_shuffle/`](time_shuffle/) | Time-shuffled trajectory | Permute **sentence-aligned activation steps** before SDS/EM; CEBRA still trained on real CoT order |
| [`correct_only/`](correct_only/) | Correctness confound | **Paired-correct** (both base & RFT right on same problems) |


