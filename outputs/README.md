# Outputs

Generated outputs from experiments: logs, metrics, checkpoints, and intermediate results.

## Structure

- **logs/** - Training logs and experiment logs
- **metrics/** - Computed metrics and evaluation results
- **checkpoints/** - Model checkpoints (symlinked to `../checkpoints/`)
- **cache/** - Cached computations

## Note

This directory is typically added to `.gitignore` as contents are regenerated.
Symlink important results to `../results/` for retention.
