# Representation experiment workflow

## Stable entry points

- Keep main usable for the released 384D Fusion50 checkpoints and the Fish2 Reviewed81 evaluator.
- Run Point-MAE reconstruction first. Record the data manifest and body split, initialization, width, attention heads, global batch, and training budget.
- Run Point CLS as a separate validated stage. The native-768 path standardizes the two CLS outputs at the encoder boundary. Block 9-11 use per-layer readout normalization and Block 12 uses the native final LayerNorm.
- Assemble the no-balance fusion baseline after CLS validation. Export normalized Point and Image, Concat1536 and Project384 as separate matrices with one record-ID mapping. Treat PairBalance or a change to the training loss as its own experiment.
- Use the same records, held-out body folds, and readout positions for comparisons. Save epoch-level total and component losses, UMAP parameters, and Fish2 Cell Type and Compartment metrics for both concat and projected readouts. A comparison across unequal training budgets is an endpoint comparison.

## Branch and checkpoint lifecycle

1. Start shared improvements in an isolated worktree from origin/main. Keep changes for historical experiments on their own branches. Use a clean PR for reusable commits that were developed on a stacked experiment branch.
2. A training-path PR records a smoke run with the intended model dimensions, real inputs, expected gradients, checkpoint reload, and export contracts. A multi-GPU path also needs an end-to-end test at the intended rank count. Use the established unit and integrity checks for downstream changes.
3. A run records the exact Git revision, config, checkpoint, dataset identity, and commands. Keep datasets, weights, caches, and regenerated results outside Git.
4. Before local branch cleanup, verify ancestry or the pinned commit, local worktree cleanliness, and any launcher, inference job, or shared entry referring to that worktree. Keep worktrees used by running jobs and shared entries. Shared remote branches follow the repository's retention rules.
