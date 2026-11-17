# LayerNorm loss routing

## Joint 50-epoch fusion
At L9–L12, Point-MAE and Image-MAE CLS are normalized before any objective or fusion (L9–L11 feature LayerNorm; L12 native final LayerNorm). Point/image variance-covariance regularizers (weights 1.25/0.05) act on those normalized CLS vectors before concatenation. PairBalance is off. The 1536D concatenation enters the fusion MLP; VICReg (25/5/0.2) and filtered InfoNCE act on the fused output.

## Fixed 50-epoch image-only ablation
Run point and image branches separately for 50 epochs each. In each run, only that modality’s CLS and Transformer blocks 9–12 train; the other encoder and fusion MLP are frozen. Point CLS starts from the training-set point-feature mean with small noise. Per-layer VICReg and filtered InfoNCE act on the LayerNorm'd image CLS. No extra CLS regularizer is added, avoiding duplicate variance/covariance pressure. PairBalance is off.

InfoNCE positives are the other view of the same instance. The top-20 nearest other instances are excluded from negatives; remaining instances in the global batch are negatives at temperature 0.2.

Upstream already had the joint loss route, but lacked this fixed image-only entry point. This patch adds it without changing VICReg or neighbor selection.
