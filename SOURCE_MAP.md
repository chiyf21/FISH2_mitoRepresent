# Fusion50 source provenance

- **Surface Point-MAE** comes from the 113 run `mito_e16_reinfer_fillholes_lt200_pointmae_20261105_01/code` and its sibling `ddp_mae.py`. The Point-MAE source and license are bundled under `fusion50/vendor/point_mae`.
- **Image MAE** comes from `zqh/code/mask2former/MAE` on 113. Its `models_mae.py` and positional embedding helper match the Fusion50 runtime copies byte for byte. The source license is bundled beside the training entry point.
- **NNA and fusion** come from `mito_newmae_vicreg_reg02_cls005_20261105_01` on 113. The model, losses and image token interface are from its `code` and `vendor` directories. The training entry point comes from its top-level `train.py`. The upstream MIM-Refiner components and license used by NNA are bundled under `fusion50/vendor/mim_refiner`.
- **Export** uses the same Fusion50 forward path and L12 readout as the earlier full-manifest inference branch. Its input and output are configured per run.

Historical initial weights were the surface Point-MAE `epoch_100.pt` and image MAE `checkpoint-119.pth`. Historical Fusion50 inference used `refine_epoch050.pt`. Checkpoints are external to the repository.
