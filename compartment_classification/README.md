# Fixed-representation compartment classification

This experiment compares explicit 22-column handcrafted inputs and 384-dimensional
canonical L12 fused vectors, separately for Fish2 and FlyEM. Targets are axon (0)
and dendrite (1). Identity, cell type, labels, skeleton projection/QC metadata and
body identifiers are never feature inputs. Fish2 additionally requires QC=ok;
FlyEM uses only author exact-center matches.

Retain the encoder test bodies as an independent holdout. Fish2 bodies absent from
the older encoder cohort are also holdout. Combine the remaining original train
and validation bodies as development. Use five outer StratifiedGroupKFold splits
on development and three inner group folds for tuning, seed 20261113. The final
holdout model is tuned by three group folds using development only. Every feature
and classifier consumes exactly the same frozen row indices. Predictions include
both development out-of-fold and independent holdout rows, clearly distinguished.

L2 logistic uses float64 inputs, balanced class weights, train-only StandardScaler,
C in [0.01,0.1,1,10], lbfgs, max_iter=10000, tol=1e-6; convergence warnings fail.
Random Forest uses 300 trees, balanced_subsample, no standardization, leaf size in
[5,1], max_features in [sqrt,0.1]. Inner mean balanced accuracy selects parameters;
stable grid order breaks ties. A most-frequent Dummy baseline shares the splits.
Input assertions reject duplicates, metadata conflicts, missing/nonfinite values,
unmatched author targets, incorrect model versions and any group overlap.

The export changes wrap the existing records/load_points/core.prepare/System
inference. Optional smoke indices retain original cohort_index and point_row;
repeated inference must be bitwise identical. Full export may verify the same
smoke rows. No retraining or monitor-vector substitution is used.

Example (all asset/output variables point outside this checkout):

```sh
python fusion50/refine/export.py --config "$MODEL_CONFIG" --checkpoint "$CHECKPOINT" \
  --expected-epoch 33 --indices 0,1,2,3 --verify-repeat --output "$OUT/smoke"
python fusion50/refine/export.py --config "$MODEL_CONFIG" --checkpoint "$CHECKPOINT" \
  --expected-epoch 33 --output "$OUT/flyem_embeddings"
python compartment_classification/experiment.py prepare --dataset fish2 \
  --asset-root "$ASSETS" --output "$OUT"
python compartment_classification/experiment.py prepare --dataset flyem \
  --asset-root "$ASSETS" --run-root "$MODEL_RUN" --output "$OUT"
python compartment_classification/experiment.py fit --dataset fish2 \
  --feature handcrafted --output "$OUT" --workers 2
```

Run each dataset with handcrafted, embedding and dummy features. Raw arrays,
instance manifests, fitted models, predictions and logs stay in the output tree,
not Git. Runtime provenance records hashes, versions and split identities.
Skeleton-derived length in the handcrafted features is contextual information;
these features are not all mask-only morphology. Different dataset units and
self-supervised recipes prohibit a single-factor causal interpretation of
cross-dataset score differences. A body-disjoint downstream test also does not
prove that the entire frozen encoder never encountered its bodies: historical
image-pretraining exposure remains a limitation. Fish2 review-source provenance
is incomplete. Scores apply to single mitochondria, not neurite-segment aggregation.

After all three feature jobs finish for both datasets:

```sh
python compartment_classification/validate.py --output "$OUT"
python compartment_classification/summarize.py --output "$OUT" --bootstrap 5000
```

The report separates retained holdout results from development OOF results, gives
per-class precision/recall, per-fold metrics, count/row-normalized confusion
matrices, and 5,000 paired body cluster bootstrap intervals. Bootstrap uncertainty
is conditional on the fixed fitted models and does not include retraining or
label uncertainty. Per-body balanced accuracy averages only classes present in
that body; pooled balanced accuracy remains the main reported metric.

For resource interruptions, stop only the owned exporter. The exporter supports
`--resume` on the same output after verifying the flushed batch boundary, fixed
checkpoint/config, point input manifest and inference source hashes. It does not
resample the cohort or change seeds. Compare the resumed full export with the
original smoke and retain the previous provenance. The exclusive GPU guard can
pause/resume the owned exporter when another process enters its card; it never
signals another user's process. Successful final validation also checks that a
recorded pre-migration prefix hash is preserved.
