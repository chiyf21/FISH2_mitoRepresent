# FlyEM exact-label three-class extension

Reuses the existing fixed epoch033 export, classifiers, C/RF grids, train-only
scaler, five outer and three inner body folds, seed 20261113, and most-frequent
Dummy. Original binary development/holdout, outer body assignments AND inner body
assignments are inherited exactly. Added cable rows inherit every membership of
their body. No holdout or fold is selected again. No encoder is fitted/exported.

Targets are axon=0, dendrite=1, connecting cable=2. Preparation verifies all 6,400
rows against published_mito_exact_center annotations, source mito IDs, existing
export hashes and instance map. All three classes must occur in every partition,
outer train/test and inner train/validation set. Coverage is saved explicitly.

Use a NEW result root; the original binary output is read-only:

```sh
python compartment_classification/prepare_threeclass.py \
  --binary-results "$BINARY_OUT" --author-directory "$AUTHOR_EXACT_DIR" --output "$OUT"
for FEATURE in handcrafted embedding dummy; do
  python compartment_classification/experiment.py fit --dataset flyem \
    --feature "$FEATURE" --output "$OUT" --workers 4
done
python compartment_classification/validate.py --output "$OUT" --datasets flyem
python compartment_classification/report_threeclass.py --output "$OUT" --bootstrap 5000
python compartment_classification/plot_holdout.py --results "$OUT" \
  --destination "$OUT/report/holdout" --datasets flyem --evaluation holdout
python compartment_classification/plot_holdout.py --results "$OUT" \
  --destination "$OUT/report/development_oof" --datasets flyem --evaluation development_oof
python compartment_classification/package.py --output "$OUT" --datasets flyem
```

The original fit and validation functions read the class list from data provenance.
Classifier hyperparameters are unchanged; CPU worker count only affects execution.
Body bootstrap shares resampled bodies across features, uses 5,000 valid draws
containing all three true classes, and records rejected class-missing draws. Main
and cable-specific precision/recall/F1 differences are reported separately. It is
conditional on fixed fitted predictions, not encoder/probe retraining or label
uncertainty. Cable holdout support is 94 instances in only six independent bodies.
No neuron-type subgrouping or segment-level aggregation is performed.
