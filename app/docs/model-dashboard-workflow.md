# Model Dashboard Workflow

`generic-v1` builds the existing dashboard bundle shape from explicit source pointers declared in `model-config.yaml`. The backend loader and the current frontend panel keep reading the same bundle files and do not need a new API contract.

## Add a model

Add an optional `dashboard` block to the model manifest:

```yaml
dashboard:
  builder: generic-v1
  notes:
    - Internal benchmark CSV excludes archived ablations.
  sources:
    runtime_config: config.json
    experiment_config: outputs/cfg_latest.json
    primary_evaluation:
      path: outputs/metrics/latest.json
      model_name: Demo Model
    benchmark:
      path: outputs/metrics/results_demo.csv
      model_name: Demo Model
    training_history: outputs/trainer_state.json
    learning_curve:
      status: not_applicable
      reason: This model is trained on one fixed dataset size.
    class_distribution:
      train: data/train.csv
      val: data/val.csv
      test: data/test.csv
      label_field: label
      dataset_field: source_dataset
    prediction_samples:
      path: outputs/prediction_samples.csv
      production_prediction:
        model: Demo Model
        label_field: production_label
        confidence_field: production_confidence
      reference_prediction:
        model: Reference Model
        label_field: reference_label
        confidence_field: reference_confidence
    confusion_matrix:
      paths:
        - outputs/cm_test.png
        - outputs/cm_external.png
```

Notes:

- `sources` stay explicit by design. The builder does not scan the repository to guess inputs.
- Simple sources can be plain strings.
- Sections that are truly out of scope can set `status: not_applicable`.
- `class_distribution` can derive both `class-distribution.json` and `source-dataset-distribution.json` when `dataset_field` is present.
- Parquet split sources are supported when the optional `pandas` dependency is available; otherwise the builder leaves that section partial instead of failing the whole bundle.

## Build a dashboard

Build one model:

```bash
python scripts/build_model_dashboard.py --model-dir app/app-models/prod-model-demo
```

Build every model that declares `dashboard.builder: generic-v1`:

```bash
python scripts/build_model_dashboard.py --show-skipped
```

The builder writes the standard bundle under `model_dir/dashboard/`:

- `dashboard/dashboard-manifest.json`
- `dashboard/metadata/model.json`
- `dashboard/metadata/experiment-config.json` when configured
- `dashboard/summary/overview.json`
- `dashboard/summary/source-audit.json`
- `dashboard/metrics/primary-evaluation.json`
- `dashboard/metrics/benchmark-test.json` when configured
- `dashboard/metrics/cross-dataset.json` when configured
- `dashboard/curves/training-history.json` when configured
- `dashboard/curves/learning-curve.json` when configured
- `dashboard/distributions/class-distribution.json` when configured
- `dashboard/distributions/source-dataset-distribution.json` when available
- `dashboard/samples/prediction-samples.json` when configured
- `dashboard/figures/*.plotly.json` for supported chart sections
- `dashboard/confusion/*` for copied confusion-matrix images

## Partial generation

Missing or unreadable sources do not abort the build. Each section becomes one of:

- `available`
- `image_only`
- `missing`
- `not_applicable`

Metadata and summary files are still generated, so a model can expose a lightweight skeleton dashboard even before every artifact is attached.

## Legacy migration

`scripts/build_dashboard_data.py` remains the legacy, domain-specific path for the existing hardcoded prod dashboards.

The migration path is:

1. Move the model's hardcoded source paths into `model-config.yaml`.
2. Mark the model with `dashboard.builder: generic-v1`.
3. Run `scripts/build_model_dashboard.py`.
4. Compare the generated bundle with the legacy one and remove the old hardcoded builder only after parity is good enough for that model.

Already generated legacy dashboards continue to load because the backend loader still reads the same `dashboard/dashboard-manifest.json` contract.
