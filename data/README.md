# Data folder

This folder contains the example/synthetic data files provided with the repository.

The original real surveillance data should not be uploaded here if they are subject to access restrictions.

If you run the code on another machine, update the paths in:

```text
config/dataset/dataset.yaml
```

Recommended relative paths:

```yaml
path: data/synthetic_dataset_daily_aggregated.parquet.gzip
translation_path: data/translation_df.csv
```
