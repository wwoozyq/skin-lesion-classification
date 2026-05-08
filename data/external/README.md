# External Data

Do not commit external images to this repository.

Recommended local layout:

```text
data/external/isic2018_task3/
  images/
    ISIC_0024306.jpg
    ...
  label.csv
  manifest.csv
```

Use the script below to convert the official ISIC 2018 Task 3 ground truth file:

```bash
python scripts/prepare_isic2018_labels.py \
  --ground_truth_csv /path/to/ISIC2018_Task3_Training_GroundTruth.csv \
  --output_dir data/external/isic2018_task3
```

Only `MEL`, `NV`, and `VASC` are kept, because they map directly to this project:

```text
MEL  -> mel
NV   -> nv
VASC -> vasc
```

The generated files are small and may be used for local experiments. The image
files themselves should stay local.
