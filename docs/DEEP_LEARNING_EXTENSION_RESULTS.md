# Deep Learning Extension Results

This document records the optional deep-learning extension experiments. These
results are for the presentation/report extension only. The official
quantitative submission should still use the traditional image-processing and
machine-learning pipeline.

## Evaluation Protocol

The deep-learning experiments use the same leakage-control principle as the
traditional ML pipeline:

- Split method: `StratifiedGroupKFold`
- Group key: original lesion id from `base_id_from_image_id`
- Random state: `127`
- Current deep-learning validation: the first grouped fold
- Train/validation size: 480 train images, 120 validation images

This means original images and their augmented versions stay in the same side
of the split, avoiding augmentation leakage.

Important limitation:

The traditional ML result is 5-fold grouped OOF over all 600 images, while the
current deep-learning result is a single grouped validation fold. Therefore, the
deep-learning numbers should be reported as extension/search results, not as a
direct replacement for the traditional ML main result.

## Best Current Result

Best run:

```text
run                  = score_mobilenet_192_clean_pad035_acc_tta
model                = MobileNetV2
pretrained           = ImageNet pretrained
device               = MPS
image_size           = 192
mask_mode            = clean
crop_pad             = 0.35
augment_strength     = standard
batch_size           = 8
max_epochs           = 30
freeze_backbone      = 4 epochs
early_stop_patience  = 8 epochs
monitor              = accuracy
eval_tta             = true
```

Metrics on the grouped validation fold:

| metric | score |
|---|---:|
| Accuracy | 0.8750 |
| Macro-F1 | 0.8623 |
| Balanced Accuracy | 0.8533 |

Confusion matrix:

| true \ pred | mel | nv | vasc |
|---|---:|---:|---:|
| mel | 39 | 7 | 2 |
| nv | 3 | 54 | 0 |
| vasc | 0 | 3 | 12 |

Classification report:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| mel | 0.93 | 0.81 | 0.87 | 48 |
| nv | 0.84 | 0.95 | 0.89 | 57 |
| vasc | 0.86 | 0.80 | 0.83 | 15 |

The model correctly classifies 105 out of 120 validation images. Reaching 90%
accuracy on this fold would require 108 out of 120 correct predictions, so the
current best result is 3 validation images away from 90%.

## Experiment Table

| run | device | image size | mask | crop pad | monitor | TTA | accuracy | macro-F1 | balanced acc |
|---|---|---:|---|---:|---|---|---:|---:|---:|
| `score_mobilenet_192_clean_pad035_acc_tta` | MPS | 192 | clean | 0.35 | accuracy | yes | 0.8750 | 0.8623 | 0.8533 |
| `score_mobilenet_192_clean_pad035_macro_tta` | MPS | 192 | clean | 0.35 | macro-F1 | yes | 0.8583 | 0.8665 | 0.8721 |
| `score_mobilenet_192_clean_acc_tta` | MPS | 192 | clean | 0.25 | accuracy | yes | 0.8500 | 0.8436 | 0.8357 |
| `score_mobilenet_192_clean_macro_tta` | MPS | 192 | clean | 0.25 | macro-F1 | yes | 0.8500 | 0.8436 | 0.8357 |
| `score_mobilenet_224_clean_macro_tta` | MPS | 224 | clean | 0.25 | macro-F1 | yes | 0.8333 | 0.8136 | 0.8240 |
| `score_mobilenet_192_clean_pad015_macro_tta` | MPS | 192 | clean | 0.15 | macro-F1 | yes | 0.8250 | 0.8221 | 0.8018 |
| `night_lowload_mobilenet_clean_macro_f1` | MPS | 160 | clean | 0.25 | macro-F1 | no | 0.8167 | 0.8133 | 0.8254 |
| `score_mobilenet_192_raw_macro_tta` | MPS | 192 | raw | 0.25 | macro-F1 | yes | 0.8000 | 0.8103 | 0.8323 |
| `night_lowload_mobilenet_clean_acc` | MPS | 160 | clean | 0.25 | accuracy | no | 0.7833 | 0.7863 | 0.8195 |
| `score_mobilenet_160_clean_macro_tta` | CPU | 160 | clean | 0.25 | macro-F1 | yes | 0.7500 | 0.7556 | 0.7645 |
| `night_lowload_mobilenet_clean_light_acc` | MPS | 160 | clean | 0.25 | accuracy | no | 0.7417 | 0.7536 | 0.7531 |
| `mobilenet_v2_pretrained_probe` | auto | 160 | clean | 0.25 | accuracy | no | 0.6250 | 0.6093 | 0.6178 |

## Training Rounds

In deep learning, one training round is usually called an `epoch`. One epoch
means the model has seen the full training split once.

In the current experiments:

- Each run allows up to 30 epochs.
- The first 4 epochs train only the classifier head while the backbone is
  frozen.
- After epoch 4, the MobileNetV2 backbone is unfrozen and fine-tuned.
- Early stopping stops training after 8 epochs without improvement.

For the current best run, the maximum was 30 epochs, but early stopping stopped
the run at epoch 17. The best validation accuracy was reached at epoch 9.

So the answer is: yes, the deep model is trained for multiple epochs, but we are
not training endlessly. We train several controlled variants, each with early
stopping, and keep the best validation checkpoint.

## Reporting Wording

Recommended presentation wording:

> As an optional deep-learning extension, we trained an ImageNet-pretrained
> MobileNetV2 with mask-based lesion cropping and grouped validation. The best
> single-fold extension result reached 87.5% accuracy and 0.8623 macro-F1 using
> MPS acceleration, 192-pixel input, clean-mask crop with 0.35 padding, and
> test-time augmentation. This result is not used as the official quantitative
> submission because the course evaluation emphasizes traditional image
> processing and simple machine learning.

Reviewer-safe caveat:

> The deep-learning result uses the same grouped split principle and avoids
> augmentation leakage, but it is currently a single grouped validation fold.
> Full grouped deep cross-validation would be needed before claiming stable
> generalization.
