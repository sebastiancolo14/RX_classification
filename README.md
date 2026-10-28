# COVID-19 Detection on Chest X-ray Images (Task 1.1: Multi-class Classification)

End-to-end deep learning pipeline classifying 9,961 chest radiographs into three diagnostic categories (Normal, Viral Pneumonia, COVID-19), built around a nested validation protocol for an unbiased performance estimate. The project is part of the Examination Project 2025 (Image Processing Task).

## Highlights

- **Validation:** 3×3 nested stratified K-fold. Outer folds are used for performance estimation and inner folds for batch-size selection, for a total of 30 full training runs. Decoupling the hyperparameter search from the evaluation removes the indirect leakage that inflates results under plain cross-validation.
- **Architecture:** CNN designed from scratch, with 5 convolutional blocks (32→256 filters), batch normalisation and progressive dropout, followed by a dense softmax classifier. AdamW optimiser, early stopping with best-weight restoration, adaptive learning-rate reduction.
- **Preprocessing:** CLAHE contrast enhancement (clip limit 2.0, 8×8 grid), chosen over global histogram equalisation because pulmonary opacities are low-contrast and spatially localised. Augmentation is restricted to anatomically plausible transforms (vertical flip deliberately excluded).
- **Class imbalance:** addressed through inverse-frequency class weights rather than resampling, preserving the native data distribution.

## Dataset

COVID-19 Radiography Database, 9,961 chest X-rays in PNG format.

| Class | Label | Images | Share |
|---|---|---|---|
| Normal | 0 | ~5,015 | ~50% |
| Viral Pneumonia | 1 | ~1,345 | ~14% |
| COVID-19 | 2 | ~3,601 | ~36% |
| **Total** | | **9,961** | 100% |

## Preprocessing

| Step | Operation | Details |
|---|---|---|
| 1 | Grayscale loading | `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` |
| 2 | CLAHE | `clipLimit=2.0`, `tileGridSize=(8, 8)` |
| 3 | Resize | 150×150 pixels |
| 4 | Reshape | add channel dimension → (150, 150, 1) |
| 5 | Normalisation | divide by 255 → range [0, 1] |

CLAHE is preferred over global equalisation because it preserves the local detail of lung lesions, which a uniform equalisation can wash out.

### Data augmentation

Applied on-the-fly and **only to the training set**:

| Transform | Range |
|---|---|
| Rotation | ±20° |
| Horizontal / vertical shift | ±15% |
| Shear | ±15% |
| Zoom | ±15% |
| Horizontal flip | enabled |
| Vertical flip | disabled (anatomically implausible) |

## Model

Sequential CNN, ~1.245M parameters.

| Block | Composition | Output shape |
|---|---|---|
| Input | Input layer | (150, 150, 1) |
| Block 1 | Conv2D(32) + BN + MaxPool | (75, 75, 32) |
| Block 2 | Conv2D(64) + Dropout(0.1) + BN + MaxPool | (38, 38, 64) |
| Block 3 | Conv2D(64) + BN + MaxPool | (19, 19, 64) |
| Block 4 | Conv2D(128) + Dropout(0.2) + BN + MaxPool | (10, 10, 128) |
| Block 5 | Conv2D(256) + Dropout(0.2) + BN + MaxPool | (5, 5, 256) |
| Classifier | Flatten + Dense(128) + Dropout(0.2) | (128) |
| Output | Dense(3, softmax) | (3) |

All convolutions use 3×3 filters with ReLU activation. Dropout rates increase towards the deeper layers.

### Class imbalance

Inverse-frequency class weights are used:

```
weight[i] = n_samples / (n_classes × n_samples_class_i)
```

Approximate values: Normal 0.66, COVID-19 0.92, Viral Pneumonia 2.47. A misclassified Viral Pneumonia sample therefore contributes about 3.7× more to the loss than a Normal one.

### Training configuration

| Parameter | Value |
|---|---|
| Optimiser | AdamW (Adam with weight decay) |
| Loss | Categorical cross-entropy |
| Epochs | 25 (max) |
| Batch size | tuned over {16, 32, 64} |
| Early stopping | patience = 7 on `val_loss`, restores best weights |
| ReduceLROnPlateau | factor = 0.3, patience = 4 |

## Validation Strategy

3×3 nested stratified K-fold cross-validation (`StratifiedKFold`, which preserves class proportions in every fold):

- **Outer loop (3 folds):** estimates the final performance. Each outer test fold is never used during hyperparameter selection.
- **Inner loop (3 folds):** selects the batch size among {16, 32, 64}, which gives 27 training runs.
- **Final training:** the best batch size of each outer fold is used to train the model on the full outer-training set, for 3 additional runs (30 in total).

**Hyperparameter tuned:** batch size.

## Results

Mean ± standard deviation across the 3 outer folds:

| Metric | Value |
|---|---|
| Accuracy | 90.92% ± 2.89% |
| Macro Precision | 90.12% ± 3.26% |
| Macro Recall | 91.27% ± 2.57% |
| Macro F1-score | 90.53% ± 3.06% |

A macro F1 of about 90% on three imbalanced classes is a solid result for a custom CNN without transfer learning, and the low standard deviation (~3%) indicates stable behaviour across folds. Switching the optimiser from RMSprop to AdamW improved accuracy by about 8% in our experiments, thanks to the weight decay regularisation.

## Limitations and Future Work

**Limitations:** relatively small dataset (~10K images), single data source, no clinical validation, simplified class set.

**Possible improvements:** transfer learning, model ensembles, attention mechanisms, Grad-CAM for explainability.

