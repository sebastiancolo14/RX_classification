End-to-end deep learning pipeline classifying 9,961 chest radiographs into three diagnostic categories (Normal, Viral Pneumonia, COVID-19), built around a nested validation protocol for an unbiased performance estimate.

• Validation: 3×3 nested stratified K-fold — outer folds for performance estimation, inner folds for batch-size selection, totalling 30 full training runs. Decoupling hyperparameter search from evaluation removes the indirect leakage that inflates results under plain cross-validation.
• Architecture: CNN designed from scratch — 5 convolutional blocks (32→256 filters) with batch normalisation and progressive dropout, followed by a dense softmax classifier. AdamW optimiser, early stopping with best-weight restoration, adaptive learning-rate reduction.
• Preprocessing: CLAHE contrast enhancement (clip limit 2.0, 8×8 grid), chosen over global histogram equalisation because pulmonary opacities are low-contrast and spatially localised; augmentation restricted to anatomically plausible transforms (vertical flip deliberately excluded).
• Class imbalance: addressed through inverse-frequency class weights rather than resampling, preserving the native data distribution.
• Results (mean ± SD across outer folds): Accuracy 90.92% ± 2.89% · Macro Precision 90.12% ± 3.26% · Macro Recall 91.27% ± 2.57% · Macro F1 90.53% ± 3.06%.
