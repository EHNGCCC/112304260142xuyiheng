# Modification Summary

This file records the main changes made during the project.

## Version 1
- built a formal project structure under `formal/`
- implemented data loading, preprocessing, `Word2Vec`, classical models, and `BiLSTM`
- switched evaluation logic to `ROC-AUC`
- exported probability-based Kaggle submission files

## Version 2
- added caching for preprocessing results
- added caching for `Word2Vec` feature matrices
- reduced repeated feature generation time for later experiments

## Version 3
- added stronger sparse text baselines
- introduced `TF-IDF + LinearSVC`
- introduced `NB-SVM`
- compared dense embedding methods with sparse text methods

## Version 4
- added rank-average blending across sparse models
- added cross-validation sparse ensemble
- generated multiple Kaggle submission files for comparison
- achieved the current best score `0.97166`

## Version 5
- tested multi-seed cross-validation blending
- tested pseudo-labeling with high-confidence test samples
- preserved these experiments because they help document the full optimization process
- confirmed that they did not surpass the current best online score
