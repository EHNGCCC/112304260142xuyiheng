# Code Directory

## Main Files
- `main.py`: original formal baseline pipeline with Word2Vec, classical models, and optional BiLSTM
- `generate_highscore_submission.py`: sparse high-score pipeline
- `generate_multiseed_cv_submission.py`: multi-seed CV sparse ensemble
- `generate_pseudolabel_submission.py`: pseudo-label experiment
- `src/`: reusable modules

## Important Note
This GitHub version keeps the source code only.

It does **not** include:
- raw Kaggle dataset files
- local training caches
- large intermediate model artifacts

To run the code, prepare the competition data locally and install dependencies from `requirements.txt`.
