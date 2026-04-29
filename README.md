# Machine Learning Kaggle Projects

This repository contains my machine learning coursework projects prepared for GitHub tracking and deployment.

## Projects

- `code/`, `report/`, `results/`: IMDB sentiment classification for Kaggle `Bag of Words Meets Bags of Popcorn`.
- `digit-recognizer-cnn/`: CNN handwritten digit recognition for Kaggle `Digit Recognizer`, including a Flask web app and Render deployment config.

---

# IMDB Sentiment Classification for Kaggle

## Project Overview
This repository contains the formal part of my Kaggle project for the competition `Bag of Words Meets Bags of Popcorn`.

The task is binary sentiment classification on IMDB movie reviews, and the official evaluation metric is `ROC-AUC`.

This repository only keeps the code, reports, and experiment results needed for GitHub version control. Large raw datasets, model caches, and temporary artifacts are intentionally excluded.

## Best Result
- Best Kaggle submission in this repository: `submission_cv_sparse_rank_blend_auc.csv`
- Best online score observed so far: `0.97166`
- Best model family: sparse text ensemble with `TF-IDF`, `LinearSVC`, `NB-SVM`, and rank blending

## Repository Structure
```text
formal_github_repo/
|-- code/
|   |-- main.py
|   |-- generate_highscore_submission.py
|   |-- generate_multiseed_cv_submission.py
|   |-- generate_pseudolabel_submission.py
|   |-- requirements.txt
|   `-- src/
|-- report/
|   |-- experiment_report.md
|   |-- modification_summary.md
|   `-- github_setup_guide.md
|-- results/
|   |-- metrics/
|   |-- plots/
|   |-- submissions/
|   `-- score_summary.md
|-- .gitignore
`-- README.md
```

## Main Methods
The project includes several experiment stages:
- `Word2Vec + Logistic Regression / Random Forest / BiLSTM`
- weighted fusion between classical models and BiLSTM
- sparse text models based on `word TF-IDF`, `char TF-IDF`, `NB-SVM`, and `LinearSVC`
- cross-validation rank blending
- multi-seed cross-validation blending
- pseudo-label experiments

The best online result came from the `cv sparse rank blend` pipeline.

## How To Run
Install dependencies first:

```bash
pip install -r code/requirements.txt
```

Run the main baseline pipeline:

```bash
python code/main.py
```

Run the sparse high-score pipeline:

```bash
python code/generate_highscore_submission.py
```

Run the multi-seed CV ensemble:

```bash
python code/generate_multiseed_cv_submission.py
```

Run the pseudo-label experiment:

```bash
python code/generate_pseudolabel_submission.py
```

## Notes
- This repository is prepared for GitHub upload and experiment tracking.
- Large data files and local caches are excluded on purpose.
- Kaggle submission CSV files, metric reports, and key plots are preserved to show the full experiment process.
