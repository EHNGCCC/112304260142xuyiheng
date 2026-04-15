# Experiment Report

## 1. Task Description
This project targets the Kaggle competition `Bag of Words Meets Bags of Popcorn`.

- Dataset: IMDB movie reviews
- Task: binary sentiment classification
- Official metric: `ROC-AUC`

## 2. Data Processing
The text preprocessing pipeline includes:
- removing HTML tags and `<br />`
- converting text to lowercase
- expanding common negation forms
- tokenization
- removing English stopwords while keeping important negation words

Two text views are used in later experiments:
- `clean_review`: normalized text after preprocessing
- `soft_review`: softer cleaned text that keeps more original token boundaries for sparse models

## 3. Model Evolution

### Stage 1: Word2Vec baseline
Early experiments followed the competition theme more closely:
- train `Word2Vec` on labeled and unlabeled reviews
- build sentence vectors from word embeddings
- train `Logistic Regression`, `Random Forest`, and `BiLSTM`

Representative results:
- `submission_bilstm_auc.csv`: Kaggle score `0.95000`
- `submission_blend_auc.csv`: Kaggle score `0.95385`

### Stage 2: Sparse text models
Later experiments showed that sparse text models performed better on this task:
- `word TF-IDF + LinearSVC`
- `char TF-IDF + LinearSVC`
- `NB-SVM` on word n-grams
- `NB-SVM` on character n-grams

Representative result:
- `submission_tfidf_svm_blend_auc.csv`: Kaggle score `0.96739`

### Stage 3: Cross-validation rank blending
The best online result came from a cross-validation sparse ensemble:
- word-level sparse features
- character-level sparse features
- `LinearSVC`
- `NB-SVM`
- rank-average blending

Best online result:
- `submission_cv_sparse_rank_blend_auc.csv`
- Kaggle score: `0.97166`

### Stage 4: Further optimization
Additional experiments were tested:
- mega rank blend
- multi-seed cross-validation blend
- pseudo-labeling

These methods slightly improved local validation in some cases, but did not surpass the best Kaggle score already achieved by the `cv sparse rank blend`.

## 4. Best Final Method
The best-performing submission in this repository is:

- File: `results/submissions/submission_cv_sparse_rank_blend_auc.csv`
- Score: `0.97166`

Core idea:
- use multiple sparse text representations
- combine `LinearSVC` and `NB-SVM`
- use cross-validation predictions instead of a single validation split
- blend model outputs by rank to improve stability

## 5. Conclusion
For this IMDB sentiment task, traditional sparse text models turned out to be more effective than the original Word2Vec-centered pipeline.

The final result shows that:
- careful text preprocessing matters
- sparse n-gram features are very strong for sentiment classification
- cross-validation and model blending are more important than small parameter changes
- online validation must be trusted more than tiny improvements on local validation
