# Score Summary

## Kaggle Submission Scores

| Submission File | Main Idea | Kaggle Score |
|---|---|---:|
| `submission_bilstm_auc.csv` | Word2Vec + BiLSTM | 0.95000 |
| `submission_blend_auc.csv` | BiLSTM + classical blend | 0.95385 |
| `submission_tfidf_svm_blend_auc.csv` | TF-IDF sparse blend | 0.96739 |
| `submission_mega_rank_blend_auc.csv` | rank-average merge of strong submissions | 0.97083 |
| `submission_cv_sparse_rank_blend_auc.csv` | sparse text 5-fold CV rank blend | **0.97166** |
| `submission_multiseed_cv_rank_blend_auc.csv` | multi-seed CV rank blend | 0.97157 |
| `submission_pseudolabel_rank_blend_auc.csv` | pseudo-label rank blend | 0.97137 |

## Best Submission
- Best file: `submission_cv_sparse_rank_blend_auc.csv`
- Best observed score: `0.97166`

## Interpretation
- Sparse models outperformed Word2Vec-based dense models on this task.
- Cross-validation blending improved stability.
- Multi-seed blending and pseudo-labeling did not surpass the best online score, even though they slightly improved some local validation results.
