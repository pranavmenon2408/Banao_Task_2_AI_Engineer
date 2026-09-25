# Model comparison (walk-forward, 9 monthly folds Oct 2025 - Jun 2026)

Same features and folds for every model. Stochastic models averaged over 3 seeds. Sorted by post-May PR-AUC, the regime the test set is in.

| Model | PR-AUC all | PR-AUC pre-May | PR-AUC post-May | Recall@40 | R-precision | Desk net Rs/month | Train+score time |
|---|---|---|---|---|---|---|---|
| LightGBM, monotone (chosen) | 0.460 | 0.559 | 0.267 | 0.584 | 0.542 | 25,720 | 4s |
| Logistic regression (C=1) | 0.417 | 0.535 | 0.265 | 0.591 | 0.491 | 22,601 | 0s |
| Logistic regression (C=0.1) | 0.408 | 0.545 | 0.224 | 0.572 | 0.505 | 23,070 | 0s |
| SVM linear (C=0.1) | 0.306 | 0.371 | 0.213 | 0.540 | 0.375 | 18,946 | 1s |
| Random forest (500 trees, leaf 3) | 0.422 | 0.546 | 0.197 | 0.575 | 0.531 | 23,656 | 23s |
| Extra trees (500 trees, leaf 3) | 0.414 | 0.541 | 0.180 | 0.573 | 0.520 | 24,216 | 19s |
| SVM RBF (C=10) | 0.304 | 0.403 | 0.170 | 0.541 | 0.388 | 22,914 | 4s |
| Random forest (500 trees, leaf 10) | 0.392 | 0.548 | 0.163 | 0.563 | 0.523 | 22,842 | 23s |
| Logistic regression, balanced (C=0.1) | 0.316 | 0.417 | 0.157 | 0.544 | 0.415 | 21,954 | 1s |
| SVM RBF (C=1) | 0.277 | 0.388 | 0.131 | 0.536 | 0.397 | 21,588 | 8s |
| SVM polynomial deg 3 (C=1) | 0.242 | 0.373 | 0.078 | 0.481 | 0.415 | 17,811 | 3s |
| SVM polynomial deg 2 (C=1) | 0.267 | 0.417 | 0.064 | 0.522 | 0.390 | 20,523 | 5s |
| SVM sigmoid (C=1) | 0.037 | 0.037 | 0.055 | 0.122 | 0.005 | 1,782 | 15s |
