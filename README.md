# End-to-End Heart Disease Classification

Predicting whether a patient has heart disease from clinical data, taking the problem from
a raw CSV through EDA, model comparison, hyperparameter tuning, and evaluation.

The data is the Cleveland set from the UCI Machine Learning Repository — 303 patients, 13
clinical predictors (age, sex, chest pain type, resting blood pressure, cholesterol, max
heart rate, ST depression, and so on) plus a binary target.

## Results

Baseline comparison, test set accuracy:

| Model | Accuracy |
|---|---|
| Logistic Regression | 88.52% |
| Random Forest | 83.61% |
| K-Nearest Neighbors | 68.85% |

After tuning, Logistic Regression stayed on top at 88.52% (`C=0.204`,
`solver="liblinear"`), with a tuned Random Forest at 86.89% and a hand-tuned KNN peaking
at 75.41%.

Five-fold cross-validated metrics for the tuned Logistic Regression:

| Metric | Score |
|---|---|
| Accuracy | 84.47% |
| Precision | 82.08% |
| Recall | 92.12% |
| F1 | 86.73% |

I set a 95% accuracy target up front as the bar for taking the project further, and didn't
hit it. That's part of the exercise. The last section of the notebook works through what
I'd try next: collecting more data, reaching for something like CatBoost or XGBoost, or
engineering better features.

## What the notebook covers

The project follows a six-step framework — define the problem, look at the data, decide how
to evaluate, understand the features, model, then experiment.

The EDA is most of it: `value_counts()` and `crosstab()`, checking for missing values, and
plotting heart disease frequency against sex, age against max heart rate, and chest pain
type. A correlation matrix rendered as a Seaborn heatmap shows which features move
together.

Modelling starts with a reusable `fit_and_score()` function running three classifiers side
by side, then tunes: `n_neighbors` from 1 to 20 for KNN by hand, plotting train against
test scores to spot the overfitting; `RandomizedSearchCV` over hyperparameter grids for
Logistic Regression and Random Forest; then `GridSearchCV` exhaustively over the `C` and
solver space for the winner.

Evaluation goes past accuracy: ROC curve and AUC, a confusion matrix, a classification
report, and `cross_val_score()` with five folds so the numbers don't depend on one lucky
split. Reading the Logistic Regression `coef_` values shows which clinical attributes push
a prediction toward disease.

## Files

```
end-to-end-heart-disease-classification.ipynb   the full analysis
heart-disease.csv                               Cleveland dataset, 303 rows, 14 columns
```

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook end-to-end-heart-disease-classification.ipynb
```

## What I took from it

The model is the small part. Most of the notebook is spent understanding the data before a
single classifier is fit, and that's where the useful intuitions came from. Plotting age
against max heart rate made the separation between classes visible in a way no accuracy
score does, and the correlation heatmap flagged which features were doing real work.

Comparing three models side by side made it obvious that more complexity isn't
automatically better. KNN, which sounds intuitive, was the worst performer by a wide
margin, while plain Logistic Regression won and stayed winning through both
`RandomizedSearchCV` and `GridSearchCV`. Tuning gave back almost nothing over the baseline,
which was a useful reality check on how much hyperparameter search actually buys you.

The evaluation section changed how I read results. Recall came out at 92% against 82%
precision, and for a heart disease screen that's the right trade — missing a sick patient
is far worse than flagging a healthy one for follow-up. The cross-validated accuracy of 84%
landed several points below the single-split 89%, which is a concrete demonstration that
one train/test split can flatter a model.

Not hitting the 95% target was the most honest part. Knowing when a proof of concept hasn't
earned the next step matters more than forcing a number.

## Related

- [End-to-End Bulldozer Price Regression](https://github.com/ThaiBenjamin/end-to-end-bulldozer-price) — regression on a 400,000-row time series
- [End-to-End Dog Vision](https://github.com/ThaiBenjamin/end-to-end-dog-vision) — 120-class image classification with transfer learning
