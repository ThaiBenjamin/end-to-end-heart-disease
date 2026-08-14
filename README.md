# ❤️ End-to-End Heart Disease Classification

An end-to-end machine learning project — using clinical patient data from the Cleveland Heart Disease dataset to predict whether or not a patient has heart disease, taking the problem from raw CSV through EDA, model comparison, hyperparameter tuning, and evaluation.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📚 What I Was Learning

The full machine learning workflow from start to finish — not just fitting a model, but framing the problem, exploring the data, establishing a baseline across multiple models, tuning hyperparameters, and evaluating with the metrics that actually matter for a medical classification problem.

The project follows a 6-step framework:

1. **Problem Definition** — *Given clinical parameters about a patient, can we predict whether or not they have heart disease?*
2. **Data** — Cleveland dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) (303 patients, 14 attributes)
3. **Evaluation** — Target of 95% accuracy to pursue the project further
4. **Features** — Building a data dictionary of the 13 clinical predictors (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, ST depression, etc.)
5. **Modelling** — Comparing three classifiers, then tuning the best one
6. **Experimentation** — Where to go next if the metric isn't hit

---

## 🔑 Key Things Practiced

- **Exploratory Data Analysis** — `value_counts()`, `crosstab()`, checking for missing values, and plotting heart disease frequency against sex, age vs. max heart rate, and chest pain type
- **Correlation Analysis** — Building a correlation matrix and visualizing it as a Seaborn heatmap to see which features move together
- **Baseline Model Comparison** — Fitting Logistic Regression, K-Nearest Neighbors, and Random Forest through a reusable `fit_and_score()` function
- **Hyperparameter Tuning by Hand** — Looping `n_neighbors` from 1–20 for KNN and plotting train vs. test scores to spot overfitting
- **RandomizedSearchCV** — Randomly sampling hyperparameter grids for Logistic Regression and Random Forest
- **GridSearchCV** — Exhaustively searching the `C` and `solver` space for the best-performing model
- **Evaluation Beyond Accuracy** — ROC curve and AUC, confusion matrix, classification report, precision, recall, and F1
- **Cross-Validation** — Using `cross_val_score()` with 5 folds to get metrics that don't depend on one lucky train/test split
- **Feature Importance** — Reading the Logistic Regression `coef_` values to see which clinical attributes push a prediction toward disease

---

## 📊 Results

**Baseline model comparison (test set accuracy):**

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 88.52% |
| Random Forest | 83.61% |
| K-Nearest Neighbors | 68.85% |

**After tuning** — Logistic Regression stayed on top at **88.52%** (`C=0.204`, `solver="liblinear"`), with tuned Random Forest at 86.89% and hand-tuned KNN peaking at 75.41%.

**Cross-validated metrics (5-fold) for the tuned Logistic Regression:**

| Metric | Score |
|--------|-------|
| Accuracy | 84.47% |
| Precision | 82.08% |
| Recall | 92.12% |
| F1 | 86.73% |

The 95% accuracy goal wasn't reached — which is itself part of the exercise. The final section works through what to try next: collecting more data, reaching for a stronger model like CatBoost or XGBoost, or engineering better features.

---

## 🗂️ What's Here

```
end-to-end-heart-diesease-classification.ipynb   # The full analysis
heart-disease.csv                                # Cleveland dataset (303 rows, 14 columns)
```

---

## 🚀 Running It

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook end-to-end-heart-diesease-classification.ipynb
```

---

## 💡 What It Taught Me

The biggest lesson was that the model is the small part. Most of the notebook is spent understanding the data before a single classifier is fit — and that's where the useful intuitions came from. Plotting age against max heart rate made the separation between the classes visible in a way no accuracy score does, and the correlation heatmap flagged which features were doing real work.

Comparing three models side by side made it obvious that more complexity isn't automatically better: KNN, which sounds intuitive, was the worst performer by a wide margin, while plain Logistic Regression won and stayed winning even after both RandomizedSearchCV and GridSearchCV. Tuning gave back almost nothing on top of the baseline, which was a useful reality check on how much hyperparameter search actually buys you.

The evaluation section changed how I read results. Recall came out at 92% against 82% precision — for a heart disease screen, that's the right trade: missing a sick patient is far worse than flagging a healthy one for follow-up. And the cross-validated accuracy (84%) landed several points below the single-split accuracy (89%), which was a concrete demonstration that one train/test split can flatter a model. Finally, not hitting the 95% target was the most honest part of the project: knowing when a proof of concept hasn't earned the next step matters more than forcing a number.
