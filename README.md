# MSCS-634 Lab 4: Regression Analysis with Regularization Techniques

**Student:** Bharath Kumar Origanti  
**Course:** MSCS-634 — Advanced Big Data and Data Mining

## Purpose
Compare Simple Linear, Multiple Linear, Polynomial, Ridge, and Lasso regression on the scikit-learn Diabetes dataset. Evaluate held-out performance and explore regularization and overfitting.

## Files
- `MSCS_634_Lab_4_Regression.ipynb` — fully executed notebook with code, analysis, metrics, and charts.
- `model_metrics.csv` — measured performance for all five models.
- `figures/` — saved plots (target distribution, predictions, errors, regularization, and polynomial degree).

## Run instructions
1. Install Python 3.10 or newer.
2. Install packages: `pip install -r requirements.txt`
3. Open the notebook using Jupyter: `jupyter notebook MSCS_634_Lab_4_Regression.ipynb`
4. Select **Run All**. The notebook regenerates `model_metrics.csv` and `figures/`.

## Method
- Load `load_diabetes(as_frame=True)` from scikit-learn; inspect features, target, and missing values.
- Use an 80% training / 20% test split (`random_state=42`).
- Use BMI for simple and degree-2 polynomial regression, and all ten features for multiple, Ridge, and Lasso regression.
- Use `StandardScaler` in training pipelines for multiple, Ridge, and Lasso models.
- Compare MAE, MSE, RMSE, and R² on the same test set.

## Measured test results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Lasso (alpha=1) | 42.80 | 2824.57 | 53.15 | 0.467 |
| Ridge (alpha=1) | 42.81 | 2892.01 | 53.78 | 0.454 |
| Multiple Linear | 42.79 | 2900.19 | 53.85 | 0.453 |
| Simple Linear (BMI) | 52.26 | 4061.83 | 63.73 | 0.233 |
| Polynomial Degree 2 (BMI) | 52.38 | 4085.03 | 63.91 | 0.229 |

## Key insights
- **Lasso (alpha=1)** achieved the lowest test RMSE (53.15) for this split and the selected model settings.
- BMI-only models use less information than models trained on all ten features; the comparison illustrates how the number of predictors affects predictive accuracy.
- Ridge uses L2 shrinkage, while Lasso uses L1 shrinkage and may set some coefficients to zero. Stronger penalties may reduce overfitting but can also cause underfitting.
- Increasing polynomial degree makes the BMI model more flexible, but added complexity does not guarantee better test performance.

## Challenges and design decisions
- Avoided train/test leakage by using preprocessing pipelines fitted only on training data.
- Used a fixed random seed so results can be reproduced.
- Used polynomial features on BMI only to avoid a very large set of interactions.
- Alpha sweeps are exploratory; proper hyperparameter selection would use training-set cross-validation, not the held-out test set.

## References
- Scikit-learn developers. (n.d.). *Diabetes dataset*. https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
- Scikit-learn developers. (n.d.). *Linear models*. https://scikit-learn.org/stable/modules/linear_model.html

## Submission
Upload all files in this project to a **public** GitHub repository named `MSCS_634_Lab_4`, then paste its repository URL into Blackboard Lab 4.
