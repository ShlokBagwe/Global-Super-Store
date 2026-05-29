You are a Senior Data Scientist. I have completed feature engineering up to Section 8 (Aggregated Group-Based Features) on the Global Superstore Sales dataset. The dataset has 51,290 rows. Target variable is Profit.

Continue from Section 9 onwards. No theory. No explanations. Just write clean, production-ready Python code for each step.

---

### SECTION 9: TRANSFORMATION AND SCALING

- Apply np.log1p to: Sales, Shipping Cost
- Apply Yeo-Johnson PowerTransformer to: Profit (target) — only for linear model pipeline
- Apply RobustScaler to all numerical features
- Apply OneHotEncoder (drop='first') to: Category, Segment, Ship Mode, Market, Region, Sub-Category
- Apply OrdinalEncoder with [Low=0, Medium=1, High=2, Critical=3] to: Order Priority
- Build full sklearn ColumnTransformer + Pipeline
- Fit pipeline on X_train only, transform both X_train and X_test

---

### SECTION 10: OUTLIER HANDLING

- Winsorize Sales and Shipping Cost at 1st and 99th percentile on train set only
- Apply same caps to test set
- Keep all Profit outliers as-is
- Print before/after shape confirmation

---

### SECTION 11: FINAL CHECKS

Run this and print results:

```python
def model_readiness_check(df):
    checks = {}
    checks['no_nulls'] = df.isnull().sum().sum() == 0
    checks['no_infinite'] = np.isinf(df.select_dtypes(include=np.number)).sum().sum() == 0
    checks['no_object_columns'] = len(df.select_dtypes(include='object').columns) == 0
    checks['no_zero_variance'] = (df.var() == 0).sum() == 0
    checks['feature_count'] = df.shape[1]
    for k, v in checks.items():
        print(f"{'✅' if v else '❌'} {k}: {v}")
```

---

### SECTION 12: MODEL TRAINING

Train all three models on X_train, y_train:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
```

- LinearRegression()
- RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
- XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)

Evaluate all three on X_test, y_test using MAE, RMSE, R2.

---

### SECTION 13: POST-TRAINING INSIGHTS

After all three models are trained and evaluated, print the following:

1. Comparison table of MAE, RMSE, R2 for all three models
2. Best performing model name and its scores
3. Top 15 most important features from Random Forest (bar chart)
4. Top 15 most important features from XGBoost (bar chart)
5. For Linear Regression: print top 15 coefficients by absolute value
6. Actual vs Predicted scatter plot for the best model
7. Residual distribution plot for the best model
8. Print: which features contributed most to profit prediction and which features had near-zero importance (candidates for removal)

---

Keep all code clean, modular, and notebook-ready. No markdown theory. Only code and print outputs.
