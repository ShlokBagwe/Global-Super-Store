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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- STEP 1: LOG TRANSFORM SKEWED COLUMNS ---
df['log_sales'] = np.log1p(df['Sales'])
df['log_shipping_cost'] = np.log1p(df['Shipping Cost'])

# --- STEP 2: DEFINE FEATURES AND TARGET ---
X = df.drop(columns=['Profit', 'Sales', 'Shipping Cost'])
y = df['Profit']

# --- STEP 3: TRAIN TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: DEFINE COLUMN GROUPS ---
num_cols = ['Quantity', 'Discount', 'log_sales', 'log_shipping_cost', 'Year', 'weeknum']

cat_cols = ['Category', 'Segment', 'Ship Mode', 'Market', 'Region', 'Sub-Category']

ord_col = ['Order Priority']

# --- STEP 5: PREPROCESSOR ---
preprocessor = ColumnTransformer(transformers=[
    ('num', RobustScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
    ('ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Critical']]), ord_col)
])

# --- STEP 6: TRAIN ALL 3 MODELS ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[name] = {
        'MAE': round(mean_absolute_error(y_test, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'R2': round(r2_score(y_test, y_pred), 4)
    }

# --- STEP 7: COMPARE RESULTS ---
results_df = pd.DataFrame(results).T
print(results_df)
print(f"\nBest Model by R2: {results_df['R2'].idxmax()}")

Keep all code clean, modular, and notebook-ready. No markdown theory. Only code and print outputs.
