You are a Senior Data Scientist. I completed EDA on Global Superstore Sales dataset (51,290 rows).

Current columns:
Category, Country, Discount, Market, Order Priority, Profit, Quantity, Region, Sales, Segment, Ship Mode, Shipping Cost, Sub-Category, Year, weeknum, shipping_days

Target: Profit
Previous R2: Linear Regression=0.53, Random Forest=0.68
Goal: R2 >= 0.80

Rules:
- Log transformation only on TARGET variable (Profit) — it gives 0.4 skewness after transformation
- Do NOT create ratio or product features from existing columns — causes multicollinearity
- Only create features that add NEW information
- Plot heatmap after all feature engineering and encoding — BEFORE model training

---

### STEP 1: DROP USELESS COLUMNS

Drop: shipping_days (zero variance), Country (147 unique values, already covered by Market and Region)

---

### STEP 2: TRANSFORM TARGET VARIABLE

Profit has negative values so shift first then apply log:

```python
shift_value = abs(df['Profit'].min()) + 1
df['Profit_log'] = np.log1p(df['Profit'] + shift_value)
# Store shift_value to inverse transform predictions later
```

Use Profit_log as target y. After predictions inverse transform:
```python
y_pred_actual = np.expm1(y_pred) - shift_value
```

---

### STEP 3: FEATURE ENGINEERING (No Multicollinearity)

```python
# Binary flags from Discount — new information, not a copy
df['high_discount_flag'] = (df['Discount'] > 0.2).astype(int)
df['aggressive_discount_flag'] = (df['Discount'] >= 0.5).astype(int)

# Time features from weeknum — new information
df['quarter'] = pd.cut(df['weeknum'], bins=[0,13,26,39,53], labels=[1,2,3,4]).astype(int)
df['is_holiday_season'] = (df['weeknum'] >= 40).astype(int)
```

---

### STEP 4: ENCODING

```python
# Ordinal — Order Priority
priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
df['Order Priority'] = df['Order Priority'].map(priority_map)

# One Hot — nominal categoricals
df = pd.get_dummies(df, columns=['Category', 'Segment', 'Ship Mode', 'Market', 'Region', 'Sub-Category'], drop_first=True)
```

---

### STEP 5: TRAIN TEST SPLIT

```python
X = df.drop(columns=['Profit', 'Profit_log'])
y = df['Profit_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### STEP 6: AGGREGATION FEATURES (TRAIN ONLY)

Compute on X_train only, map to both train and test:

```python
# These bring in external group-level signal — not derived from same row columns
agg_features = {
    'avg_profit_by_subcategory': ('Sub-Category columns not available after OHE — use before OHE'),
}
```

NOTE: Compute aggregation features BEFORE one hot encoding in Step 4. Add these before encoding:

```python
# Run these BEFORE pd.get_dummies step
# Do train test split first, then compute on X_train

agg1 = X_train.groupby('Sub-Category')['Discount'].mean()
agg2 = X_train.groupby('Region')['Sales'].mean()
agg3 = X_train.groupby('Category')['Quantity'].mean()

X_train['avg_discount_by_subcat'] = X_train['Sub-Category'].map(agg1)
X_test['avg_discount_by_subcat'] = X_test['Sub-Category'].map(agg1).fillna(agg1.mean())

X_train['avg_sales_by_region'] = X_train['Region'].map(agg2)
X_test['avg_sales_by_region'] = X_test['Region'].map(agg2).fillna(agg2.mean())

X_train['avg_qty_by_category'] = X_train['Category'].map(agg3)
X_test['avg_qty_by_category'] = X_test['Category'].map(agg3).fillna(agg3.mean())
```

Then apply OHE on X_train and X_test separately.

---

### STEP 7: SCALING

Apply RobustScaler only on numerical columns (not binary flags, not OHE columns):

```python
from sklearn.preprocessing import RobustScaler

num_cols = ['Discount', 'Sales', 'Quantity', 'Shipping Cost', 'Year', 'weeknum',
            'avg_discount_by_subcat', 'avg_sales_by_region', 'avg_qty_by_category']

scaler = RobustScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

---

### STEP 8: HEATMAP (BEFORE MODEL TRAINING)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Only numerical columns for heatmap
num_df = X_train.select_dtypes(include=[np.number])
corr = num_df.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, annot_kws={'size': 7})
plt.title('Feature Correlation Heatmap After Engineering', fontsize=14)
plt.tight_layout()
plt.show()

# Print any feature pairs with correlation > 0.75
high_corr = [(i, j, corr.loc[i,j]) 
             for i in corr.columns 
             for j in corr.columns 
             if i < j and abs(corr.loc[i,j]) > 0.75]
print("High Correlation Pairs (>0.75):")
for pair in high_corr:
    print(f"  {pair[0]} — {pair[1]}: {pair[2]:.2f}")
```

Drop any high correlation pairs identified before proceeding to model training.

---

### STEP 9: MODEL TRAINING

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    
    # Inverse transform
    y_pred_actual = np.expm1(y_pred_log) - shift_value
    y_test_actual = np.expm1(y_test) - shift_value
    
    results[name] = {
        'MAE': round(mean_absolute_error(y_test_actual, y_pred_actual), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)), 2),
        'R2': round(r2_score(y_test_actual, y_pred_actual), 4)
    }

results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print(results_df)
print(f"\nBest Model: {results_df['R2'].idxmax()}")
```

---

### STEP 10: BEST MODEL INSIGHTS

```python
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10,8))
plt.title(f'Top 20 Feature Importances — {best_model_name}')
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.3)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title(f'Actual vs Predicted — {best_model_name}')
plt.tight_layout()
plt.show()
```
