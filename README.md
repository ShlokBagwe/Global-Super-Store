# Global-Super-Store

You are a Senior Data Scientist with 10+ years of experience in retail analytics and ML engineering. I have completed full EDA on the Global Superstore Sales dataset. Now perform ADVANCED FEATURE ENGINEERING to prepare this dataset for ML model training (Linear Regression, Random Forest, XGBoost).

---

## DATASET CONTEXT (Already Completed)

**Current columns after cleaning (51,290 rows × 16 features):**
- Numerical: Discount, Profit, Quantity, Sales, Shipping Cost, Year, weeknum, shipping_days
- Categorical: Category, Country, Market, Order Priority, Region, Segment, Ship Mode, Sub-Category

**Critical EDA findings you must respect:**
- Target variable: Profit (skewness=4.16, kurtosis=291.41 — highly non-normal)
- Sales skewness=8.14, Shipping Cost skewness=5.86 — log transformation required
- Discount range: 0.0–0.85; profitability cliff confirmed beyond 20% discount
- 24.46% of transactions are loss-making (Profit < 0)
- Sales–Shipping Cost correlation = 0.77 (multicollinearity risk)
- Discount–Profit correlation = -0.32 (primary profit killer)
- Country has 147 unique values (high cardinality — handle carefully)
- shipping_days has zero variance (std=0.0) — drop immediately
- Top loss sub-categories: Tables, Machines; top profit: Copiers, Phones
- Canada/North Asia = high-margin low-discount markets; Southeast Asia/EMEA = opposite

---

## YOUR TASK: COMPLETE FEATURE ENGINEERING PIPELINE

Work through each section below **step by step**, like a senior data scientist writing a production notebook. For EVERY feature you create or reject, provide:
1. The Python code
2. WHY this feature is created (business reasoning) — 2-4 lines
3. WHY this feature improves ML performance (statistical/ML reasoning) — 2-4 lines
4. WHY alternatives were NOT chosen — 2-4 lines
5. Risk warnings (leakage, overfitting, multicollinearity) — 2-4 lines

---

### SECTION 1: PRE-ENGINEERING AUDIT

Before creating any features:

1.1 — Print column dtypes, cardinalities, and null counts.

1.2 — Identify and DROP zero-variance features. Explain in 2-4 lines why zero-variance features hurt models.

1.3 — Identify potential data leakage columns. Explain in 2-4 lines what data leakage is, why it's dangerous, and which columns could cause it.

1.4 — Separate EDA-only features. Explain in 2-4 lines why binned versions of continuous variables should NOT replace the original in ML.

1.5 — Define the train-test split timing rule. Explain in 2-4 lines why ALL group-based aggregation features must be computed on TRAINING DATA ONLY. Show the correct code pattern.

---

### SECTION 2: CATEGORICAL ENCODING

For each categorical column, choose the correct encoding strategy:

2.1 — **Category** (3 values: Furniture, Office Supplies, Technology)
- Apply One-Hot Encoding.
- Explain in 2-4 lines the dummy variable trap and why drop_first=True matters for linear models.
- Explain in 2-4 lines why tree-based models are less sensitive but consistency is still good practice.

2.2 — **Segment** (3 values: Consumer, Corporate, Home Office)
- Apply One-Hot Encoding.
- Explain in 2-4 lines why Label Encoding would be WRONG here.

2.3 — **Ship Mode** (4 values: Standard Class, Second Class, First Class, Same Day)
- Apply One-Hot Encoding.
- Explain in 2-4 lines why ordinal encoding could be debated but OHE is still preferred.

2.4 — **Order Priority** (4 values: Low, Medium, High, Critical)
- Apply ORDINAL Encoding with mapping: Low=0, Medium=1, High=2, Critical=3.
- Explain in 2-4 lines why this IS appropriate and show code using a manual map.

2.5 — **Market** (7 values: APAC, EU, US, LATAM, Africa, EMEA, Canada)
- Apply One-Hot Encoding.
- Explain in 2-4 lines why frequency encoding is a valid alternative for tree-based models.

2.6 — **Region** (13 values)
- Apply One-Hot Encoding.
- Explain in 2-4 lines why target encoding requires careful cross-validation to avoid leakage.

2.7 — **Sub-Category** (17 values)
- Apply One-Hot Encoding.
- Explain in 2-4 lines the dimensionality tradeoff and when target encoding is preferred.

2.8 — **Country** (147 values — HIGH CARDINALITY)
- DO NOT one-hot encode directly. Explain in 2-4 lines why 147 binary columns cause problems.
- Create a country_profit_tier feature using mean profit grouped by country from TRAINING DATA ONLY.
- Explain in 2-4 lines when to drop Country entirely if Market and Region already capture geographic signal.
- Show BOTH approaches.

Show the complete encoding pipeline using sklearn's ColumnTransformer. Explain in 2-4 lines why it is preferred over manual pandas encoding for production.

---

### SECTION 3: DATE AND TIME FEATURES

3.1 — **Year** (2011–2014): Keep as-is. Explain in 2-4 lines why treating it as continuous vs categorical matters differently for linear vs tree models.

3.2 — **weeknum** (1–53): Keep as-is. Explain in 2-4 lines how week number captures seasonality compactly.

3.3 — Create **quarter** from weeknum:
```python
df['quarter'] = pd.cut(df['weeknum'], bins=[0,13,26,39,53], labels=[1,2,3,4]).astype(int)
```
Explain in 2-4 lines why quarter captures macro-seasonality that weeknum alone may not expose to linear models.

3.4 — Create **is_holiday_season** flag (Q4 = weeks 40–53):
```python
df['is_holiday_season'] = (df['weeknum'] >= 40).astype(int)
```
Explain in 2-4 lines why this is NOT leakage and what business behavior it captures.

3.5 — Create **is_year_end** flag (weeks 48–53). Explain in 2-4 lines the business logic around end-of-year budget flush behavior.

3.6 — Explain in 2-4 lines each why you do NOT create:
- Day of week: already lost when Order Date was dropped.
- Month: redundant with quarter and weeknum — parsimony principle.
- Days since first order: Order Date was dropped; reconstructing from proxies introduces noise.

---

### SECTION 4: PROFITABILITY FEATURES

4.1 — **profit_margin** = Profit / Sales:
```python
df['profit_margin'] = np.where(df['Sales'] > 0, df['Profit'] / df['Sales'], 0)
```
Explain in 2-4 lines why profit margin normalizes profit by order size and exposes efficiency.

⚠️ LEAKAGE WARNING — Explain in 2-4 lines why profit_margin is LEAKY when predicting Profit and when it is safe to use.

4.2 — **is_loss** = (Profit < 0).astype(int). Explain in 2-4 lines when this is a valid auxiliary feature vs when it is leaky.

4.3 — **high_discount_flag** = (Discount > 0.2).astype(int). Explain in 2-4 lines why the 20% threshold is supported by EDA and why this helps linear models capture threshold effects.

4.4 — **aggressive_discount_flag** = (Discount >= 0.5).astype(int). Explain in 2-4 lines why the top-20 loss orders (avg 65.25% discount) justify this flag.

4.5 — Explain in 2-4 lines each why you do NOT create:
- loss_amount = abs(Profit) when Profit < 0: transformation of the target.
- profit_quintile: binning the target destroys information for regression.

---

### SECTION 5: SALES AND REVENUE FEATURES

5.1 — **sales_per_unit** = Sales / Quantity. Explain in 2-4 lines why average unit price captures value density differently than raw Sales.

5.2 — **log_sales** = np.log1p(Sales). Explain in 2-4 lines why Sales skewness=8.14 requires this and why log1p is safer than log.

5.3 — **high_value_order** = (Sales > Sales.quantile(0.90)).astype(int). Explain in 2-4 lines why the top-10% revenue cluster needs a flag given the EDA finding that 422 top-5% orders were loss-making.

5.4 — Explain in 2-4 lines each why you do NOT create:
- revenue_rank: rank-based features are dataset-order dependent and can leak.
- sales_zscore computed on full dataset before split: leaks test distribution into training.

---

### SECTION 6: SHIPPING FEATURES

6.1 — **log_shipping_cost** = np.log1p(Shipping Cost). Explain in 2-4 lines why skewness=5.86 and the $933 outlier require this transformation.

6.2 — **shipping_cost_ratio** = Shipping Cost / Sales. Explain in 2-4 lines why this ratio signals unsustainable logistics costs, especially for Furniture.

6.3 — **is_expensive_shipment** = (Shipping Cost > Shipping Cost.quantile(0.90)).astype(int). Explain in 2-4 lines why top-10% shipping costs disproportionately affect margins.

6.4 — Explain in 2-4 lines why shipping_days is DROPPED and show the verification code.

6.5 — Multicollinearity Warning: Explain in 2-4 lines why Sales–Shipping Cost r=0.77 is a problem for linear models and show how to compute VIF.

---

### SECTION 7: DISCOUNT INTERACTION FEATURES

7.1 — **discount_x_sales** = Discount × Sales. Explain in 2-4 lines why this interaction captures absolute revenue loss that linear models cannot find on their own.

7.2 — **discount_x_quantity** = Discount × Quantity. Explain in 2-4 lines why bulk-discount behavior needs an explicit interaction term.

7.3 — **effective_price** = Sales × (1 - Discount). Explain in 2-4 lines why this is the economically meaningful revenue figure, and note the assumption to verify about whether Sales is pre- or post-discount.

7.4 — **discount_bucket** (ordinal):
```python
bins = [0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.85]
labels = [0, 1, 2, 3, 4, 5]
df['discount_bucket'] = pd.cut(df['Discount'], bins=bins, labels=labels, include_lowest=True).astype(int)
```
Explain in 2-4 lines why this ordinal version is valid for ML unlike the EDA visualization bin.

7.5 — Explain in 2-4 lines why you do NOT create discount_squared: no theoretical reason for a quadratic relationship given EDA shows a consistent negative linear trend.

---

### SECTION 8: AGGREGATED GROUP-BASED FEATURES

⚠️ CRITICAL WARNING: Explain in 2-4 lines why ALL aggregated features MUST be computed on TRAINING DATA ONLY. Show the correct split-first pattern at the start of this section.

8.1 — **avg_profit_by_category**: Mean profit per Category (train set). Explain in 2-4 lines why category-level baseline profitability is a powerful contextual signal.

8.2 — **avg_profit_by_subcategory**: Mean profit per Sub-Category (train set). Explain in 2-4 lines why sub-category granularity captures the Copiers vs Tables disparity.

8.3 — **avg_profit_by_region**: Mean profit per Region (train set). Explain in 2-4 lines why regional profitability baseline reflects structural market differences.

8.4 — **avg_profit_by_market**: Mean profit per Market (train set).

8.5 — **avg_discount_by_subcategory**: Mean discount per Sub-Category (train set). Explain in 2-4 lines why systematic over-discounting at sub-category level is a structural signal.

8.6 — **category_sales_share**: Each order's Sales / total sales for that category (TRAIN ONLY). Explain in 2-4 lines why relative order size within a category adds context beyond raw Sales.

8.7 — Show the complete correct implementation:
```python
from sklearn.model_selection import train_test_split

X = df.drop('Profit', axis=1)
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

agg_map = X_train.groupby('Category')['Sales'].mean().to_dict()
X_train['avg_sales_by_category'] = X_train['Category'].map(agg_map)
X_test['avg_sales_by_category'] = X_test['Category'].map(agg_map)
X_test['avg_sales_by_category'].fillna(X_train['avg_sales_by_category'].mean(), inplace=True)
```

8.8 — Explain in 2-4 lines what happens if aggregations are computed before the split, with a concrete numerical example showing leakage.

---

### SECTION 9: TRANSFORMATION AND SCALING

9.1 — **Log transformation** for right-skewed features: Apply np.log1p to Sales and Shipping Cost. Explain in 2-4 lines why this matters more for Linear Regression than tree models.

9.2 — **Target variable transformation** (Profit — skewness=4.16, kurtosis=291.41):
- Log transform: CANNOT apply directly (negatives exist). Show why.
- Yeo-Johnson PowerTransformer: Handles negatives. Show implementation.
- Winsorization: Cap at 1st and 99th percentile. Show implementation.
- Leave untransformed: Valid for tree models.
Explain in 2-4 lines which approach suits which model type.

9.3 — **Scaling recommendations**:
- StandardScaler: Explain in 2-4 lines when to use and its outlier sensitivity.
- MinMaxScaler: Explain in 2-4 lines why NOT recommended for this dataset.
- RobustScaler: Explain in 2-4 lines why this is RECOMMENDED given extreme outliers.
- Explain in 2-4 lines why tree models do NOT require scaling.

9.4 — Build a complete sklearn Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

numeric_features = ['Sales', 'Quantity', 'Discount', 'Shipping Cost', ...]
ohe_features = ['Category', 'Segment', 'Ship Mode', ...]
ordinal_features = ['Order Priority']

preprocessor = ColumnTransformer(transformers=[
    ('num', RobustScaler(), numeric_features),
    ('ohe', OneHotEncoder(drop='first', sparse_output=False), ohe_features),
    ('ord', OrdinalEncoder(categories=[['Low','Medium','High','Critical']]), ordinal_features)
])
```
Explain in 2-4 lines why this pipeline pattern prevents leakage by fitting only on train.

---

### SECTION 10: OUTLIER HANDLING STRATEGY

10.1 — **Sales outliers** (max=$22,638, median=$85): Explain in 2-4 lines why keeping and log-transforming is better than removing for this dataset.

10.2 — **Shipping Cost outliers** (max=$933, 5,909 IQR-flagged): Explain in 2-4 lines why Winsorization at 99th percentile or log transform is the right strategy.

10.3 — **Profit outliers** (range: -$6,600 to +$8,400): Explain in 2-4 lines why negative extremes must be KEPT as they represent the most critical business cases.

10.4 — Show the complete outlier assessment code:
```python
for col in ['Sales', 'Shipping Cost', 'Profit']:
    Q1, Q3 = df[col].quantile([0.01, 0.99])
    print(f"{col}: 1st pct={Q1:.2f}, 99th pct={Q3:.2f}, 
          outliers outside range={((df[col]<Q1)|(df[col]>Q3)).sum()}")
```

---

### SECTION 11: FEATURES TO EXPLICITLY NOT CREATE

For each of the following, explain in 2-4 lines why a beginner might create it and why it is wrong:

11.1 — profit_per_unit = Profit / Quantity: Contains the TARGET variable directly. Leakage.
11.2 — profit_margin = Profit / Sales (when predicting Profit): Leakage.
11.3 — revenue_rank or profit_rank: Rank-based features computed before split leak test distribution.
11.4 — customer_lifetime_value: Customer ID was dropped. Recreating from remaining features adds noise without real signal.
11.5 — shipping_days (all zeros): Zero-variance, no information.
11.6 — country_encoded (147 OHE columns): Curse of dimensionality.
11.7 — Year-over-year growth features without proper time ordering: Can cause temporal leakage.

---

### SECTION 12: FINAL MODEL-READINESS CHECKLIST

Generate a complete checklist with code verification for each item:

```python
def model_readiness_check(df):
    checks = {}
    checks['no_nulls'] = df.isnull().sum().sum() == 0
    checks['no_infinite'] = np.isinf(df.select_dtypes(include=np.number)).sum().sum() == 0
    checks['no_object_columns'] = len(df.select_dtypes(include='object').columns) == 0
    checks['target_not_in_features'] = 'Profit' not in df.columns
    checks['no_zero_variance'] = (df.var() == 0).sum() == 0
    checks['feature_count'] = df.shape[1]
    for k, v in checks.items():
        print(f"{'✅' if v else '❌'} {k}: {v}")
    return checks
```

12.1 — No null values
12.2 — No infinite values (log transformations can create -inf if 0 values exist — check)
12.3 — No object/string columns remaining
12.4 — Target variable separated from features
12.5 — No zero-variance features
12.6 — No data leakage features
12.7 — Train-test split completed BEFORE aggregation features
12.8 — Scaling applied ONLY to numerical features (not binary flags)
12.9 — OHE applied ONLY to nominal categoricals
12.10 — Ordinal encoding applied ONLY to ordinal categoricals
12.11 — Feature correlation matrix recomputed after engineering
12.12 — Feature importance preview using Random Forest before full model training

---

### SECTION 13: FEATURE IMPORTANCE PREVIEW

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

rf_preview = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_preview.fit(X_train_encoded, y_train)

importances = pd.Series(rf_preview.feature_importances_, index=X_train_encoded.columns)
importances.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10,8))
plt.title('Top 20 Feature Importances (RF Preview)')
plt.show()
```
Explain in 2-4 lines why this preview helps identify useful engineered features vs noise before final model training.

---

### SECTION 14: COMMON BEGINNER MISTAKES — PREVENT THEM

For each mistake, show a WRONG example and a CORRECT example. Explain in 2-4 lines why the wrong approach fails:

14.1 — Fitting the scaler on the full dataset before splitting.
14.2 — One-hot encoding ordinal variables.
14.3 — Ordinal encoding nominal variables.
14.4 — Including the target variable as a feature.
14.5 — Computing group aggregations on the full dataset before splitting.
14.6 — Forgetting to handle Sales=0 when creating ratio features.
14.7 — Applying log transform to columns with zero or negative values without log1p.
14.8 — Dropping all outliers without checking business meaning.
14.9 — Using the same encoding for all categoricals regardless of their nature.
14.10 — Not verifying VIF after encoding.

---

### FINAL DELIVERABLES

Produce:
1. A single, clean, modular Python function: `engineer_features(df_train, df_test)` that accepts raw train and test dataframes and returns fully engineered versions.
2. A final engineered feature list with a 1-line annotation for each feature.
3. The filled model-readiness checklist.
4. Top 3 features predicted to have highest importance with 2-4 lines of reasoning each.

The final engineered dataset must be ready for direct use in:
- sklearn LinearRegression
- sklearn RandomForestRegressor
- XGBRegressor (xgboost library)

Format all code as clean, commented, notebook-ready Python cells. Use consistent variable naming. Add section headers as markdown cells.
