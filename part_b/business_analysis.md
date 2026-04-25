# Part B — Business Case Analysis: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation (3 marks)

**Target Variable:** `items_sold` — the number of items sold at a given store in a given month under a given promotion.

**Candidate Input Features:**

| Feature Category | Examples |
|---|---|
| Store attributes | store_id, store_size, location_type (urban/semi-urban/rural), monthly footfall |
| Promotion attributes | promotion_type (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points) |
| Competitive context | local competition density |
| Temporal features | month, year, is_weekend, is_festival, is_month_end |
| Customer demographics | age distribution index, income proxy for the store's catchment area |
| Historical performance | trailing 3-month average items_sold per store |

**Type of ML Problem:** This is a **supervised regression** problem. The target (`items_sold`) is a continuous numerical quantity, and we have labelled historical observations pairing store-promotion combinations with their observed sales volumes. 

**Justification:** Because we are predicting a numerical outcome (not a category), regression is more appropriate than classification. If the business only needed to rank promotions (best/worst), classification or ranking models could be considered — but since the team wants to *maximise* items sold, a regression output allows direct comparison of predicted volumes across promotion options and enables selecting the promotion with the highest predicted value for each store-month.

---

### B1(b) — Why Items Sold is More Reliable Than Revenue (3 marks)

**Revenue = Price × Quantity.** Using revenue as the target conflates two signals: the discount depth applied by the promotion and the actual volume response of customers. For example, a Flat Discount promotion mechanically reduces unit price and therefore reduces revenue per item even if it drives substantially higher footfall. A model trained to maximise revenue may therefore avoid recommending deep-discount promotions that actually deliver the greatest volume uplift.

`items_sold` isolates the **volume response** to a promotion — the true measure of whether a promotion changed customer behaviour — from the pricing mechanism.

**Broader Principle — Target Variable Selection:** The target variable should be the quantity the business actually wants to influence, not a proxy that is contaminated by other variables. In this case, revenue is contaminated by price changes that are themselves a feature of the promotion. Choosing a contaminated target leads to a model that optimises the wrong objective. The principle is: *the target must be causally downstream of the decision being modelled, and must not be mechanically linked to the inputs in a way that hides the true signal.*

---

### B1(c) — Alternative to a Single Global Model (2 marks)

A single global model assumes that all 50 stores share the same relationship between promotions and sales — an assumption violated by the problem statement (stores in different locations respond very differently).

**Proposed Strategy: Hierarchical / Clustered Modelling**

1. **Cluster stores** into groups of similar profile (location type, size, footfall, competition density) using unsupervised methods (e.g., K-Means on store attributes).
2. **Train one model per cluster.** Each cluster model learns the promotion-response pattern specific to that store profile. Urban flagship stores and rural smaller stores will have fundamentally different coefficients.
3. **Optionally use a mixed-effects (hierarchical) model** with store as a random effect, which allows partial pooling: stores with little data borrow strength from similar stores, while stores with rich histories are fitted more individually.

**Justification:** This approach captures location-level heterogeneity without requiring a fully separate model per store (which would overfit for small stores). It also reduces the risk that majority store types dominate a global model's parameter estimates and produce poor recommendations for minority store types.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables (4 marks)

**Grain of the Final Dataset:** One row = **one store × one month × one promotion** combination, representing the observed items sold under that promotion at that store in that calendar month.

**Join Strategy:**

| Step | Action |
|---|---|
| 1 | Join `transactions` (store_id, date, items_sold, promotion_id) with `promotion_details` on `promotion_id` to get promotion type and mechanics for each transaction. |
| 2 | Aggregate to store-month-promotion grain: `SUM(items_sold)`, `COUNT(transactions)`, `AVG(basket_size)` grouped by (store_id, year-month, promotion_type). |
| 3 | Join aggregated result with `store_attributes` (store_id → size, location_type, footfall, competition_density) on `store_id`. |
| 4 | Join with `calendar` (date → is_weekend, is_festival, month, year) on year-month to attach temporal flags. |

**Key aggregations before modelling:**
- Total `items_sold` per store-month-promotion (target).
- `transaction_count` — proxy for footfall under each promotion.
- `avg_basket_size` — whether the promotion drives larger baskets.
- `promotion_duration_days` — how many days the promotion ran (from promotion_details).

**Important:** Each store-month should have **one row per promotion type observed**. If a store ran only one promotion in a given month, there is one row. Months with no promotion can be encoded as promotion_type = "None" and used as a baseline.

---

### B2(b) — EDA Strategy (4 marks)

| Analysis / Chart | What to Look For | Influence on Modelling |
|---|---|---|
| **1. Items sold distribution by promotion type** (box plot, one box per promotion) | Whether any promotion systematically drives higher or lower volume; outliers per promotion | If one promotion dominates in volume, class imbalance is relevant; extreme outliers may need Winsorisation or log-transformation of the target |
| **2. Promotion performance by location type** (grouped bar chart: promotion_type × location_type, y = mean items_sold) | Whether BOGO works in urban stores but not rural ones, or vice versa | Confirms the need for location-segmented models or interaction features (promotion_type × location_type) |
| **3. Monthly seasonality heatmap** (store cluster × month, colour = mean items_sold) | Peak months (festival season), slow months, and whether different store types peak at different times | Guides creation of cyclical month features (sin/cos encoding) and prompts inclusion of is_festival flag |
| **4. Correlation heatmap of numerical store attributes vs items_sold** | Which store attributes (footfall, competition_density, store_size) most strongly predict sales volume | Helps prioritise features; high multicollinearity between store attributes may suggest PCA or VIF-based pruning |
| **5. Promotion adoption timeline** (line chart of how often each promotion was used per month over the 3-year period) | Whether certain promotions were phased in/out — this affects data availability and potential concept drift | If Loyalty Points was only introduced in year 2, the model has less data for it; worth flagging or modelling separately for the early period |

---

### B2(c) — Handling Promotion Imbalance (2 marks)

With 80% of transactions occurring without a promotion, a naïve model may learn to predict "no promotion" as the default and under-fit the promotional periods.

**Impact on the model:**
- The model will be trained predominantly on non-promotional patterns and may not learn the differential effect of promotion types reliably.
- Metrics computed on the full dataset will be dominated by non-promotional rows and may look good while predictions on promotional rows are poor.

**Steps to address:**
1. **Stratified sampling:** When splitting train/test, stratify by a `has_promotion` flag to ensure both splits have the same promotion representation.
2. **Separate evaluation:** Report RMSE/MAE specifically on promotional-month rows, not only on the full test set, so promotion-period predictive quality is explicitly measured.
3. **Oversample or focus the model:** Train the model specifically on promotional rows if the goal is promotion selection (the deployment task is always a promotion month). Non-promotional rows can still be included as the "None" baseline level of `promotion_type`.
4. **Feature engineering:** Create a `promotion_uplift` feature (items_sold in promotion month ÷ trailing 3-month average) if sufficient history exists, which isolates the incremental effect.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics (4 marks)

**Train-Test Split Setup:**

With 3 years × 50 stores × monthly data ≈ 1,800 store-month observations:

- **Temporal split:** Use the most recent 6 months (approximately the last 17% of the timeline) as the hold-out test set. Train on the preceding 30 months.
- **Why not random split:** A random split allows the model to train on December 2024 data and test on January 2023 data. The model "sees the future" during training, producing overoptimistic test metrics. In production, the model always predicts a future month it has never seen — the evaluation protocol must mirror this.
- **Optional — Walk-forward validation:** For more robust estimates, use 3-fold time-series cross-validation (also called rolling-origin evaluation): Train on months 1–24, validate on 25–30; then train on 1–30, validate on 31–36, etc. Average metrics across folds.

**Evaluation Metrics:**

| Metric | Formula | Business Interpretation |
|---|---|---|
| **RMSE** | √(mean of squared errors) | Penalises large mispredictions heavily. A large RMSE signals the model is badly wrong on some store-months — costly if those are peak-season decisions. |
| **MAE** | Mean of absolute errors | Average number of items the forecast is off by. Easily communicated to the marketing team: "On average, our forecast is off by X items." |
| **MAPE** | Mean absolute % error | Scale-invariant; useful when comparing across stores of very different size. A 10% MAPE has the same meaning for a 50-item store and a 5,000-item store. |
| **Promotion Ranking Accuracy** | % of store-months where the model correctly identifies the best promotion | Most directly aligned with the deployment goal: does the model pick the right promotion? Computed by ranking model predictions across the 5 promotion types and checking if rank-1 matches the historically best performer. |

---

### B3(b) — Investigating Different Recommendations via Feature Importance (4 marks)

The model recommends **Loyalty Points Bonus** for Store 12 in December and **Flat Discount** in March. To investigate and explain this:

**Step 1 — SHAP Values (SHapley Additive exPlanations)**

Compute SHAP values for Store 12's December and March predictions. SHAP assigns each feature a contribution score (positive = pushes prediction up, negative = pushes it down) for a specific prediction.

- For December: SHAP likely shows `is_festival = 1`, `month = 12`, and `promotion_type_LoyaltyPoints` all have large positive SHAP values. The model has learned that in high-footfall festival months, customers are already motivated to buy — they respond better to a rewards mechanism that encourages return visits than to an immediate price cut.
- For March: SHAP likely shows `competition_density` and `promotion_type_FlatDiscount` dominating. In a quieter month with local competitors running discounts, a direct price reduction is needed to attract footfall.

**Step 2 — Communicating to the Marketing Team**

Produce a **SHAP waterfall chart** for each Store 12 prediction showing the top 5 features and their direction of influence. Frame it narratively:

> *"In December, our model recommends Loyalty Points because Store 12 sits in a high-footfall festival period (is_festival = 1) and customers visiting during peak season are already purchase-ready. The loyalty mechanism is predicted to add 14% more items compared to a flat discount, primarily because of the festival effect and the store's urban demographic responding well to long-term rewards. In March — a quieter month with three competing stores in the same catchment — a Flat Discount is predicted to win footfall from price-sensitive shoppers who might otherwise visit a competitor."*

This approach is transparent, avoids jargon, and ties the model's logic to business intuition the marketing team already holds.

---

### B3(c) — End-to-End Deployment and Monitoring (4 marks)

**Saving the Model**

```python
import joblib
joblib.dump(pipeline, 'promotion_model_v1.pkl')
```

Store the serialised pipeline (including the preprocessing ColumnTransformer) in a versioned model registry (e.g., MLflow, AWS S3 with versioning). Tag it with the training date, data version, and evaluation metrics.

**Monthly Inference Process**

At the start of each month:

1. **Data preparation:** The data engineering team runs a scheduled job that pulls the latest store attributes, calendar flags (is_festival, is_weekend for the upcoming month), and computes trailing features (e.g., 3-month average items_sold) from the transactions database.
2. **Feature construction:** The same `ColumnTransformer` pipeline applied during training is loaded from the saved pipeline object — this guarantees identical preprocessing without rewriting code.
3. **Inference:** For each of the 50 stores, generate 5 rows (one per promotion type) with the features constructed above. Run `pipeline.predict()` on all 250 rows.
4. **Recommendation output:** For each store, select the promotion type with the highest predicted `items_sold`. Output a recommendation CSV: `store_id, recommended_promotion, predicted_items_sold`. Deliver to the marketing team's dashboard.

**Monitoring and Retraining Triggers**

| Monitoring Check | Method | Action Threshold |
|---|---|---|
| **Prediction accuracy** | Each month, once actual items_sold is available (~30 days later), compute rolling MAE on the last 3 months. | If MAE rises > 20% above the baseline test-set MAE, flag for retraining. |
| **Data drift** | Track the statistical distribution of input features monthly (mean, std, % nulls) using tools like Evidently AI or a custom drift detector. Compare against the training-set distribution. | If Population Stability Index (PSI) > 0.2 for any key feature, investigate and potentially retrain. |
| **Concept drift** | Monitor whether the model's promotion ranking accuracy (which promotion it recommends vs which actually performed best) is declining. | If ranking accuracy drops below 60% for two consecutive months, retrain on the most recent 24 months of data. |
| **Scheduled retraining** | Regardless of drift signals, retrain annually at the start of the fiscal year to incorporate the full previous year of promotional data. | Annual — incorporate the latest data and re-run hyperparameter tuning. |

**Retraining Protocol:** When retraining is triggered, run the full pipeline — data join, aggregation, feature engineering, train-test temporal split, model fit, evaluation — in an automated workflow (e.g., Airflow DAG or GitHub Actions). Gate deployment on the new model outperforming the current production model on the most recent 3-month hold-out before replacing it.
