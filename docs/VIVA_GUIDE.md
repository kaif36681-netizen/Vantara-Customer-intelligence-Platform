# Viva Guide - Customer Behavior Prediction (Churn Core)

## 1. One-minute summary (memorize this)
"I built a pipeline that takes about 1 million retail transactions, cleans and validates them, and turns
them into one row per customer with behavioural features (recency, frequency, monetary value, trends,
returns, product mix). I pick a cutoff date, compute features only from the past, and label a customer
as churned if they make no purchase in the next 90 days. I train four+ models, tune the decision
threshold on a validation set for 70% recall, and report final results on a held-out test set. The key
design concern is avoiding data leakage, which I verify with automated tests."

## 2. How the program works (data flow)
```
online_retail_II.xlsx -> loader (rename cols) -> cleaning (flag/clean) -> validation (fail loudly)
   -> builder.build_features (data BEFORE cutoff) + build_labels (data AFTER cutoff)
   -> customer_features.csv -> train_baselines (split, CV, threshold, test) -> model + metrics
```

**Timeline (the single most important diagram):**
```
|------ history: features computed here ------|cutoff|--- 90-day window: label computed here ---|
2009-12-01                               2011-09-09                                    2011-12-08
```

## 3. Key concepts and decisions

| Topic | What the code does | Why |
|---|---|---|
| Churn definition | `churned = 1` if no purchase in `[cutoff, cutoff+90d)` | Retail has no cancellation event, so churn = inactivity |
| Eligible customers | Only those with at least 1 purchase before cutoff | You cannot "lose" someone who was never a customer |
| Returns | Invoices starting with "C" or negative quantity are **flagged, not dropped** | Return behaviour is a feature (`return_rate`); returns do not count as purchases or orders |
| Missing Customer ID (~23%) | Kept in the clean table but excluded from customer-level features | Cannot attribute them to a person |
| Non-product codes | POST, BANK CHARGES, etc. flagged `is_product=False` | They are fees, not buying behaviour |
| Outliers | Flagged by IQR (factor 3), never deleted | Wholesale buyers are real customers, not errors |
| Duplicates | Only rows identical in every column are removed | Trade-off: a genuinely repeated line is indistinguishable (configurable) |
| Class imbalance | `class_weight="balanced"` | No resampling, so no leakage risk from oversampling before the split |
| Threshold | Chosen on **validation** for recall >= 0.70 at best precision | Missing a churner (false negative) costs more than a wasted retention offer |
| Model selection | By **validation** AUC, not test | Selecting on test would inflate the reported score |

## 4. Features (what each means)
- **recency_days** - days since last order (strongest churn signal)
- **frequency / monetary_total / monetary_avg** - orders count, total and average order value (the RFM trio)
- **avg_basket_size** - mean items per order
- **tenure_days** - days since first order
- **gap_variance** - variance of days between orders (irregular buyers churn more)
- **frequency_trend** - OLS slope of orders per 30-day bin over the last 12 bins (negative = slowing down)
- **seasonal_concentration** - Herfindahl index of calendar-month order shares (1/12 = even, 1 = all in one month)
- **return_rate, n_return_lines** - share of spend returned
- **clv_historical** - net revenue so far (purchases minus returns)
- **discount_sensitivity** - PROXY: share of lines priced under 90% of the SKU's median price (the data has no discount field)
- **n_unique_products, cat_share_*** - breadth and keyword-based category mix
- **engagement_score** - 0-100 average of percentile ranks of recency (inverted), frequency, monetary

## 5. Data leakage - the examiner's favourite topic
**Definition:** using information at training time that would not be available at prediction time, which
makes metrics look great and the model useless in production.

**How this project prevents it**
1. `build_features` filters `invoice_date < cutoff` as its first step.
2. The reference price used for `discount_sensitivity` is computed from pre-cutoff rows only.
3. `build_labels` raises an error if the label window is not fully observed (otherwise everyone would look churned).
4. Splitting is done *after* features are built, and scaling happens inside a Pipeline (fit on train only).
5. **Tests:** features must be identical if (a) future rows are deleted, or (b) future rows are multiplied by 100.
   A row at exactly the cutoff is excluded from features but counts in the label window.

**Known limitation to admit honestly:** `engagement_score` uses percentile ranks across the population, so for
live scoring of one customer you must save and reuse the training distribution.

## 6. Evaluation metrics
- **Accuracy** - misleading when classes are imbalanced.
- **Precision** - of customers flagged as churners, how many truly churned (cost of wasted offers).
- **Recall** - of true churners, how many we caught (cost of missed churners).
- **F1** - harmonic mean of the two.
- **ROC-AUC** - probability that a random churner is ranked above a random non-churner; threshold-independent.
- **Confusion matrix** - TN, FP, FN, TP counts (saved in `model_comparison.csv`).
- **CV AUC** - 5-fold stratified cross-validation on the training set only, to estimate stability.

## 7. Models - what to say about each
- **Logistic Regression** - linear, interpretable coefficients, strong baseline; needs scaling.
- **Decision Tree** (depth 5) - fully interpretable rules, but high variance.
- **KNN** (k=25) - distance-based; needs scaling; slow at scale; weak baseline.
- **Random Forest** - many decorrelated trees averaged (bagging); handles non-linearity; tuned by RandomizedSearchCV.
- **XGBoost / LightGBM** - gradient boosting; trees built sequentially to fix prior errors. Optional here.

## 8. Viva questions with model answers

**Q1. What problem are you solving and why does it matter?**
Predict customers likely to stop buying in the next 90 days so the business can target retention offers. Retaining a customer is generally cheaper than acquiring one.

**Q2. How did you define churn, given the dataset has no "cancelled" field?**
Non-contractual retail, so churn is inactivity: no purchase within 90 days after a cutoff date. The 90-day length is a business parameter in the config, not a universal truth.

**Q3. What is data leakage and how did you prevent it?**
See section 5. Features use only data before the cutoff; labels only after; tests prove future data cannot change features.

**Q4. Why not a random train/test split of transactions?**
Rows from the same customer would land in both sets. I split by customer (one row per customer), after feature construction, stratified on the label.

**Q5. A random customer split still mixes time periods. Is that a problem?**
Since all customers share the same cutoff, every row describes the same moment, so there is no temporal leakage inside this dataset. The limitation is that I validate on one time slice only; a stronger design uses several cutoffs and tests on a later one (a "temporal backtest"). I would list that as future work.

**Q6. Why did you flag returns instead of deleting them?**
Return behaviour carries signal and deleting it would also distort spend. But a return is not a purchase, so it does not count as an order or as "activity" in the churn label.

**Q7. 23% of rows have no Customer ID. What did you do?**
Kept them in the cleaned table, but excluded them from customer features since they cannot be tied to a person. Validation allows up to 35% missing and fails loudly beyond that.

**Q8. How do you handle class imbalance?**
Class weights. I avoided SMOTE because oversampling before cross-validation leaks synthetic copies of validation points into training; if used, it must be inside the CV folds.

**Q9. Why is accuracy not enough?**
If 80% of customers churn, predicting "everyone churns" gets 80% accuracy with zero value. I report precision, recall, F1 and AUC.

**Q10. Why tune the threshold on validation and not test?**
Tuning on test makes the test score optimistic. The test set is used once for the final estimate.

**Q11. Why recall >= 0.70?**
Business assumption: missing a churner loses the customer's future revenue, while a false alarm costs only a discount. The target is configurable; with real costs you would pick the threshold that maximises expected profit.

**Q12. Explain RFM.**
Recency (how recently), Frequency (how often), Monetary (how much). A classic customer-value framework; here they are model inputs, and the composite engagement score is their average percentile.

**Q13. How is frequency_trend computed?**
Orders are counted in twelve 30-day bins before the cutoff; I fit a least-squares line through the counts and use the slope. Positive means accelerating, negative means fading.

**Q14. Why Random Forest over a single tree?**
Averaging many decorrelated trees reduces variance; a single tree overfits easily.

**Q15. What does class_weight="balanced" do mathematically?**
It scales each sample's loss by n / (n_classes * n_class_samples), so the minority class counts more.

**Q16. How would you explain a prediction to a business user?**
Feature importance for global view; SHAP values for per-customer reasons (e.g. "recency 140 days, falling frequency trend"). SHAP is not implemented yet.

**Q17. Why scale features for Logistic Regression and KNN but not trees?**
Those two depend on feature magnitudes (gradients / distances); trees only compare thresholds. Scaling lives inside a Pipeline so it is fitted on training data only.

**Q18. What are the limitations of your project?**
- Discount sensitivity is a price-based proxy, not true discount data.
- Categories come from description keywords, not an official taxonomy.
- One cutoff date; no temporal backtest.
- Churn is defined over a fixed 90 days, which is arbitrary.
- Customers with one order have little history (gap_variance set to 0).
- Not yet built from the full PRD: CLV regression model, segmentation, SHAP, API, dashboard, LSTM.

**Q19. How do you know your features are correct?**
Unit tests with hand-computed values (recency 3.5 days, monetary 80, return rate 0.2) and leakage tests.

**Q20. What would you do next?**
Multiple cutoffs and a temporal backtest, calibration of probabilities, CLV regression, SHAP explanations, a FastAPI scoring endpoint with the saved model and threshold, and monitoring for drift.

**Q21. What is cross-validation and why stratified?**
Splits training data into k folds, trains on k-1 and validates on the remaining one, rotating. Stratified keeps the churn ratio equal in every fold.

**Q22. What is ROC-AUC = 0.5 vs 1.0?**
0.5 is random ranking; 1.0 is perfect separation.

**Q23. Why median price as the discount reference?**
Robust to outliers. It is computed from pre-cutoff data only to avoid leakage.

**Q24. Why do the printed results here not prove the model works?**
I developed against synthetic data because the real dataset was not available offline. Real results must come from running on Online Retail II.

## 9. Things to say if asked "what did you actually test?"
13 automated tests: cleaning rules, validation errors, hand-computed features, label logic, cutoff boundary,
unobserved-window guard, and 3 leakage tests. Tested on synthetic data in Python 3.12 / pandas 3.0 /
scikit-learn 1.8. XGBoost and LightGBM paths were **not** executed (not installable offline).
