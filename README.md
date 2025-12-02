# taobao-ctr-prediction
---

## Dataset Description

The dataset originates from the **Taobao Display Advertising Challenge (2018)**.  
It contains:

- **26M+** impression records  
- User demographic attributes  
- Ad metadata (price, brand, category, advertiser, campaign)  
- Page context information (PID)  
- Click labels (`0 = no click`, `1 = click`)

Dataset source: https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom?select=user_profile.csv

---

## Key Feature Engineering Steps

### 1. Temporal & Contextual Features  
- `hour`, `weekday`, `part_of_day`, `is_weekend`

### 2. User Behaviour Aggregates  
- `user_impressions`, `user_clicks`, `user_ctr`  
- `user_avg_ad_price`

### 3. Leakage-Free Sequential Features (Strict Past-Only)  
- `user_impressions_past`  
- `user_clicks_past`  
- `user_ctr_past`  
- `user_avg_price_past`  
> Built using expanding windows + `.shift(1)` to ensure **no future information leakage**.

### 4. Ad-Level CTR Statistics  
- `ad_ctr`, `cate_ctr`, `brand_ctr`, `campaign_ctr`, `customer_ctr`

### 5. Personalisation Feature  
- `user_cate_ctr` — user–category interaction history.

### 6. Price Buckets  
Useful for capturing non-linear effects in ad pricing.

---

## Models Trained & Benchmarking

### Baseline Algorithms
- Logistic Regression  
- Naive Bayes  
- K-Nearest Neighbours  
- Decision Tree  
- SVM  
- Random Forest  

### Advanced Ensemble Models
- Random Forest (tuned via RandomizedSearchCV)
- XGBoost
- CatBoost (final model)

Performance was measured using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Confusion Matrix**

---

## Final Model: CatBoost

After hyperparameter optimisation and threshold tuning:

| Metric | Score |
|--------|--------|
| Accuracy | **0.963** |
| Precision | **0.635** |
| Recall | **0.624** |
| F1-score | **0.629** |
| ROC-AUC | **0.969** |

The **ROC curve** demonstrates strong separation between classes, confirming excellent discriminative power.

---

## Handling Class Imbalance

Original dataset:
- **95% non-click**
- **5% click**

Used:
- Random under-sampling (1:1 balanced training set)
- Threshold tuning after probability prediction
- `class_weight='balanced'` during hyperparameter search

---
