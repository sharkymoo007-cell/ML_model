# Modular End-to-End ML Pipeline & Feature Engineering Framework

A comprehensive machine learning pipeline constructed for the Kaggle Titanic classification benchmark. This repository demonstrates end-to-end data processing, advanced feature engineering, target encoding, dimensionality reduction, and modern gradient boosting within leak-free Scikit-Learn pipeline architectures.

---

## Technical Highlights

* **Leak-Free Pipeline Architecture:** Fully automated feature extraction and transformations using `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`.
* **Advanced Feature Engineering:**
  * **Custom Parsing:** Extracted ticket group dynamics (`Average_Fare`) and family surname frequencies (`Surname_count`).
  * **Unsupervised Clustering:** Applied `KMeans` clustering ($k=15$) on scaled numerical features to engineer distance-to-cluster centroid metrics.
  * **Principal Component Analysis (PCA):** Synthesized principal components on scaled feature pairs to capture variance across correlated dimensions.
  * **Smoothing Target Encoding:** Utilized `category_encoders.MEstimateEncoder` ($m=5.0$) on high-cardinality categorical variables to prevent overfitting while boosting signal strength.
* **Feature Selection & Analysis:** Leveraged Mutual Information (`mutual_info_regression` / `mutual_info_classif`) to inspect feature importance scores prior to training.
* **Gradient Boosting:** Trained an `XGBClassifier` tuned with early stopping on validation sets to maximize predictive performance over base random forest benchmarks.

---

## Model Pipeline Architecture

                                      ┌── Target Encoding (MEstimate)
                                      ├── One-Hot Encoders
    Raw Input Data ──> Preprocessor ──┼── K-Means Clustering Metrics ──> XGBoost Classifier ──> Predictions
                                      ├── PCA Components
                                      └── Custom Feature Transformers

---

## Experimental Benchmarks & Results

| Architecture | Feature Set | Model | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **Baseline** | Raw Features + One-Hot Encoding | `RandomForestClassifier` | **80.5970%** |
| **Optimized** | Engineered (PCA, Target Enc, KMeans, MI) | `XGBClassifier` | **83.1683%** |

---

## Core Dependencies

* **Language:** Python 3.12
* **Data Manipulation:** `numpy`, `pandas`
* **Machine Learning Frameworks:** `scikit-learn`, `xgboost`
* **Categorical Encoding:** `category-encoders`

---

## Repository Structure & Usage

1. **`notebook.ipynb`**: Contains the complete data prep, feature engineering pipelines, feature validation step, and model prediction logic.
2. **Execution:** Run the Jupyter Notebook directly or execute via Kaggle environment. Output submission files are automatically saved to `/kaggle/working/submission.csv`.
