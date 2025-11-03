# Student Habits and Academic Performance Analysis

## Project overview
This project investigates how student lifestyle and study habits relate to academic performance, and builds predictive models to forecast exam outcomes.  
It combines data cleaning, exploratory data analysis (EDA), supervised learning, and model explainability (SHAP & LIME).

---

## Repository structure
├── main_script.py # Main pipeline orchestrator
├── student_analysis.py # Custom classes for loading, cleaning, EDA, visualization, and modeling
├── student_habits_performance.csv # Original dataset
├── README.md # This file
└── Report_SM_Bango # Final report
└── student_clean.csv

---

## Dataset
- **Records:** 1,000 student rows  
- **Columns:** 16 student characteristics
  - Academic: `exam_score`, `attendance_percentage`, `study_hours_per_day`  
  - Lifestyle: `sleep_hours`, `exercise_frequency`, `diet_quality`, `social_media_hours`, `netflix_hours`  
  - Demographics: `gender`, `parental_education_level`, `part_time_job`, `internet_quality`  
  - Psychological: `mental_health_rating`

> **Target:** `exam_score` (used as continuous and converted to categorical for classification)

---

## Technical workflow
1. **Data validation & ingestion** (`DataLoaderIO_nException`)  
2. **Cleaning & preprocessing** (`DataCleaner`) — missing values, duplicates, encodings  
3. **Exploratory Data Analysis** (`StudentAnalyser`, `VisualisationEngine`) — correlations, scatter/violin/stacked-bar plots  
4. **Modeling** (`ScorePredictor`, `MakeModel`) — Logistic Regression, Decision Tree with GridSearchCV + Stratified K-Fold  
5. **Explainability** — SHAP (global) and LIME (local) analyses

---

## Key findings (summary)
- **Study hours per day** is the strongest predictor of exam success (correlation r ≈ **0.83**).  
- **Mental health rating** has a moderate positive relationship (r ≈ **0.32**).  
- **Social media** and **Netflix** hours show small negative correlations (r ≈ **-0.17**).  
- Lifestyle factors (sleep, exercise, diet) show smaller positive effects.

---

## Model performance (test set)
| Model               | Accuracy | Precision | Recall | F1-score | AUC  |
|---------------------|:--------:|:---------:|:------:|:--------:|:----:|
| Logistic Regression | 0.89     | 0.913     | 0.913  | 0.91     | 0.88 |
| Decision Tree       | 0.885    | 0.919     | 0.897  | 0.91     | 0.88 |

**Takeaway:** Both models perform similarly. Logistic Regression preferred for simplicity and interpretability.

---

## Explainability
- **SHAP (global):** Identifies top drivers across the dataset — study hours and mental health show the strongest positive impact; social-media/streaming time are negative contributors.  
- **LIME (local):** Provides human-readable explanations for individual predictions (which features increased or decreased the predicted probability).

---

## How to run
1. Create and activate a Python environment (recommended: `venv` or `conda`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt


