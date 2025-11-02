#🧠 Student Habits and Academic Performance Analysis
📋 Project Overview

This project explores the relationship between student lifestyle habits and academic performance using data science techniques.
The aim is twofold:

To identify which habits most influence exam outcomes, and

To build predictive models capable of forecasting academic performance based on these habits.

The project was developed as part of a Data Science learning journey — combining data cleaning, exploratory data analysis (EDA), machine learning, and model interpretability (SHAP and LIME).

🧩 Project Structure
├── main_script.py               # Main pipeline orchestrator
├── student_analysis.py          # Custom classes for data loading, cleaning, analysis, visualization, and modeling
├── ROC_AUC.png                  # ROC curve comparison of models
├── SHAP.png                     # Global feature importance visualization
├── LIME.png                     # Example of local model explanation
├── data/
│   └── student_habits_performance.csv   # Dataset (not included here for privacy)
├── README.md                    # Project documentation
└── reports/
    └── Student_Habits_Report.docx       # Final report (if applicable)

🧠 Dataset Description

The dataset consists of 1,000 student records, each containing a mix of behavioral, lifestyle, and academic attributes.
Key features include:

Category	Variables
Academic	exam_score, attendance_percentage, study_hours_per_day
Lifestyle	sleep_hours, exercise_frequency, diet_quality, social_media_hours, netflix_hours
Demographic	gender, parental_education_level, part_time_job, internet_quality
Psychological	mental_health_rating

Target Variable: exam_score (used for prediction and classification tasks)

⚙️ Technical Workflow

The project follows a modular, class-based structure:

Stage	Module/Class	Key Functions
Data Validation	DataLoaderIO_nException	File format and read checks
Data Cleaning	DataCleaner	Missing value imputation, duplicates removal, negative value checks
EDA & Visualization	VisualisationEngine	Correlation matrix, scatter, violin, and stacked bar plots
Statistical Analysis	StudentAnalyser	Grouping, averaging, and outlier detection
Modeling	ScorePredictor, MakeModel	Logistic Regression, Decision Tree, cross-validation
Explainability	—	SHAP (global), LIME (local)
📊 Key Findings
🔹 Correlation Insights

Top correlations with exam_score:

Feature	Correlation (r)	Relationship
study_hours_per_day	0.83	Strong positive
mental_health_rating	0.32	Moderate positive
social_media_hours	-0.17	Weak negative
netflix_hours	-0.17	Weak negative
exercise_frequency	0.16	Mild positive
🔹 Visualization Highlights

Scatter plots: Clear trend between study hours and exam performance.

Violin plots: Poor diet quality slightly associated with lower median scores.

Stacked bar plot: Gender differences in mental health distribution, though minor.

🤖 Model Performance
Model	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.89	0.913	0.913	0.91	0.88
Decision Tree	0.885	0.919	0.897	0.91	0.88

Takeaway:
Both models performed almost identically, but Logistic Regression was chosen for its simplicity, stability, and interpretability.

🔍 Model Explainability

Explainability techniques provided transparency into how predictions were made.

SHAP (Global)

Identified study hours per day and mental health rating as top positive drivers.

Highlighted minor negative influence of social media and Netflix hours.

LIME (Local)

Explained individual student predictions.

Showed how specific factors (e.g., low study hours or poor mental health) directly impact predicted performance.

💡 Lessons Learned

Data quality and preprocessing determine downstream model reliability.

Simplicity > complexity — Logistic Regression performed on par with a tuned Decision Tree.

Explainability matters — SHAP and LIME turned abstract models into understandable tools for educators.

🧭 Recommendations

Encourage consistent study routines — the strongest predictor of success.

Support student mental health and lifestyle balance.

Monitor screen time — modest improvements can lead to better academic outcomes.

Integrate explainable AI tools in educational analytics for transparency and fairness.

🧰 Technologies Used

Python 3.11+

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, shap, lime

Development: Visual Studio Code

Version Control: Git + GitHub

🧑🏽‍💻 Author

Sesethu M. Bango
Data Analyst | Data Scientist in Training | Mechanical Engineering Background
📍 South Africa

🔗 LinkedIn
 (add your link here)

📧 [bsesethu@outlook.com]
