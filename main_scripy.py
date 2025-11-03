import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from student_analysis import (
    DataLoaderIO_nException,
    DataCleaner,
    StudentAnalyser,
    VisualisationEngine,
    MakeModel
)
# *************** Data collection and preparation ******************
# ==========================================================================
# Handling File I/O 
# ==========================================================================
# Using the DataLoader_IO_n_Excep class
print('\n')
print('Checking file format from file path/name')
filePath = 'student_habits_performance.csv'
file_test = DataLoaderIO_nException(filePath)
print(file_test.format_check())

print('\n')
print('Checking file readibility and permissions')
print(file_test.readFile_check())

# Create DataFrame from 'student_habits_performance.csv'
# Read CSV with delimiter (;)
df = pd.read_csv(filePath, delimiter= ';') # Values in the file are seperated by delimiter ';'. Hence the file is now readable as a df.
print('\n')
print('Original DataFrame, first 5 rows:')
print(df.head())  

# ==========================================================================
# Check for missing values, Remove duplicates, Validate data ranges
# ============================================================================

print('\n')
print('DataFrame information:')
print(df.info()) # Print column names

clean_1 = DataCleaner
# Check empty DataFrame
print('\nCheck for empty DataFrame:')
clean_1.check_emptyDataFrame(df)

# Using the DataCleaner class
print('\nCheck for missing values:')
clean_1.check_missing(df)

# Using the 'check_missing' method is how we find where the null/None values are, in the 'parental_education_level' column.
# Replacing None values with the value 'No Education'
clean_1.parentalNulls_fix(df) # Should return a null free dataframe
# To check whether it's really null free:
print('\nCheck DataFrame after replacing Null/None values:')
clean_1.check_missing(df)

# Removing duplicates
print('\nDataFrame after removing duplicate rows, of which there were none (showing first 5 rows):')
df = clean_1.dropDuplicates(df)
print(df.head())
print('Number of rows and columns respectively:', df.shape)
print('Result says there are zero duplicates in the original dataframe. Number of rows before are equal to the number of rows after implementing the duplicateFree method.')

print('\nDatatypes for each column:')
print(clean_1.dataTypes(df))

# Check negatives. Check each number value column individually
print('\n')
for col in df.columns: # Checks each column for negative values
    if is_numeric_dtype(df[col]):
        print(f'Column {col} contains negative values: {clean_1.check_negatives(df, col)}')
    else:
        print(f'Column {col} is a non numeric datatype')      

# Encode categorical variables
# ===============================================
# Instantiate
Mm = MakeModel 
print('\nApplying the string_to_Int function to encode the categorical variables to numerical.')
# Adding more 
new_df = Mm.string_to_Int(df, 'gender', arg1='Female', arg2='Male', arg3= 'Other') # String to integer the gender column, do the same to the other string columns
new_df = Mm.string_to_Int(new_df, 'part_time_job', arg1='No', arg2='Yes', arg3= 'Other')
new_df = Mm.string_to_Int(new_df, 'diet_quality', arg1='Good', arg2='Fair', arg3= 'Poor', arg4= 'Other')
new_df = Mm.string_to_Int(new_df, 'parental_education_level', arg1='Master', arg2='High School', arg3= 'Bachelor', arg4= 'No Education')
new_df = Mm.string_to_Int(new_df, 'internet_quality', arg1='Average', arg2='Poor', arg3= 'Good', arg4= 'Other')
df_encoded = Mm.string_to_Int(new_df, 'extracurricular_participation', arg1='No', arg2='Yes', arg3= 'Other')

print(df_encoded.head())

# Correlation matrix, printed table and correlation heatmap plot
correlation_matrix = VisualisationEngine.correlation_Matrix(df_encoded) 

# Standardise/normalise numeric variables
# ===================================================
scaler_Standard = StandardScaler() # Instantiate

X = df_encoded.drop(['exam_score', 'student_id'], axis= 1) 
# Testing the use of only the features with the highest correlation with exam_score. Only where r >= 0.05 wrt exam_score
X = X[['study_hours_per_day', 'mental_health_rating', 'social_media_hours', 'netflix_hours', 'exercise_frequency', 'sleep_hours', 'attendance_percentage', 'internet_quality']]
y = df_encoded['exam_score']

# Change target variables from numerical to categorical. 
y_categorical = Mm.target_toCategorical(y)

# Split train/test features
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size= 0.2, random_state= 42)

scaler_Standard.fit(X_train) # Fit scaling function

# Transform datasets using StandardScaler
X_train_scaledStandard = scaler_Standard.transform(X_train)
X_test_scaledStandard = scaler_Standard.transform(X_test)

# Create cleaned dataset
# ================================================
# new_dataset = df.to_csv('students_clean.csv')


# ***************************** Data exploration *************************************
# ==============================================================================================
# Perform EDA
# ====================================================================================

StA = StudentAnalyser
# Find outliers for each of the numeric columns. Assume they are all normally distributed
all_outliers = StA.findOutliers(df)

# Perform EDA with 2D and 3D plots

Vis = VisualisationEngine

# Scatter plot of sleep vs final scores
col_1 = 'sleep_hours'
col_2 ='exam_score'
skatter_1 = Vis.scatterPlot(df, col_1, col_2)

# Scatter plot of study_hours_per_day vs final scores
col_1 = 'study_hours_per_day'
col_2 ='exam_score'
skatter_1 = Vis.scatterPlot(df, col_1, col_2)

# Scatter plot of social_media_hours vs final scores
col_1 = 'social_media_hours'
col_2 ='exam_score'
skatter_1 = Vis.scatterPlot(df, col_1, col_2)

# 3d scatter. Not a useful graph
# threeD = Vis.threeD_plot(df['gender'], df['age'], df['exam_score'], '3D')

# Stacked bar plot of gender by mental_health rating
bar = Vis.stacked_bar(df)

# Violin plot
violinplot_1 = Vis.violin_plot(df, 'diet_quality', 'exam_score')

# Parellel coodinates plot. Messy.
parallel_coords_plot = Vis.parallel_coords_plot(df, 'exam_score', 'sleep_hours', 'study_hours_per_day')

# # Radar chart for a single student
radar = Vis.radar_plot(df, 'S1001') # exam_score has been re-scaled to a value out of 10, to match the scale of other values
# we can add an input student id program. We can compare two or more students.

# ********************************* Model Building and Evaluation *************************************
# Train Decision tree and Logistic regression models
# =============================================================================================================
# Logistic regression
# Initialize Logistic regression model
model_Logic = LogisticRegression(random_state= 42)

# Train model
model_Logic.fit(X_train_scaledStandard, y_train) # Logistic

# Make predictions on test data
y_pred_Logic = model_Logic.predict(X_test_scaledStandard)
# -------------------------------------------------------------------------------

Mm = MakeModel
# Decision tree model
# Get the best model by implementing the following function
model_Decision = Mm.decision_tree_setup(X_train_scaledStandard, y_train)

# Make predictions on test data
y_pred_Decision = model_Decision.predict(X_test_scaledStandard)

# ===========================================================================================
# Model evaluation
# ====================================================================================================
# Check accuracy
accuracy_Logic = accuracy_score(y_test, y_pred_Logic)
class_report_Logic = classification_report(y_test, y_pred_Logic, target_names= ['Fail (0)', 'Pass (1)'])

accuracy_Decision = accuracy_score(y_test, y_pred_Decision)
class_report_Decision = classification_report(y_test, y_pred_Decision, target_names= ['Fail (0)', 'Pass (1)'])

print('\nCheck accuracy of Logistic regression')
print(f'Accuracy score: {accuracy_Logic}')
print('Classification report:')
print(class_report_Logic)

print('\nCheck accuracy of Decision tree')
print(f'Accuracy score: {accuracy_Decision}')
print('Classification report:')
print(class_report_Decision)

#NOTE Logistic Regression is the slightly better model

# Evaluate model using Confusion matrices, ROC curves and AUC

# Evaluate Logistic regression using confusion matrix, and plot it
precision_Logic, recall_Logic = Mm.evaluateNPlot_Logistic_confusion(y_test, y_pred_Logic, model_Logic)

print('\nEvaluation using the Confusion matrices')
print('For the Logistic regession model:')
print('Precisiion =', precision_Logic)
print('Recall =', recall_Logic)

# Evaluate Decision tree using confusion matrix, and plot it
precision_Decision, recall_Decision = Mm.evaluateNPlot_Decision_confusion(y_test, y_pred_Decision, model_Decision)

print('\nFor the Decision tree model:')
print('Precisiion =', precision_Decision)
print('Recall =', recall_Decision)

# Evaluate both models using ROC curves
# Plot ROC curve and print AUC score
plot_ROC = Mm.ROC_curvePlot(y_test, y_pred_Logic, y_pred_Decision)
#------------------------------------------------------------------------------------------------------------------------

# ===========================================================================================
# Explainability
# ====================================================================================================

# Apply SHAP and LIME. Apply it to the Logistic regression model, the better of the two
# Initialize the SHAP explainer, create an explainer object
explainer = shap.Explainer(model_Logic.predict, X_test_scaledStandard)

# Compute SHAP values for the test dataset
shap_values = explainer(X_test_scaledStandard)

# # Plot of SHAP values summary
feature_names_list = X_train.columns.tolist() # list of feature names
shap.summary_plot(shap_values, X_test_scaledStandard, feature_names= feature_names_list)

# For LIME; plot and explain ...
plot_nExplain_LIME = Mm.LIME_explanation_NPlot(X_train_scaledStandard, X_test_scaledStandard, model_Logic, feature_names_list)
