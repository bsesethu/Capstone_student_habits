import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from student_analysis import (
    DataLoaderIO_nException,
    DataCleaner,
    StudentAnalyser,
    VisualisationEngine,
    ScorePredictor
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
        print(f'Column {col} contains negative values: {clean_1.check_negatives(df, col)}') # NOTE Use a loop later
    else:
        print(f'Column {col} is a non numeric datatype')        

# Encode categorical variables
# ===============================================
Sco = ScorePredictor   
print('\nApplying the string_to_Int function to encode the categorical variables to numerical.')
# Adding more 
new_df = Sco.string_to_Int(df, 'gender', arg1='Female', arg2='Male', arg3= 'Other') # String to integer the gender column, do the same to the other string columns
new_df = Sco.string_to_Int(new_df, 'part_time_job', arg1='No', arg2='Yes', arg3= 'Other')
new_df = Sco.string_to_Int(new_df, 'diet_quality', arg1='Good', arg2='Fair', arg3= 'Poor', arg4= 'Other')
new_df = Sco.string_to_Int(new_df, 'parental_education_level', arg1='Master', arg2='High School', arg3= 'Bachelor', arg4= 'No Education')
new_df = Sco.string_to_Int(new_df, 'internet_quality', arg1='Average', arg2='Poor', arg3= 'Good', arg4= 'Other')
df_encoded = Sco.string_to_Int(new_df, 'extracurricular_participation', arg1='No', arg2='Yes', arg3= 'Other')

print(df_encoded.head())

# Standardise/normalise numeric variables
# ===================================================
scaler_Standard = StandardScaler() # Instantiate

X = df_encoded.drop(['exam_score', 'student_id'], axis= 1)
y = df_encoded['exam_score']

# For Logistic regression
# Also for Decision trees, so we're comparing apples to apples
    # Must make target features categorical. 
    # Use 0: Fail is y < 50
    #     1: Pass Satisfactory is 50 <= y < 75
    #     2: Pass Distinction y >= 75
y_categorical= [] # New target variable
for row in y:
    if row < 65:
        y_categorical.append(0)
    elif row >= 65: # and row < 75:
        y_categorical.append(1)
    # elif row >= 75:
    #     y_categorical.append(2)
    else:
        y_categorical.append(None)

# Split train/test features
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size= 0.2, random_state= 42)

scaler_Standard.fit(X_train) # Fit scaling function

# Transform datasets using StandardScaler
X_train_scaledStandard = scaler_Standard.transform(X_train)
X_test_scaledStandard = scaler_Standard.transform(X_test)

# Create cleaned dataset
# ================================================
# new_dataset = df.to_csv('students_clean.csv')


# *************** Data exploration and model building *********************
# ==============================================================================================
# Perform EDA
# ====================================================================================
# Correlation matrix
df_corr = df_encoded.drop(['student_id'], axis= 1) # Removing the non numeric columns
correlation = df_corr.corr()
print(correlation)

# Plot the correlation matrix
fig = plt.figure()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Correlation matrix heatmap')
# plt.show()
#-------------------------------------------------------------------------------------------------------------------

# Five Number summary of all each of the numerical columns
print(df.describe())
# five_numSummary = df.describe()
# saved_summary = five_numSummary.to_csv('five_numSummary.csv') # Saved csv


StA = StudentAnalyser
df_grouped_1 = StA.isolateColumns(df, 'mental_health_rating', 'study_hours_per_day')

print('\nAverage study hours per day, grouped by mental health rating')
df_grouped_1_mean = StA.group_nFind(df_grouped_1, 'mental_health_rating', 'mean') 
print(df_grouped_1_mean)

print('\nMedian study hours per day, grouped by mental health rating')
df_grouped_1_median = StA.group_nFind(df_grouped_1, 'mental_health_rating', 'median')
print(df_grouped_1_median)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Identify correlation between sleep and exam scores
# Isolate the two columns
df_sl_ex = StA.isolateColumns(df, 'sleep_hours', 'exam_score')
# print(df_sl_ex)

print('\nIdentify correlation between sleep and exam scores by finding exam_score grouped by sleep_hours.') # Best way to do this is to visualize it.. Later
# df_sl_ex_grouped = StA.group_nFind(df_sl_ex, 'sleep_hours', 'mean')
# print(df_sl_ex_grouped.head()) # Not a very useful result
new_df = StA.groupDistinct_avgScore(df_sl_ex)
print('\nCorrelation between sleep_time and exam_scores')
print(new_df)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('\nDetect outliers in social media usage')
sigma = StA.isolate_nStd(df, 'social_media_hours')
print('\nStandard dev. of social media hours', sigma)

# Find outlers
outlier_value, index_value = StA.findOutliers(df, 'social_media_hours', sigma)
print('\nOutlier values:', outlier_value)
print('Outlier index values:', index_value)

Vis = VisualisationEngine
# Histogram of Study Time frequency
column_name = 'exam_score'
x_label = 'exam_score'
y_label = 'Number of Students'
# histo_1 = Vis.histogram(df, column_name, 48, x_label, y_label)

# Skatter plot of sleep vs final scores
col_1 = 'sleep_hours'
col_2 ='exam_score'
# skatter_1 = Vis.scatterPlot(df, col_1, col_2)

# Box plots of scores by diet quality
# boxplot_1 = Vis.boxplots(df, 'diet_quality', 'exam_score')

# 3d scatter 
# threeD = Vis.threeD_plot(df['gender'], df['age'], df['exam_score'], '3D') # Not a useful graph

# NOTE More plots later

# Train Decision tree and Logistic regression models
# =============================================================================================================
# Logistic regression
# Initialize Logistic regression model
model_Logic = LogisticRegression(random_state= 42)

# Train model
model_Logic.fit(X_train_scaledStandard, y_train) # Logistic

# Make predictions on test data
y_pred_Logic = model_Logic.predict(X_test_scaledStandard)
# -------------------------------------------------------------

# Decision tree model
# Hyperparameter tuning
decision_instance = DecisionTreeClassifier(random_state= 42)

# Define parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4] # No ccp alpha, add later to chech difference
}
# Define cross validation strategy
cross_validation = StratifiedKFold(n_splits= 5, shuffle= True, random_state = 42)
gridS = GridSearchCV(estimator= decision_instance,
                     param_grid= param_grid,
                     cv= cross_validation,
                     scoring= 'accuracy',
                     n_jobs= -1)
gridS.fit(X_train_scaledStandard, y_train)

# Get the best model from gridS
model_Decision = gridS.best_estimator_

# Make predictions on test data
y_pred_Decision = model_Decision.predict(X_test_scaledStandard)
#----------------------------------------------------------------------------------

# Check accuracy
# ====================================================================
# Logic regression using Standard scalar function
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

# ===========================================================================================
# Model evaluation and explainability
# ====================================================================================================
# Evaluate model using confusion matrices, ROC curves and AUC
# Using confusion matrices
confuse_Logic = confusion_matrix(y_test, y_pred_Logic)

TP_L = confuse_Logic[1, 1] # extract True Negatives
FP_L = confuse_Logic[0, 1] # extract False Positives
FN_L = confuse_Logic[1, 0] # extracy False Negative
# Calculate precition and recall
precision_Logic = TP_L / (TP_L + FP_L)
recall_Logic = TP_L / (TP_L + FN_L)
print('\nFor the Logistic regession model:')
print('Precisiion =', precision_Logic)
print('Recall =', recall_Logic)

confuse_Decision = confusion_matrix(y_test, y_pred_Decision)

TP_D = confuse_Decision[1, 1] # extract True Positives
FP_D = confuse_Decision[0, 1] # extract False Positives
FN_D = confuse_Decision[1, 0] # extracy False Negative

precision_Decision = TP_D / (TP_D + FP_D)
recall_Decision = TP_D / (TP_D + FN_D)
print('\nFor the Decision tree model:')
print('Precisiion =', precision_Decision)
print('Recall =', recall_Decision)

display_L = ConfusionMatrixDisplay(confusion_matrix= confuse_Logic, display_labels= model_Logic.classes_)
display_L.plot(cmap= plt.cm.Blues) # Using blue colour map
plt.title('Confusion matrix for Logistic Regression')
plt.show

display_D = ConfusionMatrixDisplay(confusion_matrix= confuse_Decision, display_labels= model_Decision.classes_)
display_D.plot(cmap= plt.cm.Blues)
plt.title('Confusion matrix for Decision tree')
plt.show()
#------------------------------------------------------------------------------------------------------------------------

# Using ROC curves