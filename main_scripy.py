import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import shap
from lime.lime_tabular import LimeTabularExplainer
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
    ScorePredictor,
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
        
# Nje, checking
print(df[['gender', 'mental_health_rating']].head())   

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

# Instantiate
Mm = MakeModel
# Make target features categorical. 
y_categorical = Mm.target_toCategorical(y)

# Split train/test features
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size= 0.2, random_state= 42)

scaler_Standard.fit(X_train) # Fit scaling function

print('\nBefore Standardising')
print(X_train.describe())
print('\nAfter Standardising')

# Transform datasets using StandardScaler
X_train_scaledStandard = scaler_Standard.transform(X_train)
X_test_scaledStandard = scaler_Standard.transform(X_test)

# For the PB graph I generated. Not essential step
# Just to see the X_train data in it's columns. 
# Convert back to DataFrame with original column names
# X_train_scaled_df = pd.DataFrame(X_train_scaledStandard, columns=X_train.columns) # Otherwise X_train_scaledStandard is an array datatype
# print(X_train_scaled_df.describe())
# print('\n')

# # Plotting Before vs after sscaling
# fig, axes = plt.subplots(1, 2, figsize=(10,4))

# X_train.describe().T[['mean','std']].plot(kind='bar', ax=axes[0], title='Before Scaling') # NOTE Interesting way of plotting
# X_train_scaled_df.describe().T[['mean','std']].plot(kind='bar', ax=axes[1], title='After Scaling')

# plt.tight_layout()
# plt.show()

# Create cleaned dataset
# ================================================
# new_dataset = df.to_csv('students_clean.csv')


# *************** Data exploration and model building *********************
# ==============================================================================================
# Perform EDA
# ====================================================================================
# Correlation matrix, printed table and correlation heatmap plot
correlation_matrix = VisualisationEngine.correlation_Matrix(df_encoded)

#-------------------------------------------------------------------------------------------------------------------

# *************************UNNECESSARY SECTION**************************************
# Some analysis. Not prescribed in the report, holdover from assignment 1
StA = StudentAnalyser
# Isolate the following columns
df_grouped_1 = StA.isolateColumns(df, 'mental_health_rating', 'study_hours_per_day')

print('\nAverage study hours per day, grouped by mental health rating')
df_grouped_1_mean = StA.group_nFind(df_grouped_1, 'mental_health_rating', 'mean') 
print(df_grouped_1_mean)

print('\nMedian study hours per day, grouped by mental health rating')
df_grouped_1_median = StA.group_nFind(df_grouped_1, 'mental_health_rating', 'median')
print(df_grouped_1_median)

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


print('\nDetect outliers in social media usage')
sigma = StA.isolate_nStd(df, 'social_media_hours')
print('\nStandard dev. of social media hours', sigma)

# Find outlers
outlier_value, index_value = StA.findOutliers(df, 'social_media_hours', sigma)
print('\nOutlier values:', outlier_value)
print('Outlier index values:', index_value)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Perform EDA with 2D and 3D plots

Vis = VisualisationEngine
# Histogram of Study Time frequency
column_name = 'exam_score'
x_label = 'exam_score'
y_label = 'Number of Students'
histo_1 = Vis.histogram(df, column_name, 48, x_label, y_label)

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

df_counts = df[['gender', 'mental_health_rating']]
counts = df_counts.groupby('gender')['mental_health_rating'].value_counts().sort_index()
# df_save = counts.to_csv('bar_plot_data.csv') # File saved, had to save it
# df_bar = pd.read_csv('bar_plot_data.csv')

# bar = Vis.stacked_bar(df_bar, 'gender', 'mental_health_rating')

# Violin plot
violinplot_1 = Vis.violin_plot(df, 'diet_quality', 'exam_score')

parallel_coords_plot = Vis.parallel_coords_plot(df, 'exam_score', 'sleep_hours')

# Radar chart
radar = Vis.radar_plot(df, 'S1001')
# we can add an input student id program. We can compare two or more students.

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

    # For plotting the decision tree
feature_names_list = X_train.columns.tolist() 
class_names_list = ['0', '1']

# Decision tree plot
plt.figure(figsize=(24, 16)) # Adjust figure size for clarity
plot_tree(model_Decision,
          max_depth= 2,
          feature_names=feature_names_list, 
          class_names=class_names_list,
          filled=True, 
          rounded=True, 
          fontsize=9)
plt.title("Decision Tree Visualization")
plt.tight_layout()
# plt.show()
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

#NOTE Logistic Regression is the slightly better model

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
# plt.show()

display_D = ConfusionMatrixDisplay(confusion_matrix= confuse_Decision, display_labels= model_Decision.classes_)
display_D.plot(cmap= plt.cm.Blues)
plt.title('Confusion matrix for Decision tree')
# plt.show()

# Using ROC curves
# Compute ROC curve and AUC
fpr_L, tpr_L, thresholds_L = roc_curve(y_test, y_pred_Logic) 
fpr_D, tpr_D, thresholds_D = roc_curve(y_test, y_pred_Decision) 

auc_L = auc(fpr_L, tpr_L)
auc_D = auc(fpr_D, tpr_D)


# Plot ROC curve
plt.figure()

plt.plot(fpr_L, tpr_L, color= 'blue', label= f'ROC curve Logistic (AUC = {auc_L:.2f})') # Plot Logic 
plt.plot(fpr_D, tpr_D, color= 'red', label= f'ROC curve Decision (AUC = {auc_D:.2f})') # Plot Decision

plt.plot([0, 1], [0, 1], color= 'gray', linestyle= '--', label= 'Random Guess') # Diagonal line
plt.xlabel('False posivive rate')
plt.ylabel('Receiver Operating Characteristic')
plt.title('ROC curve of Two Models')
plt.legend(loc= 'lower right')
# plt.show()

roc_auc = roc_auc_score(y_test, y_pred_Logic)
print('\nROC AUC Score:', roc_auc)
# Doing the samw thing using roc auc (Trapezoidal rule)
print('AUC:', auc_L)
#NOTE AUC values are displayed in the ROC curve figure
#------------------------------------------------------------------------------------------------------------------------

# Apply SHAP and LIME. Apply it to the Logistic regression model, the better of the two
# Initialize the SHAP explainer, create an explainer object
explainer = shap.Explainer(model_Logic.predict, X_test_scaledStandard)

# Compute SHAP values for the test dataset
# shap_values = explainer(X_test_scaledStandard)

# Plot of SHAP values summary
# feature_names_list = X_train.columns.tolist() # list of feature names
# shap.summary_plot(shap_values, X_test_scaledStandard, feature_names= feature_names_list)

# For LIME 
# Initialize the Explainer
explainer_lime = LimeTabularExplainer(training_data= X_train_scaledStandard, feature_names= feature_names_list,
                                      class_names= ['Fail', 'Pass'], mode= 'classification')
# Select an instance
instance_idx = 0
instance = X_test_scaledStandard[instance_idx]
print('Instance to explain:', instance)

# Generate the explanation
explanation = explainer_lime.explain_instance(data_row= instance, predict_fn= model_Logic.predict_proba)

# Inspect the explanation
print(explanation.as_list())

# Visualising the LIME results
# plt.figure()
# plt.barh(feature_names_list, )