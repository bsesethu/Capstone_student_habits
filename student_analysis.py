import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer

import pickle


class DataLoaderIO_nException:
    def __init__(self, file_path):
        self.file_path = file_path
        
    # Error handling check whether file being loaded is of CSV format
    # (Function) Check file format
    def format_check(self): 
        try:
            if self.file_path[-4:] != '.csv':
                return 'Invalid file format used, please input a CSV format file'
            else:
                return self.file_path, ' Correct format detected'
        except TypeError:
            return 'Invalid file, please provide a valid file'      
        
    # (Function) Check file readability and permissions
    def readFile_check(self):
        try:
            with open(self.file_path, 'r') as f:
                contents = f.read()
                return 'File read successfully'
        except FileNotFoundError:
            return 'Error: The file was not found.'
        except PermissionError:
            return 'Error: Permission denied to read the file.'
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

# Implement a datacleaner class to:
class DataCleaner:
    # Part of Validation process
    def check_emptyDataFrame(df): # Check of an empty dataframe, hence an empty CSV file
        if(df.empty):
            print('Empty DataFrame, empty CSV file') 
        else:
            print('DataFrame, CSV file not empty')
        return df
    
    # Check for missing values
    def check_missing(df):
        df_check = df.isna()
        count1 = 0 # To count the number of missing and non missing values
        count2 = 0
        for col in df_check.columns:
            for row in df_check[col]:
                if row == True: # Returns a missing value
                    count1 += 1
                    # print(row, col) # Shows where the None values are. NOTE They are in the 'parental_education_level'
                else:
                    count2 += 1# Returns a non missing value
        print('Number of missing values', count1)
        print('Total number of values in the table', count1 + count2)

        missing_percentage = 100.0 * count1 / (count1 + count2)
        print(f'Proportion of missing data to the total: {round(missing_percentage, 2)}% of data is missing.') # percentage is well below 1%, Unlikely to have a major impact on overall findings
        # print(df.isna()) # Returns missing values as true
        
    def parentalNulls_fix(df): # Rename null/'None' values in 'parental_education_level' column to 'No Education' 
        rows_total = df.shape[0] # Total number of rows

        for i in range(rows_total):
            if pd.isna(df.loc[i, 'parental_education_level']): 
                df.loc[i, 'parental_education_level'] = 'No Education'
        return df
        # Could've done all this using one line
        # df['parental_education_level'].fillna('No Education', inplace= True)

    # Remove duplicates
    def dropDuplicates(df):
        df_noDuplicates = df.drop_duplicates() # Removes any row that is a duplicate of any other, leaving obly the first occurance
        return df_noDuplicates
    
    # Validate data ranges
    def dataTypes(df): # return the datatypes of each column
        return df.dtypes
    
        # Validate, check for negative values where applicable
    def check_negatives(df, column_name):
        has_negative = (df[column_name] < 0).any()
        return has_negative
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class StudentAnalyser:
    #  (Function) Isolate and return only two specific columns from the DF
    def isolateColumns(df, column_name1, column_name2):
        df_condensed = df[[column_name1, column_name2]] # If we are more than one column, we must have double square brackets
        return df_condensed
    
    # (Funtion) Group and calculate various aggregations
    def group_nFind(df, grouped_column, agg): # grouped_column is the column the groupby function is applied to. agg is the choice of aggregation used.
        if agg == 'mean':
            df_grouped = df.groupby([grouped_column]).mean().round(2) 
        elif agg == 'median':
            df_grouped = df.groupby([grouped_column]).median().round(2) 
        return df_grouped
    
    # (Funtion) Group the sleep hours into distinct intervals and find the average exam score for each group
    def groupDistinct_avgScore(df):
        sum_4 = 0; cnt4 = 0 # Initialize some counts for each group
        sum_6 = 0; cnt6 = 0
        sum_7 = 0; cnt7 = 0
        sum_9 = 0; cnt9 = 0
        sum_10 = 0; cnt10 = 0
        
        for i, row in (df.iterrows()): # Iterate over indices and rows
            if row['sleep_hours'] >= 3.0 and row['sleep_hours'] < 4.5:
                sum_4 += row['exam_score']
                cnt4 += 1
            elif row['sleep_hours'] >= 4.5 and row['sleep_hours'] < 6.0:
                sum_6 += row['exam_score']
                cnt6 += 1
            elif row['sleep_hours'] >= 6.0 and row['sleep_hours'] < 7.5:
                sum_7 += row['exam_score']
                cnt7 += 1
            elif row['sleep_hours'] >= 7.5 and row['sleep_hours'] < 9.0:
                sum_9 += row['exam_score']
                cnt9 += 1
            elif row['sleep_hours'] >= 9.0:
                sum_10 += row['exam_score']
                cnt10 += 1

        mean_4 = sum_4 / cnt4
        mean_6 = sum_6 / cnt6
        mean_7 = sum_7 / cnt7
        mean_9 = sum_9 / cnt9
        mean_10 = sum_10 / cnt10
        # total_count = cnt4+ cnt6 + cnt7 + cnt9 + cnt10 # To ensure we're accounting for all 1000 rows
        # print(total_count)
        new_df = pd.DataFrame({
            'sleep_intervals, hrs': ['3.0 - 4.5', '4.5 - 6.0', '6.0 - 7.5', '7.5 - 9.0', '>9.0'],
            'mean_exam_score': [mean_4, mean_6, mean_7, mean_9, mean_10]
        })
        return new_df
        
    # (Function) Find outlier values and indices for each column in the dataframe. NOTE it may need to be normally distributed, find out
    def findOutliers(df): 
        # Find the standard deviation of a each and all numeric column
        df_numeric = df.select_dtypes(include='number') # select only numeric dtype columns into a df
        for col in df_numeric.columns.to_list():  
            sigma = round(np.std(df[col]), 2) # Returns the standard deviation 
            
            mean_value = df[col].mean() # Mean value
            # print('Mean value:', mean_value)
            
            # Upper and lower boundaries, beyond which we can identify outliers
            upper_b = round(mean_value + (3 * sigma), 2)
            lower_b = round(mean_value - (3 * sigma), 2)
            # print('Upper and lower boundaries', upper_b, lower_b) # There are no values for 'social_media_hours' that are less than 0, lower_b is less than that . Hence there won't be any lower bound outliers
            
            # Find outliers
            rows_total = df_numeric.shape[0]
            index_val = []
            outlier_val = []
            for i in range(rows_total):
                if df.loc[i, col] > upper_b:
                    index_val.append(i) # Returns index of outlier
                    outlier_val.append(df.loc[i, col]) # Returns outlier value
                    
                elif df.loc[i, col] < lower_b:
                    index_val.append(i) # Returns index of outlier
                    outlier_val.append(df.loc[i, col]) # Returns outlier value
            
            print('\n')
            print(f'Detect outliers for {col} column')
            print(f'Standard deviation for {col} = {sigma}')

            # Find outlers
            print('\nOutlier values:', outlier_val)
            print('Outlier index values:', index_val)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class VisualisationEngine:
    # (Function) Create a histogram of one of the columns of a dataframe
    def histogram(df, column_name, num_of_bins, xlabel, ylabel): # specify these characteristics of your histogram
        study_time = df[column_name]
        title = 'Histogram of ' + column_name
        plt.hist(study_time, bins= num_of_bins, color= 'skyblue', edgecolor= 'black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
    
    # (Function) Create a scatter plot of one column vs another
    def scatterPlot(df, column_name1, column_name2): # Specify these characteristics of your skatter plot
        x = df[column_name1]
        y = df[column_name2]
        x_threshold = np.mean(x)
        y_threshold = np.mean(y)
        title_name = 'Scatter plot of ' + column_name1 +  ' vs ' +  column_name2

        plt.scatter(x, y, alpha= 0.7, cmap= 'viridis', marker= '.')
        plt.xlabel(column_name1)
        plt.ylabel(column_name2)
        plt.title(title_name)
        
        # Add horizontal and vertical lines for the quadrants. Each line is the mean value.
        # plt.axvline(x=x_threshold, color='r', linestyle='--', label=f'X Threshold ({x_threshold:.2f})')
        # plt.axhline(y=y_threshold, color='r', linestyle='--', label=f'Y Threshold ({y_threshold:.2f})')

        # # Add quadrant labels
        # plt.text(x_threshold * 0.7, y_threshold * 1.2, 'Q2', fontsize=16, ha='center', va='center')
        # plt.text(x_threshold * 1.3, y_threshold * 1.2, 'Q1', fontsize=16, ha='center', va='center')
        # plt.text(x_threshold * 0.7, y_threshold * 0.6, 'Q3', fontsize=16, ha='center', va='center')
        # plt.text(x_threshold * 1.3, y_threshold * 0.6, 'Q4', fontsize=16, ha='center', va='center')

        plt.show()
    
    # (Function) Create boxplots of scores by diet quality
    def boxplots(df, column_1, column_2):
        sns.boxplot(x= column_1, y= column_2, data= df[[column_1, column_2]])
        title_str = 'Box Plot of ' + column_1 + ' by ' + column_2
        plt.title(title_str)
        plt.xlabel(column_1)
        plt.ylabel(column_2)
        plt.show()
        
    def violin_plot(df, column_1, column_2):
        """
        Create a violin plot
        Args:
            df: DataFrame
            column_1 (string): Name of categories column
            column_2 (string): Name of values column
        Return:
            None
        """
        # Rename the diet_quality values from numeric categorical string value categorical
        df[column_1].replace({1 : 'Good', 2 : 'Fair', 3 : 'Poor'}, inplace= True)
        
        sns.violinplot(x= column_1, y= column_2, data= df[[column_1, column_2]])
        title_str = 'Violin Plot of ' + column_1 + ' by ' + column_2
        plt.title(title_str)
        plt.xlabel(column_1)
        plt.ylabel(column_2)
        plt.show()
    
    def threeD_plot(x, y, z, title):
        fig = plt.figure()
        ax = plt.axes(projection= '3d')
        
        ax.scatter(x, y, z)
        ax.set_title(title)
        plt.show()
        
    def stacked_bar(df): 
        """ 
        Create a stacked bar chart of gender by mental_health_rating
        Args:
            df: DataFrame
        Returns:
            None
        """
        # Change genders from numeric to categorical
        
        # Group data by gender and mental health rating, count occurrences
        grouped = df.groupby(['gender', 'mental_health_rating']).size().unstack(fill_value=0)

        # Plot stacked bar
        grouped.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='coolwarm')

        plt.title('Distribution of Mental Health Ratings by Gender', fontsize=14, pad=10)
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Number of Students', fontsize=12)
        plt.legend(title='Mental Health Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
                
    def parallel_coords_plot(df, class_value, *args):
        """
        Create parallel coordinates plot
        Args:
            df: DataFrame
            class_value (String): Class column name
            *args (Strings): Features column names
        Returns
            None
        """
        df = df[[class_value, *args]]
        plt.figure(figsize=(8, 6))
        parallel_coordinates(df, class_column= class_value, color= ['blue', 'green', 'red', 'yellow'])
        plt.title('Parallel Coordinates Plot')
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.grid(True)
        plt.show()
    
    def correlation_Matrix(df):
        """
        Prints the correlation matrix and plots the correlation matrix heatmap
        Args:
            df: Dataframe
        Returns
            None
        """
        # Correlation matrix
        df_corr = df.drop(['student_id'], axis= 1) # Removing the non numeric columns
        correlation = df_corr.corr()
        print('\nCorrelation Martix')
        print(correlation)

        # Plot the correlation matrix
        fig = plt.figure()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title('Correlation matrix heatmap')
        plt.show()
        
    def radar_plot(df, student_id):
        """
        
        """
        categories = ['exam_score', 'study_hours_per_day', 'netflix_hours', 'sleep_hours', 'social_media_hours']
        # Use one student's values at a time
        df_values = df[['student_id', 'exam_score', 'study_hours_per_day', 'netflix_hours', 'sleep_hours', 'social_media_hours']]
        
        # Finding the values for a specific student
        for ind, val in enumerate(df_values['student_id']):
            if val == student_id:
                # Create a list of values for this student
                values = [df_values.iloc[ind, 1] / 10, df_values.iloc[ind, 2], df_values.iloc[ind, 3], df_values.iloc[ind, 4], df_values.iloc[ind, 5]] # First value (exam_score) / 10 to fit the scale
            
        # Repeat the first value to close the circle
        values = np.concatenate((values, [values[0]]))

        # Set the angle for each axis
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        # Create a polar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True)) # Making plot coordinates polar

        # Plot the data
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        # Set labels for each axis
        # ax.set_yticklabels([]) # Hide radial ticks
        # ax.set_xticks(angles[:-1])
        # ax.set_xticklabels(categories)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels=categories)

        # Add a title
        plt.title(f'Radar plot of student ID {student_id}')

        plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class MakeModel:
    def target_toCategorical(y):
        # For Logistic regression
        # Must make target features categorical. 
        # Use 0: Fail is y < 65
        #     1: Pass is y >= 65
        y_categorical= [] # New target variable
        for row in y:
            if row < 65:
                y_categorical.append(0)
            elif row >= 65:
                y_categorical.append(1)
            else:
                y_categorical.append(None)
        return y_categorical
    
    # (Function) Convert specific string value to an integer using kwargs. Hence we can then include string value coulumns in the model
    def string_to_Int(df, column_name, **kwargs):
        for i, value in enumerate(kwargs.values(), start= 1):
            df.loc[df[column_name] == value, column_name] = i
        return df
    
    def decision_tree_setup(X_train, y_train):
        """
        
        """
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
        gridS.fit(X_train, y_train)

        # Get the best model from gridS
        model_Decision = gridS.best_estimator_
        
        return model_Decision
    
    def evaluateNPlot_Logistic_confusion(y_test, y_pred_Logic, model_Logic):
        """
        
        """
        confuse_Logic = confusion_matrix(y_test, y_pred_Logic)

        TP_L = confuse_Logic[1, 1] # extract True Negatives
        FP_L = confuse_Logic[0, 1] # extract False Positives
        FN_L = confuse_Logic[1, 0] # extracy False Negative
        # Calculate precition and recall
        precision_Logic = TP_L / (TP_L + FP_L)
        recall_Logic = TP_L / (TP_L + FN_L)
        
        # Plot confusion matrix
        display_L = ConfusionMatrixDisplay(confusion_matrix= confuse_Logic, display_labels= model_Logic.classes_)
        display_L.plot(cmap= plt.cm.Blues) # Using blue colour map
        plt.title('Confusion matrix for Logistic Regression')
        plt.show()
        
        return precision_Logic, recall_Logic

    def evaluateNPlot_Decision_confusion(y_test, y_pred_Decision, model_Decision):
        """
        
        """
        confuse_Decision = confusion_matrix(y_test, y_pred_Decision)

        TP_D = confuse_Decision[1, 1] # extract True Positives
        FP_D = confuse_Decision[0, 1] # extract False Positives
        FN_D = confuse_Decision[1, 0] # extracy False Negative

        precision_Decision = TP_D / (TP_D + FP_D)
        recall_Decision = TP_D / (TP_D + FN_D)
        
        # Plot confusion matrix
        display_D = ConfusionMatrixDisplay(confusion_matrix= confuse_Decision, display_labels= model_Decision.classes_)
        display_D.plot(cmap= plt.cm.Blues)
        plt.title('Confusion matrix for Decision tree')
        plt.show()
        
        return precision_Decision, recall_Decision

    def ROC_curvePlot(y_test, y_pred_Logic, y_pred_Decision):
        """
        
        """
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
        plt.title('ROC curves of Two Models')
        plt.legend(loc= 'lower right')
        plt.show()

        roc_auc_L = roc_auc_score(y_test, y_pred_Logic)
        print('\nROC AUC Score (Logistic):', roc_auc_L)
        
        roc_auc_D = roc_auc_score(y_test, y_pred_Decision)
        print('\nROC AUC Score (Decision):', roc_auc_D)

        # Doing the same thing using roc auc (Trapezoidal rule)
        print('AUC (Logistic):', auc_L)
        print('AUC (Decision):', auc_D)
        #NOTE AUC values are displayed in the ROC curve figure

    def LIME_explanation_NPlot(X_train_scaledStandard, X_test_scaledStandard, model_Logic, feature_names_list):
        """
        
        """
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
        # Get the explanation as a list of (feature, weight) tuples
        exp_list = explanation.as_list()
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]

        # Sort features by absolute weight for better visualization
        sorted_indices = np.argsort(np.abs(weights))[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, sorted_weights, color=['red' if w < 0 else 'green' for w in sorted_weights])
        plt.xlabel("Feature Importance (LIME Weight)")
        plt.ylabel("Feature")
        plt.yticks(fontsize= 6)
        plt.title(f"LIME Explanation for Instance Predicted")
        plt.gca().invert_yaxis() # Puts most important feature at the top
        plt.show()
#==========================================================================================================================================================================
# FIN