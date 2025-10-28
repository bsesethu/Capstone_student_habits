import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
    
    # (Function) Find the standard deviation of a single column
    def isolate_nStd(df, column_name):
        sigma = round(np.std(df[column_name]), 2) # Returns the standard deviation 
        return sigma
    
    # (Function) Find outlier values and indices of a single column. NOTE it may need to be normally distributed, find out
    def findOutliers(df, column_name, sigma): #NOTE Returns two variables, sigma is the standard deviation. 
        mean_value = df[column_name].mean() # Mean value
        print('Mean value:', mean_value)
        
        # Upper and lower boundaries, beyond which we can identify outliers
        upper_b = round(mean_value + (3 * sigma), 2)
        lower_b = round(mean_value - (3 * sigma), 2)
        print('Upper and lower boundaries', upper_b, lower_b) # There are no values for 'social_media_hours' that are less than 0, lower_b is less than that . Hence there won't be any lower bound outliers
        
        # Find outliers
        rows_total = df.shape[0]
        index_val = []
        outlier_val = []
        for i in range(rows_total):
            if df.loc[i, column_name] > upper_b:
                index_val.append(i) # Returns index of outlier
                outlier_val.append(df.loc[i, column_name]) # Returns outlier value
                
            elif df.loc[i, column_name] < lower_b:
                index_val.append(i) # Returns index of outlier
                outlier_val.append(df.loc[i, column_name]) # Returns outlier value
        
        return outlier_val, index_val
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
        title_name = 'Skatter plot of ' + column_name1 +  ' vs ' +  column_name2

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
            matplotlib.axes object
        """
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
        
    def stacked_bar(counts, gender, mental): # Rewrite """ """"
        """ 
        Create a stacked bar chart
        Args:
            df: DataFrame
            gender (string): Category column
            mental (string): mental_health_rating column
        Returns:
            None
        """
        l1 = l2 = l3 = l4 = l5 = l6 = l7 = l8 = l9 = l10 = [] # Corresponds to each level
        levels = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
        gender_category = ['Female', 'Male', 'Other']
        for i in range(10): # iterate over the first 10 rows
            if counts.iloc[i, 1] == i + 1 and counts.iloc[i, 0] == 1:
                if counts.iloc[i + 10, 1] == i + 1 and counts.iloc[i + 10, 0] == 2: 
                    if counts.iloc[i + 20, 1] == i + 1 and counts.iloc[i + 20, 0] == 3:
                        if i == 0:
                            l1.append(counts.iloc[i, 2])
                            l1.append(counts.iloc[i + 10, 2])
                            l1.append(counts.iloc[i + 20, 2])                           
                        elif i == 1:
                            l2.append(counts.iloc[i, 2])
                            l2.append(counts.iloc[i + 10, 2])
                            l2.append(counts.iloc[i + 20, 2])
                        elif i == 2:
                            l3.append(counts.iloc[i, 2])
                            l3.append(counts.iloc[i + 10, 2])
                            l3.append(counts.iloc[i + 20, 2])
                        elif i == 3:
                            l4.append(counts.iloc[i, 2])
                            l4.append(counts.iloc[i + 10, 2])
                            l4.append(counts.iloc[i + 20, 2])
                        elif i == 4:
                            l5.append(counts.iloc[i, 2])
                            l5.append(counts.iloc[i + 10, 2])
                            l5.append(counts.iloc[i + 20, 2])
                        elif i == 5:
                            l6.append(counts.iloc[i, 2])
                            l6.append(counts.iloc[i + 10, 2])
                            l6.append(counts.iloc[i + 20, 2])
                        elif i == 6:
                            l7.append(counts.iloc[i, 2])
                            l7.append(counts.iloc[i + 10, 2])
                            l7.append(counts.iloc[i + 20, 2])
                        elif i == 7:
                            l8.append(counts.iloc[i, 2])
                            l8.append(counts.iloc[i + 10, 2])
                            l8.append(counts.iloc[i + 20, 2])
                        elif i == 8:
                            l9.append(counts.iloc[i, 2])
                            l9.append(counts.iloc[i + 10, 2])
                            l9.append(counts.iloc[i + 20, 2])
                        elif i == 9:
                            l10.append(counts.iloc[i, 2])
                            l10.append(counts.iloc[i + 10, 2])
                            l10.append(counts.iloc[i + 20, 2])
        # Now to put this into a df
        data_new = {
            'Gender': gender_category,
            'Mental health rating 1': l1,
            'Mental health rating 2': l2,
            'Mental health rating 3': l3,
            'Mental health rating 4': l4,
            'Mental health rating 5': l5,
            'Mental health rating 6': l6,
            'Mental health rating 7': l7,
            'Mental health rating 8': l8,
            'Mental health rating 9': l9,
            'Mental health rating 10': l10,
        }
        print(data_new)
        
        df_bar = pd.DataFrame(data_new)
        # counts.drop('gender', inplace= True) # LAter
        df_new = df_bar.set_index(gender)
        df_new.plot(kind= 'bar', stacked= True, figsize= (8, 6))
        
        print(df_bar)
        
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
        print(correlation)

        # Plot the correlation matrix
        fig = plt.figure()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title('Correlation matrix heatmap')
        plt.show()
        
    def radar_plot(df, student_id):
        """
        
        """
        categories = ['exam_score', 'mental_health_rating', 'diet_quality', 'sleep_hours', 'social_media_hours']
        # Use one student's values at a time
        df_values = df[['student_id', 'exam_score', 'mental_health_rating', 'diet_quality', 'sleep_hours', 'social_media_hours']]
        
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
        ax.set_yticklabels([]) # Hide radial ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Add a title
        plt.title(f'Radar plot of student ID {student_id}')

        plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ScorePredictor:
    # (Function) Model function. Linear regression
    def model_LinearReg(df, X): # x_new is the new individual student characteristics. X is the input features, prediction_num is the index of the prediction required
        y = df['exam_score']
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = LinearRegression() 
        model.fit(X_train, y_train)

        y_prediction = model.predict(X_test)
        return y_prediction, y_test
   
    # (Function) Convert specific string value to an integer using kwargs. Hence we can then include string value coulumns in the model
    def string_to_Int(df, column_name, **kwargs):
        for i, value in enumerate(kwargs.values(), start= 1):
            df.loc[df[column_name] == value, column_name] = i
        return df
    
    # (Function) Creates DF of actual and predicted values and their percentage differance
    def compareActual_toPredicted(df, X, actualColumn_name): # actualValues is the actual value column name
        df_test_case_1 = X
        y = df[actualColumn_name]
        length_df = df_test_case_1.shape[0]
        list_predConsumption = []
        model = LinearRegression()
        model.fit(X, y)

        for i in range(length_df):
            list_predConsumption.append(round(model.predict(df_test_case_1)[i], 2)) # Returns list of predicted values

        # Convert list to df
        df_prediction = pd.DataFrame({'Predicted Values': list_predConsumption})  
        df_actual = df[actualColumn_name] # Original Energy Consumption column
        # print('\n', df_prediction, df_Energy_Cons)

        # Find percentage difference of predicted value from the actual value
        df_joined = df_prediction.join(df_actual) 
        # print('\n', df_joined)
        df_joined['Percent_Difference'] = round(((df_joined[actualColumn_name] - df_joined['Predicted Values']) / df_joined[actualColumn_name]) * 100.0, 2)
        return df_joined
    
        # (Function) Saving model using pickle
    def save_usingPickle(df, X, actualColumn_name): # actualValues is the actual value column nam
        y = df[actualColumn_name]
        # Trained model
        model = LinearRegression()
        model.fit(X, y)
        model_file = 'trained_model.pkl'
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {model_file}")
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
    
    