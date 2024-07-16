import pandas as pd
import numpy as np

def preprocess_data(data):
    try:
        print("Initial data shape:", data.shape)
        
        # Fill missing values
        print("Filling missing values for 'MonthlyIncome' and 'NumberOfDependents'...")
        data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())
        data['NumberOfDependents'] = data['NumberOfDependents'].fillna(data['NumberOfDependents'].median())
        
        print("Checking for column 'NumberOfTime30-59DaysPastDueNotWorse'...")
        # Ensure the correct column name is used
        if 'NumberOfTime30-59DaysPastDueNotWorse' in data.columns:
            data['NumberOfTime30_59DaysPastDueNotWorse'] = data['NumberOfTime30-59DaysPastDueNotWorse']
        else:
            print("Column 'NumberOfTime30-59DaysPastDueNotWorse' not found.")

        print("Checking for column 'NumberOfTime60-89DaysPastDueNotWorse'...")
        if 'NumberOfTime60-89DaysPastDueNotWorse' in data.columns:
            data['NumberOfTime60_89DaysPastDueNotWorse'] = data['NumberOfTime60-89DaysPastDueNotWorse']
        else:
            print("Column 'NumberOfTime60-89DaysPastDueNotWorse' not found.")
        
        # Handle the duplicate edges issue in qcut
        print("Binning 'RevolvingUtilizationOfUnsecuredLines'...")
        data['RevolvingUtilizationOfUnsecuredLines'] = pd.qcut(data['RevolvingUtilizationOfUnsecuredLines'].values, 5, duplicates='drop').codes
        
        print("Binning 'age'...")
        data['age'] = pd.qcut(data['age'].values, 5, duplicates='drop').codes
        
        print("Binning 'NumberOfTime30_59DaysPastDueNotWorse'...")
        data['NumberOfTime30_59DaysPastDueNotWorse'] = pd.cut(data['NumberOfTime30_59DaysPastDueNotWorse'].values, bins=[-1, 0, 1, 2, 3, 4, 5, 6], labels=False)
        
        print("Binning 'DebtRatio'...")
        data['DebtRatio'] = pd.qcut(data['DebtRatio'].values, 5, duplicates='drop').codes
        
        print("Binning 'MonthlyIncome'...")
        data['MonthlyIncome'] = pd.qcut(data['MonthlyIncome'].values, 5, duplicates='drop').codes
        
        print("Binning 'NumberOfOpenCreditLinesAndLoans'...")
        data['NumberOfOpenCreditLinesAndLoans'] = pd.qcut(data['NumberOfOpenCreditLinesAndLoans'].values, 5, duplicates='drop').codes
        
        print("Binning 'NumberOfTimes90DaysLate'...")
        data['NumberOfTimes90DaysLate'] = pd.cut(data['NumberOfTimes90DaysLate'].values, bins=[-1, 0, 1, 2, 3, 4, 5, 6], labels=False)
        
        print("Binning 'NumberRealEstateLoansOrLines'...")
        data['NumberRealEstateLoansOrLines'] = pd.cut(data['NumberRealEstateLoansOrLines'].values, bins=[-1, 0, 1, 2, 3, 4, 5, 6], labels=False)
        
        print("Binning 'NumberOfTime60_89DaysPastDueNotWorse'...")
        data['NumberOfTime60_89DaysPastDueNotWorse'] = pd.cut(data['NumberOfTime60_89DaysPastDueNotWorse'].values, bins=[-1, 0, 1, 2, 3], labels=False)
        
        print("Binning 'NumberOfDependents'...")
        data['NumberOfDependents'] = pd.cut(data['NumberOfDependents'].values, bins=[-1, 0, 1, 2, 3, 4], labels=False)

        # Generate dummy variables
        print("Generating dummy variables...")
        data = pd.get_dummies(data, columns=[
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30_59DaysPastDueNotWorse', 
            'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
            'NumberRealEstateLoansOrLines', 'NumberOfTime60_89DaysPastDueNotWorse', 'NumberOfDependents'
        ])

        print("Final processed data shape:", data.shape)
        return data
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage for testing
    print("Reading input data...")
    input_data = pd.read_csv("data/cs-test.csv")
    print("Starting preprocessing...")
    preprocessed_data = preprocess_data(input_data)
    print("Preprocessing completed.")
    print(preprocessed_data.head())
