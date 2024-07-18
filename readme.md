# 🧮 Credit Scoring Model API

This repository contains a FastAPI application for predicting credit default probabilities using a pre-trained Random Forest model.

## Directory Structure

```
├── a_preprocessing_featurepipeline.py
├── b_model_gen.py
├── c_model_prediction.py
├── main.py
├── Dockerfile
├── requirements.txt
├── models
│ └── (ignored) random_forest_model.pkl
├── data
│ ├── cs-training.csv
│ └── cs-test.csv
└── README.md
```


## Setup

1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Generate the model file**

    ```bash
    python a_preprocessing_featurepipeline.py
    python b_model_gen.py
    ```

5. **Build the Docker container**

    ```bash
    docker build -t credit-scoring-api .
    ```

6. **Run the Docker container**

    ```bash
    docker run -d -p 8888:8888 credit-scoring-api
    ```

7. **Test the API**

    Use the following example payload to test the `/predict` endpoint using `curl`:

    ```bash
    curl -X POST "http://0.0.0.0:8888/predict" -H "Content-Type: application/json" -d '{
        "ID": 1,
        "RevolvingUtilizationOfUnsecuredLines": 0.766126609,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.802982129,
        "MonthlyIncome": 9120,
        "NumberOfOpenCreditLinesAndLoans": 13,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 6,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }'
    ```


## .gitignore

```
/models/*
!/models
.ipynb_checkpoints/
__pycache__/
```




# 🧱 Original Task

'''
["metadata":{ 
"author": "Nadia Zaheer",
"gitlink": "https://github.com/nadiazk/assessments.git",
"version": "18062024v2",
"department": "Data Science and AI" 
}]
'''



**Congratulations** for making it to the Challenge stage of the application process! 

The goal of the challenge is to give you a small taste of what it's like to work with our data as well as an opportunity for us to understand your process! **Understanding your process is just as important to us as your answers.**

*Before continuing, please check to make sure that the exercise is in your preferred language and that the filename is labeled with your name (e.g., if your name is Lucy, the filename should read "Vanquis ML Exercise - Python [Lucy].ipynb")*. You will see the name as VBL-PFG in some documentation as well, so don't get confused.

Please spend no more than 3 hours total on the exercise.

**About this notebook:**

This notebook from the current team of data scientists at Vanquis creates an ML model that predicts the probability that our customer will experience financial distress (delinquency) in the next two years. That is, they won't be able to pay back the credit/loan. This notebook uses a mock dataset of credit repayment difficulty rates among customers. Please note that the topic, data and problem are reflective of cases we have solved in the past.

---

**Background:**
Vanquis plays a crucial role in market economies. We, as a subprime lender, decide who can get finance and on what terms and can make or break investment decisions. We empathise with our customers. For markets and society to function, individuals and companies need access to credit. Credit scoring algorithms, which make a guess at the probability of default, are the method we use to determine whether or not a loan or credit should be granted. This model is within the domain of credit scoring decisions i.e. it predicts the probability that somebody will experience financial distress in the next two years.

Historical data are provided on 250,000 borrowers.

---
**About the data:**
Data is in two csv files:
cs-training.csv,
cs-test.csv

The variables are the following: SeriousDlqin2yrs is the target variable

**SeriousDlqin2yrs** Person experienced 90 days past due delinquency or worse (Target variable / label)

**RevolvingUtilizationOfUnsecuredLines**: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits

**age** Age of borrower in years

**NumberOfTime30-59DaysPastDueNotWorse**: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.

**DebtRatio**: Monthly debt payments, alimony,living costs divided by monthy gross income

**MonthlyIncome**: Monthly income

**NumberOfOpenCreditLinesAndLoans**: Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)

**NumberOfTimes90DaysLate**: Number of times borrower has been 90 days or more past due.

**NumberRealEstateLoansOrLines**: Number of mortgage and real estate loans including home equity lines of credit

**NumberOfTime60-89DaysPastDueNotWorse**: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.

**NumberOfDependents**: Number of dependents in family excluding themselves (spouse, children etc.)

---
NDA: By undertaking this challenge, you are automatically bound by the non-disclosure agreement of Vanquis.



**Objective of your challenge:**
Vanquis data scientists have developed this scoring model and handed over this notebook to you. The goal of this tech challenge is to deploy and productionize this ML scoring model. The model predicts the probability that somebody will experience financial distress in the next two years. 

REMEMBER: Task is not to build the best model but to take it the next step i.e. properly ship and deploy this machine learning model so that we can continuously improve our models and ship out experiments faster.

**Your Deliverable:** 

*   Your submission is an API that is a train-offline and predict-realtime via rest API POST request e.g.
```@app.post("/predict", response_model=PredictResponse)```
* input_json: 
```
[
    {
        "ID": "1",
        "RevolvingUtilizationOfUnsecuredLines": "0.766126609",
        "age": "45",
        "NumberOfTime30-59DaysPastDueNotWorse": "2",
        "DebtRatio": "0.802982129",
        "MonthlyIncome": "9120",
        "NumberOfOpenCreditLinesAndLoans": "13",
        "NumberOfTimes90DaysLate": "0",
        "NumberRealEstateLoansOrLines": "6",
        "NumberOfTime60-89DaysPastDueNotWorse": "0",
        "NumberOfDependents": "2",
        "SeriousDlqin2yrs": "1"
    },
    {
        "ID": "2",
        "RevolvingUtilizationOfUnsecuredLines": "0.957151019",
        "age": "40",
        "NumberOfTime30-59DaysPastDueNotWorse": "0",
        "DebtRatio": "0.121876201",
        "MonthlyIncome": "2600",
        "NumberOfOpenCreditLinesAndLoans": "4",
        "NumberOfTimes90DaysLate": "0",
        "NumberRealEstateLoansOrLines": "0",
        "NumberOfTime60-89DaysPastDueNotWorse": "0",
        "NumberOfDependents": "1",
        "SeriousDlqin2yrs": "0"
    }]
```
* PredictResponse in json:
```
[{
    "Id": "1",
    "Probability": 0.29
  },
  {
    "Id": "2",
    "Probability": 0.03
  }]
```
Ensure to have the following modules for submission. Please email the git repo link, which sould be private. Oh! but don't forget to give us the access when you email the submission link! :)

1.   00_preprocessing_featurepipeline.py <-- should be the pipeline to preprocess the data into what model accepts
2.   01_model_gen.py [optional if you have time]
3.   02_model_prediction.py
4.   wrap the model in an API, dockerise and containerize this application so that we can readily test the rest API once we run your docker container using a post request in json.
5.  We love postman to test. 
6.  Make an output folder. It should attach a snapshot of the post request prediction output from model
7. Ensure to stick to OOP and make your code modular and reusable.
8. We like staying organized so we want to see how you organize your directory structure
9. readme.md should be self-explantory

---

Data Scientists at Vanquis used a random forest classifier for two reasons: firstly, because it would allow to quickly and easily change the output to a simple binary classification problem. Secondly, because the predict_proba functionality allows to output a probability score (probability of 1), this score is what is used for predicting the probability of 90 days past due delinquency or worse in 2 years time by the stakeholder teams and is built within product.

Furthermore, data scienitists have predominantly adopted a quantiles based approach in order to streamline the process as much as possible so that hypothetical credit checks can be returned as easily and as quickly as possible.

**Deadline** 5 days **Time Cap:** [3 hours MAX] We do NOT want you to spend more than 3 hours time on this. In case time runs out, please make a list of things you would like to do that we can discuss in the tech interview. This task is designed to give you insight into the kind of problems we work on. This is a take home task, where we want to see how you solve problems in real life when you have ALL the resources at your hand.

**Notes and Tips:**
* This notebook can be run on google colab and takes ~2 mins. 
* Feel free to run and save the model pickle/joblib from colab to save time
* Not looking for perfectionism as we want to ship things fast, ideally it should be a working piece of application.
* Wrap it in some API web framework (Hint: Fast API or Flask API). No need for a front end HTML or streamlit. Keep it simple! 
* If you want to deploy on some cloud, feel free to do so. Modify your run instructions accordingly
* Feel free to ask questions
* If in doubt, feel free to make an assumption and the reasoning behind it. Ideally, I would prefer if you take note of it in your readme.md

**Preparing for the Tech interview:**

Some design considerations to keep on top of your mind for the next stage of tech interview:
* How would you test for incorrect user input?
* If you had more time what would you do? 
* How would you scale your API?
* How would you improve the architecture and your submitted design?
* How would you build monitoring around this? 
* Comment on this modelling approach taken by the data scientists. 
* What would you improve?



---
# 🔬 Project Review
---


---
# 🍏 (1/8) cs-training.csv
---

- **Description**: This file contains the training data used for building the credit scoring model. It includes various financial and demographic attributes of the borrowers along with the target variable indicating if the borrower experienced financial distress.
- **Columns**:
  - **SeriousDlqin2yrs**: Binary target variable indicating if the borrower experienced 90 days past due delinquency or worse (1: Yes, 0: No).
  - **RevolvingUtilizationOfUnsecuredLines**: The ratio of total balance on credit cards and personal lines of credit to the sum of credit limits.
  - **age**: The age of the borrower in years.
  - **NumberOfTime30-59DaysPastDueNotWorse**: The number of times the borrower has been 30-59 days past due but no worse in the last 2 years.
  - **DebtRatio**: The ratio of monthly debt payments, alimony, living costs to monthly gross income.
  - **MonthlyIncome**: The monthly income of the borrower.
  - **NumberOfOpenCreditLinesAndLoans**: The number of open loans (e.g., car loan or mortgage) and lines of credit (e.g., credit cards).
  - **NumberOfTimes90DaysLate**: The number of times the borrower has been 90 days or more past due.
  - **NumberRealEstateLoansOrLines**: The number of mortgage and real estate loans including home equity lines of credit.
  - **NumberOfTime60-89DaysPastDueNotWorse**: The number of times the borrower has been 60-89 days past due but no worse in the last 2 years.
  - **NumberOfDependents**: The number of dependents in the family excluding the borrower (e.g., spouse, children).

---
# 🍎 (2/8) cs-test.csv
---

- **Description**: This file contains the test data used for evaluating the credit scoring model. It includes similar attributes as the training data but without the target variable, which the model needs to predict.
- **Columns**:
  - **SeriousDlqin2yrs**: This column is empty as it is supposed to be predicted by the model.
  - **RevolvingUtilizationOfUnsecuredLines**: The ratio of total balance on credit cards and personal lines of credit to the sum of credit limits.
  - **age**: The age of the borrower in years.
  - **NumberOfTime30-59DaysPastDueNotWorse**: The number of times the borrower has been 30-59 days past due but no worse in the last 2 years.
  - **DebtRatio**: The ratio of monthly debt payments, alimony, living costs to monthly gross income.
  - **MonthlyIncome**: The monthly income of the borrower.
  - **NumberOfOpenCreditLinesAndLoans**: The number of open loans (e.g., car loan or mortgage) and lines of credit (e.g., credit cards).
  - **NumberOfTimes90DaysLate**: The number of times the borrower has been 90 days or more past due.
  - **NumberRealEstateLoansOrLines**: The number of mortgage and real estate loans including home equity lines of credit.
  - **NumberOfTime60-89DaysPastDueNotWorse**: The number of times the borrower has been 60-89 days past due but no worse in the last 2 years.
  - **NumberOfDependents**: The number of dependents in the family excluding the borrower (e.g., spouse, children).




---
# 🎲 (3/8) Model (sklearn.ensemble.**RandomForestClassifier**)
---

## Why RandomForestClassifier was Chosen

**Advantages:**
1. **Robustness**: Random forests are robust against overfitting, especially when dealing with large datasets with many features.
2. **Versatility**: They can handle both classification and regression tasks.
3. **Feature Importance**: Random forests provide estimates of feature importance, helping to understand the impact of different variables.
4. **Handling Missing Values**: They can handle missing data effectively by using median values and proximity in trees.
5. **Ensemble Learning**: By combining the predictions of multiple trees, random forests improve predictive accuracy.

**Disadvantages:**
1. **Complexity**: They can be computationally intensive and require more memory compared to simpler models.
2. **Interpretability**: While more interpretable than some black-box models, understanding the specific impact of each feature can still be challenging.

## Current Hyperparameters

- `n_estimators=1000`: The number of trees in the forest. A higher number can improve performance but increases computational cost.
- `random_state=42`: Ensures reproducibility by controlling the randomness in the model.
- Default values are used for other parameters such as `criterion='gini'`, `max_depth=None`, `min_samples_split=2`, etc.

## How the Dataset was Split

The dataset was divided into training and test sets:
- **Training Data**: Contained in `cs-training.csv` and used to train the model.
- **Test Data**: Contained in `cs-test.csv` and used to evaluate the model's performance.

## How RandomForestClassifier Builds and Works as a Model

1. **Bootstrap Sampling**: RandomForestClassifier creates multiple subsets of the training data by sampling with replacement (bootstrap sampling).
2. **Decision Trees**: For each subset, it trains a decision tree. Each tree is built using a random subset of features, which introduces diversity.
3. **Node Splitting**: Each tree splits nodes based on the best possible criteria (e.g., Gini impurity) to reduce impurity in the resulting subsets.
4. **Tree Aggregation**: The model aggregates the predictions of all trees. For classification tasks, it uses majority voting to determine the final class. For regression tasks, it averages the predictions.


## Detailed Hyperparameters Description

- **`n_estimators`**: The number of trees in the forest.
- **`criterion`**: The function to measure the quality of a split (`'gini'`, `'entropy'`, or `'log_loss'`).
- **`max_depth`**: The maximum depth of each tree.
- **`min_samples_split`**: The minimum number of samples required to split an internal node.
- **`min_samples_leaf`**: The minimum number of samples required to be at a leaf node.
- **`min_weight_fraction_leaf`**: The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- **`max_features`**: The number of features to consider when looking for the best split.
- **`max_leaf_nodes`**: Grow trees with a maximum number of leaf nodes.
- **`min_impurity_decrease`**: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
- **`bootstrap`**: Whether bootstrap samples are used when building trees.
- **`oob_score`**: Whether to use out-of-bag samples to estimate the generalization accuracy.
- **`n_jobs`**: The number of jobs to run in parallel.
- **`random_state`**: Controls the randomness of the estimator.
- **`verbose`**: Controls the verbosity when fitting and predicting.
- **`warm_start`**: When set to `True`, reuse the solution of the previous call to fit and add more estimators to the ensemble.
- **`class_weight`**: Weights associated with classes.
- **`ccp_alpha`**: Complexity parameter used for Minimal Cost-Complexity Pruning.
- **`max_samples`**: If bootstrap is `True`, the number of samples to draw from X to train each base estimator.



---
# 🥞 (4/8) Preprocessing Feature Pipeline (a_preprocessing_featurepipeline.py)
---

## Importing Libraries

### Explanation
I started by importing the necessary libraries:

- **pandas**: Essential for data manipulation tasks.
- **numpy**: Provides support for numerical operations.

### Design Decisions
- **pandas**: Chosen for its powerful data manipulation capabilities.
- **numpy**: Used for efficient numerical operations.

```python
import pandas as pd
import numpy as np
```

---

## Preprocessing Function Definition

### Explanation
I defined a function `preprocess_data` that takes a DataFrame as input and starts by printing the initial shape of the data.

### Design Decisions
- **Function**: Adopted a modular approach to preprocess data.
- **Print Statement**: Included for debugging and understanding data size.

```python
def preprocess_data(data):
    try:
        print("Initial data shape:", data.shape)
```

---

## Handling Missing Values

### Explanation
I handled missing values by filling `MonthlyIncome` and `NumberOfDependents` with their median values to maintain data integrity.

### Design Decisions
- **Median Filling**: Median is robust to outliers and provides a central value for missing data.

```python
        # Fill missing values
        print("Filling missing values for 'MonthlyIncome' and 'NumberOfDependents'...")
        data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())
        data['NumberOfDependents'] = data['NumberOfDependents'].fillna(data['NumberOfDependents'].median())
```

---

## Column Name Validation

### Explanation
I validated the presence of specific columns and renamed them if necessary to ensure consistency.

### Design Decisions
- **Column Renaming**: Ensured consistency in column names throughout the process.

```python
        print("Checking for column 'NumberOfTime30-59DaysPastDueNotWorse'...")
        if 'NumberOfTime30-59DaysPastDueNotWorse' in data.columns:
            data['NumberOfTime30_59DaysPastDueNotWorse'] = data['NumberOfTime30-59DaysPastDueNotWorse']
        else:
            print("Column 'NumberOfTime30-59DaysPastDueNotWorse' not found.")

        print("Checking for column 'NumberOfTime60-89DaysPastDueNotWorse'...")
        if 'NumberOfTime60-89DaysPastDueNotWorse' in data.columns:
            data['NumberOfTime60_89DaysPastDueNotWorse'] = data['NumberOfTime60-89DaysPastDueNotWorse']
        else:
            print("Column 'NumberOfTime60-89DaysPastDueNotWorse' not found.")
```

---

## Binning and Encoding Categorical Variables

### Explanation
I binned continuous variables into discrete intervals and converted categorical variables into dummy/indicator variables. This helped in dealing with skewed data distributions and simplified the modeling process.

### Design Decisions
- **Binning**: Helped in handling skewed data distributions and converted continuous variables into discrete intervals.
- **Encoding**: Converted categorical variables into dummy variables for better model processing.

```python
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
```

---

## Final Data Shape and Return

### Explanation
Finally, I printed the processed data's shape and returned the processed DataFrame.

### Design Decisions
- **Print Statement**: Helped in verifying the final shape of the processed data.
- **Returning Data**: Ensured the processed data was available for further steps in the pipeline.

```python
        print("Final processed data shape:", data.shape)
        return data
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return pd.DataFrame()
```

---

## Example Usage for Testing

### Explanation
I included an example usage section to demonstrate how the preprocessing function could be tested with sample data.

### Design Decisions
- **Example Usage**: Helped in validating the preprocessing function independently.

```python
if __name__ == "__main__":
    # Example usage for testing
    print("Reading input data...")
    input_data = pd.read_csv("data/cs-test.csv")
    print("Starting preprocessing...")
    preprocessed_data = preprocess_data(input_data)
    print("Preprocessing completed.")
    print(preprocessed_data.head())
```


---
# ⛵️ (5/8) Model Generation Script (b_model_gen.py)
---


## Importing Libraries

### Explanation
I started by importing the necessary libraries:

- **pandas**: Essential for data manipulation tasks.
- **RandomForestClassifier** from **sklearn.ensemble**: To train the machine learning model.
- **joblib**: For saving the trained model to a file.

### Design Decisions
- **pandas**: Chosen for its powerful data manipulation capabilities.
- **RandomForestClassifier**: Selected for its robustness and ability to handle various data types.
- **joblib**: Used for efficient model serialization and deserialization.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
```

---

## Loading Training Data

### Explanation
I loaded the training data from a CSV file and printed its shape to verify successful loading.

### Design Decisions
- **CSV Loading**: Using pandas to read the training data for efficient data handling.
- **Print Statement**: Included for debugging and confirming data shape.

```python
# Load data
print("Loading training data...")
train_data = pd.read_csv("data/cs-training.csv")
print("Training data loaded. Shape:", train_data.shape)
```

---

## Preprocessing Training Data

### Explanation
I imported the preprocessing function from the previous script and applied it to the training data. This step ensures that the training data is preprocessed in the same way as the test data.

### Design Decisions
- **Function Reuse**: Ensured consistency in data preprocessing by reusing the same function.
- **Print Statement**: Helped in debugging and confirming the processed data shape.

```python
# Preprocess data
from a_preprocessing_featurepipeline import preprocess_data
print("Starting preprocessing of training data...")
train_data = preprocess_data(train_data)
print("Preprocessing completed. Processed data shape:", train_data.shape)
```

---

## Ensuring Target Column Existence

### Explanation
I checked for the presence of the target column (`SeriousDlqin2yrs`) in the training data and raised an error if it was missing.

### Design Decisions
- **Validation Check**: Ensured that the target column is present before proceeding with model training.

```python
# Ensure the target column is correctly named and exists
if 'SeriousDlqin2yrs' not in train_data.columns:
    raise KeyError("The target column 'SeriousDlqin2yrs' is missing in the training data.")
```

---

## Separating Features and Target

### Explanation
I separated the features (independent variables) and the target (dependent variable) for training the model.

### Design Decisions
- **Feature-Target Separation**: Essential step for supervised learning.

```python
# Separate features and target
print("Separating features and target...")
X_train = train_data.drop(columns=["SeriousDlqin2yrs"])
y_train = train_data["SeriousDlqin2yrs"]
print("Features shape:", X_train.shape, "Target shape:", y_train.shape)
```

---

## Training the Random Forest Model

### Explanation
I trained a Random Forest model using the preprocessed training data. The model was configured with 1000 estimators and a fixed random state for reproducibility.

### Design Decisions
- **Random Forest**: Chosen for its robustness and ability to handle high-dimensional data.
- **Hyperparameters**: Set to 1000 estimators for better performance and reproducibility.

```python
# Train model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")
```

---

## Saving the Trained Model

### Explanation
I saved the trained model to a file using joblib for future use in predictions.

### Design Decisions
- **Model Serialization**: Used joblib for efficient model saving and loading.

```python
# Save model
print("Saving the model...")
joblib.dump(model, "models/random_forest_model.pkl")
print("Model saved successfully.")
```



---
# 🛰️ (6/8) Prediction Script (c_model_prediction.py)
---

## Importing Libraries

### Explanation
I began by importing the necessary libraries:

- **pandas**: For data manipulation tasks.
- **joblib**: For loading the trained model.
- **FastAPI**: To create the API.
- **BaseModel** from **pydantic**: For data validation.
- **List** from **typing**: To handle lists of input data.

### Design Decisions
- **pandas**: Essential for handling input data.
- **joblib**: Used for loading the pre-trained model.
- **FastAPI**: Chosen for its fast performance and ease of use in creating APIs.
- **pydantic**: Ensures data validation and structured input handling.
- **typing.List**: Necessary for handling list inputs in API requests.

```python
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
```

---

## Loading the Model

### Explanation
I loaded the trained Random Forest model from the file using joblib.

### Design Decisions
- **Model Loading**: Used joblib to deserialize the trained model for making predictions.

```python
# Load the model
model = joblib.load("models/random_forest_model.pkl")
```

---

## Defining the Preprocessing Function

### Explanation
I imported the preprocessing function from the preprocessing script to ensure the input data is preprocessed in the same way as the training data.

### Design Decisions
- **Function Reuse**: Maintains consistency in data preprocessing by reusing the same function.

```python
# Define the function to preprocess data
from a_preprocessing_featurepipeline import preprocess_data
```

---

## Defining the Input and Output Data Models

### Explanation
I defined the structure of the input and output data using pydantic's BaseModel. This ensures the data passed to the API is validated and structured correctly.

### Design Decisions
- **Data Validation**: Ensures that the input data adheres to the expected format and types.
- **Structured Output**: Provides a clear and structured format for the API responses.

```python
# Define the input data structure
class InputData(BaseModel):
    ID: int
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

class PredictResponse(BaseModel):
    Id: int
    Probability: float
```

---

## Initializing the FastAPI Application

### Explanation
I initialized the FastAPI application to create the API.

### Design Decisions
- **FastAPI Initialization**: Sets up the FastAPI application for handling API requests.

```python
app = FastAPI()
```

---

## Defining the Prediction Endpoint

### Explanation
I created an endpoint to handle POST requests for predictions. This endpoint converts the input data to a DataFrame, preprocesses it, checks for missing columns, reorders the columns, makes predictions, and returns the results.

### Design Decisions
- **POST Endpoint**: Handles prediction requests by accepting a list of input data.
- **DataFrame Conversion**: Facilitates data manipulation and preprocessing.
- **Preprocessing**: Ensures the input data is prepared in the same way as the training data.
- **Column Reordering**: Matches the order of features used during model training.
- **Prediction**: Uses the trained model to make predictions and returns the results in a structured format.

```python
@app.post("/predict", response_model=List[PredictResponse])
async def predict(data: List[InputData]):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([item.dict() for item in data])

        # Preprocess the input data
        processed_data = preprocess_data(input_df)

        # Check for missing columns and add them if necessary
        expected_columns = model.feature_names_in_
        missing_cols = set(expected_columns) - set(processed_data.columns)
        for col in missing_cols:
            processed_data[col] = 0

        # Reorder columns to match the training order
        processed_data = processed_data[expected_columns]

        # Make predictions
        predictions = model.predict_proba(processed_data)[:, 1]

        # Prepare response
        response = [{"Id": item.ID, "Probability": prob} for item, prob in zip(data, predictions)]
        return response
    except Exception as e:
        print(f"Error in prediction: {e}")
        return []
```

---

## Running the FastAPI Application

### Explanation
I included a block to run the FastAPI application using uvicorn when the script is executed directly.

### Design Decisions
- **Application Execution**: Ensures the FastAPI server runs when the script is executed, allowing for handling prediction requests.

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
```



---
# 🐳 (7/8) Dockerfile (Dockerfile)
---

## Base Image and Working Directory

### Explanation
I started by using the official Python 3.10-slim image as the base image for the Docker container. Then, I set the working directory in the container to `/app`.

### Design Decisions
- **Python 3.10-slim**: Chosen for its lightweight nature, reducing the overall size of the Docker image.
- **Working Directory**: Setting a specific working directory helps in organizing the files within the container.

```dockerfile
# Use the official Python image.
FROM python:3.10-slim

# Set the working directory in the container.
WORKDIR /app
```

---

## Copying and Installing Dependencies

### Explanation
I copied the `requirements.txt` file into the container and installed the dependencies using `pip`. This ensures that all necessary libraries are available in the container.

### Design Decisions
- **Copy requirements.txt**: Copies the requirements file to the container to ensure all dependencies are listed and can be installed.
- **Install Dependencies**: Uses `pip` to install the dependencies listed in `requirements.txt`.

```dockerfile
# Copy the requirements file into the container.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt
```

---

## Copying Application Files

### Explanation
I copied the rest of the application files into the container. This includes all scripts and models needed for the application to run.

### Design Decisions
- **Copy Application Files**: Ensures that all necessary files are included in the Docker image, allowing the application to run correctly.

```dockerfile
# Copy the rest of the working directory contents into the container.
COPY . .
```

---

## Exposing Port and Running the Application

### Explanation
I exposed port 8888, which is the port the FastAPI server runs on, and set the default command to start the FastAPI server using `uvicorn`.

### Design Decisions
- **Expose Port**: Makes the application accessible on port 8888.
- **Start FastAPI Server**: Uses `uvicorn` to run the FastAPI application, specifying the host and port.

```dockerfile
# Expose the port the app runs on.
EXPOSE 8888

# Run the command to start the FastAPI server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
```

---

### Purpose of `main.py` in the Docker Container

#### Overview:
The `main.py` file serves as the entry point for the FastAPI application, handling incoming requests, preprocessing data, making predictions using the trained model, and returning the results. It plays a critical role in the workflow of the machine learning API deployed in the Docker container.

#### Detailed Explanation:

1. **Initialization and Setup:**
   - The file starts by importing necessary libraries and modules, including FastAPI, joblib, pandas, numpy, and the custom preprocessing function from `a_preprocessing_featurepipeline.py`.
   - It initializes a FastAPI app instance to handle incoming HTTP requests.

    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel
    import joblib
    import pandas as pd
    import numpy as np
    from a_preprocessing_featurepipeline import preprocess_data

    # Initialize FastAPI app
    app = FastAPI()
    ```

2. **Loading the Trained Model:**
   - The pre-trained RandomForest model is loaded using joblib, which is essential for making predictions.

    ```python
    # Load the trained model
    model = joblib.load("models/random_forest_model.pkl")
    ```

3. **Defining Data Models:**
   - `PredictRequest` and `PredictResponse` classes are defined using Pydantic to enforce data validation for incoming requests and outgoing responses.
   - This ensures that the input data conforms to the expected format and types, reducing errors during prediction.

    ```python
    # Define request and response models
    class PredictRequest(BaseModel):
        ID: int
        RevolvingUtilizationOfUnsecuredLines: float
        age: int
        NumberOfTime30_59DaysPastDueNotWorse: int
        DebtRatio: float
        MonthlyIncome: float
        NumberOfOpenCreditLinesAndLoans: int
        NumberOfTimes90DaysLate: int
        NumberRealEstateLoansOrLines: int
        NumberOfTime60_89DaysPastDueNotWorse: int
        NumberOfDependents: int

    class PredictResponse(BaseModel):
        ID: int
        Probability: float
    ```

4. **Prediction Endpoint:**
   - The `/predict` endpoint is defined, which accepts POST requests containing the input data for prediction.
   - The input data is converted to a pandas DataFrame, preprocessed, and used to make predictions using the loaded model.
   - The response is then structured as a `PredictResponse` object and returned to the client.

    ```python
    # Define prediction endpoint
    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        try:
            # Convert request to DataFrame
            input_data = pd.DataFrame([request.dict()])
            print("Input data received:", input_data)

            # Preprocess input data
            preprocessed_data = preprocess_data(input_data)
            print("Preprocessed data:", preprocessed_data)

            # Ensure correct columns order
            feature_names = model.feature_names_in_
            preprocessed_data = preprocessed_data.reindex(columns=feature_names, fill_value=0)

            # Make prediction
            prediction = model.predict_proba(preprocessed_data)[:, 1][0]
            print("Prediction:", prediction)

            # Create response
            response = PredictResponse(ID=request.ID, Probability=prediction)
            return response
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)}
    ```

5. **Running the FastAPI Server:**
   - The `if __name__ == "__main__":` block ensures that the FastAPI server runs when the script is executed directly.
   - Uvicorn is used to serve the FastAPI application, making it accessible at the specified host and port.

    ```python
    # To run the server, use the command: uvicorn main:app --host 0.0.0.0 --port 8888
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8888)
    ```


---
# 🧮 (8/8) Requirements File (requirements.txt)
---

## Required Libraries

### Explanation
The `requirements.txt` file lists all the necessary Python libraries required for this project. Each library serves a specific purpose, ensuring that the project runs smoothly and efficiently.

### Design Decisions
- **fastapi**: Chosen as the web framework for building the API due to its speed and ease of use.
- **uvicorn**: A lightning-fast ASGI server, essential for running the FastAPI application.
- **pandas**: Used for data manipulation and preprocessing, making it easier to handle and process data.
- **scikit-learn**: Provides machine learning tools, including the RandomForestClassifier used in this project.
- **joblib**: Used for saving and loading the trained machine learning model.
- **numpy**: Supports numerical operations, which are crucial for data processing and manipulation.
- **pydantic**: Ensures data validation and settings management using Python type annotations, making it essential for FastAPI request and response models.



---
# 🏗 Design Consideration Questions
---

## **Questions**:
	1. How would you test for incorrect user input?
	2. If you had more time what would you do? 
	3. How would you scale your API?
	4. How would you improve the architecture and your submitted design?
	5. How would you build monitoring around this? 
	6. Comment on this modelling approach taken by the data scientists. 
	7. What would you improve?

---
## Short Answers

### 1. How would you test for incorrect user input?
Testing for incorrect user input involves several steps:
- **Validation at the API Level**: Use Pydantic models to enforce data types and constraints. For example, ensure that `age` is an integer and `MonthlyIncome` is a float.
- **Unit Testing**: Create test cases for various scenarios, including valid and invalid inputs.

### 2. If you had more time what would you do?
- **Enhance the Model**: Experiment with other machine learning algorithms, hyperparameter tuning, and feature engineering to improve the model's accuracy.
- **Detailed Documentation**: Provide more comprehensive documentation on the API, model, and data preprocessing steps.
- **CI/CD Pipeline**: Implement continuous integration and deployment pipelines to automate testing and deployment.
- **User Authentication**: Add user authentication to secure the API.
- **Logging and Monitoring**: Implement detailed logging and monitoring for the API to track performance and issues.

### 3. How would you scale your API?
- **Load Balancing**: Use load balancers to distribute incoming requests across multiple instances of the API server.
- **Horizontal Scaling**: Deploy multiple instances of the API server across different machines or containers.
- **Caching**: Implement caching mechanisms for frequently accessed data to reduce the load on the API server.
- **Auto-scaling**: Use auto-scaling services to automatically adjust the number of API server instances based on demand.

### 4. How would you improve the architecture and your submitted design?
- **Microservices Architecture**: Break down the monolithic application into smaller microservices for better scalability and maintainability.
- **Service Mesh**: Use a service mesh for managing microservices communication, security, and monitoring.
- **Database Optimization**: Optimize database queries and indexing for faster data access.
- **Asynchronous Processing**: Use asynchronous processing for long-running tasks to improve the API's responsiveness.

### 5. How would you build monitoring around this?
- **APM Tools**: Use Application Performance Monitoring (APM) tools like New Relic, Datadog, or Prometheus to monitor API performance and detect issues.
- **Logging**: Implement structured logging using tools like ELK stack (Elasticsearch, Logstash, Kibana) for centralized log management and analysis.
- **Health Checks**: Implement health check endpoints and use monitoring tools to regularly check the API's health status.
- **Alerts**: Set up alerts for critical metrics such as high response times, error rates, and resource usage to proactively address issues.

### 6. Comment on this modeling approach taken by the data scientists.
- **Random Forest Model**: The use of a Random Forest classifier is appropriate due to its robustness and ability to handle various data types. It also provides feature importance, which helps in understanding the model.
- **Quantile Binning**: The quantile-based binning approach simplifies the handling of skewed data distributions and can improve model performance.

### 7. What would you improve?
- **Feature Engineering**: Explore additional feature engineering techniques to capture more information from the data.
- **Model Evaluation**: Perform more extensive model evaluation using cross-validation and other metrics like ROC-AUC.
- **Handling Imbalanced Data**: Implement techniques for handling imbalanced data, such as SMOTE or adjusting class weights.
- **Explainability**: Use tools like SHAP or LIME to explain the model's predictions and make the model more interpretable.




