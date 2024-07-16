import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load the model
model = joblib.load("models/random_forest_model.pkl")

# Define the function to preprocess data
from a_preprocessing_featurepipeline import preprocess_data

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

app = FastAPI()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
