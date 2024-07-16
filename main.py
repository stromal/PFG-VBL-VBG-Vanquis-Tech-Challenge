from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from a_preprocessing_featurepipeline import preprocess_data

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

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

# To run the server, use the command: uvicorn main:app --host 0.0.0.0 --port 8888
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
