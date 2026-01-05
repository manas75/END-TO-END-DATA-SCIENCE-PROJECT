from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(age: int, salary: int, experience: int):
    data = np.array([[age, salary, experience]])
    prediction = model.predict(data)[0]

    return {"Purchased": int(prediction)}
