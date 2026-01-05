import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/dataset.csv")


X = df.drop("Purchased", axis=1)
y = df["Purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])


pipeline.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.pkl")

print("Model trained and saved successfully")
