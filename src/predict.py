import pandas as pd
import joblib

MODEL_PATH = "models/insurance_cost_model_rf.joblib"

def main():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "age": 30,
        "sex": "male",
        "bmi": 28.5,
        "children": 1,
        "smoker": "no",
        "region": "southeast"
    }])

    pred = model.predict(sample)[0]
    print("Predicted insurance charges:", round(pred, 2))

if __name__ == "__main__":
    main()