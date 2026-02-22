import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

DATA_PATH = "data/processed/insurance_clean.csv"
MODEL_PATH = "models/insurance_cost_model_rf.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["charges"])
    y = df["charges"]

    cat_cols = ["sex", "smoker", "region"]
    num_cols = ["age", "bmi", "children"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

if __name__ == "__main__":
    main()