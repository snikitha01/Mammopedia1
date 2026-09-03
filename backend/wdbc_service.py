import joblib
import numpy as np
from pathlib import Path


# Find the trained model file
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "mammopedia_wdbc_final.joblib"
)


# Load the trained Mammopedia model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]


def predict_wdbc(features):
    """
    Make a breast-cancer prediction using the
    trained WDBC Logistic Regression model.
    """

    # Convert input to numpy array
    X = np.array(features, dtype=float).reshape(1, -1)

    # Check that exactly 30 WDBC features were supplied
    if X.shape[1] != 30:
        raise ValueError(
            f"Expected 30 features, but received {X.shape[1]}"
        )

    # Apply the SAME scaler used during training
    X_scaled = scaler.transform(X)

    # Prediction
    prediction = int(model.predict(X_scaled)[0])

    # Probability
    probabilities = model.predict_proba(X_scaled)[0]

    malignant_probability = float(probabilities[0])
    benign_probability = float(probabilities[1])

    if prediction == 0:
        diagnosis = "malignant"
    else:
        diagnosis = "benign"

    return {
        "prediction": prediction,
        "diagnosis": diagnosis,
        "malignant_probability": malignant_probability,
        "benign_probability": benign_probability
    }
