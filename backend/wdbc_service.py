
import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "mammopedia_wdbc_final.joblib"
)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def _prepare_features(features):
    X = np.asarray(features, dtype=float).reshape(1, -1)

    if X.shape[1] != 30:
        raise ValueError(
            f"Expected 30 features, but received {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise ValueError("All 30 features must be finite numbers.")

    return X


def _class_probability_map(probabilities):
    classes = list(model.classes_)
    return {
        int(label): float(probabilities[i])
        for i, label in enumerate(classes)
    }


def predict_wdbc(features):
    X = _prepare_features(features)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]
    probability_map = _class_probability_map(probabilities)

    if 0 not in probability_map or 1 not in probability_map:
        raise ValueError("The trained model must contain classes 0 and 1.")

    return {
        "prediction": prediction,
        "diagnosis": "malignant" if prediction == 0 else "benign",
        "malignant_probability": probability_map[0],
        "benign_probability": probability_map[1],
    }


def explain_wdbc(features):
    """
    Return signed Logistic Regression score contributions.

    Positive contributions move the model score toward class 0
    (malignant); negative contributions move it toward class 1
    (benign), assuming the model's classes are [0, 1].
    """
    X = _prepare_features(features)
    X_scaled = scaler.transform(X)

    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(
            "Explanation currently requires model classes [0, 1]."
        )

    if not hasattr(model, "coef_") or model.coef_.shape != (1, 30):
        raise ValueError(
            "The loaded model is not the expected binary Logistic Regression model."
        )

    # For classes [0, 1], sklearn's binary logistic score
    # represents class 1 relative to class 0.
    contributions = (
        X_scaled[0] * model.coef_[0]
    )

    # Reverse the sign so positive means movement toward malignant (0).
    contributions = -contributions

    ranked = sorted(
        [
            {
                "feature": FEATURE_NAMES[i],
                "contribution": float(contributions[i]),
                "direction": (
                    "toward_malignant"
                    if contributions[i] > 0
                    else "toward_benign"
                    if contributions[i] < 0
                    else "neutral"
                ),
            }
            for i in range(30)
        ],
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )

    return {
        "method": "Logistic Regression signed score contributions",
        "interpretation": (
            "Positive values move the model score toward malignant; "
            "negative values move it toward benign. These are score "
            "contributions, not causal effects or clinical risk estimates."
        ),
        "features": ranked,
    }
