
import time
import threading

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


RANDOM_STATE = 42
_cached_result = None
_cache_lock = threading.Lock()


def run_qsvm_experiment():
    global _cached_result

    with _cache_lock:
        if _cached_result is not None:
            return {**_cached_result, "cached": True}

        started = time.time()

        # Load WDBC; sklearn uses 0=malignant, 1=benign.
        dataset = load_breast_cancer()
        X = dataset.data
        y = dataset.target

        # Reproduce notebook's 80/20 development/test split,
        # then 80/20 train/validation split.
        X_dev, X_final_test, y_dev, y_final_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE,
            stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=0.20,
            random_state=RANDOM_STATE, stratify=y_dev
        )

        # Fit preprocessing on training data only.
        standard_scaler = StandardScaler()
        X_train_scaled = standard_scaler.fit_transform(X_train)
        X_val_scaled = standard_scaler.transform(X_val)

        # Reduce to four quantum input features.
        pca = PCA(n_components=4)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        quantum_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_train_q = quantum_scaler.fit_transform(X_train_pca)
        X_val_q = quantum_scaler.transform(X_val_pca)

        # Match the notebook's stratified 120-sample subset.
        X_qml_train, _, y_qml_train, _ = train_test_split(
            X_train_q,
            y_train,
            train_size=120,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        feature_map = zz_feature_map(
            feature_dimension=4,
            reps=1,
            entanglement="linear",
        )
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        K_train = quantum_kernel.evaluate(x_vec=X_qml_train)
        K_val = quantum_kernel.evaluate(
            x_vec=X_val_q,
            y_vec=X_qml_train,
        )

        classifier = SVC(
            kernel="precomputed",
            probability=True,
            random_state=RANDOM_STATE,
        )
        classifier.fit(K_train, y_qml_train)

        predictions = classifier.predict(K_val)
        probabilities = classifier.predict_proba(K_val)
        malignant_index = list(classifier.classes_).index(0)
        malignant_scores = probabilities[:, malignant_index]

        matrix = confusion_matrix(y_val, predictions, labels=[0, 1])

        result = {
            "status": "completed",
            "model": "QSVM",
            "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "execution": "Qiskit quantum-kernel simulator",
            "evaluation_split": "validation",
            "training_samples": int(len(X_qml_train)),
            "validation_samples": int(len(y_val)),
            "quantum_features": 4,
            "qubits": 4,
            "accuracy": float(accuracy_score(y_val, predictions)),
            "precision_malignant": float(
                precision_score(y_val, predictions, pos_label=0, zero_division=0)
            ),
            "sensitivity_malignant": float(
                recall_score(y_val, predictions, pos_label=0, zero_division=0)
            ),
            "specificity_benign": float(
                recall_score(y_val, predictions, pos_label=1, zero_division=0)
            ),
            "f1_malignant": float(
                f1_score(y_val, predictions, pos_label=0, zero_division=0)
            ),
            "roc_auc_malignant": float(
                roc_auc_score((y_val == 0).astype(int), malignant_scores)
            ),
            "confusion_matrix_labels": ["malignant", "benign"],
            "confusion_matrix": matrix.tolist(),
            "pca_variance_retained": float(
                pca.explained_variance_ratio_.sum()
            ),
            "duration_seconds": round(time.time() - started, 2),
            "cached": False,
            "disclaimer": (
                "Research experiment on a classical simulator; "
                "not a clinically validated diagnostic tool. "
                "Final test data was not used for these metrics."
            ),
        }

        _cached_result = result
        return result
