"""
tools/train_model.py
---------------------
Train an SVM gesture classifier on data/gesture_data.csv and
save the resulting model to models/gesture_model.pkl.

Usage:
    python tools/train_model.py

Requirements:
    - data/gesture_data.csv must exist (run collect_gestures.py first).
    - Each row: 63 float landmark values + 1 string label.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Paths relative to project root
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH  = os.path.join(ROOT, "data",   "gesture_data.csv")
MODEL_PATH = os.path.join(ROOT, "models", "gesture_model.pkl")


def train():
    # ── Load dataset ──
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at '{DATA_PATH}'.")
        print("Run tools/collect_gestures.py first to create it.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, header=None)
    print(f"Loaded {len(df)} samples from {DATA_PATH}")
    print(f"Gesture classes: {df.iloc[:, -1].unique().tolist()}")

    X = df.iloc[:, :-1].values          # 63 landmark features
    y = df.iloc[:, -1].values           # gesture labels (strings)

    # ── Encode labels ──
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ── Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ── Train SVM ──
    print("Training SVM (RBF kernel) …")
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print(f"\nTest accuracy: {acc:.4f} ({acc * 100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_))

    # ── Save model + encoder ──
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    payload = {"model": model, "label_encoder": le}
    joblib.dump(payload, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
