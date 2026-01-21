from pathlib import Path
import joblib # pentru încărcarea modelului salvat
import pandas as pd
import numpy as np
# ------------------------------
# Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # proiect/
MODELS_DIR = BASE_DIR / "models"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "fraud_model_xgb.pkl"

if not SCALER_PATH.is_file():
    raise FileNotFoundError(f"Nu găsesc scaler-ul: {SCALER_PATH}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Nu găsesc modelul: {MODEL_PATH}")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# ------------------------------
# Feature columns (exact ca în dataset)
# ------------------------------
FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

# Câte feature-uri știe scaler-ul? (la tine e 30)
SCALER_FEATURES = getattr(scaler, "n_features_in_", None)


def simulate_transaction() -> pd.DataFrame:
    """
    Simulează o singură tranzacție (DEMO):
    - Time: între 0 și 48 de ore (în secunde)
    - V1–V28: valori normale N(0, 1.5)
    - Amount: lognormal (mai multe valori între ~10 și ~200)
    """
    time = np.random.uniform(0, 172800)  # 48h în secunde
    pca_features = np.random.normal(loc=0.0, scale=1.5, size=28)
    amount = np.random.lognormal(mean=3, sigma=0.7)

    values = np.concatenate([[time], pca_features, [amount]])
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def prepare_for_model(df_tx: pd.DataFrame) -> pd.DataFrame:
    """
    Pregătește tranzacția conform scaler-ului salvat.
    În cazul tău (n_features_in_=30) scalează toate coloanele.
    """
    df_tx = df_tx.copy()
    df_tx = df_tx[FEATURE_COLUMNS].fillna(0)

    if SCALER_FEATURES == 30:
        df_tx[FEATURE_COLUMNS] = scaler.transform(df_tx[FEATURE_COLUMNS])
    elif SCALER_FEATURES == 2:
        df_tx[["Time", "Amount"]] = scaler.transform(df_tx[["Time", "Amount"]])
    else:
        # fallback: încercăm întâi 30, apoi 2
        try:
            df_tx[FEATURE_COLUMNS] = scaler.transform(df_tx[FEATURE_COLUMNS])
        except Exception:
            df_tx[["Time", "Amount"]] = scaler.transform(df_tx[["Time", "Amount"]])

    return df_tx


def predict_fraud(df_tx_prepared: pd.DataFrame, threshold: float = 0.5):
    """
    Returnează:
    - pred (0/1) folosind pragul threshold
    - proba de fraudă (float)

    Compatibil cu:
    - XGBClassifier (sklearn API) -> predict_proba(X)
    - xgboost.Booster -> predict(DMatrix)
    """
    X = df_tx_prepared[FEATURE_COLUMNS].to_numpy()

    # 1) Dacă modelul are predict_proba (de obicei sklearn/XGBClassifier)
    if hasattr(model, "predict_proba"):
        # IMPORTANT: îi dăm numpy array, ca să evităm probleme de tipuri
        proba = model.predict_proba(X)[:, 1]
    else:
        # 2) Dacă modelul e Booster (xgboost)
        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
        proba = model.predict(dmat)

    pred = (proba >= threshold).astype(int)
    return pred, proba


def simulate_and_predict(n: int = 10, threshold: float = 0.5) -> pd.DataFrame:
    """
    Generează n tranzacții + predicții și întoarce un DataFrame cu rezultate.
    """
    results = []
    for _ in range(n):
        tx = simulate_transaction()
        tx_prepared = prepare_for_model(tx)

        pred, proba = predict_fraud(tx_prepared, threshold=threshold)

        tx["Prediction"] = int(pred[0])
        tx["Fraud_Probability"] = float(proba[0])
        results.append(tx)

    return pd.concat(results, ignore_index=True)


def main():
    print("[INFO] Proiect:", BASE_DIR)
    print("[INFO] Model:", MODEL_PATH.name)
    print("[INFO] Scaler:", SCALER_PATH.name)
    print("[INFO] Scaler n_features_in_ =", SCALER_FEATURES)

    # Debug util: să vedem ce tip de model ai încărcat
    print("[DEBUG] Tip model:", type(model))
    print("[DEBUG] Are predict_proba?", hasattr(model, "predict_proba"))

    df = simulate_and_predict(n=100, threshold=0.5)

    cols_to_show = ["Time", "Amount", "Prediction", "Fraud_Probability"]
    print("\n[EXEMPLE]")
    print(df[cols_to_show].head(15))

    print("\n[REZUMAT]")
    print(df["Prediction"].value_counts(dropna=False))
    print("Probabilitate medie fraudă:", df["Fraud_Probability"].mean())


if __name__ == "__main__":
    main()