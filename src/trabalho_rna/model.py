from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from trabalho_rna.constants import BUILD_FEATURES, NUMERIC_COLUMNS, WEAPON_FEATURES


def train_model(df: pd.DataFrame):
    X = df[BUILD_FEATURES + WEAPON_FEATURES]
    y = df["compatibility_score"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    quality = "Excelente" if r2 >= 0.95 else "Boa" if r2 >= 0.85 else "Regular"
    print("\n=== 4. Métricas do Treinamento ===")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f} -> Qualidade: {quality}")

    joblib.dump(model, "modelo_recomendador.pkl")
    joblib.dump(scaler, "scaler_recomendador.pkl")

    return model, scaler


def recomendar_armas(build: dict, weapons_df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    rows = []
    for _, w in weapons_df.iterrows():
        feat = {
            "build_str": build["STR"],
            "build_dex": build["DEX"],
            "build_int": build["INT"],
            "build_fai": build["FAI"],
            "build_arc": build["ARC"],
            "build_end": build.get("END", build.get("Endurance", 20)),
        }
        for col in NUMERIC_COLUMNS:
            feat[f"weapon_{col}"] = w[col]
        rows.append(feat)

    X = pd.DataFrame(rows, columns=BUILD_FEATURES + WEAPON_FEATURES)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    weapons_df = weapons_df.copy()
    weapons_df["recommend_score"] = np.clip(preds, 0.0, 1.0)
    return weapons_df.sort_values("recommend_score", ascending=False)
