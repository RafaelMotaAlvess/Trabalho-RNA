from __future__ import annotations

from pathlib import Path

import pandas as pd

from trabalho_rna.constants import BUILD_PRESETS
from trabalho_rna.data import build_feature_dataset
from trabalho_rna.logs import log_base_dataset, log_preprocess, log_synthetic
from trabalho_rna.model import recomendar_armas, train_model
from trabalho_rna.scoring import build_training_dataset


def main() -> None:
    dataset_name = "weapons.csv"
    print("Carregando dataset original...")
    raw = pd.read_csv(dataset_name)

    log_base_dataset(raw, dataset_name)

    pre_weapons = build_feature_dataset(raw)
    nan_total_before = pre_weapons.isna().sum().sum()
    weapons = pre_weapons.fillna(0.0).drop_duplicates(subset="name")
    nan_after = weapons.isna().sum().sum()
    log_preprocess(raw, weapons, nan_total_before, nan_after)

    train_df = build_training_dataset(weapons)
    log_synthetic(train_df)

    print("\n=== 4. Treinando MLPRegressor ===")
    model, scaler = train_model(train_df)

    print("\n=== 5. Recomendação ===")
    print("Builds pré-set disponíveis:", list(BUILD_PRESETS.keys()))
    choice = input("Digite o nome da build pré-set ou 'custom' para informar manualmente: ").strip().upper()
    if choice in BUILD_PRESETS:
        build = BUILD_PRESETS[choice]
        print(f"Usando build pré-set: {choice} -> {build}")
    else:
        print("Informe seus atributos (enter vazio assume 0):")

        def read_val(label: str) -> float:
            raw = input(f"{label}: ").strip()
            try:
                return float(raw) if raw else 0.0
            except ValueError:
                return 0.0

        build = {
            "STR": read_val("STR"),
            "DEX": read_val("DEX"),
            "INT": read_val("INT"),
            "FAI": read_val("FAI"),
            "ARC": read_val("ARC"),
            "END": read_val("END"),
        }
        print(f"Build custom: {build}")

    ranked = recomendar_armas(build, weapons, model, scaler)
    top10 = ranked[["name", "category", "recommend_score"]].head(10)
    print("Top 10 armas recomendadas (ordenadas por recommend_score):")
    print(top10.to_string(index=False, formatters={"recommend_score": "{:.4f}".format}))


if __name__ == "__main__":
    main()
