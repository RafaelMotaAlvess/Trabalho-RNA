from __future__ import annotations

from pathlib import Path

import pandas as pd

from trabalho_rna.constants import BUILD_PRESETS, BUILD_FEATURES, WEAPON_FEATURES
from trabalho_rna.data import format_nested_cell
from trabalho_rna.model import recomendar_armas


def log_base_dataset(raw: pd.DataFrame, dataset_name: str) -> None:
    print("\n=== 1. Definição da Base de Dados ===")
    print(f"Arquivo: {dataset_name}")
    print(f"Linhas: {len(raw)}, Colunas: {len(raw.columns)}")
    print("Colunas:", list(raw.columns))
    cols_raw = ["name", "category", "attack", "requiredAttributes", "scalesWith", "weight"]
    sample_raw = raw[cols_raw].head(10).copy()
    for col in ["attack", "requiredAttributes", "scalesWith"]:
        sample_raw[col] = sample_raw[col].apply(format_nested_cell)
    print("\nAmostra (10 linhas) - colunas críticas:")
    print(sample_raw.to_string(index=False))
    print("\nCategorias (top 10):")
    print(raw["category"].value_counts().head(10))


def log_preprocess(raw: pd.DataFrame, weapons: pd.DataFrame, nan_total_before: int, nan_after: int) -> None:
    print("\n=== 2. Pré-processamento ===")
    print(f"Armas no bruto: {len(raw)}")
    print(f"Armas após preprocess (unnest + drop_duplicates): {len(weapons)}")
    print(
        f"Total de valores NaN substituídos: {nan_total_before - nan_after if nan_total_before >= nan_after else nan_total_before}"
    )
    print("Categorias após pré-processamento (top 10):")
    print(weapons["category"].value_counts().head(10))


def log_synthetic(train_df: pd.DataFrame) -> None:
    print("\n=== 3. Dataset sintético (build + arma → score) ===")
    print(f"Shape do train_df: {train_df.shape[0]} linhas x {train_df.shape[1]} colunas")
    raw_min = train_df.attrs.get("score_min_raw", None)
    raw_max = train_df.attrs.get("score_max_raw", None)
    if raw_min is not None and raw_max is not None:
        print(f"Scores normalizados para 0–1 (min bruto={raw_min:.6f}, max bruto={raw_max:.6f})")
    print("Distribuição de builds no dataset sintético:")
    if "build_name" in train_df.columns:
        print(train_df["build_name"].value_counts())
    print("Amostra (10 linhas):")
    print(train_df.head(10)[BUILD_FEATURES + WEAPON_FEATURES + ["build_name", "weapon_name", "weapon_category", "compatibility_score"]])


def log_recommendations(weapons: pd.DataFrame, model, scaler) -> None:
    pass
