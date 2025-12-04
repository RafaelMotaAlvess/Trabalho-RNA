from __future__ import annotations
import ast
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# =============================
# CONFIGURAÇÕES
# =============================

NUMERIC_COLUMNS = [
    "physical", "magic", "fire", "lightning", "holy",
    "crit", "guard_boost",
    "str_req", "dex_req", "int_req", "fai_req", "arc_req",
    "weight",
    "str_scale", "dex_scale", "int_scale", "fai_scale", "arc_scale",
    "base_damage_weight", "scaling_weight",
]
BUILD_FEATURES = ["build_str", "build_dex", "build_int", "build_fai", "build_arc", "build_end"]
WEAPON_FEATURES = [f"weapon_{col}" for col in NUMERIC_COLUMNS]

# Builds sintéticas típicas de Elden Ring (valores altos para perfis fortes):
BUILD_PRESETS = {
    "STR":      {"STR": 80, "DEX": 20, "INT": 10, "FAI": 10, "ARC": 20, "END": 35},
    "DEX":      {"STR": 20, "DEX": 80, "INT": 10, "FAI": 10, "ARC": 10, "END": 10},
    "INT":      {"STR": 10, "DEX": 20, "INT": 80, "FAI": 20, "ARC": 10, "END": 20},
    "FAI":      {"STR": 10, "DEX": 10, "INT": 20, "FAI": 80, "ARC": 10, "END": 20},
    "ARC":      {"STR": 10, "DEX": 20, "INT": 20, "FAI": 10, "ARC": 80, "END": 20},
    "QUALITY":  {"STR": 60, "DEX": 60, "INT": 20, "FAI": 20, "ARC": 20, "END": 30},
    "PALADIN":  {"STR": 50, "DEX": 20, "INT": 20, "FAI": 60, "ARC": 10, "END": 30},
    "BATTLEMAGE": {"STR": 50, "DEX": 20, "INT": 60, "FAI": 20, "ARC": 10, "END": 25},
}


# =============================
# FUNÇÕES AUXILIARES DO DATASET
# =============================

ATTACK_MAP = {"Phy": "physical", "Mag": "magic", "Fire": "fire", "Ligt": "lightning", "Holy": "holy", "Crit": "crit"}
DEFENCE_MAP = {"Boost": "guard_boost"}
REQ_MAP = {"Str": "str_req", "Dex": "dex_req", "Int": "int_req", "Fai": "fai_req", "Arc": "arc_req"}
SCALE_MAP = {"Str": "str_scale", "Dex": "dex_scale", "Int": "int_scale", "Fai": "fai_scale", "Arc": "arc_scale"}
SCALE_COEFFICIENTS = {
    "S": 1.6,
    "A": 1.2,
    "B": 1.0,
    "C": 0.75,
    "D": 0.45,
    "E": 0.2,
}
CATEGORY_BASE_DAMAGE_WEIGHTS = {
    "torch": 0.6,
    "fist": 0.75,
    "claw": 0.75,
    "dagger": 0.8,
    "straight sword": 1.0,
    "thrusting sword": 1.0,
    "curved sword": 1.0,
    "katana": 1.0,
    "axe": 1.0,
    "spear": 1.0,
    "great spear": 1.15,
    "halberd": 1.15,
    "greataxe": 1.25,
    "warhammer": 1.25,
    "great mace": 1.25,
    "curved greatsword": 1.15,
    "greatsword": 1.25,
    "colossal sword": 1.4,
}
CATEGORY_SCALING_WEIGHTS = {
    "torch": 0.3,
    "fist": 0.5,
    "claw": 0.5,
    "dagger": 0.7,
    "straight sword": 1.0,
    "thrusting sword": 1.0,
    "curved sword": 1.0,
    "katana": 1.0,
    "axe": 1.0,
    "spear": 1.0,
    "great spear": 1.2,
    "halberd": 1.2,
    "greataxe": 1.2,
    "warhammer": 1.2,
    "great mace": 1.2,
    "curved greatsword": 1.2,
    "greatsword": 1.2,
    "colossal sword": 1.4,
}


def safe_literal_list(value: object) -> List[Dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    return []


def as_float(value: object) -> float:
    try:
        return float(value)
    except:
        return 0.0


def extract_by_key(raw: object, key_map: Dict[str, str], amount_key: str = "amount") -> Dict[str, float]:
    values = {target: 0.0 for target in key_map.values()}
    for entry in safe_literal_list(raw):
        if not isinstance(entry, dict):
            continue
        source = entry.get("name")
        target = key_map.get(source)
        if not target:
            continue
        values[target] = as_float(entry.get(amount_key))
    return values


def format_nested_cell(value: object, max_items: int = 3) -> str:
    items = safe_literal_list(value)
    parts = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if "scaling" in entry:
            parts.append(f"{name}:{entry.get('scaling')}")
        elif "amount" in entry:
            parts.append(f"{name}:{entry.get('amount')}")
    if not parts:
        return ""
    if len(parts) > max_items:
        return "; ".join(parts[:max_items]) + " ..."
    return "; ".join(parts)


def build_feature_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in raw_df.iterrows():
        f = {}
        f.update(extract_by_key(row.get("attack"), ATTACK_MAP))
        f.update(extract_by_key(row.get("defence"), DEFENCE_MAP))
        f.update(extract_by_key(row.get("requiredAttributes"), REQ_MAP))
        # Scaling letters convertidos em coeficientes aproximados
        scaling_values = {target: 0.0 for target in SCALE_MAP.values()}
        for entry in safe_literal_list(row.get("scalesWith")):
            if not isinstance(entry, dict):
                continue
            target = SCALE_MAP.get(entry.get("name"))
            if target:
                scaling_values[target] = SCALE_COEFFICIENTS.get(entry.get("scaling"), 0.0)
        f.update(scaling_values)
        f["weight"] = as_float(row.get("weight"))
        f["name"] = row.get("name", "")
        category = row.get("category", "")
        f["category"] = category
        cat_key = category.lower()
        f["base_damage_weight"] = CATEGORY_BASE_DAMAGE_WEIGHTS.get(cat_key, 1.0)
        f["scaling_weight"] = CATEGORY_SCALING_WEIGHTS.get(cat_key, 1.0)
        rows.append(f)
    return pd.DataFrame(rows)


# =============================
# SCORE DE COMPATIBILIDADE
# =============================

def compute_compatibility(build: Dict[str, float], weapon: Dict[str, float]) -> float:
    build_vec = np.array([
        build["STR"], build["DEX"], build["INT"], build["FAI"], build["ARC"]
    ], dtype=float)
    build_sum = build_vec.sum() + 1e-6
    build_norm = build_vec / build_sum
    build_end = float(build.get("END", build.get("Endurance", 20)))

    req = np.array([
        weapon["str_req"],
        weapon["dex_req"],
        weapon["int_req"],
        weapon["fai_req"],
        weapon["arc_req"],
    ], dtype=float)
    req_sum = req.sum() + 1e-6
    req_norm = req / req_sum if req_sum > 0 else np.zeros_like(req)

    # Aderência de perfil: quão parecida é a distribuição de requisitos com a distribuição de atributos da build
    alignment_score = float(np.dot(build_norm, req_norm))

    # Cobertura: build >= requisitos (truncado em 1). Evita favorecer armas sem requisito nenhum.
    req_with_floor = req + 1.0
    coverage_score = float(np.minimum(build_vec / req_with_floor, 1.0).mean())

    # Penaliza déficit (quando requisito > build)
    deficit = np.clip(req - build_vec, 0, None)
    deficit_penalty = float((deficit / (req + 1e-6)).mean())

    # Considera scaling da arma (Str/Dex/Int/Fai/Arc) conforme letras
    scales = np.array([
        weapon.get("str_scale", 0.0),
        weapon.get("dex_scale", 0.0),
        weapon.get("int_scale", 0.0),
        weapon.get("fai_scale", 0.0),
        weapon.get("arc_scale", 0.0),
    ], dtype=float)
    scale_sum = scales.sum() + 1e-6
    scale_norm = scales / scale_sum if scale_sum > 0 else np.zeros_like(scales)
    scaling_alignment = float(np.dot(build_norm, scale_norm))

    # Dano ponderado pelo perfil da build, amplificado pelo alinhamento com o scaling
    physical_part = weapon["physical"] + 0.5 * weapon["crit"]
    elemental_part = weapon["magic"] + weapon["fire"] + weapon["lightning"] + weapon["holy"]
    physical_weight = (build["STR"] + build["DEX"]) / build_sum
    elemental_weight = (build["INT"] + build["FAI"] + build["ARC"]) / build_sum
    damage_total = (
        (physical_part * physical_weight) + (elemental_part * elemental_weight)
    ) / 1000.0
    damage_total *= (0.8 + 0.4 * scaling_alignment)  # boost se scaling casa com a build
    damage_total *= weapon.get("base_damage_weight", 1.0)
    base_damage = weapon["physical"] + weapon["magic"] + weapon["fire"] + weapon["lightning"] + weapon["holy"]
    low_damage_factor = min(1.0, base_damage / 250.0)

    # Peso penalizado, mas mitigado por Endurance
    effective_weight = weapon["weight"] / (1 + build_end / 30.0)
    weight_penalty = 1 / (1 + effective_weight)
    scaling_alignment *= weapon.get("scaling_weight", 1.0)

    # Combinação ponderada (peso maior em dano + scaling para priorizar armas de perfil certo)
    score = (
        0.30 * alignment_score
        + 0.20 * coverage_score
        + 0.30 * damage_total
        + 0.15 * scaling_alignment
        + 0.10 * weight_penalty
        - 0.15 * deficit_penalty
    )
    score *= low_damage_factor
    return float(score)


# =============================
# GERAR DATASET SINTÉTICO (BUILD + ARMA → SCORE)
# =============================

def build_training_dataset(weapons_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for build_name, build_stats in BUILD_PRESETS.items():
        for _, w in weapons_df.iterrows():
            weapon_feats = {col: w[col] for col in NUMERIC_COLUMNS}

            score = compute_compatibility(build_stats, weapon_feats)

            row = {}
            row["build_name"] = build_name
            # Features da build:
            row["build_str"] = build_stats["STR"]
            row["build_dex"] = build_stats["DEX"]
            row["build_int"] = build_stats["INT"]
            row["build_fai"] = build_stats["FAI"]
            row["build_arc"] = build_stats["ARC"]
            row["build_end"] = build_stats.get("END", 20)

            # Features da arma:
            for col in NUMERIC_COLUMNS:
                row[f"weapon_{col}"] = w[col]
            row["weapon_name"] = w.get("name", "")
            row["weapon_category"] = w.get("category", "")

            # Rótulo:
            row["compatibility_score"] = score

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        raw_min = df["compatibility_score"].min()
        raw_max = df["compatibility_score"].max()
        df.attrs["score_min_raw"] = raw_min
        df.attrs["score_max_raw"] = raw_max
        if raw_max > raw_min:
            df["compatibility_score"] = (df["compatibility_score"] - raw_min) / (raw_max - raw_min)
        else:
            df["compatibility_score"] = 0.0
    return df


# =============================
# TREINAR MLP E SALVAR MODELO
# =============================

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
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    quality = "Excelente" if r2 >= 0.95 else "Boa" if r2 >= 0.85 else "Regular"
    print("\n=== 4. Métricas do Treinamento ===")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f} -> Qualidade: {quality}")

    # Salvar modelo + scaler
    joblib.dump(model, "modelo_recomendador.pkl")
    joblib.dump(scaler, "scaler_recomendador.pkl")

    return model, scaler


# =============================
# RECOMENDAR ARMAS PARA UMA BUILD ESPECÍFICA
# =============================

def recomendar_armas(build: Dict[str, float], weapons_df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    rows = []

    for _, w in weapons_df.iterrows():

        # montar linha de entrada com 18 features
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


# =============================
# MAIN
# =============================

def main():
    print("=== 0. Visão Geral do Pipeline ===")
    print(
        "1) Ler dataset de armas -> 2) Pré-processar (atributos numéricos) -> "
        "3) Gerar dataset sintético (build+arma->score) -> "
        "4) Treinar MLP -> 5) Avaliar -> 6) Recomendar armas para builds."
    )

    dataset_name = "weapons.csv"
    print("Carregando dataset original...")
    raw = pd.read_csv(dataset_name)

    print("\n=== 1. Definição da Base de Dados ===")
    print(f"Arquivo: {dataset_name}")
    print(f"Linhas: {len(raw)}, Colunas: {len(raw.columns)}")
    print("Colunas:", list(raw.columns))
    print("\nAmostra (10 primeiras linhas) - colunas críticas:")
    cols_raw = ["name", "category", "attack", "requiredAttributes", "scalesWith", "weight"]
    sample_raw = raw[cols_raw].head(10).copy()
    for col in ["attack", "requiredAttributes", "scalesWith"]:
        sample_raw[col] = sample_raw[col].apply(format_nested_cell)
    print(sample_raw.to_string(index=False))
    print("\nDistribuição de categorias (top 10):")
    print(raw["category"].value_counts().head(10))
    raw_weights = pd.to_numeric(raw["weight"], errors="coerce")
    print("\nPeso (raw) - min/média/max:")
    print(raw_weights.describe()[["min", "mean", "max"]])
    scaling_counts = {}
    for scales in raw["scalesWith"]:
        for entry in safe_literal_list(scales):
            if not isinstance(entry, dict):
                continue
            key = (entry.get("name"), entry.get("scaling"))
            scaling_counts[key] = scaling_counts.get(key, 0) + 1
    if scaling_counts:
        print("\nDistribuição de scaling (atributo, letra) - top 10:")
        for k, v in sorted(scaling_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"{k}: {v}")

    print("\n=== 2. Pré-processamento ===")
    before_count = len(raw)
    weapons = build_feature_dataset(raw)
    nan_by_col_before = weapons.isna().sum()
    nan_total_before = nan_by_col_before.sum()
    weapons = weapons.fillna(0.0).drop_duplicates(subset="name")
    nan_after = weapons.isna().sum().sum()
    print(f"Armas no bruto: {before_count}")
    print(f"Armas após pre processamento: {len(weapons)}")
    print(f"Total de valores NaN substituídos: {nan_total_before - nan_after if nan_total_before >= nan_after else nan_total_before}")
    print("\nAmostra das armas pré-processadas:")
    print(
        weapons.head(10)[
            [
                "name",
                "category",
                "physical",
                "magic",
                "fire",
                "lightning",
                "holy",
                "crit",
                "str_req",
                "dex_req",
                "int_req",
                "fai_req",
                "arc_req",
                "weight",
                "str_scale",
                "dex_scale",
                "int_scale",
                "fai_scale",
                "arc_scale",
            ]
        ]
    )
    print(
        "\nConversões realizadas: attack/defence → danos/guard; requiredAttributes → requisitos; "
        "scalesWith → coeficientes str/dex/int/fai/arc; peso; categoria + pesos por categoria (dano base e scaling)."
    )
    print("\nResumo numérico (min/média/max) das features:")
    print(weapons[NUMERIC_COLUMNS].describe().loc[["min", "mean", "max"]])
    print("\nCategorias após pré-processamento (top 10):")
    print(weapons["category"].value_counts().head(10))

    print("\n=== 3. Dataset sintético (build + arma → score) ===")
    train_df = build_training_dataset(weapons)
    print(f"Shape do train_df: {train_df.shape[0]} linhas x {train_df.shape[1]} colunas")
    raw_min = train_df.attrs.get("score_min_raw", None)
    raw_max = train_df.attrs.get("score_max_raw", None)
    if raw_min is not None and raw_max is not None:
        print(f"Scores normalizados para 0–1 (min bruto={raw_min:.6f}, max bruto={raw_max:.6f})")
    print("Distribuição de builds no dataset sintético:")
    if "build_name" in train_df.columns:
        print(train_df["build_name"].value_counts())
    print("Amostra (10 linhas):")
    print(train_df.head(10))
    print("\nResumo estatístico global dos scores:")
    print(train_df["compatibility_score"].describe())
    if "build_name" in train_df.columns:
        print("\nMédia e desvio padrão dos scores por tipo de build:")
        print(train_df.groupby("build_name")["compatibility_score"].agg(["mean", "std", "min", "max"]))
    print("\nDistribuição dos scores (bins=10):")
    print(train_df["compatibility_score"].value_counts(bins=10, sort=False))

    print("\n=== 4. Treinando MLPRegressor ===")
    model, scaler = train_model(train_df)

    print("\n=== 4.1 Exemplos (score verdadeiro vs previsto) ===")
    sample = train_df.sample(min(10, len(train_df)), random_state=42)
    X_sample = sample[BUILD_FEATURES + WEAPON_FEATURES]
    y_true = sample["compatibility_score"].values
    X_sample_scaled = scaler.transform(X_sample)
    y_pred = model.predict(X_sample_scaled)
    for i in range(len(sample)):
        print(
            f"Exemplo {i+1}: true={y_true[i]:.4f}, pred={y_pred[i]:.4f}, "
            f"erro={abs(y_true[i]-y_pred[i]):.4f}, build={sample.iloc[i].get('build_name','?')}, "
            f"arma={sample.iloc[i].get('weapon_name','?')}"
        )

    print("\n=== 5. Resultado da recomendação (BATTLEMAGE) ===")
    build = BUILD_PRESETS["BATTLEMAGE"]
    print("Build usada:", build)
    ranked = recomendar_armas(build, weapons, model, scaler)
    top10 = ranked[["name", "category", "recommend_score"]].head(10)
    print("Top 10 armas recomendadas (ordenadas por recommend_score):")
    print(top10.to_string(index=False, formatters={"recommend_score": "{:.4f}".format}))

    print("\n=== 5.1 Outros exemplos de recomendação ===")
    for build_key in ["STR", "DEX", "INT"]:
        b = BUILD_PRESETS[build_key]
        rec = recomendar_armas(b, weapons, model, scaler).head(5)
        print(f"\nBuild {build_key}: {b}")
        print(rec[["name", "category", "recommend_score"]].to_string(index=False, formatters={"recommend_score": "{:.4f}".format}))


if __name__ == "__main__":
    main()
