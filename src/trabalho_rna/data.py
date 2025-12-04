from __future__ import annotations

import ast
from typing import Dict, List

import pandas as pd

from trabalho_rna.constants import (
    ATTACK_MAP,
    CATEGORY_BASE_DAMAGE_WEIGHTS,
    CATEGORY_SCALING_WEIGHTS,
    DEFENCE_MAP,
    REQ_MAP,
    SCALE_COEFFICIENTS,
    SCALE_MAP,
)


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
    except Exception:
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
