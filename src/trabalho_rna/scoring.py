from __future__ import annotations

import numpy as np
import pandas as pd

from trabalho_rna.constants import BUILD_PRESETS, NUMERIC_COLUMNS


def compute_compatibility(build: dict, weapon: dict) -> float:
    build_vec = np.array(
        [build["STR"], build["DEX"], build["INT"], build["FAI"], build["ARC"]], dtype=float
    )
    build_sum = build_vec.sum() + 1e-6
    build_norm = build_vec / build_sum
    build_end = float(build.get("END", build.get("Endurance", 20)))

    req = np.array(
        [
            weapon["str_req"],
            weapon["dex_req"],
            weapon["int_req"],
            weapon["fai_req"],
            weapon["arc_req"],
        ],
        dtype=float,
    )
    req_sum = req.sum() + 1e-6
    req_norm = req / req_sum if req_sum > 0 else np.zeros_like(req)

    alignment_score = float(np.dot(build_norm, req_norm))

    req_with_floor = req + 1.0
    coverage_score = float(np.minimum(build_vec / req_with_floor, 1.0).mean())

    deficit = np.clip(req - build_vec, 0, None)
    deficit_penalty = float((deficit / (req + 1e-6)).mean())

    scales = np.array(
        [
            weapon.get("str_scale", 0.0),
            weapon.get("dex_scale", 0.0),
            weapon.get("int_scale", 0.0),
            weapon.get("fai_scale", 0.0),
            weapon.get("arc_scale", 0.0),
        ],
        dtype=float,
    )
    scale_sum = scales.sum() + 1e-6
    scale_norm = scales / scale_sum if scale_sum > 0 else np.zeros_like(scales)
    scaling_alignment = float(np.dot(build_norm, scale_norm))

    physical_part = weapon["physical"] + 0.5 * weapon["crit"]
    elemental_part = (
        weapon["magic"] + weapon["fire"] + weapon["lightning"] + weapon["holy"]
    )
    physical_weight = (build["STR"] + build["DEX"]) / build_sum
    elemental_weight = (build["INT"] + build["FAI"] + build["ARC"]) / build_sum
    damage_total = ((physical_part * physical_weight) + (elemental_part * elemental_weight)) / 1000.0
    align_boost = max(scaling_alignment, 0.05)
    damage_total *= align_boost
    damage_total *= coverage_score
    damage_total *= 0.4 + 0.6 * alignment_score
    damage_total *= weapon.get("base_damage_weight", 1.0)
    base_damage = (
        weapon["physical"] + weapon["magic"] + weapon["fire"] + weapon["lightning"] + weapon["holy"]
    )
    low_damage_factor = min(1.0, base_damage / 250.0)

    effective_weight = weapon["weight"] / (1 + build_end / 30.0)
    weight_penalty = 1 / (1 + effective_weight)
    scaling_alignment *= weapon.get("scaling_weight", 1.0)
    req_mismatch = float(np.clip(req_norm - build_norm, 0, None).sum())
    mismatch_factor = max(0.3, 1.0 - req_mismatch)

    score = (
        0.20 * alignment_score
        + 0.20 * coverage_score
        + 0.25 * damage_total
        + 0.20 * scaling_alignment
        + 0.10 * weight_penalty
        - 0.30 * deficit_penalty
    )
    score *= low_damage_factor * mismatch_factor
    return float(max(score, 0.0))


def build_training_dataset(weapons_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for build_name, build_stats in BUILD_PRESETS.items():
        for _, w in weapons_df.iterrows():
            weapon_feats = {col: w[col] for col in NUMERIC_COLUMNS}
            score = compute_compatibility(build_stats, weapon_feats)
            row = {
                "build_name": build_name,
                "build_str": build_stats["STR"],
                "build_dex": build_stats["DEX"],
                "build_int": build_stats["INT"],
                "build_fai": build_stats["FAI"],
                "build_arc": build_stats["ARC"],
                "build_end": build_stats.get("END", 20),
                "weapon_name": w.get("name", ""),
                "weapon_category": w.get("category", ""),
                "compatibility_score": score,
            }
            for col in NUMERIC_COLUMNS:
                row[f"weapon_{col}"] = w[col]
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
