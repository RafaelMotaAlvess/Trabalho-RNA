from __future__ import annotations

from typing import Dict, List

NUMERIC_COLUMNS: List[str] = [
    "physical",
    "magic",
    "fire",
    "lightning",
    "holy",
    "crit",
    "guard_boost",
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
    "base_damage_weight",
    "scaling_weight",
]

BUILD_FEATURES = ["build_str", "build_dex", "build_int", "build_fai", "build_arc", "build_end"]
WEAPON_FEATURES = [f"weapon_{col}" for col in NUMERIC_COLUMNS]

BUILD_PRESETS = {
    "STR": {"STR": 80, "DEX": 20, "INT": 10, "FAI": 10, "ARC": 20, "END": 35},
    "DEX": {"STR": 20, "DEX": 80, "INT": 10, "FAI": 10, "ARC": 60, "END": 10},
    "INT": {"STR": 10, "DEX": 20, "INT": 80, "FAI": 20, "ARC": 10, "END": 20},
    "FAI": {"STR": 10, "DEX": 10, "INT": 20, "FAI": 80, "ARC": 10, "END": 20},
    "ARC": {"STR": 10, "DEX": 20, "INT": 20, "FAI": 10, "ARC": 80, "END": 20},
    "QUALITY": {"STR": 60, "DEX": 60, "INT": 20, "FAI": 20, "ARC": 20, "END": 30},
    "PALADIN": {"STR": 50, "DEX": 20, "INT": 20, "FAI": 60, "ARC": 10, "END": 30},
    "BATTLEMAGE": {"STR": 50, "DEX": 20, "INT": 60, "FAI": 20, "ARC": 10, "END": 25},
}

ATTACK_MAP: Dict[str, str] = {"Phy": "physical", "Mag": "magic", "Fire": "fire", "Ligt": "lightning", "Holy": "holy", "Crit": "crit"}
DEFENCE_MAP: Dict[str, str] = {"Boost": "guard_boost"}
REQ_MAP: Dict[str, str] = {"Str": "str_req", "Dex": "dex_req", "Int": "int_req", "Fai": "fai_req", "Arc": "arc_req"}
SCALE_MAP: Dict[str, str] = {"Str": "str_scale", "Dex": "dex_scale", "Int": "int_scale", "Fai": "fai_scale", "Arc": "arc_scale"}
SCALE_COEFFICIENTS: Dict[str, float] = {
    "S": 1.6,
    "A": 1.2,
    "B": 1.0,
    "C": 0.75,
    "D": 0.45,
    "E": 0.2,
}

CATEGORY_BASE_DAMAGE_WEIGHTS: Dict[str, float] = {
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

CATEGORY_SCALING_WEIGHTS: Dict[str, float] = {
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
