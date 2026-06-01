"""Perfilado de gustos sobre Nivel 1 + asignación de contramedida primaria.

Lee:
- data/data_qc_gustos/master_table_gustos_v3_aggressive.parquet
- data/data_qc_gustos/two_stage_assignments.parquet

Escribe:
- data/data_qc_gustos/player_taste_profile.parquet
- data/data_qc_gustos/countermeasure_assignments.parquet
- data/data_qc_gustos/_tercile_thresholds.json   (puntos de corte estilo_build)

Ejecutable directamente. Llama a `build_profile()` y `assign_countermeasures()`
desde el notebook si se prefiere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "data_qc_gustos"

MASTER_PATH = DATA_DIR / "master_table_gustos_v3_aggressive.parquet"
ASSIGN_PATH = DATA_DIR / "two_stage_assignments.parquet"

OUT_PROFILE = DATA_DIR / "player_taste_profile.parquet"
OUT_CM = DATA_DIR / "countermeasure_assignments.parquet"
OUT_THRESHOLDS = DATA_DIR / "_tercile_thresholds.json"

ARCH_NAMES = {
    0: "Recién Llegado Explorador",
    1: "Jugador Establecido Activo",
    2: "Hardcore End-Game",
    3: "Veterano Especializado",
    4: "Casual Dormido",
    5: "Veterano Inversor",
}
CLASE_MAP = {0: "Caballero", 1: "Asesino", 2: "Campeón", 3: "Caballero"}

ARCH_ACTIVOS = {"Jugador Establecido Activo", "Hardcore End-Game"}
ARCH_ESTABLECIDO_O_VETERANO = {
    "Jugador Establecido Activo", "Veterano Especializado", "Veterano Inversor",
}
ARCH_VETERANOS = {"Veterano Especializado", "Veterano Inversor"}


def build_profile() -> tuple[pd.DataFrame, dict]:
    """Carga master+asignaciones y calcula los 7 ejes de gusto.

    Devuelve (df_perfil, thresholds) donde thresholds documenta los puntos de
    corte calculados sobre el sample (terciles, p75).
    """
    master = pd.read_parquet(MASTER_PATH)
    asg = pd.read_parquet(ASSIGN_PATH)
    df = master.merge(asg[["user_id", "nivel1_cluster", "has_tier2"]], on="user_id", how="inner")

    df["arquetipo_n1"] = df["nivel1_cluster"].map(ARCH_NAMES)

    # === EJE 1: clase_favorita ===
    df["clase_favorita"] = df["char_class_main"].fillna(0).astype(int).map(CLASE_MAP).fillna("Caballero")

    # === EJE 2: estilo_build (terciles de items_attack_defense_ratio) ===
    t33 = float(df["items_attack_defense_ratio"].quantile(1 / 3))
    t67 = float(df["items_attack_defense_ratio"].quantile(2 / 3))
    df["estilo_build"] = np.where(
        df["items_attack_defense_ratio"] >= t67, "agresivo",
        np.where(df["items_attack_defense_ratio"] <= t33, "tanque", "equilibrado"),
    )

    # === EJE 3: monetizacion (trial PRIMERO) ===
    df["monetizacion"] = np.where(
        df["iap_trial_only"] == 1, "trial_no_convertido",
        np.where(
            df["iap_is_payer"] == 1, "pagador",
            np.where(df["reward_has_ad"] == 1, "no_pagador_ads", "no_pagador"),
        ),
    )

    # === EJE 4: perfil_enhance (mediana) ===
    df["perfil_enhance"] = np.where(
        (df["items_max_enhance_level"] > 2) | (df["pct_items_high_enhance"] > 0),
        "inversor", "no_inversor",
    )

    # === EJE 5: perfil_coleccion ===
    p75_coll = float(df["coll_total_items"].quantile(0.75))
    df["perfil_coleccion"] = np.where(
        (df["items_redundancy_ratio"] > 1.5) | (df["coll_total_items"] > p75_coll),
        "coleccionista", "funcional",
    )

    # === EJE 6: pvp_perfil (Tier 2) ===
    pvp_frustrado = (
        (df["has_tier2"] == 1)
        & (df["fights_pct_pvp"].fillna(0) > 0.3)
        & (df["fights_pct_won"].fillna(0) < 0.3)
    )
    df["pvp_perfil"] = pd.Series(["null"] * len(df), index=df.index, dtype="object")
    df.loc[df["has_tier2"] == 1, "pvp_perfil"] = "pve_focus"
    df.loc[pvp_frustrado, "pvp_perfil"] = "pvp_frustrado"

    # === EJE 7: perfil_oro (Tier 2) ===
    df["perfil_oro"] = pd.Series(["null"] * len(df), index=df.index, dtype="object")
    t2_mask = df["has_tier2"] == 1
    inflow = df["currency_pct_inflow"]
    outflow = df["currency_pct_outflow"]
    df.loc[t2_mask & (inflow > 0.85), "perfil_oro"] = "acumulador"
    df.loc[t2_mask & (df["perfil_oro"] == "null") & (outflow > 0.25), "perfil_oro"] = "gastador"
    df.loc[t2_mask & (df["perfil_oro"] == "null"), "perfil_oro"] = "neutro"

    thresholds = {
        "estilo_build": {
            "feature": "items_attack_defense_ratio",
            "t33": t33,
            "t67": t67,
            "method": "terciles globales sobre el sample de gustos (N=114.412)",
        },
        "perfil_coleccion": {
            "feature": "coll_total_items",
            "p75": p75_coll,
            "method": "percentil 75 global sobre el sample de gustos",
        },
        "perfil_enhance": {
            "rule": "items_max_enhance_level > 2 OR pct_items_high_enhance > 0",
            "rationale": "items_max_enhance_level mediana = 2; pct_items_high_enhance mediana = 0",
        },
        "monetizacion": {
            "prelacion": ["trial_no_convertido", "pagador", "no_pagador_ads", "no_pagador"],
        },
        "pvp_perfil": {
            "rule": "pvp_frustrado si fights_pct_pvp > 0.3 AND fights_pct_won < 0.3 (con has_tier2=1); pve_focus en el resto del Tier 2; null si has_tier2=0",
        },
        "perfil_oro": {
            "rule": "acumulador si currency_pct_inflow > 0.85; gastador si currency_pct_outflow > 0.25; neutro en el resto; null si has_tier2=0",
        },
    }

    cols = [
        "user_id", "arquetipo_n1", "has_tier2",
        "clase_favorita", "estilo_build", "monetizacion",
        "perfil_enhance", "perfil_coleccion", "pvp_perfil", "perfil_oro",
    ]
    return df[cols].copy(), thresholds


# =========================================================
# Catálogo de 12 contramedidas + prioridad
# =========================================================
COUNTERMEASURES = [
    # (codigo, label_corto, priority_rank)
    ("CM03", "reconversion_trial",      1),
    ("CM01", "oferta_premium_clase",    2),
    ("CM02", "oferta_premium_dirigida", 3),
    ("CM10", "oferta_conversion_oro",   4),
    ("CM11", "ajuste_pvp",              5),
    ("CM12", "reto_guiado_clase",       6),
    ("CM04", "skin_clase",              7),
    ("CM06", "reto_coleccion",          8),
    ("CM05", "evento_enhance",          9),
    ("CM07", "drop_build",             10),
    ("CM08", "recompensa_ad",          11),
    ("CM09", "regalo_moneda",          12),
]


def assign_countermeasures(profile: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada jugador su contramedida primaria por el orden de prioridad.

    El orden es de más a menos específico (alta monetización → refinamiento Tier 2
    → gusto específico → genérica por arquetipo). El primer match gana.
    """
    n = len(profile)
    code = pd.Series(["CM00"] * n, index=profile.index, dtype="object")
    label = pd.Series(["sin_asignar"] * n, index=profile.index, dtype="object")

    def apply_rule(mask: pd.Series, cm_code: str, cm_label: str) -> None:
        m = mask & (code == "CM00")
        code.loc[m] = cm_code
        label.loc[m] = cm_label

    a = profile["arquetipo_n1"]
    mon = profile["monetizacion"]
    has_t2 = profile["has_tier2"] == 1

    # 1. CM03 — reconversion_trial
    apply_rule(mon == "trial_no_convertido", "CM03", "reconversion_trial")

    # 2. CM01 — oferta_premium_clase (Hardcore + pagador)
    apply_rule((a == "Hardcore End-Game") & (mon == "pagador"), "CM01", "oferta_premium_clase")

    # 3. CM02 — oferta_premium_dirigida (resto de pagadores)
    apply_rule((mon == "pagador") & (a != "Hardcore End-Game"), "CM02", "oferta_premium_dirigida")

    # 4. CM10 — oferta_conversion_oro (Tier 2 acumulador, no pagador)
    apply_rule(has_t2 & (profile["perfil_oro"] == "acumulador"), "CM10", "oferta_conversion_oro")

    # 5. CM11 — ajuste_pvp (Tier 2 pvp_frustrado)
    apply_rule(has_t2 & (profile["pvp_perfil"] == "pvp_frustrado"), "CM11", "ajuste_pvp")

    # 6. CM12 — reto_guiado_clase (Recién Llegado) — onboarding antes que gustos
    apply_rule(a == "Recién Llegado Explorador", "CM12", "reto_guiado_clase")

    # 7. CM04 — skin_clase (Veteranos no pagador)
    apply_rule(a.isin(ARCH_VETERANOS) & (mon != "pagador"), "CM04", "skin_clase")

    # 8. CM06 — reto_coleccion (coleccionista en arquetipos Establecido/Veteranos)
    apply_rule(
        (profile["perfil_coleccion"] == "coleccionista")
        & a.isin(ARCH_ESTABLECIDO_O_VETERANO),
        "CM06", "reto_coleccion",
    )

    # 9. CM05 — evento_enhance (inversor)
    apply_rule(profile["perfil_enhance"] == "inversor", "CM05", "evento_enhance")

    # 10. CM07 — drop_build (agresivo/tanque en arquetipos activos)
    apply_rule(
        profile["estilo_build"].isin(["agresivo", "tanque"]) & a.isin(ARCH_ACTIVOS),
        "CM07", "drop_build",
    )

    # 11. CM08 — recompensa_ad (no_pagador_ads)
    apply_rule(mon == "no_pagador_ads", "CM08", "recompensa_ad")

    # 12. CM09 — regalo_moneda (Casual Dormido no pagador)
    apply_rule((a == "Casual Dormido") & (mon == "no_pagador"), "CM09", "regalo_moneda")

    # FALLBACK final: cualquiera sin asignar → CM04 (skin_clase) usando su clase
    apply_rule(code == "CM00", "CM04", "skin_clase")

    out = profile.copy()
    out["contramedida_primaria_cod"] = code
    out["contramedida_primaria_label"] = label
    return out


def main() -> None:
    print(f"[paths]")
    print(f"  master  = {MASTER_PATH}")
    print(f"  assign  = {ASSIGN_PATH}")

    profile, thresholds = build_profile()
    print(f"\n[profile] {len(profile):,} filas")
    print(f"  has_tier2=1: {(profile['has_tier2']==1).sum():,}")
    print(f"  thresholds: t33={thresholds['estilo_build']['t33']:.4f}  t67={thresholds['estilo_build']['t67']:.4f}  p75_coll={thresholds['perfil_coleccion']['p75']:.0f}")

    # Persistir thresholds
    OUT_THRESHOLDS.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False))
    print(f"\n[write] {OUT_THRESHOLDS}")

    profile.to_parquet(OUT_PROFILE, index=False)
    print(f"[write] {OUT_PROFILE}  ({profile.shape})")

    cm = assign_countermeasures(profile)
    cm.to_parquet(OUT_CM, index=False)
    print(f"[write] {OUT_CM}  ({cm.shape})")

    # Distribución final
    dist = (
        cm.groupby(["contramedida_primaria_cod", "contramedida_primaria_label"])
        .size().reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    dist["pct"] = dist["n"] / len(cm) * 100
    print(f"\n=== Distribución de contramedidas primarias (N={len(cm):,}) ===")
    print(f"{'cod':<6} {'label':<28} {'N':>8} {'%':>6} {'flag'}")
    print("-" * 60)
    for _, row in dist.iterrows():
        flag = "<500" if row["n"] < 500 else ""
        print(f"{row['contramedida_primaria_cod']:<6} {row['contramedida_primaria_label']:<28} {row['n']:>8,} {row['pct']:>5.2f}% {flag}")

    # Marginales de los 7 ejes
    print(f"\n=== Marginales de los 7 ejes ===")
    for col in ["clase_favorita", "estilo_build", "monetizacion",
                "perfil_enhance", "perfil_coleccion", "pvp_perfil", "perfil_oro"]:
        print(f"\n  {col}:")
        vc = cm[col].value_counts(dropna=False)
        for k, v in vc.items():
            print(f"    {k:<25} {v:>7,}  ({v/len(cm)*100:>5.2f}%)")


if __name__ == "__main__":
    main()
