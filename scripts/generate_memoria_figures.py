"""Generación de figuras para la memoria del TFG (capítulo 6).

Genera 5 figuras (con una variante de la 6.7) en `reports/figuras_memoria_tfg/`:

- fig_6_3_roc_curves.png            — Curvas ROC del RF L22 (3 targets, split=test)
- fig_6_6_feature_importance.png    — Top 20 features Gini para churn_14d
- fig_6_7_screening_heatmap.png     — AUC test medio por (sample, algoritmo) churn_14d
- fig_6_7_screening_boxplot.png     — AUC por spike (3 vs 7 vs 14) — artefacto temporal
- fig_6_8_arquetipos_distribucion.png — Distribución N + tasa de churn por arquetipo
- fig_6_9_calibracion.png           — Reliability diagram RF L22 (3 targets)
- fig_6_10_confusion_matrices.png   — Matrices de confusión (TEST, threshold óptimo)
"""

from __future__ import annotations

import locale
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import json

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figuras_memoria_tfg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Configuración global ----------
try:
    locale.setlocale(locale.LC_NUMERIC, "es_ES.UTF-8")
    USE_LOCALE = True
except locale.Error:
    USE_LOCALE = False

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.formatter.use_locale"] = USE_LOCALE
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Paleta sobria, coherente con los SVG conceptuales de la memoria
COLOR_BLUE = "#185FA5"        # Bloque 1 / predictivo
COLOR_TEAL = "#0F6E56"        # Bloque 2 / segmentación
COLOR_CORAL = "#D85A30"       # output / churn rate
COLOR_AMBER = "#BA7517"       # decisión operacional
COLOR_GREY = "#5F5E5A"        # neutro

# Gradiente azul para los 3 targets (claro=corto, oscuro=largo)
BLUE_GRADIENT = {
    "p_churn_7d": "#7BAFE0",
    "p_churn_14d": "#185FA5",
    "p_churn_30d": "#0B3B6E",
}
TARGET_LABEL = {
    "p_churn_7d": "churn_7d",
    "p_churn_14d": "churn_14d",
    "p_churn_30d": "churn_30d",
}

# Nombres de los 6 arquetipos Nivel 1 según convención del proyecto
ARCHETYPE_N1_NAMES = {
    0: "Recién Llegado Explorador",
    1: "Jugador Establecido Activo",
    2: "Hardcore End-Game",
    3: "Veterano Especializado",
    4: "Casual Dormido",
    5: "Veterano Inversor",
}


# ---------- Helpers de formato numérico ----------
def fmt_es(value: float, decimals: int = 2) -> str:
    """Formato con coma decimal a la española."""
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value).replace(".", ",")


def comma_axis(ax, axis: str = "both", decimals: int = 2) -> None:
    """Fuerza coma decimal en ticks aunque locale falle."""
    formatter = mticker.FuncFormatter(lambda x, _: fmt_es(x, decimals))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


# ---------- Carga de datos ----------
def load_oof() -> pd.DataFrame:
    path = ROOT / "06_modelos_publicados" / "churn" / "v1_rf_L22_2026-05-19" / "oof_predictions_L22.parquet"
    return pd.read_parquet(path)


def load_feature_importance(target: str = "ch14d") -> pd.DataFrame:
    path = ROOT / "06_modelos_publicados" / "churn" / "v1_rf_L22_2026-05-19" / f"rf_L22_v1_{target}_feature_importance.csv"
    return pd.read_csv(path)


def load_screening() -> pd.DataFrame:
    p1 = pd.read_parquet(ROOT / "04_estudio_validacion" / "data" / "screening_results" / "_screening_results.parquet")
    p3 = pd.read_parquet(ROOT / "04_estudio_validacion" / "data" / "screening_results_extended" / "_extended_results.parquet")
    cols = ["config_id", "spike", "cutoff", "min_logins", "algorithm", "cleanup", "target", "test_auc_roc"]
    return pd.concat([p1[cols], p3[cols]], ignore_index=True)


def load_archetypes() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "06_modelos_publicados" / "gustos" / "v1_kmeans_hdbscan_2026-05-12" / "two_stage_assignments.parquet")


# =========================================================
# Fig 6.3 — Curvas ROC del modelo final RF L22 (3 targets)
# =========================================================
def fig_6_3_roc_curves(oof: pd.DataFrame) -> dict:
    test = oof[oof["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 6))
    auc_values = {}

    for p_col, y_col in [
        ("p_churn_7d", "y_churn_7d"),
        ("p_churn_14d", "y_churn_14d"),
        ("p_churn_30d", "y_churn_30d"),
    ]:
        y_true = test[y_col].to_numpy()
        y_score = test[p_col].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        auc_values[p_col] = auc
        ax.plot(
            fpr, tpr,
            color=BLUE_GRADIENT[p_col],
            linewidth=2.2,
            label=f"{TARGET_LABEL[p_col]} (AUC = {fmt_es(auc, 4)})",
        )

    ax.plot([0, 1], [0, 1], color=COLOR_GREY, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    comma_axis(ax, axis="both", decimals=1)

    out = OUT_DIR / "fig_6_3_roc_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(out), "auc": auc_values, "n_test": len(test)}


# =========================================================
# Fig 6.6 — Feature importance top 20 RF L22 churn_14d
# =========================================================
def fig_6_6_feature_importance(fi: pd.DataFrame, top_n: int = 20) -> dict:
    top = fi.sort_values("importance", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["importance"], color=COLOR_BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Importancia (Gini)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    comma_axis(ax, axis="x", decimals=2)

    out = OUT_DIR / "fig_6_6_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(out),
        "top5": list(zip(top["feature"].tolist()[::-1][:5], top["importance"].tolist()[::-1][:5])),
        "total_features_csv": len(fi),
    }


# =========================================================
# Fig 6.7 — Screening 1.378 modelos (heatmap + boxplot)
# =========================================================
def fig_6_7_screening_heatmap(scr: pd.DataFrame) -> dict:
    df = scr[scr["target"] == "churn_14d"].copy()
    pivot = df.pivot_table(
        index="config_id",
        columns="algorithm",
        values="test_auc_roc",
        aggfunc="mean",
    )

    # Ordenar muestras por spike (3, 7, 14) para visualizar el artefacto temporal
    spike_by_sample = df.groupby("config_id")["spike"].first()
    cutoff_by_sample = df.groupby("config_id")["cutoff"].first()
    order = (
        pd.DataFrame({"spike": spike_by_sample, "cutoff": cutoff_by_sample})
        .sort_values(["spike", "cutoff"])
        .index.tolist()
    )
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(10, 11))
    vmin, vmax = pivot.min().min(), pivot.max().max()
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Algoritmo")
    ax.set_ylabel("Muestra (config_id)")

    # Anotaciones
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt_color = "black" if 0.7 < v < 0.95 else "white"
                ax.text(j, i, fmt_es(v, 2), ha="center", va="center",
                        fontsize=7, color=txt_color)

    # Líneas divisorias entre bloques de spike
    samples_per_spike = [
        (s, [i for i, cid in enumerate(pivot.index) if spike_by_sample[cid] == s])
        for s in [3, 7, 14]
    ]
    for s, idxs in samples_per_spike:
        if idxs:
            ax.axhline(y=idxs[-1] + 0.5, color="black", linewidth=1.2, alpha=0.8)

    # Etiquetas laterales de spike
    for s, idxs in samples_per_spike:
        if idxs:
            mid = (idxs[0] + idxs[-1]) / 2
            ax.text(
                pivot.shape[1] - 0.3, mid,
                f"spike={s}",
                rotation=270, va="center", ha="left",
                fontsize=9, color=COLOR_GREY,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.10)
    cbar.set_label("AUC test (churn_14d)")
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: fmt_es(x, 2))
    )

    out = OUT_DIR / "fig_6_7_screening_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(out),
        "n_models_ch14d": len(df),
        "samples": pivot.shape[0],
        "algorithms": pivot.columns.tolist(),
        "auc_min": float(vmin),
        "auc_max": float(vmax),
    }


def fig_6_7_screening_boxplot(scr: pd.DataFrame) -> dict:
    df = scr[scr["target"] == "churn_14d"].copy()
    spike_levels = [3, 7, 14]
    data = [df[df["spike"] == s]["test_auc_roc"].dropna().to_numpy() for s in spike_levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    box_colors = [COLOR_BLUE, COLOR_TEAL, COLOR_CORAL]
    bp = ax.boxplot(
        data,
        positions=range(len(spike_levels)),
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3.5, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)

    ax.set_xticks(range(len(spike_levels)))
    ax.set_xticklabels([f"spike={s}" for s in spike_levels])
    ax.set_xlabel("Configuración de spike (días)")
    ax.set_ylabel("AUC test (churn_14d)")
    ax.grid(axis="y", alpha=0.3)
    comma_axis(ax, axis="y", decimals=2)

    # Anotar medianas
    for i, arr in enumerate(data):
        if len(arr):
            med = float(np.median(arr))
            ax.text(i, med + 0.005, fmt_es(med, 3), ha="center", fontsize=9, color="black")

    out = OUT_DIR / "fig_6_7_screening_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(out),
        "n_per_spike": {s: int(len(arr)) for s, arr in zip(spike_levels, data)},
        "median_per_spike": {s: float(np.median(arr)) if len(arr) else None for s, arr in zip(spike_levels, data)},
    }


# =========================================================
# Fig 6.8 — Distribución de los 6 arquetipos
# =========================================================
def fig_6_8_arquetipos(asg: pd.DataFrame, oof: pd.DataFrame) -> dict:
    overlap = asg.merge(
        oof[["user_id", "y_churn_7d", "y_churn_14d", "y_churn_30d"]],
        on="user_id", how="inner",
    )

    # Orden explícito solicitado (no por tamaño nominal; fija la presentación)
    ORDER_CIDS = [4, 1, 5, 0, 2, 3]
    rows = []
    for cluster_id in ORDER_CIDS:
        name = ARCHETYPE_N1_NAMES[cluster_id]
        pop_total = int((asg["nivel1_cluster"] == cluster_id).sum())
        sub = overlap[overlap["nivel1_cluster"] == cluster_id]
        n_ovl = len(sub)
        rows.append({
            "cluster": cluster_id,
            "name": name,
            "n_pop": pop_total,
            "n_overlap": n_ovl,
            "cobertura": n_ovl / pop_total if pop_total else np.nan,
            "rate_7d": sub["y_churn_7d"].mean() if n_ovl else np.nan,
            "rate_14d": sub["y_churn_14d"].mean() if n_ovl else np.nan,
            "rate_30d": sub["y_churn_30d"].mean() if n_ovl else np.nan,
        })
    summary = pd.DataFrame(rows)

    targets = [
        ("rate_7d", "churn_7d", COLOR_BLUE),
        ("rate_14d", "churn_14d", COLOR_TEAL),
        ("rate_30d", "churn_30d", COLOR_CORAL),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    n_groups = len(summary)
    n_targets = len(targets)
    bar_width = 0.26
    group_x = np.arange(n_groups)

    for i, (col, label, color) in enumerate(targets):
        offset = (i - (n_targets - 1) / 2) * bar_width
        values_pct = summary[col].to_numpy() * 100
        bars = ax.bar(
            group_x + offset, values_pct,
            width=bar_width,
            color=color, edgecolor="white", linewidth=0.6,
            label=label,
        )
        for bar, val in zip(bars, values_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                fmt_es(val, 1),
                ha="center", va="bottom", fontsize=8, color="black",
            )

    # Etiquetas eje X: nombre + cobertura
    xtick_labels = [
        f"{row['name']}\n({fmt_es(row['cobertura'] * 100, 1)} % cobertura)"
        for _, row in summary.iterrows()
    ]
    ax.set_xticks(group_x)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=9)

    ax.set_ylabel("Tasa de churn observada (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 0)))
    ax.legend(loc="upper right", frameon=False, ncol=3, title="Target")

    out = OUT_DIR / "fig_6_8_arquetipos_distribucion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(out),
        "order": [ARCHETYPE_N1_NAMES[c] for c in ORDER_CIDS],
        "summary": summary.to_dict(orient="records"),
        "n_overlap_total": int(summary["n_overlap"].sum()),
    }


# =========================================================
# Fig 6.9 — Reliability diagram (3 targets RF L22)
# =========================================================
def fig_6_9_calibracion(oof: pd.DataFrame) -> dict:
    test = oof[oof["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 6))

    rows = []
    for p_col, y_col in [
        ("p_churn_7d", "y_churn_7d"),
        ("p_churn_14d", "y_churn_14d"),
        ("p_churn_30d", "y_churn_30d"),
    ]:
        y_true = test[y_col].to_numpy()
        y_prob = test[p_col].to_numpy()
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(
            mean_pred, frac_pos,
            color=BLUE_GRADIENT[p_col],
            marker="o", markersize=5,
            linewidth=2,
            label=TARGET_LABEL[p_col],
        )
        # Diferencia media absoluta como aproximación de ECE
        ece_approx = float(np.mean(np.abs(frac_pos - mean_pred)))
        rows.append({"target": TARGET_LABEL[p_col], "ece_approx": ece_approx})

    ax.plot([0, 1], [0, 1], color=COLOR_GREY, linestyle="--", linewidth=1, alpha=0.7,
            label="Calibración perfecta")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    comma_axis(ax, axis="both", decimals=1)

    out = OUT_DIR / "fig_6_9_calibracion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(out), "ece_approx_per_target": rows, "n_test": len(test), "n_bins": 10}


# =========================================================
# Fig 6.10 — Matrices de confusión (3 targets, TEST, threshold óptimo)
# =========================================================
def _load_threshold(target_key: str) -> float:
    path = ROOT / "06_modelos_publicados" / "churn" / "v1_rf_L22_2026-05-19" / f"rf_L22_v1_{target_key}_metrics.json"
    with open(path) as f:
        return float(json.load(f)["test_optimal_threshold"])


def fig_6_10_confusion_matrices(oof: pd.DataFrame) -> dict:
    test = oof[oof["split"] == "test"]
    n_test = len(test)

    targets = [
        ("p_churn_7d", "y_churn_7d", "churn_7d", _load_threshold("ch7d")),
        ("p_churn_14d", "y_churn_14d", "churn_14d", _load_threshold("ch14d")),
        ("p_churn_30d", "y_churn_30d", "churn_30d", _load_threshold("ch30d")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    info = []

    for ax, (p_col, y_col, label, thr) in zip(axes, targets):
        y_true = test[y_col].to_numpy()
        y_prob = test[p_col].to_numpy()
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = (tp + tn) / n_test

        im = ax.imshow(cm, cmap="Blues", aspect="equal")

        # Anotaciones por celda: conteo + (%)
        max_val = cm.max()
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = (val / n_test) * 100
                color = "white" if val > max_val * 0.55 else "black"
                ax.text(
                    j, i,
                    f"{val:,}".replace(",", ".") + f"\n({fmt_es(pct, 1)} %)",
                    ha="center", va="center", fontsize=12, color=color,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No churn\n(predicho)", "Churn\n(predicho)"], fontsize=9)
        ax.set_yticklabels(["No churn\n(real)", "Churn\n(real)"], fontsize=9, rotation=0)
        ax.set_xlabel(f"{label}    (umbral = {fmt_es(thr, 3)})", fontsize=10)

        # Métricas debajo del subplot
        metrics_txt = (
            f"Precisión = {fmt_es(prec, 3)}    "
            f"Recall = {fmt_es(rec, 3)}    "
            f"F1 = {fmt_es(f1, 3)}    "
            f"Accuracy = {fmt_es(accuracy, 3)}"
        )
        ax.text(
            0.5, -0.32, metrics_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9.5, color="black",
        )

        # Líneas divisorias claras entre celdas
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

        info.append({
            "target": label,
            "threshold": thr,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(accuracy),
        })

    plt.subplots_adjust(wspace=0.35, bottom=0.22)

    out = OUT_DIR / "fig_6_10_confusion_matrices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": str(out), "n_test": int(n_test), "per_target": info}


# ---------- Main ----------
def main() -> None:
    print(f"[locale es_ES.UTF-8] {'OK' if USE_LOCALE else 'NO disponible — formato manual'}", flush=True)

    print("Cargando datos...", flush=True)
    oof = load_oof()
    fi14 = load_feature_importance("ch14d")
    scr = load_screening()
    asg = load_archetypes()

    print(f"  OOF: {len(oof):,} filas | split test: {(oof['split']=='test').sum():,}", flush=True)
    print(f"  Feature importance ch14d: {len(fi14)} features", flush=True)
    print(f"  Screening total: {len(scr):,} filas | churn_14d: {(scr['target']=='churn_14d').sum():,}", flush=True)
    print(f"  Assignments: {len(asg):,} jugadores", flush=True)
    print()

    results = {}
    print("Fig 6.3 — ROC curves...", flush=True)
    results["6.3"] = fig_6_3_roc_curves(oof)
    print(f"  AUC test: {results['6.3']['auc']}", flush=True)

    print("Fig 6.6 — Feature importance churn_14d...", flush=True)
    results["6.6"] = fig_6_6_feature_importance(fi14, top_n=20)
    print(f"  Top 5: {results['6.6']['top5']}", flush=True)

    print("Fig 6.7 — Screening heatmap...", flush=True)
    results["6.7_heatmap"] = fig_6_7_screening_heatmap(scr)
    print(f"  {results['6.7_heatmap']['n_models_ch14d']} modelos × {results['6.7_heatmap']['samples']} muestras × {len(results['6.7_heatmap']['algorithms'])} algoritmos", flush=True)

    print("Fig 6.7 — Screening boxplot...", flush=True)
    results["6.7_boxplot"] = fig_6_7_screening_boxplot(scr)
    print(f"  Medianas por spike: {results['6.7_boxplot']['median_per_spike']}", flush=True)

    print("Fig 6.8 — Arquetipos...", flush=True)
    results["6.8"] = fig_6_8_arquetipos(asg, oof)
    print(f"  N overlap total: {results['6.8']['n_overlap_total']:,}", flush=True)
    for row in results["6.8"]["summary"]:
        print(
            f"    {row['name']:<28} N_pop={row['n_pop']:>6,} N_ovl={row['n_overlap']:>5,} "
            f"cob={row['cobertura']*100:>5.1f}% | "
            f"7d={row['rate_7d']*100:>5.2f}% 14d={row['rate_14d']*100:>5.2f}% 30d={row['rate_30d']*100:>5.2f}%",
            flush=True,
        )

    print("Fig 6.9 — Calibración...", flush=True)
    results["6.9"] = fig_6_9_calibracion(oof)
    print(f"  ECE aproximado: {results['6.9']['ece_approx_per_target']}", flush=True)

    print("Fig 6.10 — Matrices de confusión...", flush=True)
    results["6.10"] = fig_6_10_confusion_matrices(oof)
    print(f"  N test: {results['6.10']['n_test']:,}", flush=True)
    for row in results["6.10"]["per_target"]:
        print(
            f"    {row['target']:<10} thr={row['threshold']:.4f} | "
            f"TN={row['tn']:>4} FP={row['fp']:>4} FN={row['fn']:>4} TP={row['tp']:>4} | "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} Acc={row['accuracy']:.4f}",
            flush=True,
        )

    print()
    print("=" * 60)
    print("RESUMEN — archivos generados en:")
    print(f"  {OUT_DIR}")
    print("=" * 60)
    for key, val in results.items():
        print(f"  [{key}] {val['path']}")


if __name__ == "__main__":
    main()
