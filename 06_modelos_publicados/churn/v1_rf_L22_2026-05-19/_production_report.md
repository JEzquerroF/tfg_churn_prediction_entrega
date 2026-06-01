# Modelo final productivo — Reporte

**Fecha**: 2026-05-19 10:55
**Sample**: L22 (cutoff=90, spike=7, min_logins=2)
**Cleanup**: v1_conservative
**Algoritmo**: Random Forest
**N usuarios**: 33,598

## Métricas finales por target

| Target | OOF AUC | Val AUC | Test AUC | Overfit | Train n | Val n | Test n |
|---|---:|---:|---:|---:|---:|---:|---:|
| churn_14d | 0.8939 | 0.9451 | 0.9057 | -0.0118 | 23,531 | 5,027 | 5,040 |
| churn_30d | 0.8407 | 0.8949 | 0.8543 | -0.0136 | 23,531 | 5,027 | 5,040 |
| churn_7d | 0.9279 | 0.9626 | 0.9254 | +0.0025 | 23,531 | 5,027 | 5,040 |

## Predicciones OOF

- Archivo: `predictions/oof_predictions_L22.parquet`
- Total filas: 33,598
- Splits: {'train': 23531, 'test': 5040, 'val': 5027}

### Columnas de la tabla OOF

- `user_id`: identificador único del jugador
- `split`: train / val / test
- `sample`, `cleanup`, `algorithm`: metadatos
- `p_churn_7d`, `p_churn_14d`, `p_churn_30d`: probabilidades predichas
- `y_churn_7d`, `y_churn_14d`, `y_churn_30d`: targets reales (para validación)

**Uso**: cruzar con archetipos de gustos para análisis de segmentos.

## Top 10 features por target

### ch14d

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `items_days_since_last_item` | 0.2774 |
| 2 | `coll_days_since_last_item` | 0.1423 |
| 3 | `user_player_lifespan_days` | 0.1079 |
| 4 | `char_last_updated_days_ago` | 0.0809 |
| 5 | `reward_last_claim_days_ago` | 0.0591 |
| 6 | `reward_current_day_max` | 0.0513 |
| 7 | `coll_first_item_days_ago` | 0.0497 |
| 8 | `reward_first_created_days_ago` | 0.0198 |
| 9 | `items_first_item_days_ago` | 0.0157 |
| 10 | `country` | 0.0111 |

### ch30d

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `items_days_since_last_item` | 0.2559 |
| 2 | `coll_days_since_last_item` | 0.1548 |
| 3 | `user_player_lifespan_days` | 0.0804 |
| 4 | `coll_first_item_days_ago` | 0.0666 |
| 5 | `reward_last_claim_days_ago` | 0.0542 |
| 6 | `reward_current_day_max` | 0.0483 |
| 7 | `char_last_updated_days_ago` | 0.0451 |
| 8 | `reward_first_created_days_ago` | 0.0213 |
| 9 | `device_days_since_last_active` | 0.0141 |
| 10 | `country` | 0.0134 |

### ch7d

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `items_days_since_last_item` | 0.2858 |
| 2 | `coll_days_since_last_item` | 0.1419 |
| 3 | `user_player_lifespan_days` | 0.1156 |
| 4 | `char_last_updated_days_ago` | 0.0774 |
| 5 | `reward_last_claim_days_ago` | 0.0743 |
| 6 | `reward_current_day_max` | 0.0618 |
| 7 | `coll_first_item_days_ago` | 0.0492 |
| 8 | `reward_first_created_days_ago` | 0.0164 |
| 9 | `items_first_item_days_ago` | 0.0139 |
| 10 | `country` | 0.0096 |

## Hiperparámetros finales

### churn_14d

```json
{"n_estimators": 248, "max_depth": 12, "min_samples_split": 11, "min_samples_leaf": 3, "max_features": 0.5, "class_weight": null}
```

### churn_30d

```json
{"n_estimators": 187, "max_depth": 10, "min_samples_split": 8, "min_samples_leaf": 3, "max_features": 0.5, "class_weight": null}
```

### churn_7d

```json
{"n_estimators": 248, "max_depth": 12, "min_samples_split": 11, "min_samples_leaf": 3, "max_features": 0.5, "class_weight": null}
```

## Archivos generados

```
production/outputs/
├── models/
│   ├── rf_L22_v1_ch7d.pkl
│   ├── rf_L22_v1_ch14d.pkl
│   └── rf_L22_v1_ch30d.pkl
├── predictions/
│   ├── oof_predictions_L22.parquet  ← TABLA MAESTRA
│   └── oof_predictions_L22.csv
├── metrics/
│   ├── rf_L22_v1_ch7d_metrics.json
│   ├── rf_L22_v1_ch14d_metrics.json
│   └── rf_L22_v1_ch30d_metrics.json
├── feature_importance/
│   ├── rf_L22_v1_ch7d_feature_importance.csv
│   ├── rf_L22_v1_ch14d_feature_importance.csv
│   └── rf_L22_v1_ch30d_feature_importance.csv
└── _production_report.md (este archivo)
```

## Próximos pasos

1. **Cascade B (gustos)**: cruzar `oof_predictions_L22.parquet` con archetipos
2. **Despliegue**: integrar los `.pkl` en el pipeline de predicción
