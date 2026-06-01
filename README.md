# Sistema de predicción de abandono y recomendación de contramedidas en videojuegos free-to-play

**Trabajo de Fin de Grado** — Grado en Disseny i Producció de Videojocs  
**Autor:** Javier Ezquerro Fuentes  
**Tutor:** Manuel Rello Saltor  
**Curso:** 2025-2026  
**Empresa colaboradora:** Dark Curry — *Fight Legends*

---

## Resumen

Sistema de predicción de abandono (*churn*) y recomendación de contramedidas aplicado a Fight Legends, un videojuego free-to-play móvil. A partir de un dataset real de 1,16 millones de jugadores cedido por Dark Curry, el proyecto desarrolla tres bloques: un modelo predictivo de churn (Random Forest, AUC 0,91 a 14 días), una segmentación no supervisada en seis arquetipos de jugador (KMeans, silhouette 0,35), y una aplicación web desplegada que integra ambos modelos y genera informes operacionales para la empresa.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `01_preparacion_datos/` | ETL y feature engineering sobre las 19 tablas del dataset (notebooks 02a–02z) |
| `02_modelo_churn/` | Modelo predictivo de churn: comparativa de algoritmos, tuning, modelo final RF L22 |
| `03_modelo_gustos/` | Segmentación por arquetipos: clustering KMeans K=6 + sistema de perfilado de gustos |
| `04_estudio_validacion/` | Estudio experimental de 1.378 modelos sobre 45 configuraciones del grid |
| `05_cruce_y_segmentacion/` | Cruce churn × arquetipos y análisis de segmentos prioritarios |
| `06_modelos_publicados/` | Modelos serializados listos para producción |
| `data/` | Modelos auxiliares y schemas (los datos de jugadores no se incluyen por confidencialidad) |
| `reports/` | Figuras generadas para la memoria del TFG |
| `scripts/` | Scripts de generación de figuras y perfilado de gustos |

## Resultados principales

| Modelo | Métrica | Valor |
|---|---|---|
| Churn 7d | AUC-ROC test | 0,925 |
| Churn 14d | AUC-ROC test | 0,906 |
| Churn 30d | AUC-ROC test | 0,854 |
| Arquetipos N1 | Silhouette | 0,353 |

## Stack técnico

Python 3, pandas, numpy, scikit-learn, CatBoost, XGBoost, LightGBM, PyTorch, HDBSCAN, Optuna, matplotlib, seaborn.

## Repositorio de despliegue

La aplicación web (Streamlit + Hugging Face Spaces) está en un repositorio separado:

- **Código fuente:** [github.com/JEzquerroF/tfg-churn-deployment](https://github.com/JEzquerroF/tfg-churn-deployment)
- **Aplicación:** [huggingface.co/spaces/JEzquerroF/tfg-churn-prediction](https://huggingface.co/spaces/JEzquerroF/tfg-churn-prediction)
- **Modelos:** [huggingface.co/JEzquerroF/tfg-churn-models](https://huggingface.co/JEzquerroF/tfg-churn-models)

## Nota sobre los datos

Los datos de jugadores no se incluyen en el repositorio por confidencialidad (convenio universidad-empresa). Los notebooks son ejecutables si se dispone de los CSVs originales en `data/data_raw/`.