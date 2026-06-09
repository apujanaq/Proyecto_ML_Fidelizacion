# Modelo de Fidelización y Recompra
### Proyecto Final — Módulo Machine Learning
**Alejandro Pujana Quintero · Evolve Academy · 2026**

---

## Descripción del proyecto

Modelo de clasificación binaria que predice qué clientes de un retailer online londinense volverán a comprar en los tres meses siguientes a la fecha de corte (octubre – diciembre 2011).

**Pregunta de negocio:** ¿A qué clientes debería contactar el equipo comercial para maximizar el retorno de una campaña de retención?

**Resultado:** Lista priorizada de 1.088 clientes del test set con probabilidad de recompra, segmento asignado (Alta / Media / Baja) y métricas RFM en valores originales (£, días, facturas).

---

## Dataset

**Fuente:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — Kaggle / UC Irvine ML Repository

| Archivo | Período | Filas aprox. |
|---|---|---|
| `Year 2009-2010.csv` | 01/12/2009 → 09/12/2010 | ~525.000 |
| `Year 2010-2011.csv` | 01/12/2010 → 09/12/2011 | ~542.000 |

Los archivos CSV crudos están incluidos en `DATA/raw_data/`. Los archivos procesados (`.pkl`) se generan ejecutando el notebook y **no están en el repositorio** (ver `.gitignore`).

---

## Arquitectura del modelo

```
1.067.000 transacciones brutas
        ↓  Limpieza (sección 2)
  780.000 transacciones válidas
        ↓  Aggregación RFM por cliente (sección 4)
    5.438 clientes (1 fila/cliente)
        ↓  Feature Engineering + split 80/20 (sección 5)
    4.350 train  |  1.088 test
        ↓  Modelado (secciones 6–7)
    Regresión Logística / Random Forest / XGBoost / XGBoost Optuna
        ↓  Evaluación y explicabilidad (secciones 8–9)
    Lista de clientes + SHAP values
```

**Fecha de corte:** 30 de septiembre de 2011
- Período de observación (features): dic 2009 → sept 2011 (21 meses)
- Período de etiqueta (target): oct 2011 → dic 2011 (2.5 meses)

**Variable objetivo:** `recompra` — binaria (0 = no recompra, 1 = recompra)
- Distribución: 39,0% positivos / 61,0% negativos

---

## Features del modelo (11 variables)

| Feature | Tipo | Descripción |
|---|---|---|
| `recencia` | Continua | Días desde la última compra hasta la fecha de corte |
| `frecuencia` | Continua | Número de facturas únicas en el período de observación |
| `monetario` | Continua | Revenue total (£) en el período de observación |
| `ticket_medio` | Continua | Monetario / Frecuencia — detecta mayoristas |
| `antiguedad` | Continua | Días entre primera y última compra |
| `num_prod_dist` | Continua | Número de productos distintos comprados |
| `cantidad_media_factura` | Continua | Unidades medias por factura |
| `distancia_temporal_media` | Continua | Días medios entre compras consecutivas |
| `es_uk` | Binaria | 1 si el cliente es del Reino Unido (91% del dataset) |
| `ratio_cancelaciones` | Continua | Unidades canceladas / Unidades compradas totales |
| `tiene_cancelaciones` | Binaria | 1 si el cliente ha cancelado alguna vez |

Transformaciones aplicadas: `log1p` en recencia, frecuencia, monetario y antigüedad (corrección de skewness). Escalado: `StandardScaler` fit solo en train.

---

## Modelos entrenados

| Modelo | AUC-ROC | Notas |
|---|---|---|
| Regresión Logística | ~0.83 | Baseline; interpretable via coeficientes β |
| Random Forest | ~0.81 | Ensemble paralelo de árboles |
| XGBoost (default) | ~0.80 | Sin optimización de hiperparámetros |
| **XGBoost (Optuna)** | **~0.83** | 60 trials de búsqueda bayesiana, CV 5-fold |

> Los valores exactos de AUC varían ligeramente según el estado del generador aleatorio. LR y XGBoost Optuna se sitúan prácticamente empatados — resultado esperado dado el Feature Engineering que linealiza las features.

**Métrica principal:** AUC-ROC (equivalente al C-statistic de STATA para modelos logit/probit).

---

## Estructura del repositorio

```
Proyecto-ML-Fidelizacion/
├── notebook/
│   └── Fidelizacion_Recompra_ML_Pujana.ipynb   ← notebook principal (147 celdas)
├── DATA/
│   └── raw_data/
│       ├── online_retail_II.xlsx - Year 2009-2010.csv
│       └── online_retail_II.xlsx - Year 2010-2011.csv
├── requirements.txt
└── README.md
```

> Los archivos `.pkl` generados por el notebook (datos procesados, modelos entrenados) no están en el repositorio — se generan automáticamente al ejecutar el notebook completo.

---

## Reproducibilidad — cómo ejecutar el notebook

### Requisitos previos

- Python 3.10 o superior
- **macOS:** `brew install libomp` (requerido por XGBoost)
- **Linux/Windows:** sin dependencias adicionales de sistema

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/apujanaq/Proyecto_ML_Fidelizacion.git
cd Proyecto_ML_Fidelizacion

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar JupyterLab
jupyter lab
```

### Ejecución

Abrir `notebook/Fidelizacion_Recompra_ML_Pujana.ipynb` y ejecutar **Kernel → Restart & Run All**.

El notebook **crea automáticamente la carpeta `DATA/clean_data/`** y genera ahí todos los archivos procesados (`.pkl`). No es necesario crearla a mano. La primera ejecución tarda ~3–5 minutos (sección 7 — Optuna con 60 trials).

### Tiempos de ejecución estimados

| Sección | Tiempo |
|---|---|
| 1–4 (carga, limpieza, EDA, dataset) | ~60 seg |
| 5 (Feature Engineering) | ~5 seg |
| 6 (Modelado — 3 modelos) | ~10 seg |
| **7 (Optuna — 60 trials × CV 5-fold)** | **~3–4 min** |
| 8–10 (evaluación, SHAP, conclusiones) | ~30 seg |

---

## Entorno de desarrollo

| Dependencia | Versión probada |
|---|---|
| Python | 3.12 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| optuna | latest |
| shap | 0.52.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| jupyterlab | 4.5.6 |

---

## Resultados principales

- **Modelo seleccionado:** determinado dinámicamente por AUC-ROC máximo (con criterio de desempate por interpretabilidad si la diferencia es < 0.005).
- **Segmento de mayor ROI:** clientes con probabilidad 40–70% (*Media*) — la mayoría del test set se concentra aquí.
- **Variable más influyente:** recencia — con efecto no lineal identificado por SHAP (umbral crítico ~200–250 días).
- **Punto ciego del modelo:** compradores estacionales (alto `distancia_temporal_media`) — clasificados en Baja pero fielmente recurrentes cada año.
- **Entregable de negocio:** `DATA/clean_data/preprocessed/lista_clientes_probabilidad.csv` — lista completa con probabilidad, segmento y RFM para el equipo comercial (generado al ejecutar el notebook).

---

*Proyecto Final Módulo ML · Máster en Ciencia de Datos e IA · Evolve Academy · 2025*
