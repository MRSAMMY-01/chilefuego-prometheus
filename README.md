# SparkPredict AI 🌲🔥

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ML Engine](https://img.shields.io/badge/Model-XGBoost%20Ensemble-orange.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Fase%204G%20(Piloto%20Operacional)-emerald.svg)]()
[![Validation](https://img.shields.io/badge/Tests-94%20Passed%20(100%25)-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![Type Checking](https://img.shields.io/badge/Mypy-Strict%20Verified-blue.svg)](pyproject.toml)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Motor MLOps de Predicción Espacio-Temporal de Riesgo de Ignición Forestal a Escala Micro-Territorial (1 km²)**

[Tecnología](#-arquitectura-y-pipeline-el-core-técnico) • [Validación Científica](#-validación-científica-tracción) • [Tolerancia a Fallos](#-tolerancia-a-fallos-y-confiabilidad-mlops) • [Instalación Rápida](#-instalación-rápida) • [Contacto](#-autor-y-liderazgo)

</div>

---

## 📌 Descripción Ejecutiva

**SparkPredict AI** es un motor de inteligencia artificial y MLOps predictivo de última generación diseñado para estimar probabilísticamente el riesgo diario de ignición de incendios forestales con resolución micro-territorial (mallas regulares de 1 km² / 13.108 celdas en la Región de Ñuble, Chile). El sistema opera en modo sombra desatendido (*shadow production*), prediciendo simultáneamente para el mismo día (Horizonte 0) y el día siguiente (Horizonte 1 / D+1). 

Al fusionar reanálisis climático continuo ERA5-Land, pronósticos meteorológicos numéricos en tiempo real (NWP NOAA GFS), cálculo dinámico del Índice Meteorológico Canadiense de Incendios (*Fire Weather Index*, FWI) y densidades espaciales no paramétricas históricas (*priors* georreferenciados de CONAF), SparkPredict AI transforma el despacho preventivo de brigadas y la gestión de recursos de combate en una disciplina analítica de precisión matemática.

---

## ⚙️ Arquitectura y Pipeline (El Core Técnico)

El sistema implementa un flujo de trabajo productivo altamente desacoplado, diseñado para ejecutar de forma autónoma a las 07:00 CLT cada mañana:

```
[ Telemetría Meteorológica NOAA GFS ] ──> [ Monotonic Quantile Mapping ] ──┐
                                                                           │
[ Reanálisis Continuo ERA5-Land ] ────> [ Canadian FWI Engine (xclim) ] ──┼──> [ XGBoost Ensemble (10 Seeds) ] ──> [ Mapeo ECDF Calibrado ]
                                                                           │                      │                                 │
[ Catastro Geoespacial / OSM / Topo ] ─> [ Static Features 1 km² ] ────────┘                      │                                 ▼
                                                                                                  │                   [ Score Híbrido Operativo ]
                                                                                                  ▼                                 │
                                                                           [ Healthcheck UTF-8 & Schemas ]                  ▼
                                                                                                  │                   [ Outputs: Parquet / GeoJSON ]
                                                                                                  ▼                                 │
                                                                           [ Prospective Log (Trazabilidad) ] <─────────────┘
```

### 1. Orquestación Desatendida (`scripts/schedule_daily.ps1`)
* Diseñado para Windows Task Scheduler, orquesta la corrida diaria de forma estocástica e idempotente.
* Incorpora reintentos automáticos con retroceso exponencial (*exponential backoff*), captura exhaustiva de flujos `stdout` y `stderr` codificados en UTF-8 y verificación estricta de códigos de salida (`ExitCode 0`).

### 2. Motor de Inferencia en Tiempo Real (`scripts/run_daily.py`)
* **Telemetría NWP en Vivo:** Conexión automatizada a Open-Meteo GFS *seamless* para extraer a nivel de nodo ERA5 temperatura a 2m, humedad relativa, velocidad de viento y precipitación acumulada de 24 horas al mediodía solar (16:00 UTC / 12:00 CLT).
* **Mapeo de Cuantiles Monótono (*Quantile Mapping*):** Alinea contractualmente las distribuciones climáticas del pronóstico operacional con el reanálisis histórico ERA5-Land, erradicando sesgos sistemáticos entre sensores y modelos numéricos.
* **FWI Dinámico e Idempotente:** Computa serie temporal diaria del FWI (códigos de humedad de combustibles FFMC, DMC, DC y los índices de propagación ISI, BUI y FWI).
* **Ensamble XGBoost Canónico:** Inferencia probabilística basada en un ensamble de 10 semillas entrenadas bajo muestreo estratificado 20:1, evaluando 25 covariables espaciales y climáticas podadas.

### 3. Persistencia Columnar y Healthchecks Operacionales
* Genera capas espaciales de inferencia en Apache Parquet (`risk_cells.parquet`), metadatos versionados en JSON (`latest.json`) y geometrías vectoriales GeoJSON optimizadas para el visor geoespacial interactivo.
* Realiza auditorías instantáneas de integridad estructural: no negatividad de métricas, ausencia de valores nulos o infinitos (`NaN`/`Inf`), verificación de tipos de datos en memoria y cálculo de dispersión territorial de clases de riesgo (Bajo, Moderado, Alto, Extremo).

---

## 🔬 Validación Científica (Tracción y Desempeño)

La arquitectura de SparkPredict AI ha sido validada rigurosamente bajo estándares científicos y protocolos reproducibles de evaluación temporal y geoespacial.

```
       Validación Retrospectiva (10-Fold Expanding Window)
       15 Temporadas Históricas CONAF (2010/11 - 2024/25)
       23.500.000+ Celdas-Día Evaluadas (Alta Temporada) | N = 4.463 Incendios Totales | N = 378 Incendios >= 10 ha
```

### Protocolo de Evaluación Anti-Lookahead
Para evitar cualquier fuga de información temporal (*data leakage*), el sistema se sometió a un **Backtest Expansivo de 10 Pliegues** (*expanding window / forward-chaining*), donde cada pliegue se entrena exclusivamente con datos históricos estrictamente pasados ($t < s$) y se evalúa a ciegas sobre la alta temporada (noviembre a abril) del año no visto $s$.

### Resultados de Superioridad Estadística (IC 95% Pareado)
Bajo remuestreo por bloques semanales de horizonte ($B = 2.000$ réplicas bootstrap pareadas), el modelo híbrido supera categóricamente al estándar operativo internacional ($FWI \times Prior$):

| Métrica de Desempeño | SparkPredict AI (Híbrido v1.3) | Baseline Operativo (FWI × Prior) | Incremento Neto ($\Delta$) | Intervalo de Confianza (95% Pareado) | Estatus Hipótesis |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rec@5% Total (Primario)** | **39,32 %** | **34,97 %** | **+4,35 p.p.** | **[+3,20 p.p. ; +5,54 p.p.]** | **Superioridad Confirmada (p < 0,001)** |
| **Rec@10% Total** | **52,94 %** | **48,84 %** | **+4,10 p.p.** | **[+3,06 p.p. ; +5,13 p.p.]** | **Superioridad Confirmada (p < 0,001)** |
| **Rec@20% Total** | **69,65 %** | **65,83 %** | **+3,83 p.p.** | **[+2,81 p.p. ; +4,87 p.p.]** | **Superioridad Confirmada (p < 0,001)** |
| **Rec@5% Incendios Pequeños (< 1 ha)** | **41,61 %** | **35,74 %** | **+5,86 p.p.** | **[+4,42 p.p. ; +7,38 p.p.]** | **Superioridad Confirmada** |

> **Eficiencia Territorial en Eventos Severos ($\ge 10$ ha):** En las categorías de alerta máxima (Alto + Extremo, correspondiente al 5% del territorio), SparkPredict AI captura el **32,01%** de todos los incendios destructivos de gran magnitud, alcanzando un **factor de enriquecimiento territorial de 6,4×** (captura 6,4 veces más incendios mayores de lo esperado por azar territorial).

---

## 🛡️ Tolerancia a Fallos y Confiabilidad MLOps

Diseñado bajo la filosofía de operaciones críticas de infraestructura pública y defensa civil:

* **Política de Idempotencia y Estado Continuo:** El estado biológico-climático del suelo se rastrea de forma acumulativa en `fwi_state.parquet`. Si el proceso se interrumpe y se reinicia, detecta el estado previo sin duplicación de registros ni descalibración física.
* **Política Estricta de Cuarentena (S7):** Si la telemetría meteorológica externa falla tras 3 intentos con retroceso exponencial, el día se declara formalmente como **"perdido" (*lost*)**, excluyéndose irrevocablemente del análisis confirmatorio del log prospectivo. Se prohíbe explícitamente cualquier reconstrucción sintética *post-hoc* con datos dispares para garantizar la pureza del estudio.
* **Aislamiento Estricto de Ambientes:** Todo test, simulación de estrés o proceso de benchmarking corre en sandboxes temporales (`tmp_path`) con bandera `is_simulation = True`. La suite de pruebas posee aserciones de bloqueo que impiden escrituras accidentales en la base productiva `prospective_log.parquet`.
* **Calidad de Código y Tipado Fuerte:** 100% de la suite de pruebas unitarias y de integración (`pytest`, 94 tests) ejecutada con éxito, análisis estático con `ruff` libre de advertencias y verificación formal de tipos con `mypy --strict`.

---

## 🚀 Instalación Rápida

### Requisitos del Sistema
* Python 3.11+
* Sistema Operativo: Linux (Ubuntu 22.04+), macOS o Windows 11 / Server
* 8 GB RAM mínimo (16 GB recomendado para reanálisis masivo)

### Paso a Paso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/MRSAMMY-01/chilefuego-prometheus.git sparkpredict-ai
   cd sparkpredict-ai
   ```

2. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv .venv
   
   # En Linux/macOS:
   source .venv/bin/activate
   
   # En Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias de producción y desarrollo:**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. **Ejecutar la suite completa de verificación:**
   ```bash
   # Pruebas unitarias de aislamiento, modelos y contratos de datos
   pytest

   # Verificación estricta de tipos y estilo de código
   ruff check src tests scripts
   mypy --strict src
   ```

5. **Ejecutar una inferencia diaria de prueba (Horizonte 0):**
   ```bash
   python scripts/run_daily.py --date 2026-11-09 --horizon 0
   ```

---

## 👨‍💻 Autor y Liderazgo

**Ignacio Campos**  
*Ingeniero Civil en Informática*  
**Fundador & Lead AI Architect — SparkPredict AI**  
Ñuble / Concepción, Chile  

*Proyecto desarrollado con enfoque en transferencia tecnológica, ciencia de datos aplicada e impacto territorial para la prevención de emergencias forestales.*

---

<div align="center">
  <small>© 2026 SparkPredict AI. Licencia MIT. Desarrollado con rigurosidad científica para la protección de ecosistemas forestales y comunidades.</small>
</div>
