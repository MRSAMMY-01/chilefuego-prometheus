# Pre-registro Metodológico: Evaluación Prospectiva Experimental ChileFuego (Temporada 2026/27)

> **Aviso de Prototipo Experimental:** Este protocolo describe un **prototipo experimental de investigación académica**. No constituye un sistema operativo de despacho institucional ni reemplaza las plataformas de alerta y combate de los organismos públicos.

- **Versión del Documento:** `v1.3.0-preregistration-2026_27`
- **Fecha de Congelamiento Oficial:** 2026-10-27 (estrictamente antes del inicio de la temporada y antes del 1 de diciembre de 2026)
- **Hash SHA-256 del Script de Evaluación (`scripts/evaluate_preregistered_season.py`):**  
  `1889977e1239471f577a7d079c6b6677a828a5744d0c2a1880286ab5d7952b19`
- **Hash SHA-256 del Manifest del Ensamble (`models/chilefuego_xgb_ensemble_v2/manifest.json`):**  
  `73a9e61eb3c18c3b29741f1912a3a0ffda50c5390dcf6565f6142a029f7b3dc4`
- **Hash SHA-256 del Mapeador de Cuantiles (`models/quantile_mapper_nwp_era5_v1.json`):**  
  `d25c2eb5d19473545332ab608b3377f02360d489a19dc74cd5501f494e243881`
- **Hash SHA-256 del Prior Espacial Canónico v1 (`data/processed/prior_density_by_cell_season.parquet`):**  
  `57e148e8ddf2f6ff54d2366b54cdbaaeb06dc1e7ae5c06baca2ec30bab70d606`
- **Regla de Inmutabilidad Absoluta:** Este documento y su script de evaluación quedan formalmente congelados. No se admitirá ninguna modificación de código, ponderaciones o criterios de éxito después del 1 de diciembre de 2026.

---

## 1. Contexto y Objetivos

El sistema experimental mantiene el **Baseline $FWI \times Prior$** como su modelo base de referencia para el benchmark comparativo.  
El presente protocolo pre-registra de forma vinculante la prueba prospectiva fuera de muestra para determinar si el candidato de aprendizaje automático (**`chilefuego_xgb_ensemble_v2`**, ensamble de 10 semillas entrenado con 25 covariables y mapeo monótono de cuantiles NWP $\rightarrow$ ERA5) reúne evidencia empírica suficiente para superar al Baseline de referencia en la temporada 2026/27.

---

## 2. Modelos en Evaluación Concurrente

1. **Modelo de Referencia (Baseline):**
   - Puntaje continuo: $FWI \times Prior$ (con FWI diario derivado de pronóstico numérico).
   - Prior Espacial de Referencia: Versión canónica `prior_density_v1_k4` contenida en `data/processed/prior_density_by_cell_season.parquet` (KDE Gaussiano con ancho de banda $\sigma = 3.000\text{ m}$ sobre igniciones históricas validadas).
   - Mapeo de Cuantiles: Mapeador monótono empírico NWP $\rightarrow$ ERA5 (`quantile_mapper_nwp_era5_v1.json`).
   - Umbrales fijos P3: P80 (`0.00133319`), P95 (`0.00511459`), P99 (`0.01174802`).

2. **Modelo Candidato en Evaluación (Ensamble ML):**
   - Arquitectura: Promedio de predicción de 10 modelos XGBoost (`CANONICAL_PARAMS`, 106 árboles, 25 variables podadas).
   - Mapeo de Cuantiles: Mapeador monótono empírico NWP $\rightarrow$ ERA5 (`quantile_mapper_nwp_era5_v1.json`).
   - Umbrales fijos LOSO multianual: P80 (`0.00012470`), P95 (`0.00043079`), P99 (`0.00123494`).

3. **Puntaje Exploratorio Secundario (`score_baseline_prior_v2`):**
   - El pipeline registra en `prospective_log.parquet` el campo `score_baseline_prior_v2` (calculado con la versión experimental del prior espacial v2, `prior_density_2026_27_v2.parquet`).
   - **Declaración Formal:** Este puntaje es puramente exploratorio para análisis de investigación posterior y **no forma parte de los criterios confirmatorios primarios ni secundarios** del pre-registro.

---

## 3. Especificación del Pronóstico Puntuado y Horizontes de Evaluación

- **Modelo Numérico Global:** NOAA GFS seamless (vía Open-Meteo Forecast API, `models=gfs_seamless`).
- **Ciclo Meteorológico Ingerido:** **Último ciclo disponible al momento de la consulta**.  
  *Verificación de latencia y husos horarios:* A las 07:00 hora local de Chile continental (CLT), el horario UTC varía según el régimen estacional: equivale a **10:00 UTC en horario de verano (UTC-3)** y a **11:00 UTC en horario de invierno (UTC-4)**. La telemetría de Open-Meteo (`https://api.open-meteo.com/data/ncep_gfs025/static/meta.json`) evidencia empíricamente una latencia de ingestión de 5.5 a 7.2 horas desde la inicialización del modelo hasta su disponibilidad en la API. Por tanto, no se asume una hora fija de disponibilidad: el pipeline registra formalmente la hora exacta de consulta en UTC (`query_timestamp_utc`) y toma el último ciclo efectivamente disponible publicado por Open-Meteo al momento de la ejecución.
- **Hora Objetivo Diaria:** 16:00 UTC (12:00 CLT / mediodía solar de Ñuble).
- **Almacenamiento y Jerarquía de Horizontes:**
  Cada corrida diaria genera y almacena dos horizontes evaluables con sus campos `horizon` y `forecast_issue_date`:
  - **Horizonte Primario H1 (Pronóstico a 24 horas de antelación):** Emitido el día $D-1$ para el día objetivo $D$ (`horizon = 1`, `forecast_issue_date = D - 1`). El criterio primario vinculante de pre-registro evalúa de forma estricta este horizonte emitido el día anterior.
  - **Horizonte Secundario H0 (Pronóstico mismo día):** Emitido el día $D$ para el día objetivo $D$ (`horizon = 0`, `forecast_issue_date = D`).
- **Spin-up de FWI:** Estado de humedad del suelo inicializado con al menos 92 días de telemetría continua previa (`models=gfs_seamless`) para garantizar convergencia asintótica de DMC y DC.

---

## 4. Definición de Eventos y Asignación Espacial

- **Etiqueta Verdadera de Incendio:** Ocurrencias terrestres georreferenciadas provenientes del registro histórico de igniciones validadas.
- **Filtros Contractuales de Integridad (V1–V8):** Coordenadas válidas en el territorio continental de la Región de Ñuble, descarte de registros con fecha nula, descarte de quemas agrícolas autorizadas y deduplicación intradiaria de focos en la misma celda.
- **Asignación Espacial:** Proyección EPSG:32719 a la cuadrícula fija de 1 km $\times$ 1 km (`grid_1km.geojson`).
- **Unidad de Análisis:** Celda-día ($13.108\text{ celdas} \times 181\text{ días de temporada alta} \approx 2.372.548\text{ celdas-día}$).

---

## 5. Tratamiento de Días Fallidos y Disponibilidad de Datos

- **Criterio de Inclusión Primario:** La evaluación confirmativa principal se realiza **exclusivamente sobre días en que ambos modelos recibieron telemetría meteorológica fresca** (`freshness == "fresh"`).
- **Tasa de Cobertura:** Se reportará formalmente el porcentaje de cobertura diaria efectiva durante la temporada ($\text{Cobertura} = N_{\text{días\_válidos}} / N_{\text{total\_temporada}} \times 100\%$).
- **Análisis de Sensibilidad Secundario:** Se ejecutará una prueba secundaria imputando el fallback stale/persistente en días con caída de API para verificar que la regla de decisión no depende de la exclusión de días con falla de red.
- **Prohibición de Reconstrucción Dispar:** Queda estrictamente prohibido reconstruir retroactivamente días perdidos utilizando APIs o modelos numéricos diferentes a los utilizados en la ingesta diaria en vivo.

---

## 6. Plan de Contingencia ante Disponibilidad de Datos Terrestres

- **Fecha Límite:** 31 de julio de 2027.
- **Protocolo de Activación:** Si la base consolidada terrestre no está disponible al 31 de julio de 2027:
  1. Se activará automáticamente la base de contingencia basada en **NASA FIRMS (VIIRS S-NPP/NOAA-20 y MODIS)**.
  2. Filtrado estricto a clusters bajo `rule_b`: `type == 0` (fuegos de vegetación) con calidad estándar o superior.
  3. La evaluación se reportará explícitamente con la marca `is_proxy_firms = True` y tendrá validez provisional hasta la publicación del consolidado terrestre.

---

## 7. Protocolo de Desempate en Rec@K

- Para evitar distorsiones por empates de puntaje en los bordes de corte de $K=5\%$ y $K=20\%$:
  - Se aplica un desempate determinista infinitesimal basado en hash de la celda y fecha:  
    $$\tilde{s}_i = s_i + 10^{-12} \times \frac{\text{hash}(\text{cell\_id}, \text{date}) \pmod{10^6}}{10^6}$$
  - O cálculo mediante interpolación de rangos vectorizada (`FastBinnedEvaluator`), garantizando determinismo estricto e invarianza ante permutación de filas.

---

## 8. Potencia Estadística y Diferencia Mínima Detectable (MDD)

- **Muestra Esperada:** $N \approx 450$ eventos de ignición confirmados en Ñuble durante Nov–Abr (promedio histórico multianual $462 \pm 65$).
- **Prueba Estadística:** Bootstrap pareado por bloques de días ($B \ge 1.000$) sobre la diferencia de Recall diario ($\Delta \text{Rec@5\%} = \text{Rec}_{\text{ML}} - \text{Rec}_{\text{Baseline}}$).
- **Error Estándar Empírico (S1 NWP out-of-sample):** $\text{SE}_0 \approx 1.75\text{ pp}$ con $N_0 \approx 450$. El error estándar escala con el tamaño muestral según $\text{SE}(N) = 1.75 \times \sqrt{450 / N}$.
- **Fórmula de la Diferencia Mínima Detectable (MDD al 80% de potencia y $\alpha = 0.05$ bilateral):**
  $$\text{MDD}_{80\%} = (z_{1-\alpha/2} + z_{1-\beta}) \times \text{SE} = (1.95996 + 0.84162) \times \text{SE} \approx 2.8016 \times \text{SE}$$
  - Para $N \approx 450$ ($\text{SE} = 1.75\text{ pp}$): $\text{MDD}_{80\%} = 2.8016 \times 1.75 = \mathbf{4.90\text{ pp}}$.

### Tabla de Potencia Estadística Prospectiva

| Muestra ($N$) | $\text{SE}$ (pp) | $\text{MDD}_{80\%}$ | $\delta = +2\text{ pp}$ | $\delta = +3\text{ pp}$ | $\delta = +4\text{ pp}$ | $\delta = +5\text{ pp}$ | $\delta = +6\text{ pp}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **350** | $1.984$ | $5.56\text{ pp}$ | $17.1\%$ | $32.7\%$ | $52.2\%$ | $71.2\%$ | $85.6\%$ |
| **450** | $1.750$ | $4.90\text{ pp}$ | $20.7\%$ | $40.3\%$ | $62.8\%$ | **81.5%** | $92.9\%$ |
| **550** | $1.583$ | $4.43\text{ pp}$ | $24.3\%$ | $47.4\%$ | $71.5\%$ | $88.5\%$ | $96.6\%$ |

*Nota metodológica:* Con la ventaja observada fuera de muestra en el horizonte primario real H1 ($+5.74\text{ pp}$ en S1), la potencia estadística prospectiva esperada con $N=450$ es del **$89.8\%$**, garantizando una capacidad de discriminación adecuada.

---

## 9. Criterios Formales de Decisión

### 9.1 Criterio Primario (Horizonte de Evaluación Primario H1)
- **Métrica:** Recall en el 5% superior de celdas diarias ($\text{Rec@5\%}$) en **Horizonte H1** (pronóstico emitido el día $D-1$ para el día objetivo $D$).
- **Ventana Temporal:** 1 de noviembre de 2026 al 30 de abril de 2027.
- **Regla:** El candidato supera al baseline como modelo de referencia si y solo si:
  $$\text{CI}_{2.5\%}(\Delta \text{Rec@5\%}_{H1}) > 0.00\text{ pp}$$

### 9.2 Criterios Secundarios (Consistencia y Robustez Multiescala)
1. **Horizonte H0:** $\text{CI}_{2.5\%}(\Delta \text{Rec@5\%}_{H0}) > 0.00\text{ pp}$ (evaluación con pronóstico del mismo día).
2. **Cobertura Territorial Amplia:** $\Delta \text{Rec@20\%}_{H1} \ge 0.00\text{ pp}$ (verificar ausencia de degradación a escala territorial amplia).

---

## 10. Verificación Empírica con Backtest 2024/25

El script congelado `scripts/evaluate_preregistered_season.py` fue ejecutado contra el backtest de la temporada 2024/25, reproduciendo con éxito la ventaja del ensamble dentro de las tolerancias establecidas en el estudio retrospectivo Q3:
- Rec@5% Ensamble: $37.49\%$ vs Baseline: $31.97\%$ ($\Delta = +5.49\text{ pp}$, $\text{IC } 95\%: [+2.06, +8.87]$).
- Cumplimiento de criterios: `primary_criterion_met = True`, `secondary_criterion_met = True`.
