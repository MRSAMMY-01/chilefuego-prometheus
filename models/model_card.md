# Model Card: ChileFuego XGBoost Pruned Ignition Model (`v1.0.0-pruned`)

## 1. Resumen y Detalles del Modelo (Model Details)
- **Nombre:** ChileFuego Pruned Wildfire Ignition Predictor
- **Versión:** `v1.0.0-pruned`
- **Fecha de Publicación:** 2026-10-22
- **Tipo de Modelo:** Gradient Boosted Decision Trees (`xgboost.Booster`) con calibración logística univariada (Platt scaling).
- **Licencia:** MIT / Proyecto ChileFuego
- **Contacto:** Equipo Técnico ChileFuego (`samuel.campos77@outlook.com`)
- **Marco de Trabajo:** Python 3.13.5, XGBoost, Scikit-Learn, Polars.

## 2. Uso Previsto y Alcance Operativo (Intended Use)
- **Dominio Espacio-Temporal:** Grilla regular de 1 km en coordenadas UTM 19S (EPSG:32719) que cubre los 13.108 celdas de la Región de Ñuble, Chile (dominio continental).
- **Paso Temporal:** Diario (horizonte día D de reanálisis meteorológico y telemetría de mediodía).
- **Uso Primario:** Reanálisis y diagnóstico de susceptibilidad de ignición relativa para priorización táctica de recursos y análisis histórico de riesgo de fuego.
- **Fuera de Alcance:**
  - **NO es un pronóstico operativo a futuro:** El modelo opera en tiempo de reanálisis D con covariables meteorológicas instantáneas observadas a las 12:00 CLT / 15:00 UTC.
  - **NO es un disparador automático de alertas o evacuación:** Las categorías de riesgo (Bajo, Moderado, Alto, Extremo) representan una clasificación experimental de percentil relativo, no umbrales directos de evacuación comunal.

## 3. Datos de Entrenamiento y Preprocesamiento (Training Data)
- **Registros Terrestres de Incendios:** Ocurrencias CONAF (2017 a 2025) validadas bajo filtros V1-V8 y colapsadas a nivel celda-día en la Región de Ñuble (3.237 positivos efectivos fechados).
- **Año Hidrológico/Forestal:** Definición contractual estricta del 1 de julio al 30 de junio (7 temporadas fechadas: 2017_18, 2019_20 a 2024_25; temporada sin fecha 2018_19 excluida del entrenamiento con 4.784.420 celdas-día).
- **Universo Poblacional:** 33.345.071 celdas-día efectivas (tasa base efectiva = $9,71 \times 10^{-5}$).
- **Estratificación y Ponderación:** Muestreo estratificado probabilístico (N = 67.974; 3.237 positivos y 64.737 negativos en ratio 1:20 con estratos de riesgo compuestos FWI x Prior). La suma exacta de pesos reconstruye los 33.345.071 celdas-día ($N_{\text{eff}} = 46.489$).

## 4. Covariables del Modelo Podado (25 Features)
El modelo podado final descarta densidad poblacional censal (`pop_density_1km`) y los 13 rezagos/ventanas móviles meteorológicas para eliminar colinealidades y optimizar la generalización:
1. **Topografía / DEM (4):** `elevation_mean`, `slope_mean`, `northness`, `eastness`.
2. **Infraestructura Vial OSM (3):** `dist_road_nearest_m`, `dist_primary_sec_trunk_m`, `road_density_km2`.
3. **Cobertura de Suelo WorldCover (6):** `tree_cover_frac`, `shrub_cover_frac`, `grass_cover_frac`, `crop_cover_frac`, `built_cover_frac`, `dist_built_nearest_m`.
4. **Prior Espacial Histórico (1):** `prior_density` (densidad Kernel histórica acumulada out-of-fold).
5. **Meteorología Instantánea y FWI ERA5-Land (11):** `temp_noon_c`, `vpd_noon_kpa`, `rain_24h_noon_mm`, `rh_noon_pct`, `wind_noon_kmh`, `fwi`, `isi`, `bui`, `ffmc`, `dmc`, `dc`.

## 5. Rendimiento y Evaluación (Evaluation & Performance)
- **Estrategia de Evaluación:** Validación cruzada dejando una temporada completa fuera (Leave-One-Season-Out, 7 folds LOSO) evaluada exhaustivamente sobre el dominio completo de 33,3 M de celdas-día.
- **Incertidumbre:** 5 semillas aleatorias por pliegue (42 a 46) e intervalos de confianza bootstrap pareados por bloques de días (B=1000).
- **Métricas Clave:**
  - **Lift medio sobre tasa base:** $\sim 13\times$ en promedio interanual.
  - **Recall en Top 5% de celdas-día:** $\sim 44\%$ de las igniciones capturadas en el 5% de mayor score.
  - **Incendios $\ge 10$ ha:** Mayor captura de eventos de gran magnitud en ventanas de alto score.

## 6. Limitaciones Conocidas, Sesgos y Desfase de Fuentes (Limitations & Biases)
- **Sesgo de Detección CONAF:** Registros terrestres favorecen zonas habitadas, transitadas o con visibilidad desde caminos, mientras que zonas de alta cordillera despobladas tienen menor probabilidad de reporte temprano.
- **Efecto de Supresión:** Los registros CONAF reflejan igniciones reportadas, las cuales están afectadas por intervenciones de combate inicial que pueden haber evitado la propagación de ciertos eventos.
- **Independencia Espacial Intradiaria:** Días de viento extremo sincrónico (ej. evento Puelche) generan correlaciones espaciales en toda la región, cuantificadas en la evaluación mediante bootstrap por bloques de días.
- **Dominio Geográfico:** El modelo está entrenado y ajustado exclusivamente para la Región de Ñuble, Chile (EPSG:32719). No debe aplicarse a otras regiones sin un recalibrado previo.
- **Desfase de Fuentes y Latencia Operativa:**
  - ERA5-Land tiene una latencia de publicación de ~5 días (ERA5T) y 60-90 días (ERA5 consolidado), lo que impide su uso para inferencia el mismo día D en operaciones de combate activo.
  - La transición a pronósticos numéricos del tiempo (NWP, ej. ECMWF IFS, GFS, WRF DMC o Open-Meteo) requiere sincronización estricta de la precipitación acumulada de 24 h a las 12:00 CLT y corrección de sesgo local (especialmente en humedad relativa mínima y vientos catabáticos andinos).
  - Dado que los códigos de humedad del FWI (DMC y DC) retienen memoria biológica y de suelo de hasta 52 días, la inferencia con NWP debe encadenarse sobre el estado continuo previo sin reiniciar los códigos a valores por defecto.
- **Nota Contractual de Cierre sobre Conteos 444/409:** El año forestal/agronómico de CONAF corta al 1 de julio. 28 eventos ocurridos en junio de 2017 (temporada 2016_17) fueron clasificados por año calendario en 2017_18 en una versión preliminar de análisis (sumando 444 en vez de 416). Similarmente, 28 eventos invernales se transfirieron a 2023_24 en lugar de 2024_25 (reportando 409 en vez de 437). El total del panel se mantiene estrictamente invariante en 3.237 positivos. Bajo el corte oficial del 1 de julio (`season_from_date_expr`), los conteos canónicos son **2017_18 = 416** y **2024_25 = 437**. Asunto resuelto y blindado mediante test de regresión.

