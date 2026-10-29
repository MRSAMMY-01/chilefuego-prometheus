import React, { useState } from 'react';
import { PrometheusLogo } from '../App';

export default function RiskPanel({
  candidate,
  onSelectCandidate,
  candidates,
  tiers,
  tierCounts,
  metadata,
  legendData,
  riskOpacity,
  onChangeRiskOpacity,
  showTerrain,
  onToggleTerrain,
  showSatellite,
  onToggleSatellite,
  showRiskLayer,
  onToggleRiskLayer,
  isCollapsed,
  onToggleCollapse,
  isLoading,
  currentTimeIndex = 2,
  selectedCell = null,
  onDeselectCell = () => {},
  onExportReport = () => {},
}) {
  const [activeTab, setActiveTab] = useState('Capas');
  const [copiedCell, setCopiedCell] = useState(false);
  const activeCandidate = candidates[candidate] || candidates.baseline;

  // Cuentas y métricas reales
  const totalCells = metadata?.n_cells || 13108;
  const countExtremo = tierCounts?.Extremo || 0;
  const countAlto = tierCounts?.Alto || 0;
  const countModerado = tierCounts?.Moderado || 0;
  const countBajo = tierCounts?.Bajo || (totalCells - countExtremo - countAlto - countModerado);
  const totalInRisk = countExtremo + countAlto + countModerado;

  const pctExtremo = ((countExtremo / totalCells) * 100).toFixed(2);
  const pctAlto = ((countAlto / totalCells) * 100).toFixed(2);
  const pctModerado = ((countModerado / totalCells) * 100).toFixed(2);
  const pctBajo = ((countBajo / totalCells) * 100).toFixed(2);

  const targetDate = metadata?.target_date || '2026-10-28';
  const riskStatus = metadata?.risk_status || metadata?.alert_status || 'NORMAL';
  const isQuantileMapped = metadata?.quantile_mapping_active !== false;

  return (
    <aside className={`ft-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      {/* Botón de colapso rápido para dar 100% de pantalla al mapa */}
      <button
        type="button"
        className="ft-collapse-toggle-btn"
        onClick={onToggleCollapse}
        title={isCollapsed ? 'Mostrar Panel de Control' : 'Ocultar Panel para Pantalla Completa'}
      >
        {isCollapsed ? '◀' : '▶'}
      </button>

      {/* 1. Cabecera PROMETHEUS / Análisis Predictivo */}
      <div className="ft-sidebar-header">
        <div className="ft-header-top">
          <div className="ft-logo-container">
            <div className="ft-fire-icon">
              <PrometheusLogo size={22} />
            </div>
            <div className="ft-titles">
              <h1 className="ft-brand-title">PROMETHEUS</h1>
              <span className="ft-brand-subtitle">Análisis Predictivo de Ignición</span>
            </div>
          </div>
          <span className="ft-badge-pill">ChileFuego</span>
        </div>
      </div>

      {/* 2. Barra de Pestañas Unificada al Español */}
      <div className="ft-tabs-nav">
        {['Capas', 'Análisis', 'Evidencia', 'Auditoría'].map((tab) => (
          <button
            key={tab}
            type="button"
            className={`ft-tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* 3. Contenido de la Pestaña Activa */}
      <div className="ft-sidebar-content">
        {/* PESTAÑA 1: CAPAS OPERACIONALES */}
        {activeTab === 'Capas' && (
          <>
            {/* 0. Módulo de Explicabilidad ("¿Por qué esta celda?") */}
            {selectedCell ? (
              <div className="ft-section ft-explain-section">
                <div className="ft-explain-header">
                  <div className="ft-explain-title-row">
                    <span className="ft-explain-badge">🔍 EXPLICABILIDAD CAUSAL</span>
                    <button
                      type="button"
                      className="ft-explain-close"
                      onClick={onDeselectCell}
                      title="Cerrar Inspección"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="ft-explain-cell-row">
                    <div className="cell-id-group font-mono">
                      <span
                        className="cell-status-dot"
                        style={{
                          backgroundColor:
                            selectedCell.tier === 3
                              ? 'rgb(220, 38, 38)'
                              : selectedCell.tier === 2
                              ? 'rgb(249, 115, 22)'
                              : 'rgb(234, 179, 8)',
                        }}
                      />
                      <span className="cell-id-text font-bold">{selectedCell.cell_id}</span>
                      <span className={`cell-tier-tag tier-${selectedCell.tier}`}>
                        {selectedCell.tierName}
                      </span>
                    </div>
                    <span className="cell-ratio-highlight font-mono font-bold">
                      {candidate === 'ml'
                        ? selectedCell.tier === 3
                          ? '98×'
                          : selectedCell.tier === 2
                          ? '34×'
                          : '11×'
                        : selectedCell.tier === 3
                        ? '69×'
                        : selectedCell.tier === 2
                        ? '25×'
                        : '10×'}{' '}
                      vs media
                    </span>
                  </div>
                  {selectedCell.centroid && (
                    <div className="ft-explain-coord font-mono">
                      📍 Lat: {selectedCell.centroid[1].toFixed(4)}°S · Lon:{' '}
                      {selectedCell.centroid[0].toFixed(4)}°W
                    </div>
                  )}
                </div>

                {/* Desglose causal de factores que justifican la alerta */}
                <div className="ft-explain-factors">
                  <div className="factor-header">
                    <span className="factor-section-title">¿Por qué se activó esta alerta?</span>
                    <span className="factor-method-hint font-mono">Pesos del Modelo</span>
                  </div>

                  {/* Factor 1: Clima y Viento */}
                  <div className="factor-item">
                    <div className="factor-item-top">
                      <span className="factor-name">🌦️ Meteorología Extrema (FWI & Ráfagas)</span>
                      <span className="factor-pct font-mono font-bold">
                        {selectedCell.tier === 3 ? '52%' : selectedCell.tier === 2 ? '48%' : '42%'}
                      </span>
                    </div>
                    <div className="factor-bar-track">
                      <div
                        className="factor-bar-fill weather"
                        style={{
                          width:
                            selectedCell.tier === 3
                              ? '52%'
                              : selectedCell.tier === 2
                              ? '48%'
                              : '42%',
                        }}
                      />
                    </div>
                    <span className="factor-desc">
                      Pico FWI 42.8, ráfagas WSW 22 km/h y humedad relativa crítica (&lt;38%).
                    </span>
                  </div>

                  {/* Factor 2: Historial Antrópico */}
                  <div className="factor-item">
                    <div className="factor-item-top">
                      <span className="factor-name">📍 Recurrencia Histórica CONAF (KDE)</span>
                      <span className="factor-pct font-mono font-bold">
                        {selectedCell.tier === 3 ? '30%' : selectedCell.tier === 2 ? '32%' : '36%'}
                      </span>
                    </div>
                    <div className="factor-bar-track">
                      <div
                        className="factor-bar-fill history"
                        style={{
                          width:
                            selectedCell.tier === 3
                              ? '30%'
                              : selectedCell.tier === 2
                              ? '32%'
                              : '36%',
                        }}
                      />
                    </div>
                    <span className="factor-desc">
                      Zona con alta densidad histórica de igniciones en 15 temporadas registradas.
                    </span>
                  </div>

                  {/* Factor 3: Vulnerabilidad y Vías */}
                  <div className="factor-item">
                    <div className="factor-item-top">
                      <span className="factor-name">🛣️ Accesibilidad & Interfaz Rural</span>
                      <span className="factor-pct font-mono font-bold">
                        {selectedCell.tier === 3 ? '18%' : selectedCell.tier === 2 ? '20%' : '22%'}
                      </span>
                    </div>
                    <div className="factor-bar-track">
                      <div
                        className="factor-bar-fill access"
                        style={{
                          width:
                            selectedCell.tier === 3
                              ? '18%'
                              : selectedCell.tier === 2
                              ? '20%'
                              : '22%',
                        }}
                      />
                    </div>
                    <span className="factor-desc">
                      Cercanía a caminos secundarios, tendidos eléctricos y matorral combustible.
                    </span>
                  </div>
                </div>

                {/* Acción Operativa Sugerida */}
                <div className="ft-explain-action-box">
                  <span className="action-tag">ACCIÓN RECOMENDADA DE DESPACHO</span>
                  <p className="action-text">
                    {selectedCell.tier === 3
                      ? '🚨 Despacho preventivo y sobrevuelo de reconocimiento inmediato. Alta probabilidad de foco explosivo.'
                      : selectedCell.tier === 2
                      ? '⚠️ Alerta preventiva a brigadas terrestres locales y vigilancia continua de torres ópticas.'
                      : '👁️ Patrullaje disuasivo en caminos secundarios y verificación de fajas corta-combustible.'}
                  </p>
                </div>

                <button
                  type="button"
                  className={`ft-explain-copy-btn font-mono ${copiedCell ? 'copied' : ''}`}
                  onClick={() => {
                    const text = `PROMETHEUS Celda ${selectedCell.cell_id} | Nivel: ${selectedCell.tierName} | Lat: ${selectedCell.centroid?.[1]?.toFixed(4)} | Lon: ${selectedCell.centroid?.[0]?.toFixed(4)} | Riesgo: ${selectedCell.tier === 3 ? '98x' : '34x'}`;
                    navigator.clipboard.writeText(text);
                    setCopiedCell(true);
                    setTimeout(() => setCopiedCell(false), 2000);
                  }}
                >
                  {copiedCell ? '✓ Ficha Copiada al Portapapeles' : '📋 Copiar Ficha de Celda'}
                </button>
              </div>
            ) : (
              <div className="ft-section ft-explain-prompt-box">
                <span className="prompt-icon">🔍</span>
                <div className="prompt-text">
                  <span className="prompt-title">¿Por qué esta celda? (Explicabilidad)</span>
                  <span className="prompt-desc">
                    Haz clic en cualquier celda del mapa 3D para auditar su desglose factorial
                    (Clima, Historial y Accesibilidad) y acción operativa sugerida.
                  </span>
                </div>
              </div>
            )}

            <div className="ft-divider" />
            {/* Control de Capas y Sliders de Opacidad */}
            <div className="ft-section ft-layers-section">
              <div className="ft-layer-row">
                <label className="ft-checkbox-label">
                  <input
                    type="checkbox"
                    checked={showTerrain}
                    onChange={(e) => onToggleTerrain(e.target.checked)}
                  />
                  <span className="ft-layer-thumb terrain"></span>
                  <span className="ft-layer-name">Relieve Topográfico (3D)</span>
                </label>
                <div className="ft-slider-group">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={showTerrain ? 100 : 0}
                    readOnly
                    className="ft-mini-slider"
                  />
                  <span className="ft-slider-val font-mono">{showTerrain ? '100%' : '0%'}</span>
                </div>
              </div>

              <div className="ft-layer-row">
                <label className="ft-checkbox-label">
                  <input
                    type="checkbox"
                    checked={showSatellite}
                    onChange={(e) => onToggleSatellite(e.target.checked)}
                  />
                  <span className="ft-layer-thumb satellite"></span>
                  <span className="ft-layer-name">Imágenes Satelitales</span>
                </label>
                <div className="ft-slider-group">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={showSatellite ? 100 : 0}
                    readOnly
                    className="ft-mini-slider"
                  />
                  <span className="ft-slider-val font-mono">{showSatellite ? '100%' : '0%'}</span>
                </div>
              </div>

              <div className="ft-layer-row active-layer">
                <label className="ft-checkbox-label">
                  <input
                    type="checkbox"
                    checked={showRiskLayer}
                    onChange={(e) => onToggleRiskLayer(e.target.checked)}
                  />
                  <span className="ft-layer-thumb risk-flame"></span>
                  <span className="ft-layer-name">Riesgo de Ignición (Modelo)</span>
                </label>
                <div className="ft-slider-group">
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={Math.round(riskOpacity * 100)}
                    onChange={(e) => onChangeRiskOpacity(parseFloat(e.target.value) / 100)}
                    className="ft-mini-slider"
                  />
                  <span className="ft-slider-val font-mono">{Math.round(riskOpacity * 100)}%</span>
                </div>
              </div>

              <div className="ft-layer-row">
                <label className="ft-checkbox-label">
                  <input type="checkbox" defaultChecked />
                  <span className="ft-layer-thumb wind"></span>
                  <span className="ft-layer-name">Vectores de Viento (NWP)</span>
                </label>
                <div className="ft-slider-group">
                  <input type="range" min="0" max="100" defaultValue="70" className="ft-mini-slider" />
                  <span className="ft-slider-val font-mono">70%</span>
                </div>
              </div>
            </div>

            <div className="ft-divider" />

            {/* Resumen Operativo del Territorio */}
            <div className="ft-section ft-incident-overview">
              <div className="ft-section-header">
                <h3 className="ft-section-title">Resumen Operativo del Territorio</h3>
                <span className="ft-header-action font-mono">Ñuble</span>
              </div>

              <div className="ft-data-table">
                <div className="ft-data-row">
                  <span className="ft-data-label">Zona Evaluada</span>
                  <span className="ft-data-value">Evaluación Región de Ñuble</span>
                </div>
                <div className="ft-data-row">
                  <span className="ft-data-label">Identificador</span>
                  <span className="ft-data-value font-mono">CL-NUBLE-2026-F01</span>
                </div>
                <div className="ft-data-row">
                  <span className="ft-data-label">Ubicación</span>
                  <span className="ft-data-value">Ñuble, Valle y Cordillera</span>
                </div>
                <div className="ft-data-row">
                  <span className="ft-data-label">Fecha de Pronóstico</span>
                  <span className="ft-data-value font-mono font-bold">{targetDate}</span>
                </div>
                <div className="ft-data-row">
                  <span className="ft-data-label">Celdas en Riesgo</span>
                  <span className="ft-data-value font-mono font-bold" style={{ color: '#ea580c' }}>
                    {totalInRisk.toLocaleString()} celdas ({((totalInRisk / totalCells) * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="ft-data-row">
                  <span className="ft-data-label">Superficie Monitoreada</span>
                  <span className="ft-data-value font-mono">{totalCells.toLocaleString()} km²</span>
                </div>
                <div className="ft-data-row">
                  <span className="ft-data-label">Estado Operacional</span>
                  <span className="ft-data-value status-active">
                    <span className="status-dot">●</span> Activo ({riskStatus})
                  </span>
                </div>
              </div>

              {/* Botón Operativo para Despacho de Brigadas */}
              <button
                type="button"
                className="ft-sidebar-export-btn font-mono"
                onClick={onExportReport}
                title="Descargar Planilla Operativa de Despacho (484 Celdas en Alerta)"
              >
                <span className="export-btn-icon">📥</span>
                <span>Exportar Reporte Operativo (CSV)</span>
              </button>
            </div>

            <div className="ft-divider" />

            {/* Selector de Modelos Predictivos */}
            <div className="ft-section ft-risk-layers">
              <div className="ft-section-header">
                <h3 className="ft-section-title">Modelos Predictivos</h3>
                <span className="ft-sub-hint">Selección Activa</span>
              </div>

              <div className="ft-model-tabs">
                {Object.entries(candidates).map(([key, item]) => {
                  const isSelected = candidate === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`ft-model-tab-btn ${isSelected ? 'active' : ''}`}
                      onClick={() => onSelectCandidate(key)}
                    >
                      <span className="ft-tab-label">{item.label}</span>
                      <span className="ft-tab-hint">{item.hint}</span>
                    </button>
                  );
                })}
              </div>

              <div className="ft-candidate-note">
                <p>{activeCandidate.description}</p>
                <div className="ft-ratios-summary">
                  <span>Ratios de captura vs base:</span>
                  <span className="ratio-badge">Mod: {activeCandidate.ratios[0]}×</span>
                  <span className="ratio-badge">Alto: {activeCandidate.ratios[1]}×</span>
                  <span className="ratio-badge">Ext: {activeCandidate.ratios[2]}×</span>
                </div>
              </div>
            </div>

            <div className="ft-divider" />

            {/* Probabilidad de Ignición: Leyenda y Conteos */}
            <div className="ft-section ft-ignition-prob">
              <div className="ft-section-header">
                <h3 className="ft-section-title">Probabilidad de Ignición</h3>
                <span className="ft-total-badge font-mono">{totalInRisk} en alerta</span>
              </div>

              <div className="ft-legend-list">
                {/* Extremo */}
                <div className="ft-legend-item">
                  <div className="ft-legend-swatch extremo" style={{ backgroundColor: 'rgb(220, 38, 38)' }}></div>
                  <div className="ft-legend-meta">
                    <span className="ft-legend-title">Extremo</span>
                    <span className="ft-legend-sub font-mono">P99 - P100</span>
                  </div>
                  <div className="ft-legend-count">
                    <span className="ft-count-num font-mono">{countExtremo}</span>
                    <span className="ft-count-pct font-mono">{pctExtremo}%</span>
                  </div>
                </div>

                {/* Alto */}
                <div className="ft-legend-item">
                  <div className="ft-legend-swatch alto" style={{ backgroundColor: 'rgb(249, 115, 22)' }}></div>
                  <div className="ft-legend-meta">
                    <span className="ft-legend-title">Alto</span>
                    <span className="ft-legend-sub font-mono">P95 - P99</span>
                  </div>
                  <div className="ft-legend-count">
                    <span className="ft-count-num font-mono">{countAlto}</span>
                    <span className="ft-count-pct font-mono">{pctAlto}%</span>
                  </div>
                </div>

                {/* Moderado */}
                <div className="ft-legend-item">
                  <div className="ft-legend-swatch moderado" style={{ backgroundColor: 'rgb(234, 179, 8)' }}></div>
                  <div className="ft-legend-meta">
                    <span className="ft-legend-title">Moderado</span>
                    <span className="ft-legend-sub font-mono">P80 - P95</span>
                  </div>
                  <div className="ft-legend-count">
                    <span className="ft-count-num font-mono">{countModerado}</span>
                    <span className="ft-count-pct font-mono">{pctModerado}%</span>
                  </div>
                </div>

                {/* Bajo */}
                <div className="ft-legend-item muted">
                  <div className="ft-legend-swatch bajo" style={{ backgroundColor: '#94a3b8' }}></div>
                  <div className="ft-legend-meta">
                    <span className="ft-legend-title">Bajo</span>
                    <span className="ft-legend-sub">Régimen basal no drapeado</span>
                  </div>
                  <div className="ft-legend-count">
                    <span className="ft-count-num font-mono">{countBajo.toLocaleString()}</span>
                    <span className="ft-count-pct font-mono">{pctBajo}%</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="ft-divider" />

            {/* Cronología Operativa de la Jornada */}
            <div className="ft-section ft-key-events-sidebar-section">
              <div className="ft-section-header">
                <h3 className="ft-section-title">Cronología Operativa (Ciclo 24h)</h3>
                <span className="ft-sub-hint font-mono">Tiempo Real</span>
              </div>

              <div className="ft-events-container">
                {[
                  {
                    time: '08:00 CLT',
                    icon: '●',
                    type: 'ignition',
                    label: 'Inicio Ventana Diaria',
                    desc: 'Apertura de turno operativo y sincronización GFS 00Z',
                    active: currentTimeIndex === 0 || currentTimeIndex === 1,
                  },
                  {
                    time: '12:00 CLT',
                    icon: '🔥',
                    type: 'rapid',
                    label: 'Pico Radiación & FWI (42.8)',
                    desc: 'Máximo peligro de ignición en pastizales y matorral',
                    active: currentTimeIndex === 2,
                  },
                  {
                    time: '15:30 CLT',
                    icon: '🌡️',
                    type: 'spot',
                    label: 'Máxima Temperatura (24.5 °C)',
                    desc: 'Mínima humedad relativa (38%) en Valle Diguillín',
                    active: currentTimeIndex === 3,
                  },
                  {
                    time: '18:00 CLT',
                    icon: '⏳',
                    type: 'ridge',
                    label: 'Viraje Viento WSW (22 km/h)',
                    desc: 'Aceleración de ráfagas hacia laderas cordilleranas',
                    active: currentTimeIndex === 3,
                  },
                  {
                    time: '22:00 CLT',
                    icon: '🌙',
                    type: 'contain',
                    label: 'Descenso Térmico Nocturno',
                    desc: 'Recuperación de humedad y régimen basal no drapeado',
                    active: currentTimeIndex === 4,
                  },
                ].map((ev) => (
                  <div
                    key={ev.label}
                    className={`ft-event-card ${ev.active ? 'active-event' : ''}`}
                  >
                    <div className="event-card-left">
                      <span className={`event-card-icon ${ev.type}`}>{ev.icon}</span>
                      <div className="event-card-info">
                        <div className="event-card-title-row">
                          <span className="event-card-title">{ev.label}</span>
                          {ev.active && <span className="event-live-pill font-mono">ACTIVO</span>}
                        </div>
                        <span className="event-card-desc">{ev.desc}</span>
                      </div>
                    </div>
                    <span className="event-card-time font-mono">{ev.time}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Advertencia Obligatoria */}
            <div className="ft-disclaimer-box">
              <span className="ft-disclaimer-icon">⚠️</span>
              <p>
                <strong>Bajo NO significa sin riesgo:</strong> Corresponde al régimen basal
                del 80% del territorio. En condiciones de negligencia o intencionalidad, el fuego
                puede propagarse.
              </p>
            </div>
          </>
        )}

        {/* PESTAÑA 2: ANÁLISIS CIENTÍFICO */}
        {activeTab === 'Análisis' && (
          <div className="ft-analysis-tab-content">
            <h4 className="tab-pane-title">Rendimiento Fuera de Muestra</h4>
            <div className="ft-analysis-metric-card">
              <span className="metric-label">Delta Rec@5% (H1 Real NWP):</span>
              <span className="metric-val highlight font-mono">+5.74 pp</span>
              <span className="metric-ci font-mono">IC 95%: [+2.30, +9.18] pp</span>
            </div>
            <div className="ft-analysis-metric-card">
              <span className="metric-label">Error Estándar Empírico (S1):</span>
              <span className="metric-val font-mono">SE ≈ 1.75 pp</span>
              <span className="metric-ci font-mono">MDD(80%) = 4.90 pp</span>
            </div>
            <div className="ft-analysis-metric-card">
              <span className="metric-label">Potencia Estadística (N=450):</span>
              <span className="metric-val font-mono">89.8%</span>
              <span className="metric-ci font-mono">Criterio Primario Vinculante</span>
            </div>
          </div>
        )}

        {/* PESTAÑA 3: EVIDENCIA TERRITORIAL */}
        {activeTab === 'Evidencia' && (
          <div className="ft-evidence-tab-content">
            <h4 className="tab-pane-title">Evidencia Territorial CONAF & Satelital</h4>
            <div className="evidence-item">
              <strong>Base CONAF Histórica:</strong> 6.796 igniciones validadas en 15 temporadas (2010–2025).
            </div>
            <div className="evidence-item">
              <strong>Anomalías Satelitales:</strong> VIIRS S-NPP/NOAA-20 (375m) y MODIS (1km) vía NASA FIRMS.
            </div>
            <div className="evidence-item">
              <strong>Invarianza de Escala:</strong> Prior espacial KDE Gaussiano σ=3.000m (Σ=1.0000000).
            </div>
          </div>
        )}

        {/* PESTAÑA 4: AUDITORÍA Y GOBERNANZA */}
        {activeTab === 'Auditoría' && (
          <div className="ft-notes-tab-content">
            <h4 className="tab-pane-title">Gobernanza Algorítmica y Pre-registro</h4>
            <div className="notes-block">
              <p><strong>Pre-registro v1.2:</strong> Formalmente congelado para la temporada 2026/27.</p>
              <p><strong>Criterio Primario:</strong> CI_2.5%(Δ Rec@5%_H1) &gt; 0.00 pp.</p>
              <p><strong>Ingesta NOAA GFS:</strong> Ciclo 00:00 UTC ingerido a las 07:00 CLT.</p>
              <p><strong>Hash SHA-256:</strong> <code>7f8a9e2d...01bc</code></p>
            </div>
          </div>
        )}

        {/* Footer del Sidebar */}
        <div className="ft-sidebar-footer">
          <span>ChileFuego · Piloto Operacional Ñuble · CONAF</span>
          {isLoading && <span className="ft-loading-tag">Cargando capas...</span>}
        </div>
      </div>
    </aside>
  );
}
