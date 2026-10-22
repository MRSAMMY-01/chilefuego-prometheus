import React from 'react';

/**
 * @file src/components/BottomTimeline.jsx
 * @description Dock inferior de análisis forense y simulación cinemática 3D inspirado en FireTrace.
 * Incluye línea de tiempo interactiva, 3 gráficos sincronizados (Superficie Quemada, Velocidad de
 * Propagación y Perfil de Elevación Topográfica) y bitácora de Eventos Clave.
 */
export default function BottomTimeline({
  simulationHour = 0,
  onChangeSimulationHour,
  isPlaying = false,
  onTogglePlay,
  onReset,
  playbackSpeed = 1,
  onChangePlaybackSpeed,
  onSelectKeyEvent,
}) {
  const progressPct = Math.min(100, Math.max(0, (simulationHour / 24) * 100));

  // Formateo de hora cronológica a partir de las 14:26 (T+0h)
  const formatTimestamp = (hoursToAdd) => {
    const baseHour = 14 + Math.floor(hoursToAdd);
    const dayStr = baseHour >= 24 ? '19 Oct' : '18 Oct';
    const hh = String(baseHour % 24).padStart(2, '0');
    const mm = hoursToAdd % 1 === 0 ? '26' : '56';
    return `${dayStr}, ${hh}:${mm}`;
  };

  // Eventos Clave Forenses y Tácticos en la Línea de Tiempo
  const keyEvents = [
    {
      id: 'ev-ignition',
      hour: 0,
      icon: '🔴',
      label: 'Ignición Detectada (VIIRS)',
      time: '18 Oct, 14:26',
      badgeClass: 'text-red-400',
    },
    {
      id: 'ev-growth',
      hour: 4,
      icon: '🔥',
      label: 'Crecimiento Rápido en Ladera',
      time: '18 Oct, 18:26',
      badgeClass: 'text-orange-400',
    },
    {
      id: 'ev-spotfire',
      hour: 9,
      icon: '⚠️',
      label: 'Foco Secundario (Pavesas)',
      time: '18 Oct, 23:26',
      badgeClass: 'text-amber-300 bg-amber-500/15 border border-amber-500/40',
    },
    {
      id: 'ev-vlat',
      hour: 14,
      icon: '✈️',
      label: 'Descarga DC-10 VLAT (Flanco)',
      time: '19 Oct, 04:26',
      badgeClass: 'text-red-300',
    },
    {
      id: 'ev-dozer',
      hour: 19,
      icon: '🚜',
      label: 'Contención Cortafuego Dozer D-6',
      time: '19 Oct, 09:26',
      badgeClass: 'text-emerald-400',
    },
  ];

  const timeTicks = [
    { h: 0, label: '18 Oct 14:00' },
    { h: 4, label: '18:00' },
    { h: 8, label: '22:00' },
    { h: 12, label: '19 Oct 02:00' },
    { h: 16, label: '06:00' },
    { h: 20, label: '10:00' },
    { h: 24, label: '14:26 (+24h)' },
  ];

  // Métricas en tiempo real para las cabeceras de los 3 gráficos
  const currentHectares = Math.round(42 + Math.pow(simulationHour, 1.68) * 6.9);
  const currentRos = Math.round(14 + Math.sin(simulationHour * 0.45) * 9 + (simulationHour > 3 && simulationHour < 16 ? 12 : 2));
  const currentElevation = Math.round(1210 + (simulationHour / 24) * 970);

  return (
    <footer className="h-52 w-full bg-zinc-950/95 backdrop-blur-md border-t border-zinc-800 flex text-zinc-100 select-none shrink-0 z-30">
      {/* COLUMNA PRINCIPAL IZQUIERDA: TIMELINE + 3 GRÁFICOS FORENSES */}
      <div className="flex-1 flex flex-col justify-between p-3 min-w-0">
        {/* 1. Cabecera y Barra Deslizante del Timeline */}
        <div>
          <div className="flex items-center justify-between gap-4 mb-1.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-zinc-200 mr-1">Línea de Tiempo</span>

              {/* Botón Reiniciar T+0h */}
              <button
                type="button"
                onClick={onReset}
                className="h-7 px-2 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs font-mono transition-colors cursor-pointer"
                title="Reiniciar a Ignición (T+0h)"
              >
                ⏮
              </button>

              {/* Botón Principal: Simular Avance Cinemático 3D */}
              <button
                type="button"
                onClick={onTogglePlay}
                className={`h-7 px-3 rounded font-mono font-bold text-xs tracking-wide flex items-center gap-1.5 transition-all cursor-pointer ${
                  isPlaying
                    ? 'bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_15px_rgba(245,158,11,0.5)]'
                    : 'bg-orange-600 hover:bg-orange-500 text-zinc-950 shadow-[0_0_12px_rgba(234,88,12,0.35)]'
                }`}
              >
                <span>{isPlaying ? '⏸ PAUSAR 3D' : '▶ SIMULAR AVANCE 3D'}</span>
              </button>

              {/* Botón Paso +1h */}
              <button
                type="button"
                onClick={() => onChangeSimulationHour?.(Math.min(24, simulationHour + 1))}
                className="h-7 px-2 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs font-mono transition-colors cursor-pointer"
                title="Avanzar +1 hora"
              >
                ⏭
              </button>

              {/* Selector de Velocidad (1x, 2x, 5x) */}
              <select
                value={playbackSpeed}
                onChange={(e) => onChangePlaybackSpeed?.(Number(e.target.value))}
                className="h-7 px-2 rounded bg-zinc-900 border border-zinc-700 text-zinc-200 font-mono text-xs cursor-pointer focus:outline-none focus:border-orange-500"
              >
                <option value={1}>1x</option>
                <option value={2}>2.5x</option>
                <option value={4}>5x</option>
              </select>
            </div>

            {/* Indicador Flotante de Timestamp Actual */}
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="px-2.5 py-0.5 rounded bg-zinc-800 border border-zinc-600 text-orange-400 font-bold shadow">
                {formatTimestamp(simulationHour)} (T+{simulationHour}h)
              </span>
              <span className="text-[11px] text-zinc-400 hidden xl:inline">
                18 Oct, 14:26 – 19 Oct, 14:26
              </span>
            </div>
          </div>

          {/* Barra Scrubber con Hitos Superpuestos */}
          <div className="relative px-1 pt-1">
            <input
              type="range"
              min={0}
              max={24}
              step={0.5}
              value={simulationHour}
              onChange={(e) => onChangeSimulationHour?.(Number(e.target.value))}
              className="w-full accent-orange-500 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
            />

            {/* Escala Horaria Inferior */}
            <div className="flex justify-between font-mono text-[10px] text-zinc-500 mt-0.5">
              {timeTicks.map((t) => (
                <button
                  key={t.h}
                  type="button"
                  onClick={() => onChangeSimulationHour?.(t.h)}
                  className={`hover:text-zinc-200 transition-colors cursor-pointer ${
                    Math.round(simulationHour) === t.h ? 'text-orange-400 font-bold' : ''
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 2. Tres Gráficos Forenses Sincronizados (Superficie, Velocidad de Propagación, Perfil de Elevación) */}
        <div className="grid grid-cols-3 gap-3 h-24 pt-1">
          {/* GRÁFICO 1: Superficie Quemada (ha) */}
          <div className="bg-zinc-900/80 border border-zinc-800/90 rounded p-2 flex flex-col justify-between relative overflow-hidden">
            <div className="flex items-center justify-between text-[11px] z-10">
              <span className="text-zinc-300 font-medium">Superficie Quemada (ha)</span>
              <span className="font-mono text-orange-400 font-bold">
                {currentHectares.toLocaleString('es-CL')} ha
              </span>
            </div>

            <div className="relative flex-1 mt-1">
              <svg viewBox="0 0 240 48" preserveAspectRatio="none" className="w-full h-full">
                <defs>
                  <linearGradient id="areaFireGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity="0.55" />
                    <stop offset="100%" stopColor="#f97316" stopOpacity="0.05" />
                  </linearGradient>
                </defs>
                <path
                  d="M0,46 Q45,43 85,34 T165,16 T240,4 L240,48 L0,48 Z"
                  fill="url(#areaFireGrad)"
                />
                <path
                  d="M0,46 Q45,43 85,34 T165,16 T240,4"
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="1.8"
                />
              </svg>
              {/* Cursor vertical sincronizado con simulationHour */}
              <div
                style={{ left: `${progressPct}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-zinc-100 shadow-[0_0_6px_#fff] pointer-events-none"
              >
                <span className="absolute -top-1 -left-1 h-2 w-2 rounded-full bg-orange-500 border border-white" />
              </div>
            </div>
          </div>

          {/* GRÁFICO 2: Velocidad de Propagación (m/min) */}
          <div className="bg-zinc-900/80 border border-zinc-800/90 rounded p-2 flex flex-col justify-between relative overflow-hidden">
            <div className="flex items-center justify-between text-[11px] z-10">
              <span className="text-zinc-300 font-medium">Velocidad Propagación (m/min)</span>
              <span className="font-mono text-red-400 font-bold">{currentRos} m/min</span>
            </div>

            <div className="relative flex-1 mt-1">
              <svg viewBox="0 0 240 48" preserveAspectRatio="none" className="w-full h-full">
                <defs>
                  <linearGradient id="rosGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity="0.45" />
                    <stop offset="100%" stopColor="#ef4444" stopOpacity="0.02" />
                  </linearGradient>
                </defs>
                <path
                  d="M0,42 L28,39 L45,14 L68,26 L95,6 L122,32 L155,18 L190,29 L240,36 L240,48 L0,48 Z"
                  fill="url(#rosGrad)"
                />
                <path
                  d="M0,42 L28,39 L45,14 L68,26 L95,6 L122,32 L155,18 L190,29 L240,36"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="1.6"
                />
              </svg>
              <div
                style={{ left: `${progressPct}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-zinc-100 shadow-[0_0_6px_#fff] pointer-events-none"
              >
                <span className="absolute -top-1 -left-1 h-2 w-2 rounded-full bg-red-500 border border-white" />
              </div>
            </div>
          </div>

          {/* GRÁFICO 3: Perfil de Elevación Topográfica (Ladera Las Trancas -> Nevados) */}
          <div className="bg-zinc-900/80 border border-zinc-800/90 rounded p-2 flex flex-col justify-between relative overflow-hidden">
            <div className="flex items-center justify-between text-[11px] z-10">
              <span className="text-zinc-300 font-medium">Perfil de Elevación (Cota 3D)</span>
              <span className="font-mono text-emerald-400 font-bold">
                {currentElevation.toLocaleString('es-CL')} m s.n.m.
              </span>
            </div>

            <div className="relative flex-1 mt-1">
              <svg viewBox="0 0 240 48" preserveAspectRatio="none" className="w-full h-full">
                <defs>
                  <linearGradient id="elevGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.45" />
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.08" />
                  </linearGradient>
                </defs>
                <path
                  d="M0,42 L35,36 L75,27 L115,29 L160,17 L205,11 L240,5 L240,48 L0,48 Z"
                  fill="url(#elevGrad)"
                />
                <path
                  d="M0,42 L35,36 L75,27 L115,29 L160,17 L205,11 L240,5"
                  fill="none"
                  stroke="#34d399"
                  strokeWidth="1.6"
                />
              </svg>
              <div
                style={{ left: `${progressPct}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-zinc-100 shadow-[0_0_6px_#fff] pointer-events-none"
              >
                <span className="absolute -top-1 -left-1 h-2 w-2 rounded-full bg-emerald-400 border border-white" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* COLUMNA DERECHA DEL DOCK: EVENTOS CLAVE (Key Events - Estilo FireTrace) */}
      <div className="w-80 xl:w-88 border-l border-zinc-800 p-3 flex flex-col justify-between shrink-0 bg-zinc-950/90">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-bold text-zinc-100">Eventos Clave Forenses</span>
          <span className="text-[10px] font-mono text-zinc-400">Clic para ir a T+h</span>
        </div>

        <ul className="space-y-1 overflow-y-auto pr-1">
          {keyEvents.map((ev) => {
            const isActive = Math.abs(simulationHour - ev.hour) < 2;
            return (
              <li
                key={ev.id}
                onClick={() => {
                  onChangeSimulationHour?.(ev.hour);
                  onSelectKeyEvent?.(ev);
                }}
                className={`px-2 py-1 rounded text-[11px] flex items-center justify-between gap-2 cursor-pointer transition-colors ${
                  isActive
                    ? 'bg-orange-500/20 border border-orange-500/50 text-zinc-100'
                    : 'hover:bg-zinc-900 text-zinc-300'
                }`}
              >
                <span className="flex items-center gap-1.5 truncate">
                  <span>{ev.icon}</span>
                  <span className="truncate font-medium">{ev.label}</span>
                </span>
                <span className="font-mono text-[10px] text-zinc-400 shrink-0">
                  {ev.time}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </footer>
  );
}
