import React from 'react';
import { TACTICAL_INCIDENTS } from '../utils/api';

/**
 * @file src/components/TopBar.jsx
 * @description Barra superior de comando táctico y análisis forense inspirada en FireTrace.
 */
export default function TopBar({
  selectedIncident,
  onSelectIncident,
  hotspotCount = 0,
  resourceCount = 4,
  nasaSource = 'NASA FIRMS LIVE',
  lastSync = '--:--:--',
  onRefresh,
  isLoading = false,
  onFlyToIncident,
  onCycleHotspot,
  dispatchMode = null,
  onToggleDispatchMode,
  isCinematic = false,
  onToggleCinematic,
}) {
  const activeIncidentObj =
    TACTICAL_INCIDENTS.find((i) => i.id === selectedIncident) || TACTICAL_INCIDENTS[0];

  return (
    <header className="h-12 w-full z-40 bg-zinc-950/95 backdrop-blur-md border-b border-zinc-800 px-4 flex items-center justify-between gap-3 text-zinc-100 select-none shrink-0">
      {/* Izquierda: Identidad + Selector de Incidente + Ubicación + Rango Temporal */}
      <div className="flex items-center gap-3 overflow-x-auto no-scrollbar">
        <div className="flex items-center gap-2 shrink-0">
          <div className="h-7 w-7 rounded bg-orange-500/20 border border-orange-500/60 flex items-center justify-center shadow-[0_0_12px_rgba(249,115,22,0.35)]">
            <span className="text-orange-500 text-sm">🔥</span>
          </div>
          <span className="font-bold text-sm tracking-tight text-zinc-100">
            Chile<span className="text-orange-500">Fuego</span>
          </span>
        </div>

        <span className="text-zinc-700">|</span>

        <span className="text-xs text-zinc-400 hidden xl:inline shrink-0">
          Análisis Forense & Comando Táctico
        </span>

        <span className="text-zinc-700 hidden xl:inline">|</span>

        {/* Dropdown de Incidente Activo (Hace Fly-To 3D al cambiar) */}
        <select
          id="incident-selector"
          value={selectedIncident}
          onChange={(e) => onSelectIncident?.(e.target.value)}
          className="bg-zinc-900 hover:bg-zinc-800 text-zinc-100 font-medium text-xs px-2.5 py-1.5 rounded border border-zinc-700/80 focus:outline-none focus:border-orange-500 transition-colors cursor-pointer shrink-0"
        >
          {TACTICAL_INCIDENTS.map((inc) => (
            <option key={inc.id} value={inc.id} className="bg-zinc-900 text-zinc-100">
              {inc.name}
            </option>
          ))}
        </select>

        <span className="text-xs text-zinc-400 hidden lg:inline shrink-0">
          {activeIncidentObj.location}
        </span>

        <span className="text-zinc-700 hidden md:inline">|</span>

        <div className="hidden md:flex items-center gap-1.5 text-xs text-zinc-300 font-mono shrink-0">
          <span className="text-zinc-500">📅</span>
          <span>
            {activeIncidentObj.startTime} – {activeIncidentObj.endTime}
          </span>
        </div>
      </div>

      {/* Derecha: Píldoras Tácticas de Navegación, Despacho y Estado Satelital */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Botón: Enfocar Incidente en 3D */}
        <button
          type="button"
          onClick={onFlyToIncident}
          className="px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 flex items-center gap-1.5 transition-colors cursor-pointer"
          title="Volar cámara 3D al punto de ignición del incidente"
        >
          <span>🗺️</span>
          <span className="hidden sm:inline">Incidente</span>
        </button>

        {/* Botón: Recorrer Focos Satelitales NASA FIRMS */}
        <button
          type="button"
          onClick={onCycleHotspot}
          className="px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 flex items-center gap-1.5 transition-colors cursor-pointer"
          title={`Inspeccionar focos térmicos detectados por ${nasaSource} (Clic para saltar al siguiente foco y consultar su clima local)`}
        >
          <span>🛰️</span>
          <span className="hidden sm:inline">Focos NASA</span>
          <span className="px-1.5 py-0.2 rounded-full bg-orange-500/20 border border-orange-500/50 text-orange-400 font-mono text-[10px] font-bold">
            {hotspotCount}
          </span>
        </button>

        {/* Botón: Recursos y Maquinaria */}
        <button
          type="button"
          onClick={onFlyToIncident}
          className="hidden lg:flex px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 items-center gap-1.5 transition-colors cursor-pointer"
        >
          <span>🚜</span>
          <span>Recursos</span>
          <span className="px-1.5 py-0.2 rounded-full bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 font-mono text-[10px] font-bold">
            {resourceCount}
          </span>
        </button>

        {/* Botón Interactivo: Despachar Descarga Aérea / Cortafuego en el Mapa */}
        <button
          type="button"
          onClick={() =>
            onToggleDispatchMode?.(dispatchMode === 'retardant' ? null : 'retardant')
          }
          className={`px-2.5 py-1 rounded border text-xs flex items-center gap-1.5 transition-colors cursor-pointer ${
            dispatchMode
              ? 'bg-red-600/30 border-red-500 text-red-200 shadow-[0_0_12px_rgba(239,68,68,0.4)]'
              : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-700 text-zinc-200'
          }`}
          title="Haz clic aquí y luego en la montaña 3D para despachar una línea de descarga del Avión DC-10 VLAT"
        >
          <span>🎯</span>
          <span className="hidden md:inline">
            {dispatchMode ? 'Clic en Mapa 3D...' : 'Despachar Descarga'}
          </span>
        </button>

        {/* Botón: Cámara Cinemática 3D */}
        <button
          type="button"
          onClick={onToggleCinematic}
          className={`px-2.5 py-1 rounded border text-xs flex items-center gap-1.5 transition-colors cursor-pointer ${
            isCinematic
              ? 'bg-orange-500/25 border-orange-500 text-orange-300'
              : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-700 text-zinc-300'
          }`}
          title="Activar/Desactivar órbita cinemática 3D sobre la ladera"
        >
          <span>🎬</span>
          <span className="hidden md:inline">Cámara 3D</span>
        </button>

        {/* Botón Sync APIs */}
        <button
          type="button"
          onClick={onRefresh}
          disabled={isLoading}
          title={`Última sincronización: ${lastSync}`}
          className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-[11px] font-mono text-zinc-300 transition-colors disabled:opacity-50 cursor-pointer"
        >
          {isLoading ? '...' : '↻'}
        </button>

        {/* Indicador 🔴 LIVE */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-950/70 border border-red-500/50">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
          </span>
          <span className="font-mono font-bold text-[10px] tracking-wider text-red-400">
            LIVE
          </span>
        </div>
      </div>
    </header>
  );
}
