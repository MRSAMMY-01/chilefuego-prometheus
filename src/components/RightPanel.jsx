import React, { useState } from 'react';
import { TACTICAL_INCIDENTS } from '../utils/api';

/**
 * @file src/components/RightPanel.jsx
 * @description Panel lateral derecho de análisis forense y control táctico inspirado en FireTrace.
 * Incluye capas con miniaturas y sliders de opacidad, Resumen del Incidente, Clima y Comportamiento
 * del Fuego (6 métricas reales/derivadas de Open-Meteo) y selector interactivo de Recursos / Focos NASA.
 */
export default function RightPanel({
  weather,
  isLoadingWeather = false,
  layers,
  onToggleLayer,
  layerOpacities,
  onChangeLayerOpacity,
  basemap = 'satellite',
  onChangeBasemap,
  selectedIncident = 'INC-NUBLE-01',
  simulationHour = 0,
  activeResources = [],
  hotspots = [],
  onSelectResource,
  onSelectHotspot,
}) {
  const [activeTab, setActiveTab] = useState('layers'); // 'layers' | 'resources' | 'firms'

  const incident =
    TACTICAL_INCIDENTS.find((i) => i.id === selectedIncident) || TACTICAL_INCIDENTS[0];

  const temp = weather?.temperature ?? 16.7;
  const humidity = weather?.humidity ?? 33;
  const windSpeed = weather?.windSpeed ?? 6.6;
  const windDir = weather?.windDirection ?? 319;
  const windCardinal = weather?.windCardinal ?? 'NO';
  const fuelMoisture = weather?.fuelMoisture ?? 6;
  const erc = weather?.erc ?? 78;
  const flameLength = weather?.flameLength ?? '2.8–4.6 m';

  // Superficie afectada dinámica en hectáreas según la hora de simulación (T+0h a T+24h)
  const burnedHectares = Math.round(42 + Math.pow(simulationHour, 1.68) * 6.9);
  const burnedAcres = Math.round(burnedHectares * 2.471);

  const layerItems = [
    {
      key: 'terrain3d',
      label: 'Terreno (3D DEM)',
      thumbClass: 'bg-gradient-to-br from-stone-700 via-amber-900 to-emerald-950 border-stone-600',
    },
    {
      key: 'aerialImagery',
      label: 'Imagen Satelital (Esri)',
      thumbClass: 'bg-gradient-to-br from-emerald-900 via-stone-800 to-zinc-900 border-emerald-700/60',
    },
    {
      key: 'burnScar',
      label: 'Cicatriz de Quema (dNBR)',
      thumbClass: 'bg-gradient-to-br from-amber-950 via-red-950 to-stone-950 border-red-800/70',
    },
    {
      key: 'fireSpread',
      label: 'Avance del Fuego (Modelado)',
      thumbClass: 'bg-gradient-to-br from-orange-500 via-amber-600 to-red-700 border-orange-400',
    },
    {
      key: 'nasaHotspots',
      label: 'Focos NASA FIRMS (VIIRS)',
      thumbClass: 'bg-radial from-yellow-300 via-red-600 to-zinc-900 border-red-500',
    },
    {
      key: 'windVectors',
      label: 'Vectores Viento & Avance',
      thumbClass: 'bg-gradient-to-r from-zinc-900 via-orange-500/50 to-amber-400 border-orange-500/60',
    },
    {
      key: 'suppression',
      label: 'Supresión (Descargas & Dozer)',
      thumbClass: 'bg-gradient-to-r from-red-600 via-red-500 to-emerald-500 border-red-400',
    },
    {
      key: 'peaksAndLabels',
      label: 'Cumbres & Cotas Topográficas',
      thumbClass: 'bg-zinc-800 border-zinc-600',
    },
  ];

  return (
    <aside className="w-80 xl:w-88 h-full bg-zinc-950/95 backdrop-blur-md border-l border-zinc-800 flex flex-col text-zinc-100 select-none shrink-0 z-30 overflow-y-auto">
      {/* Pestañas Superiores (Estilo FireTrace) */}
      <div className="flex items-center border-b border-zinc-800 px-3 pt-2 gap-1 shrink-0 bg-zinc-950/90 sticky top-0 z-10">
        <button
          type="button"
          onClick={() => setActiveTab('layers')}
          className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'layers'
              ? 'border-orange-500 text-zinc-100'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Capas ({layerItems.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('resources')}
          className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'resources'
              ? 'border-orange-500 text-zinc-100'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Maquinaria ({activeResources.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('firms')}
          className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'firms'
              ? 'border-orange-500 text-zinc-100'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          NASA FIRMS ({hotspots.length})
        </button>
      </div>

      {/* CONTENIDO PESTAÑA 1: CAPAS CON SLIDERS DE OPACIDAD */}
      {activeTab === 'layers' && (
        <section className="p-3.5 border-b border-zinc-800/90 space-y-2.5">
          {/* Selector rápido de estilo cartográfico base */}
          <div className="grid grid-cols-3 gap-1 font-mono text-[10px] mb-2">
            {[
              { id: 'satellite', label: 'SATÉLITE 3D' },
              { id: 'tacticalDark', label: 'OSCURO C2' },
              { id: 'topo', label: 'TOPOGRÁFICO' },
            ].map((bm) => (
              <button
                key={bm.id}
                type="button"
                onClick={() => onChangeBasemap?.(bm.id)}
                className={`py-1 px-1.5 rounded border transition-colors cursor-pointer ${
                  basemap === bm.id
                    ? 'bg-orange-500/20 border-orange-500 text-orange-400 font-bold'
                    : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {bm.label}
              </button>
            ))}
          </div>

          <div className="space-y-2">
            {layerItems.map((item) => {
              const isChecked = layers?.[item.key] ?? true;
              const opacityVal = layerOpacities?.[item.key] ?? 100;

              return (
                <div
                  key={item.key}
                  className="flex items-center justify-between gap-2 text-xs py-1 px-1.5 rounded hover:bg-zinc-900/70 transition-colors"
                >
                  <label className="flex items-center gap-2 cursor-pointer min-w-0 flex-1">
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => onToggleLayer?.(item.key)}
                      className="accent-orange-500 h-3.5 w-3.5 rounded cursor-pointer shrink-0"
                    />
                    <span
                      className={`h-5 w-5 rounded border shrink-0 ${item.thumbClass}`}
                    />
                    <span
                      className={`truncate text-[11px] ${
                        isChecked ? 'text-zinc-200 font-medium' : 'text-zinc-500 line-through'
                      }`}
                    >
                      {item.label}
                    </span>
                  </label>

                  {/* Slider de Opacidad + Porcentaje */}
                  <div className="flex items-center gap-1.5 shrink-0">
                    <input
                      type="range"
                      min={10}
                      max={100}
                      step={5}
                      disabled={!isChecked}
                      value={opacityVal}
                      onChange={(e) =>
                        onChangeLayerOpacity?.(item.key, Number(e.target.value))
                      }
                      className="w-12 accent-zinc-400 h-1 bg-zinc-800 rounded cursor-pointer disabled:opacity-30"
                    />
                    <span className="font-mono text-[10px] text-zinc-400 w-8 text-right">
                      {opacityVal}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* CONTENIDO PESTAÑA 2: MAQUINARIA PESADA Y RECURSOS AÉREOS (Con Fly-To al hacer clic) */}
      {activeTab === 'resources' && (
        <section className="p-3.5 border-b border-zinc-800/90 space-y-2">
          <p className="text-[11px] text-zinc-400 mb-2">
            Haz clic en una unidad para volar la cámara 3D a su posición en terreno:
          </p>
          {activeResources.map((unit) => (
            <div
              key={unit.id}
              onClick={() => onSelectResource?.(unit)}
              className="p-2.5 rounded bg-zinc-900/90 border border-zinc-800 hover:border-orange-500/60 transition-colors cursor-pointer"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-xs font-bold text-zinc-100">
                  {unit.icon} {unit.name}
                </span>
                <span
                  className={`font-mono text-[9px] font-semibold px-1.5 py-0.5 rounded border ${unit.statusColor}`}
                >
                  {unit.status}
                </span>
              </div>
              <p className="font-mono text-[10px] text-zinc-400 mt-1">{unit.type}</p>
              <div className="flex items-center justify-between font-mono text-[10px] text-zinc-500 mt-1.5 pt-1 border-t border-zinc-800">
                <span>GPS: {unit.coords}</span>
                <span className="text-orange-400">🎯 Volar 3D →</span>
              </div>
            </div>
          ))}
        </section>
      )}

      {/* CONTENIDO PESTAÑA 3: FOCOS SATELITALES REALES NASA FIRMS */}
      {activeTab === 'firms' && (
        <section className="p-3.5 border-b border-zinc-800/90 space-y-2 max-h-68 overflow-y-auto">
          <p className="text-[11px] text-zinc-400 mb-2">
            Selecciona cualquier foco satelital de la NASA para volar hacia él y actualizar el clima de Open-Meteo en esa coordenada:
          </p>
          {hotspots.map((spot, idx) => (
            <div
              key={spot.id}
              onClick={() => onSelectHotspot?.(spot)}
              className="p-2 rounded bg-zinc-900/90 border border-zinc-800 hover:border-orange-500/60 transition-colors cursor-pointer flex items-center justify-between gap-2"
            >
              <div>
                <p className="font-mono text-xs font-bold text-orange-400">
                  🔥 #{idx + 1} · {spot.sector}
                </p>
                <p className="font-mono text-[10px] text-zinc-400">
                  Lat: {spot.latitude.toFixed(3)}° · Lon: {spot.longitude.toFixed(3)}° · FRP: {spot.frp} MW
                </p>
              </div>
              <span className="font-mono text-[10px] px-2 py-1 rounded bg-zinc-800 text-zinc-200 shrink-0">
                Ver 3D
              </span>
            </div>
          ))}
        </section>
      )}

      {/* SECCIÓN 2: RESUMEN DEL INCIDENTE (Incident Overview - Estilo FireTrace) */}
      <section className="p-3.5 border-b border-zinc-800/90 space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold text-zinc-100 tracking-wide">
            Resumen del Incidente
          </h3>
          <span className="text-[10px] font-mono text-orange-400">
            T+{simulationHour}h
          </span>
        </div>

        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between gap-2">
            <span className="text-zinc-400">Incidente</span>
            <span className="text-zinc-100 font-medium text-right">{incident.name}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-zinc-400">ID CONAF / SENAPRED</span>
            <span className="text-zinc-200 font-mono">{incident.code}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-zinc-400">Ubicación</span>
            <span className="text-zinc-200 text-right truncate max-w-[180px]">
              {incident.location}
            </span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-zinc-400">Ignición (Est.)</span>
            <span className="text-zinc-200 font-mono">{incident.startTime}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-zinc-400">Superficie Quemada</span>
            <span className="text-orange-400 font-mono font-bold">
              {burnedHectares.toLocaleString('es-CL')} ha ({burnedAcres.toLocaleString('es-CL')} ac)
            </span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-zinc-400">Causa Preliminar</span>
            <span className="text-zinc-300">{incident.cause}</span>
          </div>
          <div className="flex justify-between items-center gap-2 pt-0.5">
            <span className="text-zinc-400">Estado Táctico</span>
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-500/15 border border-amber-500/40 text-amber-300 text-[11px] font-medium">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
              {incident.status}
            </span>
          </div>
        </div>
      </section>

      {/* SECCIÓN 3: CLIMA & COMPORTAMIENTO DEL FUEGO (6 Tarjetas Estilo FireTrace) */}
      <section className="p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs font-bold text-zinc-100">
              Clima & Comportamiento del Fuego
            </h3>
            <p className="text-[10px] font-mono text-zinc-400 truncate max-w-[250px]">
              📍 {weather?.location || 'Las Trancas, Pinto, Ñuble'}
            </p>
          </div>
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400">
            Open-Meteo
          </span>
        </div>

        {isLoadingWeather ? (
          <div className="grid grid-cols-3 gap-2 animate-pulse">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 bg-zinc-900 rounded border border-zinc-800" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-2">
            {/* 1. Viento */}
            <div className="bg-zinc-900/90 border border-zinc-800/90 rounded p-2 flex flex-col justify-between">
              <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                <span>💨</span> Viento
              </span>
              <span className="font-mono text-xs font-bold text-zinc-100 mt-1">
                {windSpeed} km/h
              </span>
              <span className="font-mono text-[10px] text-zinc-400">
                {windCardinal} ({windDir}°)
              </span>
            </div>

            {/* 2. Temperatura */}
            <div className="bg-zinc-900/90 border border-zinc-800/90 rounded p-2 flex flex-col justify-between">
              <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                <span>🌡️</span> Temp
              </span>
              <span className="font-mono text-sm font-bold text-zinc-100 mt-1">
                {temp} °C
              </span>
              <span className="font-mono text-[10px] text-zinc-500">Superficie</span>
            </div>

            {/* 3. Humedad Relativa (RH) */}
            <div className="bg-zinc-900/90 border border-zinc-800/90 rounded p-2 flex flex-col justify-between">
              <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                <span>💧</span> HR
              </span>
              <span className="font-mono text-sm font-bold text-sky-400 mt-1">
                {humidity}%
              </span>
              <span className="font-mono text-[10px] text-zinc-500">Relativa</span>
            </div>

            {/* 4. Humedad del Combustible Fino */}
            <div className="bg-zinc-900/90 border border-zinc-800/90 rounded p-2 flex flex-col justify-between">
              <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                <span>🍃</span> Combust.
              </span>
              <span className="font-mono text-sm font-bold text-amber-400 mt-1">
                {fuelMoisture}%
              </span>
              <span className="font-mono text-[10px] text-zinc-500">Fino Muerto</span>
            </div>

            {/* 5. Índice ERC (Energy Release Component) */}
            <div className="bg-zinc-900/90 border border-zinc-800/90 rounded p-2 flex flex-col justify-between">
              <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                <span>🔥</span> ERC
              </span>
              <span className="font-mono text-sm font-bold text-orange-400 mt-1">
                {erc}
              </span>
              <span className="font-mono text-[10px] text-zinc-500">Energía Lib.</span>
            </div>

            {/* 6. Longitud de Llama */}
            <div className="bg-zinc-900/90 border border-zinc-800/90 rounded p-2 flex flex-col justify-between">
              <span className="text-[10px] text-zinc-400 flex items-center gap-1">
                <span>📐</span> Llama
              </span>
              <span className="font-mono text-xs font-bold text-red-400 mt-1">
                {flameLength}
              </span>
              <span className="font-mono text-[10px] text-zinc-500">Frente Activo</span>
            </div>
          </div>
        )}
      </section>
    </aside>
  );
}
