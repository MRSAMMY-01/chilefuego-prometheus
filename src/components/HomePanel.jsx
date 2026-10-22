import React from 'react';

export default function HomePanel({
  riskToday = 'ALTO',
  riskTomorrow = 'CRÍTICO',
  activeHotspotsCount = 7,
  layers,
  onToggleLayer,
}) {
  const layerItems = [
    {
      key: 'nasa',
      label: 'Focos NASA',
      dotClass: 'bg-red-500',
      isDemo: false,
    },
    {
      key: 'risk',
      label: 'Riesgo',
      dotClass: 'bg-amber-500',
      isDemo: true,
    },
    {
      key: 'incidents',
      label: 'Incidentes',
      dotClass: 'bg-red-600',
      isDemo: false,
    },
  ];

  return (
    <aside className="w-80 h-full bg-zinc-950 border-l border-zinc-800 text-zinc-100 flex flex-col justify-between p-5 select-none z-20">
      <div className="space-y-6">
        {/* Encabezado Minimalista */}
        <header className="border-b border-zinc-800 pb-4">
          <div className="flex items-center justify-between">
            <h1 className="text-base font-bold tracking-wider uppercase text-zinc-100">
              ChileFuego
            </h1>
            <span className="text-xs font-mono text-zinc-400">
              Región de Ñuble
            </span>
          </div>
          <p className="text-xs text-zinc-500 mt-1">
            Monitoreo regional de ignición y focos satelitales
          </p>
        </header>

        {/* 3 Tarjetas de Información Directa (Máximo 3) */}
        <section className="space-y-3" aria-label="Indicadores principales">
          {/* Tarjeta 1: Riesgo hoy (Pronóstico -> Ámbar + Cinta DEMO) */}
          <div className="relative overflow-hidden rounded-lg bg-zinc-900 border border-zinc-800 p-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                Riesgo hoy
              </span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider rounded bg-amber-500/15 text-amber-400 border border-amber-500/40">
                DEMO
              </span>
            </div>
            <div className="text-2xl font-bold tracking-tight text-amber-400">
              {riskToday}
            </div>
          </div>

          {/* Tarjeta 2: Riesgo mañana (Pronóstico -> Ámbar + Cinta DEMO) */}
          <div className="relative overflow-hidden rounded-lg bg-zinc-900 border border-zinc-800 p-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                Riesgo mañana
              </span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider rounded bg-amber-500/15 text-amber-400 border border-amber-500/40">
                DEMO
              </span>
            </div>
            <div className="text-2xl font-bold tracking-tight text-amber-500">
              {riskTomorrow}
            </div>
          </div>

          {/* Tarjeta 3: Focos activos (últimas 24 h) (Estado activo -> Rojo) */}
          <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                Focos activos (últimas 24 h)
              </span>
              <span className="h-2 w-2 rounded-full bg-red-500" />
            </div>
            <div className="text-3xl font-mono font-bold tracking-tight text-red-500">
              {activeHotspotsCount}
            </div>
          </div>
        </section>

        {/* Control de Capas Simplificadas (Máximo 3 capas, switch simple) */}
        <section className="pt-2 border-t border-zinc-800" aria-label="Control de capas">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-3">
            Capas del mapa
          </h2>

          <div className="space-y-2">
            {layerItems.map((item) => {
              const isChecked = Boolean(layers[item.key]);
              return (
                <div
                  key={item.key}
                  className="flex items-center justify-between rounded-md bg-zinc-900/70 border border-zinc-800 px-3 py-2.5"
                >
                  <div className="flex items-center gap-2.5">
                    <span className={`h-2.5 w-2.5 rounded-full ${item.dotClass}`} />
                    <span className="text-sm text-zinc-200">{item.label}</span>
                    {item.isDemo && (
                      <span className="px-1.5 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider rounded bg-amber-500/15 text-amber-400 border border-amber-500/40">
                        DEMO
                      </span>
                    )}
                  </div>

                  <button
                    type="button"
                    role="switch"
                    aria-checked={isChecked}
                    aria-label={`Alternar capa ${item.label}`}
                    onClick={() => onToggleLayer(item.key)}
                    className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      isChecked ? 'bg-zinc-200' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-zinc-950 shadow transition duration-200 ease-in-out ${
                        isChecked ? 'translate-x-4' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      {/* Pie de panel sobrio en gris */}
      <footer className="border-t border-zinc-800 pt-3 text-[11px] text-zinc-500 flex items-center justify-between">
        <span>Vista general 2D</span>
        <span className="font-mono">VIIRS · MODIS</span>
      </footer>
    </aside>
  );
}
