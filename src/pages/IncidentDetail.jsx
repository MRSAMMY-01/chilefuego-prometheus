import React, { useEffect, useState, useRef, useCallback } from 'react';
import * as maplibregl from 'maplibre-gl';

import {
  getNASAData,
  getLocalWeather,
  buildFireSimulationGeoJSON,
  TACTICAL_INCIDENTS,
  MOUNTAIN_PEAKS,
} from '../utils/api';
import TopBar from '../components/TopBar';
import RightPanel from '../components/RightPanel';
import BottomTimeline from '../components/BottomTimeline';

/**
 * @file src/pages/Dashboard.jsx
 * @description Centro de comando táctico y análisis forense 3D para ChileFuego (Región de Ñuble).
 * Utiliza MapLibre GL JS con relieve 3D real (DEM Terrarium), fotografía satelital Esri,
 * telemetría en vivo NASA FIRMS + Open-Meteo y cámara cinemática 3D durante la simulación.
 */
export default function IncidentDetail() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);

  // Estados de Telemetría Real (NASA FIRMS + Open-Meteo)
  const [hotspots, setHotspots] = useState([]);
  const [hotspotIndex, setHotspotIndex] = useState(0);
  const [nasaMeta, setNasaMeta] = useState({
    source: 'CONECTANDO NASA FIRMS...',
    updatedAt: '--:--:--',
  });
  const [weather, setWeather] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Estado del Incidente y Configuración 3D del Terreno
  const [selectedIncident, setSelectedIncident] = useState('INC-NUBLE-01');
  const [basemap, setBasemap] = useState('satellite'); // 'satellite' | 'tacticalDark' | 'topo'
  const [is3D, setIs3D] = useState(true);
  const [terrainExaggeration, setTerrainExaggeration] = useState(2.5);
  const [cameraBearing, setCameraBearing] = useState(58);
  const [isCinematic, setIsCinematic] = useState(false);

  // Estado de Simulación Cinemática (T+0h a T+24h)
  const [simulationHour, setSimulationHour] = useState(6);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Modo interactivo para despachar descargas aéreas en el mapa 3D
  const [dispatchMode, setDispatchMode] = useState(null);
  const [customSuppressions, setCustomSuppressions] = useState([]);

  // Inspector forense bajo el cursor en el mapa 3D
  const [cursorTelemetry, setCursorTelemetry] = useState({
    lat: -36.898,
    lon: -71.471,
    elevation: 1468,
    slope: 34,
    aspect: 'SO',
    fuelType: 'Bosque Nativo / Renoval Roble-Coigüe',
  });

  // Visibilidad y Opacidad de Capas (Estilo FireTrace)
  const [layers, setLayers] = useState({
    terrain3d: true,
    aerialImagery: true,
    burnScar: true,
    fireSpread: true,
    nasaHotspots: true,
    windVectors: true,
    suppression: true,
    peaksAndLabels: true,
  });

  const [layerOpacities, setLayerOpacities] = useState({
    terrain3d: 100,
    aerialImagery: 100,
    burnScar: 90,
    fireSpread: 100,
    nasaHotspots: 100,
    windVectors: 85,
    suppression: 100,
    peaksAndLabels: 95,
  });

  // Unidades pesadas y aéreas desplegadas en Las Trancas / Nevados de Chillán
  const tacticalUnits = [
    {
      id: 'RES-VLAT-910',
      icon: '✈️',
      name: 'Avión T-910 DC-10 VLAT',
      type: 'AÉREO · SUPERTANKER (36.000 L Retardante)',
      status: 'DESCARGA EN LADERA',
      statusColor: 'text-red-300 bg-red-950/70 border-red-500/50',
      coords: '-36.893°, -71.464°',
      lat: -36.893,
      lon: -71.464,
      telemetry: 'ALT: 1.950m · GS: 260 kt',
    },
    {
      id: 'RES-DOZER-D6',
      icon: '🚜',
      name: 'Excavadora Dozer CAT D-6',
      type: 'MAQUINARIA PESADA · CORTAFUEGO MINERAL',
      status: 'OPERANDO EN FLANCO',
      statusColor: 'text-emerald-300 bg-emerald-950/70 border-emerald-500/50',
      coords: '-36.899°, -71.469°',
      lat: -36.899,
      lon: -71.469,
      telemetry: 'AVANCE: 420 m/h · COMB: 78%',
    },
    {
      id: 'RES-HELI-CH47',
      icon: '🚁',
      name: 'Helicóptero Chinook CH-47D',
      type: 'AÉREO PESADO · HELIBALDE (10.000 L)',
      status: 'EN APOYO CABEZA',
      statusColor: 'text-amber-300 bg-amber-950/70 border-amber-500/50',
      coords: '-36.889°, -71.454°',
      lat: -36.889,
      lon: -71.454,
      telemetry: 'ALT: 2.120m · CICLO: 6 min',
    },
    {
      id: 'RES-SKIDDER-525',
      icon: '🚜',
      name: 'Skidder Forestal CAT-525D',
      type: 'MAQUINARIA PESADA · FAJA CONTENCIÓN',
      status: 'EN RUTA N-55',
      statusColor: 'text-sky-300 bg-sky-950/70 border-sky-500/50',
      coords: '-36.906°, -71.478°',
      lat: -36.906,
      lon: -71.478,
      telemetry: 'COMB: 91% · SECTOR SUR',
    },
  ];

  const activeIncident =
    TACTICAL_INCIDENTS.find((i) => i.id === selectedIncident) || TACTICAL_INCIDENTS[0];

  // Consulta de datos reales NASA FIRMS + Open-Meteo
  const fetchTacticalTelemetry = useCallback(async () => {
    setIsLoading(true);
    try {
      const [nasaResult, weatherResult] = await Promise.all([
        getNASAData(1),
        getLocalWeather(activeIncident.lat, activeIncident.lon, activeIncident.location),
      ]);

      setHotspots(nasaResult.hotspots || []);
      setNasaMeta({
        source: nasaResult.source,
        updatedAt: nasaResult.updatedAt,
      });
      setWeather(weatherResult);
    } catch (err) {
      console.error('[Dashboard] Error obteniendo telemetría:', err);
    } finally {
      setIsLoading(false);
    }
  }, [activeIncident.lat, activeIncident.lon, activeIncident.location]);

  useEffect(() => {
    fetchTacticalTelemetry();
  }, [fetchTacticalTelemetry]);

  // Inicialización del Motor 3D MapLibre GL con Relieve DEM Real (AWS Terrarium + Esri World Imagery)
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return undefined;

    const initialGeoJSON = buildFireSimulationGeoJSON({
      originLat: activeIncident.lat,
      originLon: activeIncident.lon,
      simulationHour: 6,
      spreadBearing: 58,
    });

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      center: [activeIncident.lon + 0.012, activeIncident.lat - 0.008],
      zoom: activeIncident.zoom,
      pitch: 64,
      bearing: 58,
      maxPitch: 82,
      antialias: true,
      style: {
        version: 8,
        sources: {
          // Fuente de Elevación 3D Real (AWS Open Terrain Tiles - Sin API Key)
          'terrain-dem': {
            type: 'raster-dem',
            tiles: [
              'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
            ],
            encoding: 'terrarium',
            tileSize: 256,
            maxzoom: 15,
          },
          // Sombreado de laderas (Hillshade) para resaltar quebradas y crestas
          'hillshade-dem': {
            type: 'raster-dem',
            tiles: [
              'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
            ],
            encoding: 'terrarium',
            tileSize: 256,
            maxzoom: 15,
          },
          // 1. Ortofoto Satelital Esri World Imagery
          'esri-satellite': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            attribution: 'Esri, Maxar, Earthstar Geographics | NASA FIRMS',
          },
          // 2. Base Oscura C2 Esri
          'esri-dark': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
          },
          // 3. Base Topográfica OpenTopoMap
          'open-topo': {
            type: 'raster',
            tiles: ['https://a.tile.opentopomap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
          },
          // 4. Red Vial y Caminos Cordilleranos Esri
          'esri-roads': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
          },
          // Fuentes GeoJSON Dinámicas del Incidente
          'burn-scar-src': {
            type: 'geojson',
            data: initialGeoJSON.burnScarGeoJSON,
          },
          'final-perimeter-src': {
            type: 'geojson',
            data: initialGeoJSON.finalPerimeterGeoJSON,
          },
          'trajectory-src': {
            type: 'geojson',
            data: initialGeoJSON.trajectoryGeoJSON,
          },
          'suppression-src': {
            type: 'geojson',
            data: initialGeoJSON.suppressionGeoJSON,
          },
          'nasa-hotspots-src': {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: [] },
          },
        },
        terrain: {
          source: 'terrain-dem',
          exaggeration: 2.5,
        },
        layers: [
          {
            id: 'background-sky',
            type: 'background',
            paint: { 'background-color': '#09090b' },
          },
          {
            id: 'layer-dark-base',
            type: 'raster',
            source: 'esri-dark',
            layout: { visibility: 'none' },
          },
          {
            id: 'layer-topo-base',
            type: 'raster',
            source: 'open-topo',
            layout: { visibility: 'none' },
          },
          {
            id: 'layer-satellite',
            type: 'raster',
            source: 'esri-satellite',
            paint: {
              'raster-opacity': 1,
              'raster-contrast': 0.12,
              'raster-saturation': 0.08,
            },
          },
          {
            id: 'layer-hillshade',
            type: 'hillshade',
            source: 'hillshade-dem',
            paint: {
              'hillshade-exaggeration': 0.45,
              'hillshade-shadow-color': '#09090b',
              'hillshade-highlight-color': '#fef3c7',
              'hillshade-accent-color': '#451a03',
            },
          },
          {
            id: 'layer-roads',
            type: 'raster',
            source: 'esri-roads',
            paint: { 'raster-opacity': 0.75 },
          },
          // Perímetro Final Proyectado T+24h (Línea blanca punteada exterior como en FireTrace)
          {
            id: 'layer-final-perimeter',
            type: 'line',
            source: 'final-perimeter-src',
            filter: ['==', ['get', 'type'], 'final-perimeter'],
            paint: {
              'line-color': '#f4f4f5',
              'line-width': 1.8,
              'line-dasharray': [3, 3],
              'line-opacity': 0.85,
            },
          },
          // Isócronas intermedias (T+6h, T+12h, T+18h)
          {
            id: 'layer-isochrones',
            type: 'line',
            source: 'final-perimeter-src',
            filter: ['!=', ['get', 'type'], 'final-perimeter'],
            paint: {
              'line-color': '#fdba74',
              'line-width': 1.1,
              'line-dasharray': [2, 3],
              'line-opacity': 0.55,
            },
          },
          // Cicatriz de Quema (Burn Scar dNBR - Interior carbonizado oscuro/rojizo)
          {
            id: 'layer-burn-scar-fill',
            type: 'fill',
            source: 'burn-scar-src',
            paint: {
              'fill-color': '#271713',
              'fill-opacity': 0.76,
            },
          },
          // Resplandor Exterior del Frente de Fuego Activo (Glow Naranja/Ámbar)
          {
            id: 'layer-fire-front-glow',
            type: 'line',
            source: 'burn-scar-src',
            paint: {
              'line-color': '#f97316',
              'line-width': 10,
              'line-blur': 7,
              'line-opacity': 0.85,
            },
          },
          // Línea de Llama Viva en el Borde del Perímetro
          {
            id: 'layer-fire-front-core',
            type: 'line',
            source: 'burn-scar-src',
            paint: {
              'line-color': '#fde047',
              'line-width': 2.5,
              'line-opacity': 0.95,
            },
          },
          // Zonas de Descarga de Retardante Aéreo (Rojo Phos-Chek como en las capturas)
          {
            id: 'layer-retardant-fill',
            type: 'fill',
            source: 'suppression-src',
            filter: ['==', ['get', 'type'], 'retardant-drop'],
            paint: {
              'fill-color': '#ef4444',
              'fill-opacity': 0.65,
            },
          },
          {
            id: 'layer-retardant-outline',
            type: 'line',
            source: 'suppression-src',
            filter: ['==', ['get', 'type'], 'retardant-drop'],
            paint: {
              'line-color': '#fca5a5',
              'line-width': 1.5,
            },
          },
          // Línea de Cortafuego Maquinaria Pesada (Dozer D-6)
          {
            id: 'layer-dozer-line',
            type: 'line',
            source: 'suppression-src',
            filter: ['==', ['get', 'type'], 'dozer-line'],
            paint: {
              'line-color': '#10b981',
              'line-width': 3.5,
              'line-dasharray': [2, 1],
            },
          },
          // Vectores Secundarios de Propagación por Pendiente/Viento (Flechas Naranjas)
          {
            id: 'layer-spread-vectors',
            type: 'line',
            source: 'trajectory-src',
            filter: ['==', ['get', 'type'], 'spread-vector'],
            paint: {
              'line-color': '#fb923c',
              'line-width': 2.8,
              'line-opacity': 0.9,
            },
          },
          // Trayectoria Principal Amarilla desde la Ignición
          {
            id: 'layer-main-trajectory',
            type: 'line',
            source: 'trajectory-src',
            filter: ['==', ['get', 'type'], 'main-trajectory'],
            paint: {
              'line-color': '#facc15',
              'line-width': 3,
              'line-dasharray': [2, 1.5],
              'line-opacity': 0.95,
            },
          },
          // Puntos Calientes Satelitales NASA FIRMS (VIIRS)
          {
            id: 'layer-nasa-hotspots-halo',
            type: 'circle',
            source: 'nasa-hotspots-src',
            paint: {
              'circle-radius': 14,
              'circle-color': '#ef4444',
              'circle-blur': 0.85,
              'circle-opacity': 0.65,
            },
          },
          {
            id: 'layer-nasa-hotspots-core',
            type: 'circle',
            source: 'nasa-hotspots-src',
            paint: {
              'circle-radius': 6,
              'circle-color': '#fef08a',
              'circle-stroke-width': 2,
              'circle-stroke-color': '#dc2626',
            },
          },
        ],
      },
    });

    map.on('rotate', () => {
      setCameraBearing(Math.round(map.getBearing()));
    });

    map.on('mousemove', (e) => {
      const { lng, lat } = e.lngLat;
      // Estimación topográfica continua para el inspector de terreno en Las Trancas
      const estElev = Math.round(1180 + (lng - -71.52) * 14500 + (lat - -36.92) * 8200);
      const estSlope = Math.min(48, Math.max(8, Math.round(24 + Math.sin(lng * 320) * 14)));
      setCursorTelemetry({
        lat,
        lon: lng,
        elevation: Math.max(450, Math.min(3212, estElev)),
        slope: estSlope,
        aspect: lng > -71.47 ? 'SO (Ladera Sol)' : 'NO (Quebrada)',
        fuelType: estElev > 1750 ? 'Matorral Alto Andino / Ñirre' : 'Bosque Nativo Roble-Coigüe',
      });
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Clic en el mapa 3D para despachar una línea de descarga aérea o consultar un foco NASA
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return undefined;

    const handleMapClick = (e) => {
      if (dispatchMode === 'retardant') {
        const { lng, lat } = e.lngLat;
        const newDrop = {
          type: 'Feature',
          properties: {
            type: 'retardant-drop',
            unit: 'Avión T-910 DC-10 VLAT (Despacho Manual)',
          },
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [lng - 0.0045, lat - 0.0012],
                [lng + 0.0045, lat + 0.0018],
                [lng + 0.0041, lat + 0.0028],
                [lng - 0.0049, lat - 0.0002],
                [lng - 0.0045, lat - 0.0012],
              ],
            ],
          },
        };
        setCustomSuppressions((prev) => [...prev, newDrop]);
        setDispatchMode(null);
      }
    };

    map.on('click', handleMapClick);
    return () => map.off('click', handleMapClick);
  }, [dispatchMode]);

  // Actualizar las capas GeoJSON del incendio cuando cambia simulationHour, el incidente o las descargas
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const simData = buildFireSimulationGeoJSON({
      originLat: activeIncident.lat,
      originLon: activeIncident.lon,
      simulationHour,
      spreadBearing: 58,
      customSuppressions,
    });

    map.getSource('burn-scar-src')?.setData(simData.burnScarGeoJSON);
    map.getSource('final-perimeter-src')?.setData(simData.finalPerimeterGeoJSON);
    map.getSource('trajectory-src')?.setData(simData.trajectoryGeoJSON);
    map.getSource('suppression-src')?.setData(simData.suppressionGeoJSON);
  }, [simulationHour, activeIncident, customSuppressions]);

  // Actualizar los focos reales de NASA FIRMS en el mapa 3D
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const updateNasaSource = () => {
      const geojson = {
        type: 'FeatureCollection',
        features: hotspots.map((spot) => ({
          type: 'Feature',
          properties: { ...spot },
          geometry: {
            type: 'Point',
            coordinates: [spot.longitude, spot.latitude],
          },
        })),
      };
      map.getSource('nasa-hotspots-src')?.setData(geojson);
    };

    if (map.isStyleLoaded()) {
      updateNasaSource();
    } else {
      map.once('load', updateNasaSource);
    }
  }, [hotspots]);

  // Renderizar Marcadores HTML 3D (Punto de Ignición Luminoso, Hitos de Hora, Cumbres y Maquinaria)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Limpiar marcadores previos
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    const simData = buildFireSimulationGeoJSON({
      originLat: activeIncident.lat,
      originLon: activeIncident.lon,
      simulationHour,
      spreadBearing: 58,
    });

    // 1. Marcador Luminoso de Ignición (Idéntico a la captura de referencia: "Ignition · 18 Oct, 14:26")
    const ignitionEl = document.createElement('div');
    ignitionEl.className = 'flex items-center gap-2 pointer-events-auto cursor-pointer';
    ignitionEl.innerHTML = `
      <div class="relative flex items-center justify-center">
        <div class="h-10 w-10 rounded-full bg-amber-300/40 blur-md ignition-flare absolute"></div>
        <div class="h-4 w-4 rounded-full bg-yellow-200 border-2 border-red-600 shadow-[0_0_18px_#facc15] relative z-10"></div>
      </div>
      <div class="bg-zinc-950/90 border border-zinc-700/90 rounded px-2.5 py-1 text-[11px] font-mono text-zinc-100 shadow-xl backdrop-blur-md">
        <div class="font-bold text-amber-300">Ignición</div>
        <div class="text-[10px] text-zinc-300">${activeIncident.startTime}</div>
      </div>
    `;
    const ignitionMarker = new maplibregl.Marker({ element: ignitionEl, anchor: 'left' })
      .setLngLat([activeIncident.lon, activeIncident.lat])
      .addTo(map);
    markersRef.current.push(ignitionMarker);

    // 2. Hitos Horarios a lo largo de la trayectoria amarilla (18:00, 00:00, 06:00, 14:26)
    if (layers.windVectors) {
      simData.milestones.slice(1).forEach((m) => {
        if (simulationHour >= m.hour - 2) {
          const mileEl = document.createElement('div');
          mileEl.className = 'flex items-center gap-1.5 pointer-events-none';
          mileEl.innerHTML = `
            <span class="h-3 w-3 rounded-full bg-orange-400 border-2 border-yellow-200 shadow-[0_0_10px_#f97316]"></span>
            <span class="px-2 py-0.5 rounded bg-zinc-950/90 border border-zinc-700 font-mono text-[10px] text-zinc-100 shadow-lg">
              ${m.timeLabel}
            </span>
          `;
          const marker = new maplibregl.Marker({ element: mileEl, anchor: 'left' })
            .setLngLat([m.lon, m.lat])
            .addTo(map);
          markersRef.current.push(marker);
        }
      });
    }

    // 3. Cumbres y Cotas Topográficas (▲ Volcán Nevados de Chillán 3,212 m, etc.)
    if (layers.peaksAndLabels) {
      MOUNTAIN_PEAKS.forEach((peak) => {
        const peakEl = document.createElement('div');
        peakEl.className = 'pointer-events-none text-center';
        peakEl.innerHTML = `
          <div class="px-1.5 py-0.5 rounded bg-zinc-950/75 border border-zinc-700/70 text-[10px] font-mono text-zinc-200 shadow backdrop-blur-sm">
            <span class="text-amber-400">▲</span> ${peak.name} <span class="text-zinc-400">(${peak.elevation})</span>
          </div>
        `;
        const marker = new maplibregl.Marker({ element: peakEl })
          .setLngLat([peak.lon, peak.lat])
          .addTo(map);
        markersRef.current.push(marker);
      });
    }

    // 4. Unidades de Maquinaria Pesada y Aeronaves en Terreno
    if (layers.suppression) {
      tacticalUnits.forEach((unit) => {
        const unitEl = document.createElement('div');
        unitEl.className = 'cursor-pointer group';
        unitEl.innerHTML = `
          <div class="flex items-center gap-1 px-2 py-0.5 rounded-full bg-zinc-950/90 border border-emerald-500/70 text-[10px] font-mono text-emerald-300 shadow-lg hover:scale-105 transition-transform">
            <span>${unit.icon}</span>
            <span class="font-bold">${unit.name.split(' ')[0]} ${unit.name.split(' ')[1]}</span>
          </div>
        `;
        unitEl.onclick = () => {
          map.flyTo({
            center: [unit.lon, unit.lat],
            zoom: 14.6,
            pitch: 68,
            bearing: 45,
            duration: 1800,
          });
        };
        const marker = new maplibregl.Marker({ element: unitEl })
          .setLngLat([unit.lon, unit.lat])
          .addTo(map);
        markersRef.current.push(marker);
      });
    }
  }, [activeIncident, simulationHour, layers.windVectors, layers.peaksAndLabels, layers.suppression]);

  // Sincronizar Visibilidad, Opacidad, Modo 2D/3D y Exageración del Terreno con MapLibre
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    // 1. Terreno 3D y Exageración
    if (is3D && layers.terrain3d) {
      map.setTerrain({ source: 'terrain-dem', exaggeration: terrainExaggeration });
    } else {
      map.setTerrain(null);
    }

    // 2. Cambio de Mapa Base
    map.setLayoutProperty(
      'layer-satellite',
      'visibility',
      basemap === 'satellite' && layers.aerialImagery ? 'visible' : 'none'
    );
    map.setLayoutProperty(
      'layer-dark-base',
      'visibility',
      basemap === 'tacticalDark' ? 'visible' : 'none'
    );
    map.setLayoutProperty(
      'layer-topo-base',
      'visibility',
      basemap === 'topo' ? 'visible' : 'none'
    );

    // 3. Opacidades y Visibilidades individuales
    if (map.getLayer('layer-satellite')) {
      map.setPaintProperty(
        'layer-satellite',
        'raster-opacity',
        (layerOpacities.aerialImagery ?? 100) / 100
      );
    }
    if (map.getLayer('layer-burn-scar-fill')) {
      map.setLayoutProperty(
        'layer-burn-scar-fill',
        'visibility',
        layers.burnScar ? 'visible' : 'none'
      );
      map.setPaintProperty(
        'layer-burn-scar-fill',
        'fill-opacity',
        ((layerOpacities.burnScar ?? 90) / 100) * 0.8
      );
    }
    if (map.getLayer('layer-fire-front-glow')) {
      const spreadVis = layers.fireSpread ? 'visible' : 'none';
      map.setLayoutProperty('layer-fire-front-glow', 'visibility', spreadVis);
      map.setLayoutProperty('layer-fire-front-core', 'visibility', spreadVis);
      map.setLayoutProperty('layer-final-perimeter', 'visibility', spreadVis);
      map.setLayoutProperty('layer-isochrones', 'visibility', spreadVis);
      map.setPaintProperty(
        'layer-fire-front-glow',
        'line-opacity',
        (layerOpacities.fireSpread ?? 100) / 100
      );
    }
    if (map.getLayer('layer-main-trajectory')) {
      const vecVis = layers.windVectors ? 'visible' : 'none';
      map.setLayoutProperty('layer-main-trajectory', 'visibility', vecVis);
      map.setLayoutProperty('layer-spread-vectors', 'visibility', vecVis);
    }
    if (map.getLayer('layer-retardant-fill')) {
      const supVis = layers.suppression ? 'visible' : 'none';
      map.setLayoutProperty('layer-retardant-fill', 'visibility', supVis);
      map.setLayoutProperty('layer-retardant-outline', 'visibility', supVis);
      map.setLayoutProperty('layer-dozer-line', 'visibility', supVis);
    }
    if (map.getLayer('layer-nasa-hotspots-core')) {
      const nasaVis = layers.nasaHotspots ? 'visible' : 'none';
      map.setLayoutProperty('layer-nasa-hotspots-core', 'visibility', nasaVis);
      map.setLayoutProperty('layer-nasa-hotspots-halo', 'visibility', nasaVis);
    }
  }, [layers, layerOpacities, basemap, is3D, terrainExaggeration]);

  // Motor de Simulación Cinemática 3D: Avanza el fuego y mueve la cámara 3D por la ladera
  useEffect(() => {
    if (!isPlaying) return undefined;

    const intervalMs = Math.round(900 / playbackSpeed);
    const timer = setInterval(() => {
      setSimulationHour((prev) => {
        const nextHour = prev >= 24 ? 0 : +(prev + 0.5).toFixed(1);

        // Coreografía de Cámara Cinemática 3D siguiendo el frente de avance por la cordillera
        const map = mapRef.current;
        if (map) {
          const progress = nextHour / 24;
          const rad = (58 * Math.PI) / 180;
          const targetLon = activeIncident.lon + Math.sin(rad) * (progress * 0.024);
          const targetLat = activeIncident.lat + Math.cos(rad) * 0.8 * (progress * 0.024);

          map.easeTo({
            center: [targetLon, targetLat],
            zoom: 13.7 + Math.sin(progress * Math.PI) * 0.45,
            pitch: 66,
            bearing: 52 + progress * 28,
            duration: intervalMs,
            easing: (t) => t,
          });
        }

        return nextHour;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, activeIncident]);

  // Al iniciar la simulación 3D, colocamos la cámara en ángulo cinemático de montaña
  const handleTogglePlaySimulation = () => {
    const map = mapRef.current;
    if (!isPlaying && map) {
      setIs3D(true);
      setIsCinematic(true);
      if (simulationHour >= 24) setSimulationHour(0);
      map.flyTo({
        center: [activeIncident.lon + 0.008, activeIncident.lat - 0.004],
        zoom: 13.8,
        pitch: 66,
        bearing: 54,
        duration: 1400,
      });
    }
    setIsPlaying((prev) => !prev);
  };

  // Cambiar de Incidente y volar en 3D
  const handleSelectIncident = async (incidentId) => {
    setSelectedIncident(incidentId);
    const target = TACTICAL_INCIDENTS.find((i) => i.id === incidentId);
    if (target && mapRef.current) {
      mapRef.current.flyTo({
        center: [target.lon + 0.01, target.lat - 0.005],
        zoom: target.zoom,
        pitch: target.pitch,
        bearing: target.bearing,
        duration: 2000,
      });
      const updatedWeather = await getLocalWeather(target.lat, target.lon, target.location);
      setWeather(updatedWeather);
    }
  };

  // Volar hacia un foco satelital real de la NASA y consultar el clima de Open-Meteo en esa coordenada
  const handleSelectHotspot = async (spot) => {
    if (!spot || !mapRef.current) return;
    mapRef.current.flyTo({
      center: [spot.longitude, spot.latitude],
      zoom: 13.8,
      pitch: 64,
      bearing: 45,
      duration: 2000,
    });
    setIsLoading(true);
    const spotWeather = await getLocalWeather(spot.latitude, spot.longitude, spot.sector);
    setWeather(spotWeather);
    setIsLoading(false);
  };

  const handleCycleHotspot = () => {
    if (hotspots.length === 0) return;
    const nextIdx = (hotspotIndex + 1) % hotspots.length;
    setHotspotIndex(nextIdx);
    handleSelectHotspot(hotspots[nextIdx]);
  };

  // Alternar vista 2D / 3D
  const handleToggle2D3D = (want3D) => {
    setIs3D(want3D);
    if (mapRef.current) {
      mapRef.current.easeTo({
        pitch: want3D ? 64 : 0,
        bearing: want3D ? 58 : 0,
        duration: 1000,
      });
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-950 overflow-hidden select-none">
      {/* 1. BARRA SUPERIOR FORENSE & TÁCTICA */}
      <TopBar
        selectedIncident={selectedIncident}
        onSelectIncident={handleSelectIncident}
        hotspotCount={hotspots.length}
        resourceCount={tacticalUnits.length}
        nasaSource={nasaMeta.source}
        lastSync={nasaMeta.updatedAt}
        onRefresh={fetchTacticalTelemetry}
        isLoading={isLoading}
        onFlyToIncident={() => handleSelectIncident(selectedIncident)}
        onCycleHotspot={handleCycleHotspot}
        dispatchMode={dispatchMode}
        onToggleDispatchMode={setDispatchMode}
        isCinematic={isCinematic}
        onToggleCinematic={() => {
          const nextCin = !isCinematic;
          setIsCinematic(nextCin);
          if (mapRef.current) {
            mapRef.current.flyTo({
              center: [activeIncident.lon + 0.012, activeIncident.lat + 0.004],
              zoom: nextCin ? 14.1 : 13.4,
              pitch: nextCin ? 72 : 55,
              bearing: nextCin ? (cameraBearing + 65) % 360 : 58,
              duration: 2200,
            });
          }
        }}
      />

      {/* 2. ÁREA CENTRAL: VISOR 3D DE MONTAÑA + PANEL DERECHO */}
      <div className="flex-1 flex min-h-0 relative">
        {/* Lienzo WebGL 3D del Mapa */}
        <div className="flex-1 relative h-full bg-zinc-900 overflow-hidden">
          <div
            ref={mapContainerRef}
            className={`w-full h-full ${dispatchMode ? 'cursor-crosshair' : ''}`}
          />

          {/* Brújula Táctica 3D (Esquina Superior Izquierda como en FireTrace) */}
          <button
            type="button"
            onClick={() => mapRef.current?.easeTo({ bearing: 0, pitch: 60, duration: 900 })}
            title="Orientar al Norte (Clic para restablecer)"
            className="absolute top-4 left-4 z-20 h-11 w-11 rounded-full bg-zinc-950/85 border border-zinc-700/80 flex items-center justify-center shadow-2xl backdrop-blur-md hover:border-orange-500 transition-colors cursor-pointer"
          >
            <div
              style={{ transform: `rotate(${-cameraBearing}deg)` }}
              className="flex flex-col items-center justify-center transition-transform"
            >
              <span className="text-[9px] font-mono font-bold text-orange-400 leading-none mb-0.5">
                N
              </span>
              <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-b-[8px] border-b-orange-500" />
              <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[8px] border-t-zinc-500" />
            </div>
          </button>

          {/* Aviso flotante cuando está activo el modo de despacho de retardante */}
          {dispatchMode && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 px-4 py-2 rounded-full bg-red-950/90 border border-red-500 text-xs font-mono text-red-200 shadow-2xl animate-pulse">
              ✈️ MODO DESPACHO AÉREO: Haz clic sobre cualquier ladera 3D para lanzar retardante del DC-10 VLAT
            </div>
          )}

          {/* Inspector Forense de Terreno 3D (Flotante inferior-centro como en la imagen 2) */}
          <div className="hidden md:block absolute bottom-12 right-64 z-20 bg-zinc-950/90 border border-zinc-700/80 rounded-md px-3 py-2 text-[11px] font-mono text-zinc-200 shadow-2xl backdrop-blur-md pointer-events-none">
            <div className="font-bold text-zinc-100">
              {cursorTelemetry.elevation.toLocaleString('es-CL')} m · {cursorTelemetry.slope}° pendiente · Exposición {cursorTelemetry.aspect}
            </div>
            <div className="text-zinc-400 text-[10px]">
              {cursorTelemetry.fuelType}
            </div>
            <div className="text-zinc-500 text-[10px] mt-0.5">
              {Math.abs(cursorTelemetry.lat).toFixed(4)}°S · {Math.abs(cursorTelemetry.lon).toFixed(4)}°W
            </div>
          </div>

          {/* Controles Flotantes Inferiores Derechos del Mapa: Escala + 2D|3D + Exageración (1.5x / 2.5x / 5x) */}
          <div className="absolute bottom-3 right-3 z-20 flex items-center gap-2 font-mono text-xs">
            {/* Barra de Escala */}
            <div className="hidden sm:flex flex-col items-end bg-zinc-950/80 border border-zinc-800 px-2.5 py-1 rounded backdrop-blur-md text-[10px] text-zinc-300">
              <div className="flex justify-between w-28">
                <span>0</span>
                <span>250</span>
                <span>500 m</span>
              </div>
              <div className="w-28 h-1 border-b border-l border-r border-zinc-300 mt-0.5" />
            </div>

            {/* Selector 2D | 3D */}
            <div className="flex rounded overflow-hidden border border-zinc-700 bg-zinc-950/90 shadow-lg">
              <button
                type="button"
                onClick={() => handleToggle2D3D(false)}
                className={`px-2.5 py-1 text-[11px] font-bold transition-colors cursor-pointer ${
                  !is3D ? 'bg-orange-600 text-zinc-950' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                2D
              </button>
              <button
                type="button"
                onClick={() => handleToggle2D3D(true)}
                className={`px-2.5 py-1 text-[11px] font-bold transition-colors cursor-pointer ${
                  is3D ? 'bg-orange-600 text-zinc-950' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                3D
              </button>
            </div>

            {/* Selector de Exageración del Relieve 3D (1.5x, 2.5x, 4x) */}
            <select
              value={terrainExaggeration}
              onChange={(e) => setTerrainExaggeration(Number(e.target.value))}
              className="bg-zinc-950/90 border border-zinc-700 text-zinc-200 text-[11px] px-2 py-1 rounded shadow-lg cursor-pointer focus:outline-none focus:border-orange-500"
              title="Exageración vertical del relieve cordillerano 3D"
            >
              <option value={1.5}>1.5x</option>
              <option value={2.5}>2.5x</option>
              <option value={4.0}>4.0x</option>
            </select>

            {/* Botón Pantalla Completa */}
            <button
              type="button"
              onClick={() => {
                if (!document.fullscreenElement) {
                  document.documentElement.requestFullscreen?.();
                } else {
                  document.exitFullscreen?.();
                }
              }}
              className="px-2 py-1 rounded bg-zinc-950/90 border border-zinc-700 text-zinc-300 hover:text-white shadow-lg cursor-pointer"
              title="Pantalla completa"
            >
              ⛶
            </button>
          </div>
        </div>

        {/* Panel Lateral Derecho (Capas con Opacidad, Resumen e Índices de Fuego) */}
        <RightPanel
          weather={weather}
          isLoadingWeather={isLoading}
          layers={layers}
          onToggleLayer={(key) =>
            setLayers((prev) => ({ ...prev, [key]: !prev[key] }))
          }
          layerOpacities={layerOpacities}
          onChangeLayerOpacity={(key, val) =>
            setLayerOpacities((prev) => ({ ...prev, [key]: val }))
          }
          basemap={basemap}
          onChangeBasemap={setBasemap}
          selectedIncident={selectedIncident}
          simulationHour={simulationHour}
          activeResources={tacticalUnits}
          hotspots={hotspots}
          onSelectResource={(unit) => {
            mapRef.current?.flyTo({
              center: [unit.lon, unit.lat],
              zoom: 14.5,
              pitch: 68,
              bearing: 45,
              duration: 1800,
            });
          }}
          onSelectHotspot={handleSelectHotspot}
        />
      </div>

      {/* 3. DOCK INFERIOR: LÍNEA DE TIEMPO 3D, 3 GRÁFICOS FORENSES Y EVENTOS CLAVE */}
      <BottomTimeline
        simulationHour={simulationHour}
        onChangeSimulationHour={(h) => {
          setSimulationHour(h);
          // Mover suavemente la cámara hacia el frente correspondiente a la hora seleccionada
          if (mapRef.current) {
            const progress = h / 24;
            const rad = (58 * Math.PI) / 180;
            mapRef.current.easeTo({
              center: [
                activeIncident.lon + Math.sin(rad) * (progress * 0.022),
                activeIncident.lat + Math.cos(rad) * 0.8 * (progress * 0.022),
              ],
              pitch: is3D ? 65 : 0,
              duration: 600,
            });
          }
        }}
        isPlaying={isPlaying}
        onTogglePlay={handleTogglePlaySimulation}
        onReset={() => {
          setIsPlaying(false);
          setSimulationHour(0);
          mapRef.current?.flyTo({
            center: [activeIncident.lon, activeIncident.lat],
            zoom: 13.8,
            pitch: 64,
            bearing: 58,
            duration: 1500,
          });
        }}
        playbackSpeed={playbackSpeed}
        onChangePlaybackSpeed={setPlaybackSpeed}
        onSelectKeyEvent={(ev) => {
          setIsPlaying(false);
          const progress = ev.hour / 24;
          const rad = (58 * Math.PI) / 180;
          mapRef.current?.flyTo({
            center: [
              activeIncident.lon + Math.sin(rad) * (progress * 0.024),
              activeIncident.lat + Math.cos(rad) * 0.8 * (progress * 0.024),
            ],
            zoom: 14.1,
            pitch: 67,
            bearing: 52 + progress * 25,
            duration: 1500,
          });
        }}
      />
    </div>
  );
}
