import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polygon } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import HomePanel from '../components/HomePanel';
import { getNASAData } from '../utils/api';

// Centro geográfico inicial: Región de Ñuble, Chile
const NUBLE_CENTER = [-36.75, -71.85];
const INITIAL_ZOOM = 9;

// Polígonos ilustrativos para la capa de Riesgo (DEMO) en formato [lat, lon]
const DEMO_RISK_ZONES = [
  {
    id: 'demo-zone-precordillera',
    positions: [
      [-36.68, -71.68],
      [-36.64, -71.42],
      [-36.98, -71.38],
      [-37.04, -71.65],
    ],
  },
  {
    id: 'demo-zone-secano',
    positions: [
      [-36.52, -72.52],
      [-36.48, -72.22],
      [-36.78, -72.18],
      [-36.82, -72.48],
    ],
  },
];

// Incidentes activos en monitoreo (sin identificadores inventados ni etiquetas flotantes)
const ACTIVE_INCIDENTS = [
  {
    id: 'las-trancas',
    name: 'Las Trancas, Pinto',
    position: [-36.9012, -71.4825],
  },
  {
    id: 'quillon',
    name: 'Quillón, Valle del Itata',
    position: [-36.742, -72.468],
  },
];

export default function Dashboard() {
  const [hotspots, setHotspots] = useState([]);
  const [layers, setLayers] = useState({
    nasa: true,
    risk: true,
    incidents: true,
  });

  useEffect(() => {
    let isMounted = true;

    async function loadHotspots() {
      const nasaResponse = await getNASAData();
      if (isMounted && nasaResponse?.hotspots) {
        setHotspots(nasaResponse.hotspots);
      }
    }

    loadHotspots();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleToggleLayer = (layerKey) => {
    setLayers((prev) => ({
      ...prev,
      [layerKey]: !prev[layerKey],
    }));
  };

  return (
    <div className="h-screen w-screen flex bg-zinc-950 overflow-hidden">
      {/* Mapa 2D a pantalla completa (Esri Dark Gray) */}
      <div className="flex-1 h-full relative">
        <MapContainer
          center={NUBLE_CENTER}
          zoom={INITIAL_ZOOM}
          className="h-full w-full bg-zinc-900"
          zoomControl={true}
        >
          <TileLayer
            attribution="&copy; Esri, HERE, Garmin, FAO, NOAA, USGS"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          />

          {/* Capa 1: Riesgo (DEMO) - Esquema Ámbar para pronósticos */}
          {layers.risk &&
            DEMO_RISK_ZONES.map((zone) => (
              <Polygon
                key={zone.id}
                positions={zone.positions}
                pathOptions={{
                  color: '#f59e0b',
                  weight: 1.5,
                  dashArray: '4 4',
                  fillColor: '#f59e0b',
                  fillOpacity: 0.18,
                }}
              />
            ))}

          {/* Capa 2: Focos NASA - Esquema Rojo para estados activos */}
          {layers.nasa &&
            hotspots.map((spot) => (
              <CircleMarker
                key={spot.id}
                center={[spot.latitude, spot.longitude]}
                radius={5}
                pathOptions={{
                  color: '#ef4444',
                  weight: 1,
                  fillColor: '#ef4444',
                  fillOpacity: 0.75,
                }}
              />
            ))}

          {/* Capa 3: Incidentes - Clic ejecuta únicamente console.log('Navegar a IncidentDetail') */}
          {layers.incidents &&
            ACTIVE_INCIDENTS.map((incident) => (
              <CircleMarker
                key={incident.id}
                center={incident.position}
                radius={9}
                pathOptions={{
                  color: '#fecaca',
                  weight: 2,
                  fillColor: '#dc2626',
                  fillOpacity: 0.95,
                }}
                eventHandlers={{
                  click: () => {
                    console.log('Navegar a IncidentDetail');
                  },
                }}
              />
            ))}
        </MapContainer>
      </div>

      {/* Panel Lateral Minimalista */}
      <HomePanel
        riskToday="ALTO"
        riskTomorrow="CRÍTICO"
        activeHotspotsCount={hotspots.length > 0 ? hotspots.length : 7}
        layers={layers}
        onToggleLayer={handleToggleLayer}
      />
    </div>
  );
}
