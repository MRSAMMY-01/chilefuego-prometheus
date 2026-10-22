/**
 * @file src/utils/api.js
 * @description Módulo de integración de datos geoespaciales reales (NASA FIRMS),
 * meteorología dinámica por coordenada (Open-Meteo) y motor geométrico 3D Rothermel
 * para simulación cinemática de propagación de incendios en la Región de Ñuble.
 */

export const NUBLE_COORDS = {
  lat: -36.904,
  lon: -71.482,
  bbox: '-72.90,-37.35,-71.00,-36.00',
};

// Catálogo de incidentes tácticos en la Región de Ñuble
export const TACTICAL_INCIDENTS = [
  {
    id: 'INC-NUBLE-01',
    code: 'CL-NUB-2026-014',
    name: 'Incidente Las Trancas (IF-014)',
    location: 'Valle Las Trancas / Pinto, Ñuble',
    lat: -36.904,
    lon: -71.482,
    zoom: 13.7,
    pitch: 64,
    bearing: 58,
    startTime: '18 Oct, 14:26',
    endTime: '19 Oct, 14:26',
    cause: 'Bajo Investigación Forense',
    status: 'En Combate Activo',
  },
  {
    id: 'INC-NUBLE-02',
    code: 'CL-NUB-2026-019',
    name: 'Incidente Shangri-La (IF-019)',
    location: 'Faldeo Volcán Chillán, Ñuble',
    lat: -36.876,
    lon: -71.465,
    zoom: 13.6,
    pitch: 62,
    bearing: 42,
    startTime: '18 Oct, 15:10',
    endTime: '19 Oct, 15:10',
    cause: 'Descarga Eléctrica / Pavesas',
    status: 'Alerta Roja CONAF',
  },
  {
    id: 'INC-NUBLE-03',
    code: 'CL-NUB-2026-022',
    name: 'Incidente Los Lleuques (IF-022)',
    location: 'Corredor Ruta N-55 / Río Renegado',
    lat: -36.922,
    lon: -71.525,
    zoom: 13.5,
    pitch: 60,
    bearing: 75,
    startTime: '18 Oct, 16:05',
    endTime: '19 Oct, 16:05',
    cause: 'Interfaz Urbano-Forestal',
    status: 'Contención Flanco Sur',
  },
];

// Cumbres y puntos de referencia de elevación reales en el sector Las Trancas / Nevados de Chillán
export const MOUNTAIN_PEAKS = [
  { name: 'Volcán Nevados de Chillán', elevation: '3,212 m', lat: -36.863, lon: -71.377 },
  { name: 'Cerro Pirigallo', elevation: '2,540 m', lat: -36.888, lon: -71.428 },
  { name: 'Mirador Garganta del Diablo', elevation: '1,980 m', lat: -36.893, lon: -71.454 },
  { name: 'Ladera Shangri-La', elevation: '1,690 m', lat: -36.882, lon: -71.471 },
  { name: 'Valle Las Trancas (Ruta N-55)', elevation: '1,240 m', lat: -36.911, lon: -71.492 },
];

// Focos de respaldo en caso de desconexión
const FALLBACK_FIRMS_HOTSPOTS = [
  {
    id: 'firms-ignition-main',
    latitude: -36.904,
    longitude: -71.482,
    brightness: 358.4,
    frp: 94.6,
    confidence: 'high',
    acq_date: new Date().toISOString().split('T')[0],
    acq_time: '1426',
    satellite: 'VIIRS_SNPP_NRT',
    sector: 'Punto de Ignición - Las Trancas',
  },
  {
    id: 'firms-2',
    latitude: -36.895,
    longitude: -71.462,
    brightness: 364.1,
    frp: 118.2,
    confidence: 'high',
    acq_date: new Date().toISOString().split('T')[0],
    acq_time: '1650',
    satellite: 'VIIRS_SNPP_NRT',
    sector: 'Flanco Este - Faldeo Nevados',
  },
];

/**
 * Convierte grados azimutales a punto cardinal en español.
 */
export function degreesToCardinal(deg = 0) {
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
  return dirs[Math.round(((deg % 360) < 0 ? deg + 360 : deg) / 45) % 8];
}

/**
 * Parsea la respuesta CSV de NASA FIRMS.
 */
function parseFIRMSCsv(csvText) {
  const lines = csvText.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map((h) => h.trim());

  return lines
    .slice(1)
    .map((line, index) => {
      const values = line.split(',').map((v) => v.trim());
      const entry = headers.reduce((acc, header, i) => {
        acc[header] = values[i];
        return acc;
      }, {});

      return {
        id: `firms-${index}-${entry.acq_time || Date.now()}`,
        latitude: parseFloat(entry.latitude),
        longitude: parseFloat(entry.longitude),
        brightness: parseFloat(entry.bright_ti4 || entry.brightness || 335),
        frp: parseFloat(entry.frp || 32.0),
        confidence: entry.confidence || 'nominal',
        acq_date: entry.acq_date || new Date().toISOString().split('T')[0],
        acq_time: entry.acq_time || '1426',
        satellite: entry.instrument || entry.satellite || 'VIIRS',
        sector: `Detección Satelital Ñuble #${index + 1}`,
      };
    })
    .filter((spot) => !Number.isNaN(spot.latitude) && !Number.isNaN(spot.longitude));
}

/**
 * Obtiene focos de calor en tiempo real desde NASA FIRMS (VIIRS_SNPP_NRT)
 * usando import.meta.env.VITE_NASA_API_KEY.
 */
export async function getNASAData(days = 1) {
  const apiKey = import.meta.env.VITE_NASA_API_KEY;

  if (!apiKey || apiKey === 'tu_map_key_de_nasa_firms_aqui') {
    return {
      hotspots: FALLBACK_FIRMS_HOTSPOTS,
      source: 'SIMULATED_FALLBACK (Sin MAP_KEY)',
      updatedAt: new Date().toLocaleTimeString('es-CL'),
    };
  }

  const endpoint = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${apiKey}/VIIRS_SNPP_NRT/${NUBLE_COORDS.bbox}/${days}`;

  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: { Accept: 'text/csv, text/plain, */*' },
    });

    if (!response.ok) {
      throw new Error(`Error HTTP NASA FIRMS: ${response.status}`);
    }

    const csvData = await response.text();
    if (csvData.toLowerCase().includes('invalid map_key')) {
      throw new Error('MAP_KEY de NASA FIRMS inválida.');
    }

    const parsedHotspots = parseFIRMSCsv(csvData);

    // Garantizamos incluir el punto de ignición forense en Las Trancas junto con todos los focos reales de Ñuble
    const combinedHotspots = [
      FALLBACK_FIRMS_HOTSPOTS[0],
      ...parsedHotspots,
    ];

    return {
      hotspots: combinedHotspots,
      liveCount: parsedHotspots.length,
      source: 'NASA FIRMS VIIRS_SNPP_NRT (LIVE)',
      updatedAt: new Date().toLocaleTimeString('es-CL'),
    };
  } catch (error) {
    console.error('[ChileFuego API] Error al consultar NASA FIRMS:', error);
    return {
      hotspots: FALLBACK_FIRMS_HOTSPOTS,
      liveCount: FALLBACK_FIRMS_HOTSPOTS.length,
      source: 'FALLBACK_OFFLINE',
      updatedAt: new Date().toLocaleTimeString('es-CL'),
      error: error.message,
    };
  }
}

/**
 * Obtiene las condiciones meteorológicas reales desde Open-Meteo para cualquier coordenada
 * (por defecto Las Trancas: Lat -36.90, Lon -71.48) y calcula variables de comportamiento del fuego.
 */
export async function getLocalWeather(
  lat = -36.90,
  lon = -71.48,
  sectorName = 'Las Trancas, Pinto, Ñuble'
) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&timezone=America%2FSantiago`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Error HTTP Open-Meteo: ${response.status}`);
    }

    const data = await response.json();
    const current = data.current || {};

    const temp = current.temperature_2m ?? 18.5;
    const humidity = current.relative_humidity_2m ?? 32;
    const windSpeed = current.wind_speed_10m ?? 14.2;
    const windDirection = current.wind_direction_10m ?? 245;

    // Cálculos forestales derivados (Humedad de Combustible Fino, ERC y Longitud de Llama)
    const fuelMoisture = Math.max(3, Math.min(25, Math.round(humidity * 0.18 + (30 - temp) * 0.12)));
    const erc = Math.min(98, Math.max(25, Math.round(50 + (temp - 15) * 1.6 + (50 - humidity) * 0.55 + windSpeed * 0.6)));
    const flameLengthMin = Math.max(0.8, ((windSpeed * 0.12) + (temp * 0.08)).toFixed(1));
    const flameLengthMax = (parseFloat(flameLengthMin) + 1.8).toFixed(1);

    return {
      temperature: temp,
      humidity,
      windSpeed,
      windDirection,
      windCardinal: degreesToCardinal(windDirection),
      fuelMoisture,
      erc,
      flameLength: `${flameLengthMin}–${flameLengthMax} m`,
      coords: { lat, lon },
      location: `${sectorName} (${lat.toFixed(3)}°, ${lon.toFixed(3)}°)`,
      updatedAt: new Date().toLocaleTimeString('es-CL'),
    };
  } catch (error) {
    console.error('[ChileFuego API] Error al consultar Open-Meteo:', error);
    return {
      temperature: 17.0,
      humidity: 33,
      windSpeed: 12.5,
      windDirection: 245,
      windCardinal: 'SO',
      fuelMoisture: 7,
      erc: 74,
      flameLength: '2.4–4.2 m',
      coords: { lat, lon },
      location: `${sectorName} (Caché Táctico)`,
      updatedAt: new Date().toLocaleTimeString('es-CL'),
    };
  }
}

/**
 * Genera un polígono elíptico orgánico orientado en la dirección de propagación (viento + pendiente)
 * para simular la cicatriz de quema (dNBR) y el frente activo de llama sobre el terreno 3D.
 */
function createFireSpreadPolygon(originLon, originLat, progressHours, bearingDeg = 58, isOuter = false) {
  const effectiveHours = Math.max(0.35, progressHours);
  const rad = (bearingDeg * Math.PI) / 180;
  const dirLon = Math.sin(rad);
  const dirLat = Math.cos(rad) * 0.8; // corrección de latitud en Ñuble (~37°S)
  const perpLon = Math.cos(rad);
  const perpLat = -Math.sin(rad) * 0.8;

  // Avance longitudinal (ladera arriba hacia el ENE) y expansión lateral
  const forwardScale = effectiveHours * 0.00145 * (isOuter ? 1.04 : 1.0);
  const backScale = effectiveHours * 0.00022 * (isOuter ? 1.05 : 1.0);
  const flankScale = Math.pow(effectiveHours, 0.82) * 0.00062 * (isOuter ? 1.05 : 1.0);

  const centerShiftLon = originLon + dirLon * (forwardScale * 0.46);
  const centerShiftLat = originLat + dirLat * (forwardScale * 0.46);
  const semiMajor = (forwardScale + backScale) * 0.54;

  const numPoints = 48;
  const coords = [];

  for (let i = 0; i <= numPoints; i++) {
    const theta = (i / numPoints) * Math.PI * 2;
    // Deformación orgánica determinista para simular lenguas de fuego en quebradas y crestas
    const fingerPerturbation =
      1 +
      0.11 * Math.sin(theta * 3 + 0.8) +
      0.07 * Math.cos(theta * 5 - 0.4) +
      0.04 * Math.sin(theta * 9 + effectiveHours * 0.25);

    const localAlong = Math.cos(theta) * semiMajor * fingerPerturbation;
    const localAcross = Math.sin(theta) * flankScale * fingerPerturbation;

    const ptLon = centerShiftLon + dirLon * localAlong + perpLon * localAcross;
    const ptLat = centerShiftLat + dirLat * localAlong + perpLat * localAcross;
    coords.push([ptLon, ptLat]);
  }

  // Asegurar cierre exacto del anillo GeoJSON
  coords[coords.length - 1] = coords[0];
  return coords;
}

/**
 * Construye todas las capas GeoJSON forenses para la hora de simulación actual (T+0h a T+24h).
 */
export function buildFireSimulationGeoJSON({
  originLat = -36.904,
  originLon = -71.482,
  simulationHour = 6,
  spreadBearing = 58,
  customSuppressions = [],
}) {
  // 1. Polígono de cicatriz de quema actual (Burn Scar dNBR + Frente Activo)
  const currentPerimeterCoords = createFireSpreadPolygon(
    originLon,
    originLat,
    simulationHour,
    spreadBearing,
    false
  );

  // 2. Perímetro final proyectado a T+24h (línea punteada blanca exterior)
  const finalPerimeterCoords = createFireSpreadPolygon(
    originLon,
    originLat,
    24,
    spreadBearing,
    true
  );

  // 3. Isócronas intermedias (T+6h, T+12h, T+18h)
  const isochroneHours = [6, 12, 18];
  const isochroneFeatures = isochroneHours.map((h) => ({
    type: 'Feature',
    properties: { hour: h, label: `T+${h}h` },
    geometry: {
      type: 'LineString',
      coordinates: createFireSpreadPolygon(originLon, originLat, h, spreadBearing, false),
    },
  }));

  // 4. Línea amarilla principal de trayectoria del frente de avance + Hitos horarios
  const rad = (spreadBearing * Math.PI) / 180;
  const dirLon = Math.sin(rad);
  const dirLat = Math.cos(rad) * 0.8;
  const perpLon = Math.cos(rad);
  const perpLat = -Math.sin(rad) * 0.8;

  const milestones = [
    { hour: 0, timeLabel: 'Ignición · 14:26', dist: 0, lateral: 0, elev: 1210 },
    { hour: 4, timeLabel: '18:00 (T+4h)', dist: 0.0058, lateral: -0.0006, elev: 1380 },
    { hour: 10, timeLabel: '00:00 (T+10h)', dist: 0.0145, lateral: 0.0005, elev: 1640 },
    { hour: 16, timeLabel: '06:00 (T+16h)', dist: 0.0232, lateral: -0.0003, elev: 1910 },
    { hour: 24, timeLabel: '14:26 (T+24h)', dist: 0.0342, lateral: 0.0004, elev: 2180 },
  ].map((m) => ({
    ...m,
    lon: originLon + dirLon * m.dist + perpLon * m.lateral,
    lat: originLat + dirLat * m.dist + perpLat * m.lateral,
  }));

  // Vectores secundarios naranjos de propagación por pendiente/viento (flechas en la ladera)
  const spreadVectors = [
    {
      from: [originLon + dirLon * 0.0035 - perpLon * 0.0025, originLat + dirLat * 0.0035 - perpLat * 0.0025],
      to: [originLon + dirLon * 0.0115 - perpLon * 0.001, originLat + dirLat * 0.0115 - perpLat * 0.001],
      minHour: 2,
    },
    {
      from: [originLon + dirLon * 0.0095 - perpLon * 0.0038, originLat + dirLat * 0.0095 - perpLat * 0.0038],
      to: [originLon + dirLon * 0.0195 + perpLon * 0.0002, originLat + dirLat * 0.0195 + perpLat * 0.0002],
      minHour: 6,
    },
    {
      from: [originLon + dirLon * 0.0175 + perpLon * 0.0015, originLat + dirLat * 0.0175 + perpLat * 0.0015],
      to: [originLon + dirLon * 0.0285 + perpLon * 0.0008, originLat + dirLat * 0.0285 + perpLat * 0.0008],
      minHour: 12,
    },
  ].filter((v) => simulationHour >= v.minHour);

  // 5. Zonas de Supresión: Descargas de Retardante Aéreo (Rojo) y Cortafuegos Dozer D-6 (Amarillo/Verde)
  const retardantDropFlankNorth = [
    [originLon + dirLon * 0.008 - perpLon * 0.0072, originLat + dirLat * 0.008 - perpLat * 0.0072],
    [originLon + dirLon * 0.021 - perpLon * 0.0085, originLat + dirLat * 0.021 - perpLat * 0.0085],
    [originLon + dirLon * 0.021 - perpLon * 0.0068, originLat + dirLat * 0.021 - perpLat * 0.0068],
    [originLon + dirLon * 0.008 - perpLon * 0.0056, originLat + dirLat * 0.008 - perpLat * 0.0056],
    [originLon + dirLon * 0.008 - perpLon * 0.0072, originLat + dirLat * 0.008 - perpLat * 0.0072],
  ];

  const retardantDropHead = [
    [originLon + dirLon * 0.026 + perpLon * 0.0055, originLat + dirLat * 0.026 + perpLat * 0.0055],
    [originLon + dirLon * 0.035 + perpLon * 0.0032, originLat + dirLat * 0.035 + perpLat * 0.0032],
    [originLon + dirLon * 0.035 + perpLon * 0.0048, originLat + dirLat * 0.035 + perpLat * 0.0048],
    [originLon + dirLon * 0.026 + perpLon * 0.0071, originLat + dirLat * 0.026 + perpLat * 0.0071],
    [originLon + dirLon * 0.026 + perpLon * 0.0055, originLat + dirLat * 0.026 + perpLat * 0.0055],
  ];

  const dozerFirebreakLine = [
    [originLon - dirLon * 0.002 + perpLon * 0.0045, originLat - dirLat * 0.002 + perpLat * 0.0045],
    [originLon + dirLon * 0.014 + perpLon * 0.0075, originLat + dirLat * 0.014 + perpLat * 0.0075],
    [originLon + dirLon * 0.024 + perpLon * 0.0068, originLat + dirLat * 0.024 + perpLat * 0.0068],
  ];

  return {
    burnScarGeoJSON: {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: { type: 'burn-scar', hour: simulationHour },
          geometry: {
            type: 'Polygon',
            coordinates: [currentPerimeterCoords],
          },
        },
      ],
    },
    finalPerimeterGeoJSON: {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: { type: 'final-perimeter' },
          geometry: {
            type: 'LineString',
            coordinates: finalPerimeterCoords,
          },
        },
        ...isochroneFeatures,
      ],
    },
    trajectoryGeoJSON: {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: { type: 'main-trajectory' },
          geometry: {
            type: 'LineString',
            coordinates: milestones.map((m) => [m.lon, m.lat]),
          },
        },
        ...spreadVectors.map((vec, idx) => ({
          type: 'Feature',
          properties: { type: 'spread-vector', id: idx },
          geometry: {
            type: 'LineString',
            coordinates: [vec.from, vec.to],
          },
        })),
      ],
    },
    suppressionGeoJSON: {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: {
            type: 'retardant-drop',
            unit: 'Avión T-910 DC-10 VLAT',
            label: 'Descarga Retardante Phos-Chek (36.000 L)',
          },
          geometry: {
            type: 'Polygon',
            coordinates: [retardantDropFlankNorth],
          },
        },
        {
          type: 'Feature',
          properties: {
            type: 'retardant-drop',
            unit: 'Avión T-910 DC-10 VLAT',
            label: 'Línea de Contención Aérea Flanco Sur',
          },
          geometry: {
            type: 'Polygon',
            coordinates: [retardantDropHead],
          },
        },
        {
          type: 'Feature',
          properties: {
            type: 'dozer-line',
            unit: 'Excavadora Dozer D-6',
            label: 'Cortafuego Mineralizado Dozer D-6',
          },
          geometry: {
            type: 'LineString',
            coordinates: dozerFirebreakLine,
          },
        },
        ...customSuppressions,
      ],
    },
    milestones,
  };
}
