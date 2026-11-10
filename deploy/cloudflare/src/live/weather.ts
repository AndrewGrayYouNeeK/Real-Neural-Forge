import {
  GEOCODE_TTL_SECONDS,
  WEATHER_TTL_SECONDS,
  cachedJson,
  fetchJson,
} from "./cache";
import {
  CHICAGO,
  type DateWindow,
  type Place,
  type VisitorGeo,
  extractPlaceName,
  formatPlace,
  parseDateWindow,
  visitorPlace,
} from "./geo";

type GeocodeResult = {
  results?: Array<{
    name?: string;
    latitude?: number;
    longitude?: number;
    timezone?: string;
    admin1?: string;
    country?: string;
  }>;
};

type ForecastResult = {
  timezone?: string;
  daily?: {
    time?: string[];
    temperature_2m_max?: number[];
    temperature_2m_min?: number[];
    precipitation_probability_max?: number[];
    weather_code?: number[];
  };
};

export type DayForecast = {
  date: string;
  highF: number;
  lowF: number;
  rainChance: number | null;
  condition: string;
};

const WMO: Record<number, string> = {
  0: "clear skies",
  1: "mainly clear skies",
  2: "partly cloudy skies",
  3: "overcast skies",
  45: "fog",
  48: "rime fog",
  51: "light drizzle",
  53: "drizzle",
  55: "heavy drizzle",
  61: "light rain",
  63: "rain",
  65: "heavy rain",
  71: "light snow",
  73: "snow",
  75: "heavy snow",
  80: "light rain showers",
  81: "rain showers",
  82: "heavy rain showers",
  85: "light snow showers",
  86: "snow showers",
  95: "thunderstorms",
  96: "thunderstorms with hail",
  99: "thunderstorms with heavy hail",
};

export function conditionFromCode(code: number | undefined): string {
  if (code == null || !Number.isFinite(code)) return "mixed conditions";
  return WMO[Math.trunc(code)] || "mixed conditions";
}

export async function geocodePlace(name: string): Promise<Place | null> {
  const query = name.trim();
  if (!query) return null;
  const url =
    "https://geocoding-api.open-meteo.com/v1/search?name=" +
    encodeURIComponent(query) +
    "&count=1&language=en&format=json";
  const payload = await cachedJson(`geocode:${query.toLowerCase()}`, GEOCODE_TTL_SECONDS, () =>
    fetchJson<GeocodeResult>(url),
  );
  const hit = payload.results?.[0];
  if (!hit || hit.latitude == null || hit.longitude == null) return null;
  return {
    name: hit.name || query,
    latitude: hit.latitude,
    longitude: hit.longitude,
    timezone: hit.timezone || "auto",
    source: "question",
    region: hit.admin1,
    country: hit.country,
  };
}

export async function resolvePlace(question: string, visitor: VisitorGeo): Promise<Place> {
  const asked = extractPlaceName(question);
  if (asked) {
    const found = await geocodePlace(asked);
    if (found) return found;
  }
  const fromVisitor = visitorPlace(visitor);
  if (fromVisitor) {
    if (!fromVisitor.timezone || fromVisitor.timezone === "auto") {
      // Keep lat/lon from Cloudflare; the forecast call uses timezone=auto.
      return fromVisitor;
    }
    return fromVisitor;
  }
  if (asked) {
    // Named place did not geocode and there is no visitor fix: Chicago.
    return { ...CHICAGO };
  }
  return { ...CHICAGO };
}

export async function fetchDailyForecast(place: Place): Promise<{ timezone: string; days: DayForecast[] }> {
  const url =
    "https://api.open-meteo.com/v1/forecast?" +
    new URLSearchParams({
      latitude: String(place.latitude),
      longitude: String(place.longitude),
      daily: [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
        "weather_code",
      ].join(","),
      temperature_unit: "fahrenheit",
      timezone: place.timezone && place.timezone !== "auto" ? place.timezone : "auto",
      forecast_days: "7",
    }).toString();
  const cacheKey = `weather:${place.latitude.toFixed(3)},${place.longitude.toFixed(3)}`;
  const payload = await cachedJson(cacheKey, WEATHER_TTL_SECONDS, () => fetchJson<ForecastResult>(url));
  const daily = payload.daily || {};
  const times = daily.time || [];
  const days: DayForecast[] = times.map((date, index) => ({
    date,
    highF: Number(daily.temperature_2m_max?.[index]),
    lowF: Number(daily.temperature_2m_min?.[index]),
    rainChance:
      daily.precipitation_probability_max?.[index] == null
        ? null
        : Number(daily.precipitation_probability_max[index]),
    condition: conditionFromCode(daily.weather_code?.[index]),
  }));
  return { timezone: payload.timezone || place.timezone, days };
}

export function formatTempF(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return `${Math.round(value)}°F`;
}

function formatDayLine(day: DayForecast, label: string, placeName: string, withPlace: boolean): string {
  const rain =
    day.rainChance == null
      ? ""
      : ` There's a ${Math.round(day.rainChance)}% chance of rain, with ${day.condition}.`;
  const where = withPlace ? ` in ${placeName}` : "";
  return `${capitalize(label)}${where}, the high is about ${formatTempF(day.highF)} and the low about ${formatTempF(day.lowF)}.${rain || ` Conditions look like ${day.condition}.`}`;
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function weatherReplyText(
  place: Place,
  window: DateWindow,
  days: DayForecast[],
): string {
  const placeName = formatPlace(place);
  const byDate = new Map(days.map((day) => [day.date, day]));
  const selected = window.dates
    .map((date) => byDate.get(date))
    .filter((day): day is DayForecast => Boolean(day));
  if (!selected.length) {
    return `I could not find a forecast for ${window.label} in ${placeName}. Open-Meteo's daily forecast may not reach that far yet.`;
  }
  const lines: string[] = [];
  if (place.source === "fallback") {
    lines.push(
      "I could not tell a location from your question or this connection, so I used Chicago as a default.",
    );
  }
  if (selected.length === 1) {
    const label = window.kind === "weekday" ? window.label : window.label;
    lines.push(formatDayLine(selected[0], label, placeName, true));
  } else {
    lines.push(`Here is the live forecast for ${window.label} in ${placeName}:`);
    for (let index = 0; index < selected.length; index += 1) {
      const day = selected[index];
      const label = weekdayName(day.date);
      lines.push(formatDayLine(day, label, placeName, false));
    }
  }
  lines.push("This is a live Open-Meteo forecast, not a trained model.");
  return lines.join(" ");
}

function weekdayName(isoDate: string): string {
  const [year, month, day] = isoDate.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day, 12));
  return new Intl.DateTimeFormat("en-US", { weekday: "long", timeZone: "UTC" }).format(date);
}

export async function answerWeather(question: string, visitor: VisitorGeo): Promise<{ text: string }> {
  try {
    const place = await resolvePlace(question, visitor);
    const window = parseDateWindow(question, place.timezone || "America/Chicago");
    const forecast = await fetchDailyForecast(place);
    return { text: weatherReplyText(place, window, forecast.days) };
  } catch {
    return {
      text: "I could not load a live weather forecast right now. Please try again in a moment.",
    };
  }
}
