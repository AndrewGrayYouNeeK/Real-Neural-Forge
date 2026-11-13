/** Visitor and question locations for live weather. */

export type Place = {
  name: string;
  latitude: number;
  longitude: number;
  timezone: string;
  source: "question" | "visitor" | "fallback";
  region?: string;
  country?: string;
};

export const CHICAGO: Place = {
  name: "Chicago",
  latitude: 41.85,
  longitude: -87.65,
  timezone: "America/Chicago",
  source: "fallback",
  region: "Illinois",
  country: "United States",
};

export type CfIncoming = {
  city?: string;
  latitude?: string | number;
  longitude?: string | number;
  timezone?: string;
  region?: string;
  country?: string;
};

export type VisitorGeo = {
  city?: string;
  latitude?: number;
  longitude?: number;
  timezone?: string;
  region?: string;
  country?: string;
};

export type DateWindow = {
  kind: "now" | "today" | "tomorrow" | "weekend" | "weekday" | "range";
  dates: string[];
  label: string;
};

const WEEKDAYS = [
  "sunday",
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
] as const;

const NOISE =
  /\b(what|whats|what's|will|the|a|an|weather|forecast|be|like|hows|how's|how|is|are|it|in|for|at|on|of|this|next|coming|today|tonight|tomorrow|weekend|week|morning|afternoon|evening|tonight|temperature|temp|high|low|rain|chance|conditions|please|tell|me|about|going|to|look|looking|give|show|there|right|now|currently|current)\b/gi;

const PLACE_PREFIX =
  /\b(?:in|for|at|near|around|outside)\s+([A-Za-z][A-Za-z .'-]+?(?:,\s*[A-Za-z]{2})?)(?=\s+(?:this|next|coming|today|tonight|tomorrow|right\s+now|currently|now|on|for the|this coming|over the)\b|[?!.]|$)/i;

export function visitorFromCf(cf: CfIncoming | undefined): VisitorGeo {
  if (!cf) return {};
  const latitude = Number(cf.latitude);
  const longitude = Number(cf.longitude);
  return {
    city: typeof cf.city === "string" && cf.city ? cf.city : undefined,
    latitude: Number.isFinite(latitude) ? latitude : undefined,
    longitude: Number.isFinite(longitude) ? longitude : undefined,
    timezone: typeof cf.timezone === "string" && cf.timezone ? cf.timezone : undefined,
    region: typeof cf.region === "string" && cf.region ? cf.region : undefined,
    country: typeof cf.country === "string" && cf.country ? cf.country : undefined,
  };
}

export function extractPlaceName(question: string): string | null {
  const text = question.trim();
  if (!text) return null;
  const prefixed = text.match(PLACE_PREFIX);
  if (prefixed) {
    const cleaned = cleanPlaceName(prefixed[1]);
    if (cleaned) return cleaned;
  }
  const leftover = text
    .replace(NOISE, " ")
    .replace(/[?!,]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const cleaned = cleanPlaceName(leftover);
  return cleaned;
}

function cleanPlaceName(raw: string): string | null {
  const name = raw
    .replace(/\b(the)\b/gi, " ")
    .replace(/[?!]/g, " ")
    .replace(/\s+,/g, ",")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[.,]+$/, "");
  if (!name) return null;
  if (name.length < 2 || name.length > 80) return null;
  if (!/[A-Za-z]/.test(name)) return null;
  if (/^\d+$/.test(name)) return null;
  return name;
}

export function ymdInZone(timeZone: string, at = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(at);
  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  return `${year}-${month}-${day}`;
}

export function weekdayInZone(timeZone: string, at = new Date()): number {
  const weekday = new Intl.DateTimeFormat("en-US", {
    timeZone,
    weekday: "short",
  }).format(at);
  const index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].indexOf(weekday);
  return index >= 0 ? index : at.getUTCDay();
}

export function addIsoDays(isoDate: string, days: number): string {
  const [year, month, day] = isoDate.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day + days));
  return date.toISOString().slice(0, 10);
}

/** True when the asker wants live conditions, not a forecast day. */
export function wantsCurrentConditions(question: string): boolean {
  const text = question.toLowerCase();
  if (/\b(right\s+now|currently)\b/.test(text)) return true;
  if (!/\bnow\b/.test(text)) return false;
  // Bare "now" still means current unless they also named a forecast day.
  return !/\b(tomorrow|tonight|weekend|today|this\s+(morning|afternoon|evening)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/.test(
    text,
  );
}

export function parseDateWindow(question: string, timeZone: string, at = new Date()): DateWindow {
  const text = question.toLowerCase();
  const today = ymdInZone(timeZone, at);
  const weekday = weekdayInZone(timeZone, at);
  if (wantsCurrentConditions(question)) {
    return { kind: "now", dates: [today], label: "right now" };
  }
  if (/\bthis weekend\b|\bthe weekend\b|\bweekend\b/.test(text)) {
    const saturdayOffset = weekday === 0 ? -1 : weekday === 6 ? 0 : 6 - weekday;
    const saturday = addIsoDays(today, saturdayOffset);
    const sunday = addIsoDays(saturday, 1);
    const dates = [saturday, sunday].filter((date) => date >= today);
    return { kind: "weekend", dates: dates.length ? dates : [saturday, sunday], label: "this weekend" };
  }
  if (/\btomorrow\b/.test(text)) {
    return { kind: "tomorrow", dates: [addIsoDays(today, 1)], label: "tomorrow" };
  }
  if (/\btoday\b|\btonight\b/.test(text)) {
    return { kind: "today", dates: [today], label: "today" };
  }
  for (let index = 0; index < WEEKDAYS.length; index += 1) {
    const name = WEEKDAYS[index];
    if (new RegExp(`\\b${name}\\b`).test(text)) {
      const offset = (index - weekday + 7) % 7;
      const date = addIsoDays(today, offset);
      const label = offset === 0 ? `today (${capitalize(name)})` : name;
      return { kind: "weekday", dates: [date], label };
    }
  }
  return { kind: "tomorrow", dates: [addIsoDays(today, 1)], label: "tomorrow" };
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function formatPlace(place: Place): string {
  if (place.region && place.country) return `${place.name}, ${place.region}`;
  if (place.country) return `${place.name}, ${place.country}`;
  return place.name;
}

export function visitorPlace(visitor: VisitorGeo): Place | null {
  if (
    visitor.latitude != null &&
    visitor.longitude != null &&
    Number.isFinite(visitor.latitude) &&
    Number.isFinite(visitor.longitude)
  ) {
    return {
      name: visitor.city || "your area",
      latitude: visitor.latitude,
      longitude: visitor.longitude,
      timezone: visitor.timezone || "auto",
      source: "visitor",
      region: visitor.region,
      country: visitor.country,
    };
  }
  return null;
}
