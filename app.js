const M = window.PULSESHIFT_MODEL;
const $ = (id) => document.getElementById(id);

function heatIndex(t, rh) {
  if (t < 80 || rh < 40) return t;
  const hi =
    -42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh -
    0.00683783 * t * t - 0.05481717 * rh * rh + 0.00122874 * t * t * rh +
    0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh;
  return Math.round(hi * 10) / 10;
}

const COLD_BASE = (M.stress && M.stress.cold_base_f) ?? 55;   // exported from config; fallback for older model.json
const HEAT_BASE = (M.stress && M.stress.heat_base_f) ?? 85;

function features(input) {
  const angle = (2 * Math.PI * input.hour) / 24;
  const hi = heatIndex(input.temp, input.humidity);
  return {
    heat_index_f: hi,
    cold_stress: Math.max(0, COLD_BASE - input.temp),
    heat_stress: Math.max(0, hi - HEAT_BASE),
    aqi: input.aqi,
    humidity: input.humidity,
    wind_mph: input.wind,
    precip_in: input.precip,
    visibility_mi: input.smoke ? 3 : 10,
    smoke_haze: input.smoke ? 1 : 0,
    hour_sin: Math.sin(angle),
    hour_cos: Math.cos(angle),
    is_weekend: input.weekend ? 1 : 0,
  };
}

function risk(input) {
  const f = features(input);
  let z = M.intercept;
  M.features.forEach((name, i) => {
    z += M.coef[i] * ((f[name] - M.mean[i]) / M.scale[i]);
  });
  return 1 / (1 + Math.exp(-z));
}

function riskBand(p) {
  if (p < 0.15) return { label: "Low", cls: "low" };
  if (p < 0.35) return { label: "Moderate", cls: "moderate" };
  if (p < 0.6) return { label: "High", cls: "high" };
  return { label: "Severe", cls: "severe" };
}

function recommend(p, unsafe) {
  if (unsafe) {
    return "Unsafe for outdoor activity — move indoors or reschedule.";
  }
  if (p >= 0.6) return "High suppression risk — consider a different time of day or rescheduling.";
  if (p >= 0.35) return "Elevated risk — shorten the session and watch conditions.";
  if (p >= 0.15) return "Moderate risk — plan for adverse conditions and keep it flexible.";
  return "Conditions look favorable — keep your plan.";
}

function num(id) {
  const e = $(id);
  let v = e.valueAsNumber;   // NaN when the field is empty or invalid
  if (Number.isNaN(v)) v = Number(e.defaultValue) || 0;
  const min = e.min !== "" ? Number(e.min) : -Infinity;
  const max = e.max !== "" ? Number(e.max) : Infinity;
  return Math.min(max, Math.max(min, v));
}

function read() {
  return {
    temp: num("temp"),
    humidity: num("humidity"),
    aqi: num("aqi"),
    wind: num("wind"),
    precip: num("precip"),
    hour: Number($("hour").value),
    weekend: $("weekend").checked,
    smoke: $("smoke").checked,
  };
}

function update() {
  const input = read();
  const hi = heatIndex(input.temp, input.humidity);
  const p = risk(input);
  const unsafe = hi >= M.safety.heat_unsafe_f || input.aqi >= M.safety.aqi_unsafe;
  const b = unsafe ? { label: "Unsafe conditions", cls: "severe" } : riskBand(p);

  $("result").hidden = false;
  $("risk").className = "risk " + b.cls;
  $("pct").textContent = unsafe ? "⚠" : Math.round(p * 100) + "%";
  $("band").textContent = unsafe ? "unsafe to exercise outdoors" : b.label + " suppression risk";
  $("reco").textContent = recommend(p, unsafe);
  $("detail").textContent = unsafe
    ? `Heat index ${Math.round(hi)}°F · AQI ${input.aqi} · model suppression ${Math.round(p * 100)}%`
    : `Heat index ${Math.round(hi)}°F · AQI ${input.aqi}`;
  $("besthour").textContent = "";
}

function dcHourWeekend(iso) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", hour: "numeric", hour12: false, weekday: "short",
  }).formatToParts(new Date(iso));
  let hour = Number(parts.find((x) => x.type === "hour").value);
  if (hour === 24) hour = 0;
  const wd = parts.find((x) => x.type === "weekday").value;
  return { hour, weekend: wd === "Sat" || wd === "Sun" };
}

function localHourKey(iso) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hour12: false,
  }).formatToParts(new Date(iso));
  const get = (t) => parts.find((x) => x.type === t).value;
  let h = get("hour");
  if (h === "24") h = "00";
  return `${get("year")}-${get("month")}-${get("day")}T${h}`;   // matches Open-Meteo local time keys
}

function bestSafeHour(periods, aqiByHour, fallbackAqi) {
  let best = null;
  for (const p of periods.slice(0, 24)) {
    const { hour, weekend } = dcHourWeekend(p.startTime);
    if (hour < 6 || hour > 21) continue;
    const aqi = aqiByHour && aqiByHour[localHourKey(p.startTime)] != null
      ? aqiByHour[localHourKey(p.startTime)]
      : fallbackAqi;
    if (aqi >= M.safety.aqi_unsafe) continue;   // skip hours whose air is unsafe
    const rh = p.relativeHumidity && p.relativeHumidity.value != null ? p.relativeHumidity.value : 50;
    const pop = p.probabilityOfPrecipitation && p.probabilityOfPrecipitation.value != null ? p.probabilityOfPrecipitation.value : 0;
    const hi = heatIndex(p.temperature, rh);
    if (hi >= M.safety.heat_unsafe_f) continue;
    const r = risk({ temp: p.temperature, humidity: rh, aqi, wind: parseInt(p.windSpeed, 10) || 0, precip: (pop / 100) * 0.1, hour, weekend, smoke: false });
    if (!best || r < best.risk) best = { hour, risk: r, aqi };
  }
  return best;
}

async function liveWeather() {
  const btn = $("live");
  btn.textContent = "Loading…";
  try {
    const point = await fetch("https://api.weather.gov/points/38.8951,-77.0364").then((r) => r.json());
    const hourly = await fetch(point.properties.forecastHourly).then((r) => r.json());
    const now = hourly.properties.periods[0];
    $("temp").value = now.temperature;
    if (now.relativeHumidity && now.relativeHumidity.value != null) $("humidity").value = now.relativeHumidity.value;
    $("wind").value = parseInt(now.windSpeed, 10) || $("wind").value;

    // per-hour air quality (Open-Meteo CAMS forecast, keyless), so the safest hour
    // reflects how AQI actually moves over the day rather than a single frozen value
    let aqiByHour = null;
    try {
      const aq = await fetch(
        "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=38.8951&longitude=-77.0364&hourly=us_aqi&forecast_days=2&timezone=America%2FNew_York"
      ).then((r) => r.json());
      if (aq && aq.hourly && Array.isArray(aq.hourly.time) && Array.isArray(aq.hourly.us_aqi)) {
        aqiByHour = {};
        aq.hourly.time.forEach((t, i) => { aqiByHour[t.slice(0, 13)] = aq.hourly.us_aqi[i]; });
        const nowAqi = aqiByHour[localHourKey(now.startTime)];
        if (nowAqi != null) $("aqi").value = Math.round(nowAqi);
      } else {
        console.warn("PulseShift: unexpected air-quality response; falling back to a constant AQI");
      }
    } catch (e) {
      aqiByHour = null;
      console.warn("PulseShift: hourly air-quality fetch failed; falling back to a constant AQI", e);
    }

    btn.textContent = "Live DC weather loaded";
    update();
    const fallbackAqi = Number($("aqi").value);
    const best = bestSafeHour(hourly.properties.periods, aqiByHour, fallbackAqi);
    const aqiNote = aqiByHour ? `, AQI ${Math.round(best ? best.aqi : fallbackAqi)}` : ", AQI held constant";
    $("besthour").textContent = best
      ? `Lowest-risk safe daytime hour ahead: ${String(best.hour).padStart(2, "0")}:00 (~${Math.round(best.risk * 100)}% risk${aqiNote}, est.)`
      : fallbackAqi >= M.safety.aqi_unsafe
        ? "Air quality is unsafe — stay indoors."
        : "No safe daytime hour in the forecast window — consider indoors.";
  } catch (e) {
    console.warn("PulseShift: live weather fetch failed", e);
    btn.textContent = "Live weather unavailable — enter manually";
  }
}

function init() {
  const sel = $("hour");
  for (let h = 0; h < 24; h++) {
    const o = document.createElement("option");
    o.value = h;
    o.textContent = String(h).padStart(2, "0") + ":00";
    sel.appendChild(o);
  }
  sel.value = 17;

  document.querySelectorAll("input, select").forEach((el) => el.addEventListener("input", update));
  $("live").addEventListener("click", liveWeather);
  $("meta").textContent = `Trained on ${M.meta.trained_on} · 2024 hold-out AUROC ${M.meta.auroc_2024}`;
  $("meta").title = M.meta.metrics_note;
  $("modelnote").textContent =
    "Forecast metrics are an out-of-time hold-out estimate; the served model is refit on all three years. Educational tool, not safety advice — obey official heat and air-quality advisories.";
  update();
}

init();
