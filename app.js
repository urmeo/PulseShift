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

function features(input) {
  const angle = (2 * Math.PI * input.hour) / 24;
  const hi = heatIndex(input.temp, input.humidity);
  return {
    heat_index_f: hi,
    cold_stress: Math.max(0, 55 - input.temp),
    heat_stress: Math.max(0, hi - 85),
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

function bestSafeHour(periods, aqi) {
  if (aqi >= M.safety.aqi_unsafe) return null;   // unsafe air all day
  let best = null;
  for (const p of periods.slice(0, 24)) {
    const { hour, weekend } = dcHourWeekend(p.startTime);
    if (hour < 6 || hour > 21) continue;
    const rh = p.relativeHumidity && p.relativeHumidity.value != null ? p.relativeHumidity.value : 50;
    const pop = p.probabilityOfPrecipitation && p.probabilityOfPrecipitation.value != null ? p.probabilityOfPrecipitation.value : 0;
    const hi = heatIndex(p.temperature, rh);
    if (hi >= M.safety.heat_unsafe_f) continue;
    const r = risk({ temp: p.temperature, humidity: rh, aqi, wind: parseInt(p.windSpeed, 10) || 0, precip: (pop / 100) * 0.1, hour, weekend, smoke: false });
    if (!best || r < best.risk) best = { hour, risk: r };
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
    btn.textContent = "Live DC weather loaded";
    update();
    const aqi = Number($("aqi").value);
    const best = bestSafeHour(hourly.properties.periods, aqi);
    $("besthour").textContent = best
      ? `Lowest-risk safe daytime hour ahead: ${String(best.hour).padStart(2, "0")}:00 (~${Math.round(best.risk * 100)}% risk, est.)`
      : aqi >= M.safety.aqi_unsafe
        ? "Air quality is unsafe all day — stay indoors."
        : "No safe daytime hour in the forecast window — consider indoors.";
  } catch {
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
  $("meta").textContent = `Trained on ${M.meta.trained_on} · 2024 AUROC ${M.meta.auroc_2024}`;
  update();
}

init();
