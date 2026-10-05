const M = window.PULSESHIFT_MODEL;
const $ = (id) => document.getElementById(id);
const TZ = "America/New_York";
const COLD_BASE = M.stress.cold_base_f;
const HEAT_BASE = M.stress.heat_base_f;
let editRevision = 0;

function heatIndex(t, rh) {
  const simple = (0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094) + t) / 2;
  let hi = simple;
  if (simple >= 80) {
    hi = -42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh -
      0.00683783 * t * t - 0.05481717 * rh * rh + 0.00122874 * t * t * rh +
      0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh;
    if (rh < 13 && t >= 80 && t <= 112) {
      hi -= ((13 - rh) / 4) * Math.sqrt((17 - Math.abs(t - 95)) / 17);
    } else if (rh > 85 && t >= 80 && t <= 87) {
      hi += ((rh - 85) / 10) * ((87 - t) / 5);
    }
  }
  const scaled = hi * 10;
  const floor = Math.floor(scaled);
  const rounded = scaled - floor === 0.5 ? floor + (Math.abs(floor) % 2) : Math.round(scaled);
  return rounded / 10;
}

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
    visibility_mi: input.visibility,
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
    if (!Number.isFinite(f[name]) || !Number.isFinite(M.coef[i]) || !(M.scale[i] > 0)) {
      throw new Error("Model inputs or coefficients are invalid.");
    }
    z += M.coef[i] * ((f[name] - M.mean[i]) / M.scale[i]);
  });
  if (!Number.isFinite(z)) throw new Error("Model score is invalid.");
  return 1 / (1 + Math.exp(-z));
}

function percent(p) {
  if (p < 0.01) return "<1%";
  if (p > 0.99) return ">99%";
  return Math.round(p * 100) + "%";
}

function riskBand(p) {
  if (p < 0.15) return { label: "Low", cls: "low" };
  if (p < 0.35) return { label: "Moderate", cls: "moderate" };
  if (p < 0.6) return { label: "High", cls: "high" };
  return { label: "Very high", cls: "severe" };
}

function num(id) {
  const e = $(id);
  const v = e.valueAsNumber;
  const valid = Number.isFinite(v) && e.checkValidity();
  e.setAttribute("aria-invalid", String(!valid));
  if (!valid) throw new Error("Enter a valid value for " + e.closest("label").firstChild.textContent.trim() + ".");
  return v;
}

function read() {
  const hour = Number($("hour").value);
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) throw new Error("Select a start hour.");
  return {
    temp: num("temp"), humidity: num("humidity"), aqi: num("aqi"),
    wind: num("wind"), precip: num("precip"), visibility: num("visibility"),
    hour, weekend: $("weekend").checked, smoke: $("smoke").checked,
  };
}

function update() {
  $("besthour").textContent = "";
  try {
    const input = read();
    const hi = heatIndex(input.temp, input.humidity);
    const p = risk(input);
    const excluded = hi >= M.safety.heat_unsafe_f || input.aqi >= M.safety.aqi_unsafe;
    const b = excluded ? { label: "Study threshold exceeded", cls: "severe" } : riskBand(p);
    $("result").hidden = false;
    $("error").textContent = "";
    $("risk").className = "risk " + b.cls;
    $("pct").textContent = percent(p);
    $("band").textContent = b.label + (excluded ? "" : " demand suppression likelihood");
    $("reco").textContent = excluded
      ? `Study limits: heat index below ${M.safety.heat_unsafe_f}°F and AQI below ${M.safety.aqi_unsafe}. Follow official advisories.`
      : "Estimated chance of network rides falling below half their seasonal expectation.";
    $("detail").textContent = `Heat index ${hi}°F · AQI ${Math.round(input.aqi * 10) / 10}`;
  } catch (e) {
    $("result").hidden = true;
    $("error").textContent = e.message;
  }
}

function dcHourWeekend(epoch) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: TZ, hour: "numeric", hourCycle: "h23", weekday: "short",
  }).formatToParts(new Date(epoch * 1000));
  const hour = Number(parts.find((x) => x.type === "hour").value);
  const wd = parts.find((x) => x.type === "weekday").value;
  return { hour, weekend: wd === "Sat" || wd === "Sun" };
}

function forecastInputs(weather, air, smoke, now = Date.now()) {
  const units = weather.hourly_units;
  const h = weather.hourly;
  const a = air.hourly;
  if (!units || units.time !== "unixtime" || units.temperature_2m !== "°F" ||
      units.relative_humidity_2m !== "%" || !["mp/h", "mph"].includes(units.wind_speed_10m) ||
      units.precipitation !== "inch" || !["m", "ft"].includes(units.visibility) ||
      !air.hourly_units || air.hourly_units.time !== "unixtime" || air.hourly_units.us_aqi !== "USAQI" || !h || !a) {
    throw new Error("Unexpected forecast units or fields.");
  }
  const fields = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "visibility"];
  if (!Array.isArray(h.time) || !fields.every((f) => Array.isArray(h[f]) && h[f].length === h.time.length) ||
      !Array.isArray(a.time) || !Array.isArray(a.us_aqi) || a.time.length !== a.us_aqi.length) {
    throw new Error("Incomplete forecast response.");
  }
  const aqiByTime = new Map(a.time.map((t, i) => [t, a.us_aqi[i]]));
  const start = Math.floor(now / 3600000) * 3600;
  const rows = [];
  h.time.forEach((epoch, i) => {
    if (!Number.isFinite(epoch) || epoch < start || epoch >= start + 86400) return;
    const input = {
      epoch, temp: h.temperature_2m[i], humidity: h.relative_humidity_2m[i],
      wind: h.wind_speed_10m[i], precip: h.precipitation[i], aqi: aqiByTime.get(epoch),
      visibility: Math.min(10, h.visibility[i] / (units.visibility === "ft" ? 5280 : 1609.344)),
      ...dcHourWeekend(epoch), smoke,
    };
    if (!fields.every((f) => Number.isFinite(h[f][i])) || !Number.isFinite(input.aqi) ||
        input.temp < -40 || input.temp > 130 || input.humidity < 0 || input.humidity > 100 ||
        input.aqi < 0 || input.aqi > 500 || input.wind < 0 || input.wind > 100 ||
        input.precip < 0 || input.precip > 5 || input.visibility < 0) return;
    rows.push(input);
  });
  return rows.sort((x, y) => x.epoch - y.epoch);
}

function bestForecastHour(rows) {
  let best = null;
  for (const input of rows) {
    if (input.hour < 6 || input.hour > 21 || input.aqi >= M.safety.aqi_unsafe ||
        heatIndex(input.temp, input.humidity) >= M.safety.heat_unsafe_f) continue;
    const p = risk(input);
    if (!best || p < best.risk) best = { ...input, risk: p };
  }
  return best;
}

async function fetchJson(url, ms = 8000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const response = await fetch(url, { signal: ctrl.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function liveWeather() {
  const btn = $("live");
  if (btn.disabled) return;
  const revision = editRevision;
  const now = Date.now();
  btn.disabled = true;
  btn.textContent = "Loading…";
  btn.setAttribute("aria-busy", "true");
  $("error").textContent = "";
  $("besthour").textContent = "";
  $("forecastnote").textContent = "";
  try {
    const query = "latitude=38.8951&longitude=-77.0364&forecast_hours=25&timeformat=unixtime&timezone=America%2FNew_York";
    const [weather, air] = await Promise.all([
      fetchJson("https://api.open-meteo.com/v1/forecast?" + query +
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,visibility&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"),
      fetchJson("https://air-quality-api.open-meteo.com/v1/air-quality?" + query + "&hourly=us_aqi"),
    ]);
    if (revision !== editRevision) {
      $("error").textContent = "Inputs changed during loading. Load the forecast again to replace them.";
      return;
    }
    const rows = forecastInputs(weather, air, $("smoke").checked, now);
    const first = rows.find((r) => r.epoch === Math.floor(now / 3600000) * 3600);
    if (!first) throw new Error("Current-hour weather or AQI is missing.");
    ["temp", "humidity", "aqi", "wind", "precip", "visibility", "hour"].forEach((id) => { $(id).value = first[id]; });
    $("weekend").checked = first.weekend;
    update();
    const best = bestForecastHour(rows);
    const stamp = best && new Intl.DateTimeFormat("en-US", {
      timeZone: TZ, weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZoneName: "short",
    }).format(new Date(best.epoch * 1000));
    $("besthour").textContent = best
      ? `Lowest estimated suppression within study limits: ${stamp} (${percent(best.risk)}, AQI ${Math.round(best.aqi)}).`
      : "No complete daytime forecast hour meets the study's heat and AQI limits.";
    $("forecastnote").textContent = "Open-Meteo hourly weather + CAMS AQI forecast; visibility capped at the training sensor's 10 mi limit. Smoke/haze uses your checkbox setting. Missing hours are excluded.";
    btn.textContent = "DC forecast loaded";
  } catch (e) {
    $("besthour").textContent = "";
    $("forecastnote").textContent = "";
    $("error").textContent = "Forecast unavailable: " + e.message + " Enter conditions manually.";
  } finally {
    if (btn.textContent === "Loading…") btn.textContent = "Load DC forecast";
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
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
  document.querySelectorAll("input, select").forEach((el) => el.addEventListener("input", () => {
    editRevision++;
    if (!$("live").disabled) $("live").textContent = "Load DC forecast";
    $("forecastnote").textContent = "";
    update();
  }));
  $("form").addEventListener("submit", (event) => { event.preventDefault(); update(); });
  $("live").addEventListener("click", liveWeather);
  $("meta").textContent = `DC 2022–2024 · 2024 hold-out AUROC ${M.meta.auroc_2024}`;
  $("meta").title = M.meta.metrics_note;
  $("modelnote").textContent = "DC-area bike-share network demand, not personal exercise risk. Retrospective hold-out metrics; served coefficients use all three years. Forecast inputs have not been prospectively validated.";
  update();
}

init();
