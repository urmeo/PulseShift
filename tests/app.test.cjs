const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadApp(responses) {
  const values = { temp: 85, humidity: 55, aqi: 60, wind: 6, precip: 0, hour: 17 };
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          value: values[id] ?? "", defaultValue: String(values[id] ?? ""),
          min: "", max: "", checked: false,
          get valueAsNumber() { return this.value === "" ? NaN : Number(this.value); },
          appendChild() {}, addEventListener() {}, setAttribute() {}, removeAttribute() {},
        });
      }
      return elements.get(id);
    },
    createElement() { return {}; },
    querySelectorAll() { return []; },
  };
  const context = vm.createContext({
    document, AbortController, setTimeout, clearTimeout, console: { warn() {} },
    window: { PULSESHIFT_MODEL: JSON.parse(fs.readFileSync(path.join(__dirname, "../model.json"))) },
    fetch: async () => responses.shift(),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../app.js"), "utf8"), context);
  return { context, elements };
}

function response(body, ok = true) {
  return { ok, status: ok ? 200 : 503, json: async () => body };
}

test("live weather sets the forecast's DC hour and weekday", async () => {
  const { context, elements } = loadApp([
    response({ properties: { forecastHourly: "https://api.weather.gov/forecast" } }),
    response({ properties: { periods: [{
      startTime: "2026-09-26T08:00:00-04:00", temperature: 72,
      relativeHumidity: { value: 60 }, windSpeed: "0 mph",
      probabilityOfPrecipitation: { value: 0 },
    }] } }),
    response({ hourly: { time: ["2026-09-26T08:00"], us_aqi: [40] } }),
  ]);

  await vm.runInContext("liveWeather()", context);

  assert.equal(Number(elements.get("hour").value), 8);
  assert.equal(elements.get("weekend").checked, true);
  assert.equal(Number(elements.get("wind").value), 0);
  assert.equal(Number(elements.get("aqi").value), 40);
  assert.equal(vm.runInContext("read().hour", context), 8);
});

test("HTTP failures are rejected even when the body is valid JSON", async () => {
  const { context } = loadApp([response({ properties: {} }, false)]);

  await assert.rejects(vm.runInContext('fetchJson("https://api.weather.gov/forecast")', context), /503/);
});

test("an incomplete live forecast leaves the manual conditions intact", async () => {
  const { context, elements } = loadApp([
    response({ properties: { forecastHourly: "https://api.weather.gov/forecast" } }),
    response({ properties: { periods: [{
      startTime: "2026-09-26T08:00:00-04:00", temperature: 100,
      relativeHumidity: { value: null }, windSpeed: "0 mph",
    }] } }),
  ]);

  await vm.runInContext("liveWeather()", context);

  assert.equal(Number(elements.get("temp").value), 85);
  assert.equal(Number(elements.get("hour").value), 17);
  assert.equal(elements.get("live").textContent, "Live weather unavailable — enter manually");
});

test("hours without temperature or humidity cannot be called safe", () => {
  const { context } = loadApp([]);
  const period = {
    startTime: "2026-09-26T08:00:00-04:00", temperature: 80,
    relativeHumidity: { value: 60 }, windSpeed: "0 mph",
  };

  for (const missing of [null, undefined, NaN]) {
    context.periods = [{ ...period, temperature: missing }];
    assert.equal(vm.runInContext("bestSafeHour(periods, null, 40)", context), null);
    context.periods = [{ ...period, relativeHumidity: { value: missing } }];
    assert.equal(vm.runInContext("bestSafeHour(periods, null, 40)", context), null);
  }
});

test("the hour search skips missing conditions and retains a valid safe hour", () => {
  const { context } = loadApp([]);
  context.periods = [
    { startTime: "2026-09-26T08:00:00-04:00", temperature: undefined },
    { startTime: "2026-09-26T09:00:00-04:00", temperature: 72,
      relativeHumidity: { value: 50 }, windSpeed: "0 mph" },
  ];

  assert.equal(vm.runInContext("bestSafeHour(periods, null, 40).hour", context), 9);
});
