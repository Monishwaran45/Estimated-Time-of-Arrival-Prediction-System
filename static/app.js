// ==========================================================================
// ETA Predictor Platform - Client Application Engine
// ==========================================================================

// Base API URL Resolver (supports local full-stack server, relative paths, or external Render backend on Vercel)
window.RENDER_BACKEND_URL = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? ""
  : "https://estimated-time-of-arrival-prediction.onrender.com";

function getApiUrl(path) {
  const customBase = localStorage.getItem("eta_api_base") || window.RENDER_BACKEND_URL || "";
  return customBase ? `${customBase.replace(/\/+$/, "")}${path}` : path;
}

function configureBackendUrl() {
  const current = localStorage.getItem("eta_api_base") || window.RENDER_BACKEND_URL || "(Same Origin / Localhost)";
  const input = prompt(
    "Enter your Render Backend URL (e.g., https://eta-prediction-backend.onrender.com) or leave blank for local default:",
    current.startsWith("http") ? current : ""
  );
  if (input !== null) {
    if (input.trim() === "") {
      localStorage.removeItem("eta_api_base");
      alert("Reset to default (Local / Same-Origin) API endpoint.");
    } else {
      localStorage.setItem("eta_api_base", input.trim());
      alert(`Backend API URL configured: ${input.trim()}`);
    }
    checkDatabaseStatus();
    loadPresets();
    calculateETA();
  }
}

let map;
let restaurantMarker, customerMarker, routePolyline, courierAnimMarker;
let debounceTimer = null;
let currentETA = 0;
let lastUploadedFile = null;
let isSimulating = false;

// Urban Map Center (Pickup & Dropoff)
const DEFAULT_CENTER = [40.730610, -73.935242];
let restaurantPos = [40.730610, -73.935242];
let customerPos = [40.758896, -73.985130];

// Presets Data
let presetsData = {};

document.addEventListener("DOMContentLoaded", () => {
  initLiveClock();
  checkDatabaseStatus();
  initTabs();
  initMap();
  loadPresets();
  calculateETA();
  initDragDrop();
});

// ---------------------------------------------------------
// Live Database & System Health Status
// ---------------------------------------------------------
async function checkDatabaseStatus() {
  const badgeText = document.getElementById("db-status-text");
  const pulseDot = document.getElementById("db-pulse-dot");
  const mlBadge = document.getElementById("ml-engine-badge");

  try {
    const res = await fetch(getApiUrl("/api/db/status"));
    if (res.ok) {
      const data = await res.json();
      if (data.status === "connected") {
        if (badgeText) badgeText.innerHTML = `Backend: <strong style="color: var(--uber-green);">${data.database_type}</strong>`;
        if (pulseDot) pulseDot.style.background = "var(--uber-green)";
      } else {
        if (badgeText) badgeText.innerHTML = `Backend: <strong style="color: var(--uber-yellow);">${data.database_type || "Ready"}</strong>`;
        if (pulseDot) pulseDot.style.background = "var(--uber-yellow)";
      }
    } else {
      if (badgeText) badgeText.innerHTML = `Backend: <strong style="color: var(--uber-red);">Error (${res.status})</strong>`;
      if (pulseDot) pulseDot.style.background = "var(--uber-red)";
    }
    
    // Fetch dynamic model metrics
    const healthRes = await fetch(getApiUrl("/api/health"));
    if (healthRes.ok) {
      const health = await healthRes.json();
      if (mlBadge) {
        mlBadge.innerHTML = `ML: <strong>Active (MAE ±${health.dynamic_mae_min || '2.72'}m)</strong>`;
      }
    }
  } catch (err) {
    if (badgeText) badgeText.innerHTML = `Backend: <strong style="color: var(--uber-red);">Offline / Click to config</strong>`;
    if (pulseDot) pulseDot.style.background = "var(--uber-red)";
  }
}

// ---------------------------------------------------------
// Live Clock
// ---------------------------------------------------------
function initLiveClock() {
  function updateClock() {
    const clockElem = document.getElementById("live-clock");
    if (clockElem) {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      clockElem.innerHTML = `<i class="fa-regular fa-clock" style="color: var(--uber-gray-400);"></i> ${timeStr}`;
    }
  }
  updateClock();
  setInterval(updateClock, 1000);
}

// ---------------------------------------------------------
// Tab Switching
// ---------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".uber-nav-link");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(pane => pane.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(`tab-${tab.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (tab.dataset.tab === "simulator" && map) {
        setTimeout(() => map.invalidateSize(), 150);
      }
      if (tab.dataset.tab === "history") {
        refreshHistory();
      }
    });
  });
}

// Helper for glowing pulse map markers
function createPickupPinIcon(emoji = "🍕") {
  return L.divIcon({
    className: "uber-map-pin",
    html: `<div class="custom-pulse-pin"><div class="pin-halo"></div><div class="pin-body">${emoji}</div></div>`,
    iconSize: [38, 38],
    iconAnchor: [19, 19]
  });
}

function createDropoffPinIcon(emoji = "📍") {
  return L.divIcon({
    className: "uber-map-pin",
    html: `<div class="custom-pulse-pin"><div class="pin-halo dropoff-halo"></div><div class="pin-body dropoff-body">${emoji}</div></div>`,
    iconSize: [38, 38],
    iconAnchor: [19, 19]
  });
}

// ---------------------------------------------------------
// Map Initialization
// ---------------------------------------------------------
function initMap() {
  const mapElement = document.getElementById("map");
  if (!mapElement) return;

  map = L.map("map", {
    zoomControl: false
  }).setView(DEFAULT_CENTER, 13);

  L.control.zoom({ position: 'topright' }).addTo(map);

  // Modern Dark Matter CartoDB tiles (clean, free, zero watermark)
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(map);

  // High-visibility glowing map pins
  const pickupIcon = createPickupPinIcon(currentServiceMode === "ride" ? "🚕" : "🍕");
  const dropoffIcon = createDropoffPinIcon("📍");

  restaurantMarker = L.marker(restaurantPos, { icon: pickupIcon, draggable: true }).addTo(map);
  customerMarker = L.marker(customerPos, { icon: dropoffIcon, draggable: true }).addTo(map);

  restaurantMarker.bindPopup("<b>Central Dispatch Hub</b><br>Pickup Location (Drag to change)");
  customerMarker.bindPopup("<b>Customer Dropoff</b><br>Destination (Drag to change)");

  // High-visibility glowing route line
  routePolyline = L.polyline([restaurantPos, customerPos], {
    color: '#00E676',
    weight: 5,
    opacity: 0.95,
    lineCap: 'round',
    lineJoin: 'round',
    dashArray: '8, 8',
    className: 'glowing-route-line'
  }).addTo(map);

  restaurantMarker.on('drag', updateRouteFromMap);
  customerMarker.on('drag', updateRouteFromMap);

  updateRouteFromMap();
}

function calculateDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function updateRouteFromMap() {
  const p1 = restaurantMarker.getLatLng();
  const p2 = customerMarker.getLatLng();

  routePolyline.setLatLngs([p1, p2]);

  const distKm = Math.max(0.5, Math.min(35.0, calculateDistanceKm(p1.lat, p1.lng, p2.lat, p2.lng)));
  const formattedDist = distKm.toFixed(1);

  document.getElementById("input-distance").value = formattedDist;
  document.getElementById("val-distance").innerText = `${formattedDist} km`;
  document.getElementById("map-dist-badge").innerText = `${formattedDist} km`;

  debouncedPredict();
}

function onDistanceSliderChange(value) {
  const dist = parseFloat(value);
  document.getElementById("val-distance").innerText = `${dist.toFixed(1)} km`;
  document.getElementById("map-dist-badge").innerText = `${dist.toFixed(1)} km`;

  if (restaurantMarker && customerMarker && map) {
    const p1 = restaurantMarker.getLatLng();
    const angle = 45 * Math.PI / 180;
    const earthRadius = 6371;
    const latOffset = (dist * Math.cos(angle) / earthRadius) * (180 / Math.PI);
    const lngOffset = (dist * Math.sin(angle) / (earthRadius * Math.cos(p1.lat * Math.PI / 180))) * (180 / Math.PI);
    
    const newPos = [p1.lat + latOffset, p1.lng + lngOffset];
    customerMarker.setLatLng(newPos);
    routePolyline.setLatLngs([p1, newPos]);
    map.fitBounds(routePolyline.getBounds(), { padding: [40, 40] });
  }

  debouncedPredict();
}

// ---------------------------------------------------------
// Live Uber Transit Simulation
// ---------------------------------------------------------
function simulateTrip() {
  if (isSimulating) return;
  isSimulating = true;

  const btn = document.getElementById("btn-simulate");
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Driver En Route...`;

  const p1 = restaurantMarker.getLatLng();
  const p2 = customerMarker.getLatLng();

  const vehicleType = document.querySelector('input[name="vehicle"]:checked')?.value || "Bike";
  const vehicleEmoji = vehicleType === "Car" ? "🚗" : (vehicleType === "Scooter" ? "🛵" : "🚲");

  const courierIcon = L.divIcon({
    className: "uber-driver-pin",
    html: `<div style="background: #000; width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(0,0,0,0.8); border: 2px solid #fff; font-size: 18px;">${vehicleEmoji}</div>`,
    iconSize: [38, 38],
    iconAnchor: [19, 19]
  });

  if (courierAnimMarker) {
    map.removeLayer(courierAnimMarker);
  }

  courierAnimMarker = L.marker([p1.lat, p1.lng], { icon: courierIcon, zIndexOffset: 2000 }).addTo(map);

  const totalSteps = 60;
  const durationMs = 4000;
  const intervalMs = durationMs / totalSteps;
  let currentStep = 0;

  const animInterval = setInterval(() => {
    currentStep++;
    const progress = currentStep / totalSteps;

    const lat = p1.lat + (p2.lat - p1.lat) * progress;
    const lng = p1.lng + (p2.lng - p1.lng) * progress;
    courierAnimMarker.setLatLng([lat, lng]);

    if (currentStep >= totalSteps) {
      clearInterval(animInterval);
      setTimeout(() => {
        if (courierAnimMarker) map.removeLayer(courierAnimMarker);
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Simulate Transit`;
        isSimulating = false;
      }, 1000);
    }
  }, intervalMs);
}

// ---------------------------------------------------------
// Service Mode Switching (Food Delivery vs Cab Ride)
// ---------------------------------------------------------
let currentServiceMode = "food";

const FOOD_PRESETS = [
  { id: "quick-coffee", label: "☕ Morning Coffee Run (2.5 km)", data: { Distance_km: 2.5, Weather: "Clear", Traffic_Level: "Low", Time_of_Day: "Morning", Vehicle_Type: "Scooter", Preparation_Time_min: 7.0, Courier_Experience_yrs: 4.5 } },
  { id: "rainy-dinner-rush", label: "🌧️ Monsoon Rush Hour (9.8 km)", data: { Distance_km: 9.8, Weather: "Rainy", Traffic_Level: "High", Time_of_Day: "Evening", Vehicle_Type: "Bike", Preparation_Time_min: 25.0, Courier_Experience_yrs: 1.5 } },
  { id: "suburban-night-drive", label: "🚗 Midnight Long-Range (18.2 km)", data: { Distance_km: 18.2, Weather: "Clear", Traffic_Level: "Low", Time_of_Day: "Night", Vehicle_Type: "Car", Preparation_Time_min: 14.0, Courier_Experience_yrs: 8.0 } },
  { id: "snowy-lunch-bottleneck", label: "❄️ Winter Storm Bottleneck (6.4 km)", data: { Distance_km: 6.4, Weather: "Snowy", Traffic_Level: "Medium", Time_of_Day: "Afternoon", Vehicle_Type: "Car", Preparation_Time_min: 32.0, Courier_Experience_yrs: 2.0 } }
];

const RIDE_PRESETS = [
  { id: "airport-express", label: "✈️ Airport Express (22.5 km)", data: { Distance_km: 22.5, Weather: "Clear", Traffic_Level: "Low", Time_of_Day: "Night", Vehicle_Type: "Car", Preparation_Time_min: 1.0, Courier_Experience_yrs: 7.0 } },
  { id: "downtown-rush", label: "🏢 Downtown Peak Rush (6.5 km)", data: { Distance_km: 6.5, Weather: "Clear", Traffic_Level: "High", Time_of_Day: "Evening", Vehicle_Type: "Car", Preparation_Time_min: 1.0, Courier_Experience_yrs: 3.5 } },
  { id: "rainy-commute", label: "🌧️ Rainy City Commute (11.0 km)", data: { Distance_km: 11.0, Weather: "Rainy", Traffic_Level: "High", Time_of_Day: "Morning", Vehicle_Type: "Car", Preparation_Time_min: 1.0, Courier_Experience_yrs: 4.0 } },
  { id: "moto-express", label: "🛵 Moto Taxi Quick Ride (4.0 km)", data: { Distance_km: 4.0, Weather: "Clear", Traffic_Level: "Medium", Time_of_Day: "Afternoon", Vehicle_Type: "Bike", Preparation_Time_min: 1.0, Courier_Experience_yrs: 5.0 } }
];

function setServiceMode(mode) {
  currentServiceMode = mode;

  const btnFood = document.getElementById("btn-mode-food");
  const btnRide = document.getElementById("btn-mode-ride");
  const cardTitle = document.getElementById("form-card-title");
  const cardSubtitle = document.getElementById("form-card-subtitle");
  const prepGroup = document.getElementById("group-prep-time");
  const factorPrepLabel = document.getElementById("factor-prep-label");
  const vehicleLabel = document.getElementById("vehicle-section-label");

  // Vehicle labels & emojis
  const vBikeName = document.getElementById("v-bike-name");
  const vBikeDesc = document.getElementById("v-bike-desc");
  const vBikeEmoji = document.getElementById("v-bike-emoji");

  const vScooterName = document.getElementById("v-scooter-name");
  const vScooterDesc = document.getElementById("v-scooter-desc");
  const vScooterEmoji = document.getElementById("v-scooter-emoji");

  const vCarName = document.getElementById("v-car-name");
  const vCarDesc = document.getElementById("v-car-desc");
  const vCarEmoji = document.getElementById("v-car-emoji");

  if (mode === "ride") {
    btnFood?.classList.remove("active");
    btnRide?.classList.add("active");

    if (cardTitle) cardTitle.innerHTML = "🚕 Cab Ride Request";
    if (cardSubtitle) cardSubtitle.innerText = "Real-time passenger transit ETA, traffic friction, and arrival confidence";
    if (prepGroup) prepGroup.style.display = "none";
    if (factorPrepLabel) factorPrepLabel.innerText = "🚶 Passenger Boarding";
    if (vehicleLabel) vehicleLabel.innerText = "Select Cab / Ride Fleet";

    if (vBikeName) vBikeName.innerText = "Moto Taxi";
    if (vBikeDesc) vBikeDesc.innerText = "Solo rapid trip";
    if (vBikeEmoji) vBikeEmoji.innerText = "🛵";

    if (vScooterName) vScooterName.innerText = "Comfort SUV";
    if (vScooterDesc) vScooterDesc.innerText = "Extra legroom";
    if (vScooterEmoji) vScooterEmoji.innerText = "🚙";

    if (vCarName) vCarName.innerText = "City Sedan";
    if (vCarDesc) vCarDesc.innerText = "Standard cab";
    if (vCarEmoji) vCarEmoji.innerText = "🚕";

    // Select car by default
    const carRadio = document.getElementById("v-car");
    if (carRadio) carRadio.checked = true;

    // Update map pin icon to Taxi
    if (restaurantMarker) {
      restaurantMarker.setIcon(createPickupPinIcon("🚕"));
    }

    renderPresetChips(RIDE_PRESETS);
  } else {
    btnRide?.classList.remove("active");
    btnFood?.classList.add("active");

    if (cardTitle) cardTitle.innerHTML = "Delivery Request";
    if (cardSubtitle) cardSubtitle.innerText = "Adjust trip dynamics for instantaneous machine learning prediction";
    if (prepGroup) prepGroup.style.display = "block";
    if (factorPrepLabel) factorPrepLabel.innerText = "🍳 Kitchen Preparation";
    if (vehicleLabel) vehicleLabel.innerText = "Select Fleet Vehicle";

    if (vBikeName) vBikeName.innerText = "Courier Bike";
    if (vBikeDesc) vBikeDesc.innerText = "Fast courier";
    if (vBikeEmoji) vBikeEmoji.innerText = "🚲";

    if (vScooterName) vScooterName.innerText = "Scooter";
    if (vScooterDesc) vScooterDesc.innerText = "Standard fleet";
    if (vScooterEmoji) vScooterEmoji.innerText = "🛵";

    if (vCarName) vCarName.innerText = "Delivery Car";
    if (vCarDesc) vCarDesc.innerText = "High volume";
    if (vCarEmoji) vCarEmoji.innerText = "🚗";

    // Update map pin icon to Pizza
    if (restaurantMarker) {
      restaurantMarker.setIcon(createPickupPinIcon("🍕"));
    }

    renderPresetChips(FOOD_PRESETS);
  }

  calculateETA();
}

function renderPresetChips(presetList) {
  const container = document.getElementById("presets-chip-group");
  if (!container) return;

  container.innerHTML = "";
  presetList.forEach(p => {
    presetsData[p.id] = p.data;
    const btn = document.createElement("button");
    btn.className = "uber-preset-chip";
    btn.innerText = p.label;
    btn.onclick = () => applyPreset(p.id);
    container.appendChild(btn);
  });
}

// ---------------------------------------------------------
// Preset Scenarios
// ---------------------------------------------------------
async function loadPresets() {
  renderPresetChips(FOOD_PRESETS);
  try {
    const res = await fetch(getApiUrl("/api/presets"));
    if (res.ok) {
      const presets = await res.json();
      presets.forEach(p => presetsData[p.id] = p.data);
    }
  } catch (err) {
    console.warn("Using local presets", err);
  }
}

function applyPreset(presetId) {
  const data = presetsData[presetId];
  if (!data) return;

  document.getElementById("input-distance").value = data.Distance_km;
  onDistanceSliderChange(data.Distance_km);

  const prepVal = currentServiceMode === "ride" ? 1.0 : (data.Preparation_Time_min || 15.0);
  document.getElementById("input-prep").value = prepVal;
  document.getElementById("val-prep").innerText = `${prepVal} min`;

  document.getElementById("input-exp").value = data.Courier_Experience_yrs;
  document.getElementById("val-exp").innerText = `${data.Courier_Experience_yrs} yrs`;

  const weatherRadio = document.querySelector(`input[name="weather"][value="${data.Weather}"]`);
  if (weatherRadio) weatherRadio.checked = true;

  const trafficRadio = document.querySelector(`input[name="traffic"][value="${data.Traffic_Level}"]`);
  if (trafficRadio) trafficRadio.checked = true;

  const vehicleRadio = document.querySelector(`input[name="vehicle"][value="${data.Vehicle_Type}"]`);
  if (vehicleRadio) vehicleRadio.checked = true;

  const todRadio = document.querySelector(`input[name="timeofday"][value="${data.Time_of_Day}"]`);
  if (todRadio) todRadio.checked = true;

  calculateETA();
}

// ---------------------------------------------------------
// Prediction Execution & Rendering
// ---------------------------------------------------------
function debouncedPredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    calculateETA();
  }, 200);
}

function getFormData() {
  const prepTime = currentServiceMode === "ride" ? 1.0 : parseFloat(document.getElementById("input-prep").value || 1.0);
  return {
    Distance_km: parseFloat(document.getElementById("input-distance").value),
    Preparation_Time_min: prepTime,
    Courier_Experience_yrs: parseFloat(document.getElementById("input-exp").value),
    Weather: document.querySelector('input[name="weather"]:checked')?.value || "Clear",
    Traffic_Level: document.querySelector('input[name="traffic"]:checked')?.value || "Medium",
    Vehicle_Type: document.querySelector('input[name="vehicle"]:checked')?.value || "Bike",
    Time_of_Day: document.querySelector('input[name="timeofday"]:checked')?.value || "Evening"
  };
}

async function calculateETA() {
  const payload = getFormData();

  try {
    const res = await fetch(getApiUrl("/api/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      renderPrediction(data);
    }
  } catch (err) {
    console.error("Prediction failed:", err);
  }
}

function animateNumber(element, start, end, duration = 300) {
  const startTime = performance.now();
  function update(time) {
    const elapsed = time - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const val = (start + (end - start) * progress).toFixed(1);
    element.innerText = val;
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.innerText = end.toFixed(1);
    }
  }
  requestAnimationFrame(update);
}

function renderPrediction(data) {
  const numElem = document.getElementById("eta-display-num");
  const targetEta = data.predicted_time_min;
  
  animateNumber(numElem, currentETA || targetEta, targetEta);
  currentETA = targetEta;

  document.getElementById("sla-range").innerText = `${data.lower_sla_min} to ${data.upper_sla_min} min`;
  checkDatabaseStatus();

  // Risk Badge
  const badge = document.getElementById("risk-badge-display");
  const bar = document.getElementById("risk-bar-fill");

  badge.innerText = data.risk_level;
  
  if (data.risk_level === "LOW RISK") {
    badge.style.color = "var(--uber-green)";
    badge.style.background = "var(--uber-green-bg)";
    bar.style.backgroundColor = "var(--uber-green)";
  } else if (data.risk_level === "MEDIUM RISK") {
    badge.style.color = "var(--uber-yellow)";
    badge.style.background = "var(--uber-yellow-bg)";
    bar.style.backgroundColor = "var(--uber-yellow)";
  } else {
    badge.style.color = "var(--uber-red)";
    badge.style.background = "var(--uber-red-bg)";
    bar.style.backgroundColor = "var(--uber-red)";
  }

  bar.style.width = `${Math.min(100, Math.max(15, data.risk_score))}%`;

  // Factor Breakdown Bars
  const breakdown = data.breakdown || {};
  const maxFactor = Math.max(
    breakdown.kitchen_prep_min || 15,
    breakdown.base_transit_min || 15,
    breakdown.traffic_delay_min || 15,
    breakdown.weather_delay_min || 10
  );

  document.getElementById("factor-prep").innerText = `${breakdown.kitchen_prep_min} min`;
  document.getElementById("factor-bar-prep").style.width = `${(breakdown.kitchen_prep_min / maxFactor) * 100}%`;

  document.getElementById("factor-transit").innerText = `${breakdown.base_transit_min} min`;
  document.getElementById("factor-bar-transit").style.width = `${(breakdown.base_transit_min / maxFactor) * 100}%`;

  document.getElementById("factor-traffic").innerText = `+${breakdown.traffic_delay_min} min`;
  document.getElementById("factor-bar-traffic").style.width = `${(breakdown.traffic_delay_min / maxFactor) * 100}%`;

  document.getElementById("factor-weather").innerText = `+${breakdown.weather_delay_min} min`;
  document.getElementById("factor-bar-weather").style.width = `${(breakdown.weather_delay_min / maxFactor) * 100}%`;

  document.getElementById("factor-courier").innerText = `-${breakdown.courier_tenure_bonus_min} min`;
  document.getElementById("factor-bar-courier").style.width = `${(breakdown.courier_tenure_bonus_min / 8.0) * 100}%`;
}

// ---------------------------------------------------------
// Batch Processing
// ---------------------------------------------------------
function initDragDrop() {
  const dropZone = document.getElementById("drop-zone");
  if (!dropZone) return;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.style.borderColor = "var(--uber-white)", false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.style.borderColor = "var(--uber-gray-700)", false);
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) processCSVFile(files[0]);
  });
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) processCSVFile(file);
}

async function processCSVFile(file) {
  lastUploadedFile = file;
  const formData = new FormData();
  formData.append("file", file);

  const dropZone = document.getElementById("drop-zone");
  dropZone.innerHTML = `<div style="font-size: 2rem; color: var(--uber-white); margin-bottom: 0.5rem;"><i class="fa-solid fa-spinner fa-spin"></i></div><h4 style="font-size: 1rem; font-weight: 700;">Evaluating ${file.name} with XGBoost...</h4>`;

  try {
    const res = await fetch(getApiUrl("/api/batch-predict"), {
      method: "POST",
      body: formData
    });

    if (res.ok) {
      const data = await res.json();
      renderBatchResults(data);
    } else {
      const err = await res.json();
      alert(`Batch Error: ${err.detail || "Failed to process manifest"}`);
      resetUploadZone();
    }
  } catch (err) {
    alert(`File upload error: ${err.message}`);
    resetUploadZone();
  }
}

function resetUploadZone() {
  const dropZone = document.getElementById("drop-zone");
  dropZone.innerHTML = `
    <input type="file" id="csv-file-input" accept=".csv" style="display: none;" onchange="handleFileUpload(event)">
    <div style="font-size: 2.5rem; color: var(--uber-white); margin-bottom: 0.75rem;"><i class="fa-solid fa-cloud-arrow-up"></i></div>
    <h3 style="font-size: 1.2rem; font-weight: 800; margin-bottom: 0.25rem;">Upload Manifest CSV</h3>
    <p style="color: var(--uber-gray-400); font-size: 0.85rem;">Drag and drop or browse from your computer</p>
  `;
}

function renderBatchResults(data) {
  document.getElementById("batch-results-container").style.display = "block";
  document.getElementById("batch-stat-total").innerText = data.total_records.toLocaleString();
  document.getElementById("batch-stat-avg").innerText = `${data.average_eta_min} min`;
  document.getElementById("batch-stat-low").innerText = data.low_risk_count.toLocaleString();
  document.getElementById("batch-stat-high").innerText = data.high_risk_count.toLocaleString();

  const tbody = document.getElementById("batch-table-body");
  tbody.innerHTML = "";

  (data.preview_data || []).forEach(row => {
    const tr = document.createElement("tr");
    const riskBadgeColor = row.Delay_Risk === "LOW" ? "var(--uber-green)" : (row.Delay_Risk === "MEDIUM" ? "var(--uber-yellow)" : "var(--uber-red)");
    const riskBadgeBg = row.Delay_Risk === "LOW" ? "var(--uber-green-bg)" : (row.Delay_Risk === "MEDIUM" ? "var(--uber-yellow-bg)" : "var(--uber-red-bg)");

    tr.innerHTML = `
      <td><strong class="mono">${row.Distance_km} km</strong></td>
      <td>${row.Weather}</td>
      <td>${row.Traffic_Level}</td>
      <td>${row.Vehicle_Type}</td>
      <td>${row.Preparation_Time_min} min</td>
      <td>${row.Courier_Experience_yrs} yrs</td>
      <td class="mono" style="color: var(--uber-white); font-weight: 800;">${row.Predicted_Delivery_Time_min} min</td>
      <td class="mono" style="color: var(--uber-gray-400);">${row.Lower_SLA_min} - ${row.Upper_SLA_min} min</td>
      <td><span class="uber-risk-tag" style="background: ${riskBadgeBg}; color: ${riskBadgeColor};">${row.Delay_Risk}</span></td>
    `;
    tbody.appendChild(tr);
  });

  resetUploadZone();
}

async function exportBatchCSV() {
  if (!lastUploadedFile) {
    alert("Please upload a CSV manifest first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", lastUploadedFile);

  try {
    const res = await fetch(getApiUrl("/api/batch-export"), {
      method: "POST",
      body: formData
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `eta_predictions_${Date.now()}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  } catch (err) {
    alert(`Export error: ${err.message}`);
  }
}

function downloadSampleCSV() {
  const sampleCSV = `Distance_km,Weather,Traffic_Level,Time_of_Day,Vehicle_Type,Preparation_Time_min,Courier_Experience_yrs
3.5,Clear,Low,Morning,Bike,10,4.0
12.8,Rainy,High,Evening,Bike,25,1.5
5.2,Clear,Medium,Afternoon,Scooter,15,3.0
18.0,Snowy,High,Night,Car,30,6.0
2.1,Foggy,Low,Morning,Scooter,8,5.0
8.4,Windy,Medium,Evening,Bike,18,2.0
14.5,Rainy,High,Afternoon,Car,22,3.5
6.0,Clear,Low,Night,Bike,12,1.0`;

  const blob = new Blob([sampleCSV], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sample_dispatch_manifest.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------------------------------------------------------
// History Log
// ---------------------------------------------------------
async function refreshHistory() {
  try {
    const res = await fetch(getApiUrl("/api/history"));
    if (res.ok) {
      const data = await res.json();
      const tbody = document.getElementById("history-table-body");
      if (!data.history || data.history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--uber-gray-400); padding: 2.5rem;">No dispatches simulated in this session yet. Run a prediction on the Ride & Delivery tab!</td></tr>`;
        return;
      }

      tbody.innerHTML = "";
      data.history.forEach(item => {
        const tr = document.createElement("tr");
        const riskColor = item.risk_level === "LOW RISK" ? "var(--uber-green)" : (item.risk_level === "MEDIUM RISK" ? "var(--uber-yellow)" : "var(--uber-red)");
        const riskBg = item.risk_level === "LOW RISK" ? "var(--uber-green-bg)" : (item.risk_level === "MEDIUM RISK" ? "var(--uber-yellow-bg)" : "var(--uber-red-bg)");

        tr.innerHTML = `
          <td><code class="mono" style="color: var(--uber-white); font-weight: 700;">${item.id}</code></td>
          <td class="mono">${item.time}</td>
          <td class="mono">${item.distance} km</td>
          <td>${item.weather}</td>
          <td>${item.traffic}</td>
          <td>${item.vehicle}</td>
          <td>${item.prep_time} min</td>
          <td class="mono" style="color: var(--uber-white); font-weight: 800;">${item.predicted_eta} min</td>
          <td><span class="uber-risk-tag" style="background: ${riskBg}; color: ${riskColor};">${item.risk_level}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("History fetch error:", err);
  }
}
