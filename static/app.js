// ==========================================================================
// Uber ETA Platform - Client Application Engine
// ==========================================================================

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
  initTabs();
  initMap();
  loadPresets();
  calculateETA();
  initDragDrop();
});

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

  // Uber-style Dark Matter / Positron tiles
  L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; CARTO &copy; OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map);

  // Uber Icon Style: Pickup (Circle) & Dropoff (Square)
  const pickupIcon = L.divIcon({
    className: "uber-map-pin",
    html: `<div style="background: #000; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid #fff; box-shadow: 0 4px 12px rgba(0,0,0,0.5); font-size: 14px; color: #fff;">🍕</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });

  const dropoffIcon = L.divIcon({
    className: "uber-map-pin",
    html: `<div style="background: #000; width: 32px; height: 32px; border-radius: 4px; display: flex; align-items: center; justify-content: center; border: 3px solid #fff; box-shadow: 0 4px 12px rgba(0,0,0,0.5); font-size: 14px; color: #fff;">🏠</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });

  restaurantMarker = L.marker(restaurantPos, { icon: pickupIcon, draggable: true }).addTo(map);
  customerMarker = L.marker(customerPos, { icon: dropoffIcon, draggable: true }).addTo(map);

  restaurantMarker.bindPopup("<b>Central Kitchen Hub #402</b><br>Pickup Location");
  customerMarker.bindPopup("<b>Customer Dropoff</b><br>Broadway, New York");

  // Uber Route Polyline (Bold Black with white border effect)
  routePolyline = L.polyline([restaurantPos, customerPos], {
    color: '#000000',
    weight: 5,
    opacity: 0.9,
    lineCap: 'round',
    dashArray: '8, 8'
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
// Preset Scenarios
// ---------------------------------------------------------
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
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

  document.getElementById("input-prep").value = data.Preparation_Time_min;
  document.getElementById("val-prep").innerText = `${data.Preparation_Time_min} min`;

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
  return {
    Distance_km: parseFloat(document.getElementById("input-distance").value),
    Preparation_Time_min: parseFloat(document.getElementById("input-prep").value),
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
    const res = await fetch("/api/predict", {
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
    const res = await fetch("/api/batch-predict", {
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
    const res = await fetch("/api/batch-export", {
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
    const res = await fetch("/api/history");
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
