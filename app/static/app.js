const initialOptions = JSON.parse(
  document.getElementById("initial-options").textContent,
);

const stateBadge = document.getElementById("session-state");
const sourceLabel = document.getElementById("source-label");
const formError = document.getElementById("form-error");
const videoFeed = document.getElementById("video-feed");

const sourceType = document.getElementById("source-type");
const sourceButtons = Array.from(document.querySelectorAll("[data-source]"));
const videoFields = document.getElementById("video-fields");
const rtspUrlField = document.getElementById("rtsp-url-field");
const cameraIdField = document.getElementById("camera-id-field");

const videoSelect = document.getElementById("video-select");
const videoPath = document.getElementById("video-path");
const cameraId = document.getElementById("camera-id");
const rtspUrl = document.getElementById("rtsp-url");

const configSelect = document.getElementById("config-select");
const deviceSelect = document.getElementById("device");
const runModeSelect = document.getElementById("run-mode");
const enableTelegram = document.getElementById("enable-telegram");
const saveVideo = document.getElementById("save-video");
const motSkip = document.getElementById("mot-skip");
const clsSkip = document.getElementById("cls-skip");
const motThreshold = document.getElementById("mot-threshold");
const clsThreshold = document.getElementById("cls-threshold");

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const refreshBtn = document.getElementById("refresh-btn");
const configOpenBtn = document.getElementById("config-open-btn");
const configInlineBtn = document.getElementById("config-inline-btn");
const configCloseBtn = document.getElementById("config-close-btn");
const configBackdrop = document.getElementById("config-backdrop");
const configDrawerShell = document.getElementById("config-drawer-shell");

const inferenceFpsValue = document.getElementById("inference-fps-value");
const previewFpsValue = document.getElementById("preview-fps-value");
const previewPillValue = document.getElementById("preview-pill-value");
const latencyValue = document.getElementById("latency-value");
const frameAgeValue = document.getElementById("frame-age-value");
const tracksValue = document.getElementById("tracks-value");
const framesValue = document.getElementById("frames-value");

const alertsTotal = document.getElementById("alerts-total");
const logsTotal = document.getElementById("logs-total");
const alertsList = document.getElementById("alerts-list");
const logsList = document.getElementById("logs-list");

const overviewSource = document.getElementById("overview-source");
const overviewConfig = document.getElementById("overview-config");
const overviewDevice = document.getElementById("overview-device");
const overviewRunMode = document.getElementById("overview-run-mode");
const overviewResolution = document.getElementById("overview-resolution");
const overviewPreviewTarget = document.getElementById("overview-preview-target");
const overviewFrameAge = document.getElementById("overview-frame-age");
const overviewPreviewUrl = document.getElementById("overview-preview-url");

const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
const tabPanels = {
  overview: document.getElementById("tab-overview"),
  alerts: document.getElementById("tab-alerts"),
  logs: document.getElementById("tab-logs"),
};

let currentPreviewUrl = null;


function basename(value) {
  if (!value) {
    return "-";
  }
  const text = String(value);
  const parts = text.split(/[/\\]/);
  return parts[parts.length - 1] || text;
}


function setActionState(sessionState) {
  const runningStates = new Set(["starting", "running", "stopping"]);
  const isBusy = runningStates.has(sessionState);
  startBtn.disabled = isBusy;
  stopBtn.disabled = !isBusy;
}


function setSource(mode) {
  sourceType.value = mode;
  sourceButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.source === mode);
  });
  videoFields.classList.toggle("hidden", mode !== "video");
  rtspUrlField.classList.toggle("hidden", mode !== "rtsp");
  cameraIdField.classList.toggle("hidden", mode !== "camera");
}


function setActiveTab(tabName) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  Object.entries(tabPanels).forEach(([name, panel]) => {
    panel.classList.toggle("hidden", name !== tabName);
  });
}


function openConfigDrawer() {
  configDrawerShell.classList.remove("hidden");
  configDrawerShell.setAttribute("aria-hidden", "false");
}


function closeConfigDrawer() {
  configDrawerShell.classList.add("hidden");
  configDrawerShell.setAttribute("aria-hidden", "true");
}


function formatTs(ts) {
  if (!ts) {
    return "-";
  }
  return new Date(ts * 1000).toLocaleTimeString();
}


function setEmptyState(container, text) {
  container.innerHTML = "";
  const node = document.createElement("div");
  node.className = "empty";
  node.textContent = text;
  container.appendChild(node);
}


function renderAlerts(alerts) {
  alertsList.innerHTML = "";
  if (!alerts.length) {
    setEmptyState(alertsList, "No alerts yet.");
    return;
  }

  alerts.forEach((item) => {
    const node = document.createElement("div");
    node.className = "alert-item";

    const header = document.createElement("div");
    header.className = "list-title";

    const title = document.createElement("span");
    title.textContent = `${(item.event_type || "unknown").toUpperCase()} / ${(item.alert_level || "unknown").toUpperCase()}`;

    const ts = document.createElement("span");
    ts.textContent = formatTs(item.ts);

    header.appendChild(title);
    header.appendChild(ts);

    const body = document.createElement("div");
    body.className = "list-body";
    body.textContent = `Track ${item.track_id ?? "n/a"} | ${item.display_text || item.event_kind || ""}`;

    node.appendChild(header);
    node.appendChild(body);
    alertsList.appendChild(node);
  });
}


function renderLogs(logs) {
  logsList.innerHTML = "";
  if (!logs.length) {
    setEmptyState(logsList, "No logs yet.");
    return;
  }

  logs.forEach((item) => {
    const node = document.createElement("div");
    node.className = `log-item ${item.level || "info"}`;

    const header = document.createElement("div");
    header.className = "list-title";

    const level = document.createElement("span");
    level.textContent = (item.level || "info").toUpperCase();

    const ts = document.createElement("span");
    ts.textContent = formatTs(item.ts);

    header.appendChild(level);
    header.appendChild(ts);

    const body = document.createElement("div");
    body.className = "list-body";
    body.textContent = item.message || "";

    node.appendChild(header);
    node.appendChild(body);
    logsList.appendChild(node);
  });
}


function resetPreviewFrame() {
  currentPreviewUrl = null;
  if (videoFeed.src !== "about:blank") {
    videoFeed.src = "about:blank";
  }
}


function applyPreview(snapshot) {
  const state = snapshot.state || {};
  const nextUrl = state.preview_viewer_url || null;
  const sessionState = state.session_state || "idle";

  if (!nextUrl || ["idle", "error", "stopped"].includes(sessionState)) {
    resetPreviewFrame();
    return;
  }

  if (currentPreviewUrl === nextUrl && videoFeed.src === nextUrl) {
    return;
  }

  currentPreviewUrl = nextUrl;
  videoFeed.src = "about:blank";
  window.setTimeout(() => {
    if (currentPreviewUrl === nextUrl) {
      videoFeed.src = nextUrl;
    }
  }, 10);
}


function applyState(snapshot) {
  const state = snapshot.state || {};
  const media = snapshot.media || {};
  const inferenceFps = Number(state.inference_fps ?? state.fps ?? 0);
  const publishFps = Number(state.preview_fps || 0);
  const width = state.frame_width || null;
  const height = state.frame_height || null;
  const frameAge = Number(state.server_frame_age_ms || 0);
  const previewUrl = state.preview_viewer_url || "-";

  stateBadge.textContent = state.session_state || "idle";
  sourceLabel.textContent = state.source_label || "No active source";
  inferenceFpsValue.textContent = inferenceFps.toFixed(2);
  previewFpsValue.textContent = publishFps.toFixed(2);
  previewPillValue.textContent = publishFps.toFixed(1);
  latencyValue.textContent = `${Number(state.avg_latency_ms || 0).toFixed(1)} ms`;
  frameAgeValue.textContent = `${frameAge.toFixed(1)} ms`;
  tracksValue.textContent = `${state.track_count || 0}`;
  framesValue.textContent = `${state.frame_index || 0}`;

  overviewSource.textContent = state.source_label || "No active source";
  overviewConfig.textContent = basename(state.config_path);
  overviewDevice.textContent = state.device || "-";
  overviewRunMode.textContent = state.run_mode || "-";
  overviewResolution.textContent = width && height ? `${width} x ${height}` : "-";
  overviewPreviewTarget.textContent = state.preview_transport === "webrtc"
    ? "WebRTC / MediaMTX"
    : "-";
  overviewFrameAge.textContent = `${frameAge.toFixed(1)} ms`;
  overviewPreviewUrl.textContent = previewUrl;
  alertsTotal.textContent = `${state.alerts_total || 0} total`;
  logsTotal.textContent = `${state.logs_total || 0} lines`;

  renderAlerts(snapshot.alerts || []);
  renderLogs(snapshot.logs || []);

  const messages = [];
  if (state.error) {
    messages.push(state.error);
  }
  if (!(media.available ?? true)) {
    messages.push("MediaMTX is missing. Install it or set MEDIAMTX_BIN for WebRTC preview.");
  } else if (media.error && !media.running && !["idle", "stopped"].includes(state.session_state || "")) {
    messages.push(media.error);
  }
  formError.textContent = messages.join(" ");

  setActionState(state.session_state || "idle");
  applyPreview(snapshot);
}


function makeOptionLabel(value) {
  const text = String(value || "");
  if (text.includes("\\") || text.includes("/")) {
    return basename(text);
  }
  return text;
}


function fillSelect(selectElement, values, preferredValue = null) {
  selectElement.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = makeOptionLabel(value);
    option.title = String(value);
    if (preferredValue && preferredValue === value) {
      option.selected = true;
    }
    selectElement.appendChild(option);
  });
}


async function loadOptions() {
  const response = await fetch("/api/options", { cache: "no-store" });
  const options = await response.json();
  fillSelect(configSelect, options.config_files, options.default_config);
  fillSelect(videoSelect, options.video_files, options.default_video);
  fillSelect(deviceSelect, options.device_options, "GPU");
  fillSelect(runModeSelect, options.run_mode_options, "paddle");
  if (options.media_server && !options.media_server.available) {
    formError.textContent = "MediaMTX is missing. Install it or set MEDIAMTX_BIN before starting a session.";
  }
}


function buildPayload() {
  const mode = sourceType.value;
  const payload = {
    source_type: mode,
    video_path: null,
    camera_id: Number(cameraId.value || 0),
    rtsp_url: null,
    config_path: configSelect.value || null,
    device: deviceSelect.value,
    run_mode: runModeSelect.value,
    enable_telegram: enableTelegram.checked,
    save_video: saveVideo.checked,
    output_dir: null,
    mot_skip_frame_num: motSkip.value ? Number(motSkip.value) : null,
    cls_skip_frame_num: clsSkip.value ? Number(clsSkip.value) : null,
    mot_threshold: motThreshold.value ? Number(motThreshold.value) : null,
    cls_threshold: clsThreshold.value ? Number(clsThreshold.value) : null,
    web_stream_fps: 30,
    web_stream_width: 1280,
    jpeg_quality: 80,
  };

  if (mode === "video") {
    payload.video_path = videoPath.value.trim() || videoSelect.value || null;
  } else if (mode === "rtsp") {
    payload.rtsp_url = rtspUrl.value.trim() || null;
  }
  return payload;
}


async function startSession() {
  formError.textContent = "";
  const payload = buildPayload();
  const response = await fetch("/api/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Failed to start session." }));
    formError.textContent = error.detail || "Failed to start session.";
    return;
  }

  const snapshot = await response.json();
  applyState(snapshot);
  closeConfigDrawer();
}


async function stopSession() {
  formError.textContent = "";
  await fetch("/api/session/stop", { method: "POST" });
  resetPreviewFrame();
  await refreshStatus();
}


async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    applyState(await response.json());
  } catch (error) {
    formError.textContent = "Unable to refresh session status.";
  }
}


sourceButtons.forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

configOpenBtn.addEventListener("click", openConfigDrawer);
configInlineBtn.addEventListener("click", openConfigDrawer);
configCloseBtn.addEventListener("click", closeConfigDrawer);
configBackdrop.addEventListener("click", closeConfigDrawer);
refreshBtn.addEventListener("click", loadOptions);
startBtn.addEventListener("click", startSession);
stopBtn.addEventListener("click", stopSession);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeConfigDrawer();
  }
});

window.addEventListener("beforeunload", () => {
  resetPreviewFrame();
});

fillSelect(configSelect, initialOptions.config_files, initialOptions.default_config);
fillSelect(videoSelect, initialOptions.video_files, initialOptions.default_video);
fillSelect(deviceSelect, initialOptions.device_options, "GPU");
fillSelect(runModeSelect, initialOptions.run_mode_options, "paddle");
setSource("video");
setActiveTab("overview");
setActionState("idle");
refreshStatus();
setInterval(refreshStatus, 1000);
