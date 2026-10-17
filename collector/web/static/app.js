// halo-collector UI: pick a job -> languages/options -> start -> watch stream -> answer asks -> download.
const LANG_NAMES = { bcl: "Bikol", ceb: "Cebuano", eng: "Philippine English", fil: "Filipino", hil: "Hiligaynon",
  ilo: "Ilocano", pag: "Pangasinan", pam: "Kapampangan", tsg: "Tausug", war: "Waray" };
const $ = (sel) => document.querySelector(sel);
const state = { jobs: [], current: null, es: null, seen: new Set(), asks: new Map(), status: null };

const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
};
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return n;
};

// --- compose -----------------------------------------------------------------
async function initCompose() {
  const codes = await api("/api/languages");
  const box = $("#langs");
  for (const c of codes) {
    const chip = el("span", { class: "chip", "data-code": c, onclick: () => chip.classList.toggle("on") }, LANG_NAMES[c] || c, el("code", {}, c));
    box.append(chip);
  }
  const syncKind = () => {
    const kind = $('input[name=kind]:checked').value;
    $("#opt-docs").hidden = kind !== "collect";
    $("#opt-sent").hidden = !(kind === "validate" || kind === "instruct");
  };
  document.querySelectorAll('input[name=kind]').forEach((r) => r.addEventListener("change", syncKind));
  syncKind();
  $("#push").addEventListener("change", (e) => { $("#push-warn").hidden = !e.target.checked; });
  $("#start").addEventListener("click", startJob);
  $("#new-run").addEventListener("click", () => showCompose());
}

async function startJob() {
  const languages = [...document.querySelectorAll(".chip.on")].map((c) => c.dataset.code);
  const msg = $("#start-msg");
  if (!languages.length) { msg.textContent = "Pick at least one language."; return; }
  const body = {
    kind: $('input[name=kind]:checked').value, languages,
    max_docs: +$("#max_docs").value || 300, max_sentences: +$("#max_sentences").value || 500,
    push: $("#push").checked, notes: $("#notes").value.trim(),
  };
  $("#start").disabled = true; msg.textContent = "Starting…";
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
    msg.textContent = "";
    await refreshRuns();
    openRun(job.session_id);
  } catch (e) { msg.textContent = "Could not start: " + e.message; }
  $("#start").disabled = false;
}

// --- runs list ---------------------------------------------------------------
async function refreshRuns() {
  state.jobs = await api("/api/jobs");
  const ul = $("#runs"); ul.innerHTML = "";
  if (!state.jobs.length) ul.append(el("li", { class: "muted" }, "No runs yet."));
  for (const j of state.jobs) {
    ul.append(el("li", { class: j.session_id === state.current ? "active" : "", onclick: () => openRun(j.session_id) },
      el("span", {}, j.title), el("span", { class: "pill " + (j.last_status || "") }, j.last_status || "")));
  }
}

function showCompose() {
  closeStream(); state.current = null;
  $("#compose").hidden = false; $("#run").hidden = true;
  refreshRuns();
}

// --- run view ----------------------------------------------------------------
function openRun(sid) {
  if (state.current === sid) return;
  closeStream();
  state.current = sid; state.seen = new Set(); state.asks = new Map();
  $("#feed").innerHTML = ""; $("#asks").innerHTML = ""; $("#usage").textContent = "";
  $("#outputs").innerHTML = '<li class="muted">Available once the run ends.</li>';
  $("#compose").hidden = true; $("#run").hidden = false;
  const job = state.jobs.find((j) => j.session_id === sid) || {};
  $("#run-title").textContent = job.title || sid;
  $("#console-link").href = job.console_url || "#";
  setStatus(job.last_status || "running");
  history.replaceState(null, "", "#" + sid);
  refreshRuns();
  loadStatus().then(() => { if (!["idle", "terminated", "ended", "error"].includes((state.status || {}).status)) connect(sid); else loadOutputs(); });
}

function setStatus(s) {
  const p = $("#run-status"); p.textContent = s || ""; p.className = "pill " + (s || "");
  $("#interrupt").disabled = ["idle", "terminated", "ended"].includes(s);
}

async function loadStatus() {
  try {
    const st = await api(`/api/jobs/${state.current}/status`);
    state.status = st;
    setStatus(st.status || "running");
    if (st.usage) renderUsage(st.usage);
    for (const a of st.pending_asks || []) renderAsk(a);
  } catch (e) { state.status = null; }
}

function connect(sid) {
  const es = new EventSource(`/api/jobs/${sid}/events`);
  state.es = es;
  es.onmessage = (m) => { try { handleEvent(JSON.parse(m.data)); } catch (_) {} };
  es.addEventListener("end", () => { closeStream(); loadStatus(); loadOutputs(); refreshRuns(); });
  es.onerror = () => { /* EventSource reconnects by itself; the server replays history and we dedupe by id */ };
}
function closeStream() { if (state.es) { state.es.close(); state.es = null; } }

function handleEvent(ev) {
  if (ev.id && state.seen.has(ev.id)) return;
  if (ev.id) state.seen.add(ev.id);
  const feed = $("#feed");
  const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
  const node = renderEvent(ev);
  if (node) feed.append(node);
  if (ev.type === "tool_ask") { renderAsk(ev); setStatus("needs_decision"); }
  if (ev.type === "usage" && ev.text) $("#usage").textContent = "Usage: " + ev.text;
  if (atBottom) feed.scrollTop = feed.scrollHeight;
}

function renderEvent(ev) {
  const t = ev.type || ev.raw_type || "event";
  const node = el("div", { class: "ev " + t + (ev.result ? " " + ev.result : "") });
  const label = el("span", { class: "k" }, t.replace("_", " "));
  switch (t) {
    case "text":
      node.append(label, ev.text || ""); break;
    case "tool_call": case "tool_use": case "tool_ask": case "tool_result": {
      node.append(label, el("b", {}, ev.tool || ""), " ", ev.text || "");
      const payload = ev.input != null ? ev.input : ev.result;
      if (payload != null) {
        const txt = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
        node.append(el("details", {}, el("summary", {}, t === "tool_result" ? "result" : "input"), el("pre", {}, txt)));
      }
      break;
    }
    case "outcome":
      node.append(label, el("b", {}, (ev.result || "graded") + (ev.iteration != null ? ` · iteration ${ev.iteration}` : "")),
        el("div", { class: "muted" }, ev.explanation || "")); break;
    case "usage":
      node.append(label, ev.text || JSON.stringify(ev.input || {})); break;
    case "status":
      node.append(label, ev.text || ev.raw_type || "", ev.stop_reason ? ` (${ev.stop_reason})` : ""); break;
    case "error":
      node.append(label, ev.text || "error"); break;
    default:
      node.append(label, ev.text || ev.raw_type || JSON.stringify(ev));
  }
  return node;
}

function renderUsage(u) {
  if (typeof u === "object") {
    const parts = Object.entries(u).filter(([, v]) => typeof v === "number").map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`);
    if (parts.length) $("#usage").textContent = "Usage: " + parts.join(" · ");
  }
}

function renderAsk(ev) {
  if (!ev.id || state.asks.has(ev.id)) return;
  const msg = el("input", { type: "text", placeholder: "Optional message to the agent" });
  const input = ev.input != null ? (typeof ev.input === "string" ? ev.input : JSON.stringify(ev.input, null, 2)) : "";
  const card = el("div", { class: "ask", "data-id": ev.id },
    el("h3", {}, "Needs your decision"),
    el("div", {}, "The agent wants to run ", el("b", {}, ev.tool || "a tool"), ev.text ? ` — ${ev.text}` : "."),
    input ? el("pre", {}, input) : "",
    el("div", { class: "row" },
      el("button", { class: "primary", onclick: () => answer(ev.id, true, msg.value) }, "Allow"),
      el("button", { class: "danger", onclick: () => answer(ev.id, false, msg.value) }, "Deny"),
      msg));
  state.asks.set(ev.id, card);
  $("#asks").append(card);
}

async function answer(id, allow, message) {
  const card = state.asks.get(id);
  card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  try {
    await api(`/api/jobs/${state.current}/confirm`, { method: "POST", body: JSON.stringify({ tool_event_id: id, allow, message: message || null }) });
    card.remove(); state.asks.delete(id); setStatus("running");
  } catch (e) {
    card.querySelectorAll("button").forEach((b) => (b.disabled = false));
    card.append(el("div", { class: "muted" }, "Failed: " + e.message));
  }
}

async function loadOutputs() {
  const ul = $("#outputs");
  try {
    const files = await api(`/api/jobs/${state.current}/outputs`);
    ul.innerHTML = "";
    if (!files.length) ul.append(el("li", { class: "muted" }, "No outputs."));
    for (const f of files) {
      ul.append(el("li", {}, el("a", { href: `/api/jobs/${state.current}/outputs/${encodeURIComponent(f.id)}`, download: f.filename }, f.filename),
        " ", el("span", { class: "muted" }, f.size_bytes != null ? `${f.size_bytes} bytes` : "")));
    }
  } catch (e) { ul.innerHTML = ""; ul.append(el("li", { class: "muted" }, "Outputs unavailable: " + e.message)); }
}

$("#interrupt").addEventListener("click", async () => {
  if (!state.current || !confirm("Interrupt this run?")) return;
  try { await api(`/api/jobs/${state.current}/interrupt`, { method: "POST" }); setStatus("interrupting"); }
  catch (e) { alert("Interrupt failed: " + e.message); }
});

// --- boot --------------------------------------------------------------------
(async () => {
  await initCompose();
  await refreshRuns();
  const sid = location.hash.slice(1);
  if (sid && state.jobs.some((j) => j.session_id === sid)) openRun(sid);
})();
