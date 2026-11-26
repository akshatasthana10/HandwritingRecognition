// Handwriting to text: talks to the local server in server.py.
"use strict";

const UNSURE = 0.6;              // below this confidence a letter is underlined for checking
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  model: null,        // "new" | "original"
  file: null,         // the Blob being read (kept so switching model can re-read it)
  fileName: "page",
  result: null,       // server response
  chars: [],          // flat list: {ch, box, conf, alts, line, fixed, span, rect}
  plain: false,       // editing as plain text
  popFor: null,       // index of the character the popover is open for
};

// ------------------------------------------------------------------ setup

async function init() {
  try {
    const info = await fetch("/api/info").then((r) => r.json());
    state.model = info.default;
    if (info.models.length > 1) {
      $("model-picker").hidden = false;
      for (const b of document.querySelectorAll("[data-model]")) {
        b.hidden = !info.models.includes(b.dataset.model);
        b.addEventListener("click", () => setModel(b.dataset.model));
      }
      markModel();
    }
  } catch {
    showError("The recognition server isn't responding. Start it with: .venv/bin/python server.py");
  }
  loadSamples();

  $("choose").addEventListener("click", () => $("file").click());
  $("error-retry").addEventListener("click", reset);
  $("new-page").addEventListener("click", reset);
  $("file").addEventListener("change", (e) => e.target.files[0] && read(e.target.files[0], e.target.files[0].name));
  $("show-boxes").addEventListener("change", (e) => $("photo").classList.toggle("show-boxes", e.target.checked));
  $("copy").addEventListener("click", copyText);
  $("download").addEventListener("click", downloadText);
  $("edit-plain").addEventListener("click", togglePlain);
  $("next-unsure").addEventListener("click", nextUnsure);
  $("drop-target").addEventListener("click", (e) => { if (e.target === $("drop-target")) $("file").click(); });

  // drag and drop anywhere on the page
  let depth = 0;
  window.addEventListener("dragenter", (e) => { e.preventDefault(); depth++; document.body.classList.add("dragging"); });
  window.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; document.body.classList.remove("dragging"); } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    document.body.classList.remove("dragging");
    const f = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (f) read(f, f.name);
    else if (e.dataTransfer.files.length) showError("That file isn't an image. Drop a JPG or PNG photo of the page.");
  });
  // paste an image (screenshot, copied photo)
  window.addEventListener("paste", (e) => {
    if (e.target.closest && e.target.closest("textarea, input")) return;
    const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
    if (item) { e.preventDefault(); read(item.getAsFile(), "pasted-page"); }
  });

  // popover
  document.addEventListener("click", (e) => {
    if (!$("popover").hidden && !e.target.closest("#popover, .ch, .overlay rect, #next-unsure")) closePopover();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePopover(true); });
  $("pop-input").addEventListener("input", (e) => { if (e.target.value.trim()) fixChar(state.popFor, e.target.value.trim()); });
  window.addEventListener("resize", () => closePopover());
}

async function loadSamples() {
  try {
    const samples = await fetch("/api/samples").then((r) => r.json());
    if (!samples.length) return;
    const list = $("sample-list");
    for (const s of samples) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = `<img alt="" loading="lazy"><span></span>`;
      b.querySelector("img").src = `/api/samples/${encodeURIComponent(s.name)}/thumb`;
      b.querySelector("span").textContent = s.label;
      b.setAttribute("aria-label", `Read sample page ${s.label}`);
      b.addEventListener("click", async () => {
        const blob = await fetch(`/api/samples/${encodeURIComponent(s.name)}`).then((r) => r.blob());
        read(blob, s.label);
      });
      li.append(b);
      list.append(li);
    }
    $("samples").hidden = false;
  } catch { /* samples are optional */ }
}

// ------------------------------------------------------------------ reading a page

function setModel(model) {
  if (model === state.model) return;
  state.model = model;
  markModel();
  if (state.file) read(state.file, state.fileName);
}

function markModel() {
  for (const b of document.querySelectorAll("[data-model]")) b.setAttribute("aria-checked", String(b.dataset.model === state.model));
}

async function read(blob, name) {
  state.file = blob;
  state.fileName = (name || "page").replace(/\.[^.]+$/, "");
  closePopover();
  setPlain(false);
  $("error").hidden = true;
  const img = $("photo-img");
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  img.dataset.url = URL.createObjectURL(blob);
  img.src = img.dataset.url;
  $("overlay").replaceChildren();
  setState("loading");
  $("sheet").replaceChildren();
  $("sheet").classList.add("reading");
  $("sheet").textContent = "Reading the page…";
  $("summary-text").textContent = "Finding lines and letters, then reading each one.";
  $("next-unsure").hidden = true;

  const form = new FormData();
  form.append("image", blob, name || "page");
  if (state.model) form.append("model", state.model);
  try {
    const res = await fetch("/api/recognize", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `The server couldn't read this image (error ${res.status}).`);
    if (state.file !== blob) return;            // a newer page replaced this one meanwhile
    show(data);
  } catch (err) {
    if (state.file !== blob) return;
    showError(err.message.startsWith("Failed to fetch")
      ? "The recognition server isn't responding. Start it with: .venv/bin/python server.py"
      : err.message);
  }
}

function show(result) {
  state.result = result;
  state.chars = [];
  const sheet = $("sheet");
  sheet.classList.remove("reading");
  sheet.replaceChildren();
  result.lines.forEach((line, li) => {
    const row = document.createElement("span");
    row.className = "line";
    line.words.forEach((word, wi) => {
      if (wi) row.append(" ");
      for (const c of word.chars) {
        const i = state.chars.length;
        const span = document.createElement("span");
        span.className = "ch";
        span.textContent = c.ch;
        span.dataset.i = i;
        state.chars.push({ ...c, line: li, fixed: false, span });
        row.append(span);
      }
    });
    sheet.append(row);
  });

  // boxes on the photo
  const svg = $("overlay");
  svg.setAttribute("viewBox", `0 0 ${result.width} ${result.height}`);
  svg.replaceChildren();
  for (const [i, c] of state.chars.entries()) {
    const [x0, y0, x1, y1] = c.box;
    const r = document.createElementNS(SVG_NS, "rect");
    const pad = Math.max(2, (y1 - y0) * 0.08);
    r.setAttribute("x", x0 - pad); r.setAttribute("y", y0 - pad);
    r.setAttribute("width", x1 - x0 + 2 * pad); r.setAttribute("height", y1 - y0 + 2 * pad);
    r.dataset.i = i;
    c.rect = r;
    svg.append(r);
  }
  for (const c of state.chars) mark(c);

  // link letters and boxes
  for (const el of [sheet, svg]) {
    el.onmouseover = (e) => { const i = e.target.dataset?.i; if (i !== undefined) light(+i, true); };
    el.onmouseout = (e) => { const i = e.target.dataset?.i; if (i !== undefined) light(+i, false); };
    el.onclick = (e) => { const i = e.target.dataset?.i; if (i !== undefined) openPopover(+i, e.target); };
  }
  sheet.onkeydown = (e) => {
    const i = e.target.dataset?.i;
    if (i !== undefined && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openPopover(+i, e.target); }
  };
  sheet.onfocusin = (e) => { const i = e.target.dataset?.i; if (i !== undefined) light(+i, true); };
  sheet.onfocusout = (e) => { const i = e.target.dataset?.i; if (i !== undefined) light(+i, false); };

  setState("done");
  summarise();
}

function mark(c) {
  const unsure = !c.fixed && c.conf < UNSURE && /\S/.test(c.ch);
  c.span.classList.toggle("unsure", unsure);
  c.span.classList.toggle("fixed", c.fixed);
  c.rect.classList.toggle("unsure", unsure);
  c.rect.classList.toggle("fixed", c.fixed);
  if (unsure || c.fixed) c.span.tabIndex = 0; else c.span.removeAttribute("tabindex");
  c.span.title = c.fixed ? "Corrected by you" : unsure ? "Worth checking: click to see other readings" : "";
}

function summarise() {
  const total = state.chars.length;
  const unsure = state.chars.filter((c) => c.span.classList.contains("unsure")).length;
  const p = $("summary-text");
  if (!total) {
    p.textContent = "No writing found on this page. Try a sharper, closer photo with the page filling the frame.";
    $("actions").hidden = true;
    $("next-unsure").hidden = true;
    return;
  }
  const secs = `${state.result.seconds.toFixed(1)} s`;
  p.innerHTML = "";
  const a = document.createElement("strong");
  a.textContent = `${total} characters`;
  p.append("Read ", a, ` in ${secs}. `);
  if (unsure) {
    const s = document.createElement("strong");
    s.className = "unsure-count";
    s.textContent = unsure === 1 ? "1 letter" : `${unsure} letters`;
    p.append(s, unsure === 1 ? " is worth checking." : " are worth checking.");
  } else {
    p.append("Nothing flagged for checking.");
  }
  $("next-unsure").hidden = !unsure || state.plain;
  $("actions").hidden = false;
}

function light(i, on) {
  const c = state.chars[i];
  if (!c) return;
  c.span.classList.toggle("lit", on);
  c.rect.classList.toggle("lit", on);
}

// ------------------------------------------------------------------ corrections

function openPopover(i, anchor) {
  const c = state.chars[i];
  if (!c || state.plain) return;
  state.popFor = i;
  $("pop-char").textContent = c.ch;
  const pct = Math.round(c.conf * 100);
  const conf = $("pop-conf");
  conf.textContent = c.fixed ? "Corrected by you" : `The model is ${pct}% sure`;
  conf.classList.toggle("low", !c.fixed && c.conf < UNSURE);
  const alts = $("pop-alts");
  alts.replaceChildren();
  const options = c.fixed ? [c.original, ...c.alts.map((a) => a[0])] : c.alts.map((a) => a[0]);
  for (const ch of [...new Set(options)].filter((x) => x && x !== c.ch).slice(0, 4)) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = ch;
    b.setAttribute("aria-label", `Change to ${ch}`);
    b.addEventListener("click", () => fixChar(i, ch));
    alts.append(b);
  }
  const keep = document.createElement("button");
  keep.type = "button";
  keep.className = "keep";
  keep.textContent = `Keep ${c.ch}`;
  keep.addEventListener("click", () => fixChar(i, c.ch));
  alts.append(keep);
  $("pop-input").value = "";

  const pop = $("popover");
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const w = pop.offsetWidth, h = pop.offsetHeight;
  let left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8);
  let top = r.bottom + 8;
  if (top + h > innerHeight - 8) top = Math.max(8, r.top - h - 8);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
  light(i, true);
  alts.querySelector("button").focus();
}

function closePopover(restoreFocus) {
  if ($("popover").hidden) return;
  const c = state.chars[state.popFor];
  $("popover").hidden = true;
  if (c) { light(state.popFor, false); if (restoreFocus && c.span.tabIndex === 0) c.span.focus(); }
  state.popFor = null;
}

function fixChar(i, ch) {
  const c = state.chars[i];
  if (!c) return;
  if (!c.fixed) c.original = c.ch;
  c.fixed = true;
  c.ch = ch;
  c.span.textContent = ch;
  mark(c);
  closePopover();
  summarise();
  toast(ch === c.original ? `Kept "${ch}"` : `Changed to "${ch}"`);
  nextUnsure(i);
}

function nextUnsure(after) {
  const start = typeof after === "number" ? after + 1 : 0;
  const order = [...state.chars.slice(start), ...state.chars.slice(0, start)];
  const c = order.find((c) => c.span.classList.contains("unsure"));
  if (!c) return;
  c.span.scrollIntoView({ block: "nearest" });
  c.rect.scrollIntoView?.({ block: "nearest" });
  if (typeof after !== "number") openPopover(+c.span.dataset.i, c.span);
  else c.span.focus({ preventScroll: true });
}

// ------------------------------------------------------------------ output

function currentText() {
  if (state.plain) return $("plain").value;
  return [...$("sheet").querySelectorAll(".line")].map((l) => l.textContent).join("\n");
}

async function copyText() {
  try {
    await navigator.clipboard.writeText(currentText());
    toast("Copied");
  } catch {
    toast("Couldn't copy. Select the text and press ⌘C instead.");
  }
}

function downloadText() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([currentText() + "\n"], { type: "text/plain" }));
  a.download = `${state.fileName}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast(`Downloaded ${a.download}`);
}

function togglePlain() { setPlain(!state.plain); }

function setPlain(on) {
  if (on) $("plain").value = currentText();
  state.plain = on;
  $("plain").hidden = !on;
  $("sheet").hidden = on;
  $("edit-plain").textContent = on ? "Back to checking letters" : "Edit as plain text";
  if (on) { closePopover(); $("plain").focus(); }
  if (state.result) summarise();
}

// ------------------------------------------------------------------ states and messages

function setState(s) {
  $("workspace").dataset.state = s;
  const hasPage = s !== "empty";
  $("drop").hidden = hasPage;
  $("viewer").hidden = !hasPage;
  $("progress").hidden = s !== "loading";
  $("actions").hidden = s !== "done";
}

function showError(message) {
  setState(state.file ? "error" : "empty");
  $("sheet").classList.remove("reading");
  $("sheet").replaceChildren();
  $("summary-text").textContent = "";
  $("error-text").textContent = message;
  $("error").hidden = false;
}

function reset() {
  state.file = state.result = null;
  state.chars = [];
  closePopover();
  setPlain(false);
  $("error").hidden = true;
  $("sheet").replaceChildren();
  $("summary-text").textContent = "The typed text will appear here. Letters worth checking are underlined in red.";
  $("next-unsure").hidden = true;
  setState("empty");
  $("file").value = "";
}

let toastTimer;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
}

init();
