const $ = (q) => document.querySelector(q);
const $$ = (q) => document.querySelectorAll(q);

// A minimal list of RFMiD conditions for the chip row on the Diseases tab.
const RFMID_CONDITIONS = [
  "Diabetic Retinopathy", "Age-Related Macular Degeneration", "Media Haze",
  "Drusen", "Myopia", "Branch Retinal Vein Occlusion", "Tessellation",
  "Epiretinal Membrane", "Laser Scars", "Macular Scar",
  "Central Serous Retinopathy", "Optic Disc Cupping",
  "Central Retinal Vein Occlusion", "Tortuous Vessels", "Asteroid Hyalosis",
  "Optic Disc Pallor", "Optic Disc Edema", "Shunt",
  "Anterior Ischemic Optic Neuropathy", "Parafoveal Telangiectasia",
  "Retinal Traction", "Retinitis", "Chorioretinitis", "Exudation",
  "RPE Changes", "Macular Hole", "Retinitis Pigmentosa", "Cotton Wool Spots",
  "Coloboma", "Optic Disc Pit Maculopathy", "Preretinal Hemorrhage",
  "Myelinated Nerve Fibers", "Hemorrhagic Retinopathy",
  "Central Retinal Artery Occlusion", "Tilted Disc", "Cystoid Macular Edema",
  "Post-Traumatic Choroidal Rupture", "Choroidal Folds", "Vitreous Hemorrhage",
  "Macroaneurysm", "Vasculitis", "Branch Retinal Artery Occlusion", "Plaque",
  "Hemorrhagic PED", "Collateral",
];

// ---------- tabs ----------
const PANEL_IDS = {
  predict: "predict",
  model: "model",
  diseases: "diseases",
  results: "results-tab",
  about: "about",
};
function setTab(name) {
  $$(".nav-link").forEach((el) =>
    el.classList.toggle("active", el.dataset.tab === name)
  );
  Object.entries(PANEL_IDS).forEach(([tab, id]) => {
    document.getElementById(id).classList.toggle("active", tab === name);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}
$$(".nav-link").forEach((el) =>
  el.addEventListener("click", (e) => {
    e.preventDefault();
    setTab(el.dataset.tab);
  })
);
$$("[data-goto]").forEach((el) =>
  el.addEventListener("click", () => setTab(el.dataset.goto))
);

// ---------- mode (paper default; enhanced = Future Work §VII-1) ----------
const mode = () => ($("#enhanced-toggle") && $("#enhanced-toggle").checked ? "enhanced" : "paper");

// ---------- drop zone ----------
const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const previewImg = $("#preview-img");
const predictBtn = $("#predict-btn");
let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});
function handleFile(file) {
  selectedFile = file;
  predictBtn.disabled = false;
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImg.src = ev.target.result;
    previewImg.classList.add("visible");
  };
  reader.readAsDataURL(file);
}

// ---------- predict ----------
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const showCam = $("#cam-toggle").checked;
  const fd = new FormData();
  fd.append("file", selectedFile);

  predictBtn.classList.add("loading");
  predictBtn.textContent = "Running…";
  try {
    const r = await fetch(`/api/predict?heatmap=${showCam}&mode=${mode()}`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    renderResults(data, showCam);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    predictBtn.classList.remove("loading");
    predictBtn.textContent = "🔮 Run prediction";
  }
});

async function runSample(sid) {
  predictBtn.classList.add("loading");
  try {
    const r = await fetch(`/api/sample/${sid}?mode=${mode()}`);
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    previewImg.src = data.input_image;
    previewImg.classList.add("visible");
    selectedFile = null;
    predictBtn.disabled = true;     // require new upload to re-run
    renderResults(data, true);
  } catch (e) {
    alert("Sample failed: " + e.message);
  } finally {
    predictBtn.classList.remove("loading");
  }
}

function renderResults(data, showCam) {
  $("#results-empty").classList.add("hidden");
  $("#results").classList.remove("hidden");

  const p = data.prediction;
  const sev = p.severity;

  const verdict = $("#verdict");
  verdict.className = `verdict sev-${sev}`;
  $("#verdict-grade").textContent = sev;
  $("#m-class").textContent = p.name;
  $("#m-conf").textContent = (data.confidence * 100).toFixed(1) + "%";
  $("#m-sev").textContent = sev;

  $("#clinical-note").innerHTML =
    `<b>Clinical note —</b> ${p.description}`;

  const rank = $("#ranking");
  rank.innerHTML = "";
  data.ranking.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "rank-row" + (i === 0 ? " top" : "");
    row.innerHTML = `
      <div class="rank-grade">${r.severity}</div>
      <div>
        <div class="rank-name">${r.name}</div>
        <div class="rank-bar"><span style="width:${(r.prob * 100).toFixed(1)}%"></span></div>
      </div>
      <div class="rank-prob">${(r.prob * 100).toFixed(1)}%</div>
    `;
    rank.appendChild(row);
  });

  const cw = $("#cam-wrap");
  if (showCam && data.heatmap) {
    cw.classList.remove("hidden");
    $("#cam-img").src = data.heatmap;
  } else {
    cw.classList.add("hidden");
  }

  $("#mode-note").textContent = data.mode === "enhanced"
    ? "Enhanced mode: CCER over the 2 paper members + fine-tuned members (paper Future Work §VII-1)."
    : "Paper pipeline: CCER over 2 ViT-B/16 members (circular crop + Ben Graham + TTA).";

  // scroll result into view on mobile layouts
  $("#results").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------- enhanced-mode results (Results tab) ----------
const pct = (v) => (v * 100).toFixed(1) + "%";
function renderEnhanced(enh) {
  if (!enh || !enh.available) return;
  $("#enhanced-row").hidden = false;
  const ev = enh.evaluation;
  if (!ev) return;
  $("#enhanced-card").hidden = false;
  const titles = { test: "IDRiD test — 103 ophthalmologist-graded images",
                   external: "Messidor-2 — 1,744 images (external: different country & camera)" };
  let html = "";
  for (const split of ["test", "external"]) {
    if (!ev[split]) continue;
    const rows = Object.entries(ev[split]).map(([name, m]) =>
      `<tr${name.startsWith("Enhanced") ? ' class="hl"' : ""}><td>${name}</td><td>${pct(m.accuracy)}</td>` +
      `<td>${m.qwk.toFixed(3)}</td><td>${pct(m.referable_sensitivity)}</td><td>${pct(m.referable_specificity)}</td></tr>`).join("");
    html += `<h4>${titles[split]}</h4><div class="table-wrap"><table class="data">` +
      `<tr><th>Configuration</th><th>Accuracy</th><th>QWK</th><th>Referable sens.</th><th>Referable spec.</th></tr>` +
      `${rows}</table></div>`;
  }
  html += `<p class="muted tiny">CCER weights (chosen on validation): ${Object.entries(ev.weights)
    .map(([k, w]) => `${k} ${w}`).join(" · ")}</p>`;
  $("#enhanced-tables").innerHTML = html;
}

// ---------- boot: info, samples, disease grid ----------
(async () => {
  try {
    const r = await fetch("/api/info");
    const info = await r.json();
    $("#nav-meta-text").textContent = `${info.device.toUpperCase()} · ${info.params_millions}M`;
    $("#kv-params").textContent = info.params_millions + " M";
    $("#kv-device").textContent = info.device.toUpperCase();
    $("#pill-params").textContent = Math.round(info.params_millions) + "M";

    const repos = info.ensemble_repos || [info.model_id];
    const list = $("#hf-repo-list");
    if (list) {
      list.innerHTML = "";
      repos.forEach((r) => {
        const li = document.createElement("li");
        li.innerHTML = `<a href="https://huggingface.co/${r}" target="_blank"><code>${r}</code></a>`;
        list.appendChild(li);
      });
    }
    const aboutLink = $("#about-model");
    aboutLink.textContent = info.model_id;
    aboutLink.href = `https://huggingface.co/${repos[0]}`;

    // sample grid
    const grid = $("#sample-grid");
    const expectedToSev = {
      "No DR": 0, "Mild NPDR": 1, "Moderate NPDR": 2,
      "Severe NPDR": 3, "Proliferative DR": 4,
    };
    info.samples.forEach((s) => {
      const el = document.createElement("div");
      el.className = "sample";
      const sev = expectedToSev[s.expected];
      const sevCls = sev === undefined ? "sev-x" : `sev-${sev}`;
      const badgeText = sev === undefined ? s.expected : `Grade ${sev}`;
      el.innerHTML = `
        <img src="${s.url}" alt="${s.label}" loading="lazy"/>
        <div class="sample-badge ${sevCls}">${badgeText}</div>
        <div class="sample-meta">
          <div class="sample-label">${s.label}</div>
          <div class="sample-sub">${s.caption}</div>
        </div>
      `;
      el.addEventListener("click", () => runSample(s.id));
      grid.appendChild(el);
    });

    // disease cards
    const dGrid = $("#disease-grid");
    info.classes.forEach((c) => {
      const card = document.createElement("div");
      card.className = `disease-card sev-${c.severity}`;
      card.innerHTML = `
        <div class="disease-grade">Grade ${c.severity} · ${c.code}</div>
        <div class="disease-name">${c.name}</div>
        <div class="disease-desc">${c.description}</div>
      `;
      dGrid.appendChild(card);
    });

    // RFMiD chip row
    const chips = $("#rfmid-chips");
    RFMID_CONDITIONS.forEach((name) => {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = name;
      chips.appendChild(c);
    });

    renderEnhanced(info.enhanced);
  } catch (e) {
    $("#nav-meta-text").textContent = "offline";
    console.error(e);
  }
})();
