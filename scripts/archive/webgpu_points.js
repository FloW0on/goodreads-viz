// webgpu_points.js

async function fetchArrayBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.arrayBuffer();
}

function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

function toPackedRelative(p) {
  return String(p).replace(/^.*\/packed\//, "./packed/");
}

function hslToRgb(h, s, l) {
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : (l + s - l * s);
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function makePalette(n) {
  const out = new Uint8Array(n * 4);

  const s = 0.45; 
  const l = 0.78; 

  const baseHues = [
    205, 195, 215, 185, 225, 175, 235, 165, 245, 155, 255, 145, 
    265, 135, 275, 125, 285, 115,                               
    295, 105, 305,  95, 315,  85          
  ];

  for (let i = 0; i < n; i++) {
    let hDeg;
    if (i < baseHues.length) {
      hDeg = baseHues[i];
    } else {
      const t = i - baseHues.length;
      hDeg = 170 + (t * 13.0) % 60;
    }

    const [r, g, b] = hslToRgb(hDeg / 360, s, l);
    out[i * 4 + 0] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }

  return out;
}

function rgbaToCss(r, g, b, a = 255) { return `rgba(${r},${g},${b},${a / 255})`; }

function renderLegend(legendEl, labelsJson, paletteRGBA, onSelect) {
  const labels = labelsJson.labels || {};
  const keys = Object.keys(labels).map(k => parseInt(k, 10));
  keys.sort((a, b) => (labels[b]?.size || 0) - (labels[a]?.size || 0));

  legendEl.innerHTML = "";

  const btnRow = document.createElement("div");
  btnRow.style.display = "flex";
  btnRow.style.gap = "8px";
  btnRow.style.margin = "8px 0";

  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear selection";
  clearBtn.addEventListener("click", () => onSelect(-1));

  const modeBtn = document.createElement("button");
  modeBtn.textContent = "Toggle mode (dim/only)";
  modeBtn.addEventListener("click", () => onSelect("__toggle_mode__"));

  btnRow.appendChild(clearBtn);
  btnRow.appendChild(modeBtn);
  legendEl.appendChild(btnRow);

  const h = document.createElement("h3");
  h.textContent = "Topics (cluster keywords)";
  legendEl.appendChild(h);

  for (const k of keys.slice(0, 30)) {
    const info = labels[String(k)];
    const r = paletteRGBA[k * 4 + 0], g = paletteRGBA[k * 4 + 1], b = paletteRGBA[k * 4 + 2];

    const row = document.createElement("div");
    row.className = "item";
    row.style.cursor = "pointer";
    row.addEventListener("click", () => onSelect(k));

    const sw = document.createElement("div");
    sw.className = "swatch";
    sw.style.background = rgbaToCss(r, g, b);

    const t = document.createElement("div");
    t.className = "label";

    const title = document.createElement("div");
    title.textContent = `#${k} ${info?.label || ""}`;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `size=${info?.size ?? 0}  keywords=${(info?.keywords || []).slice(0, 6).join(", ")}`;

    t.appendChild(title);
    t.appendChild(meta);

    row.appendChild(sw);
    row.appendChild(t);
    legendEl.appendChild(row);
  }

  legendEl.hidden = false;
}

export async function runWebGPUPoints({ canvas, hud, legendEl, metaUrl }) {
  if (!("gpu" in navigator)) throw new Error("WebGPU not supported in this browser.");

  // ---- load meta ----
  const meta = await (await fetch(metaUrl)).json();
  const n = meta.n;

  const pointsUrl = toPackedRelative(meta.files.points_xy);
  const idsUrl = toPackedRelative(meta.files.ids);
  const clusterUrl = toPackedRelative(meta.files.cluster);
  const labelUrl = toPackedRelative(meta.files.cluster_labels);

  // 현재는 n10000 seed 고정 파일을 쓰고 있으니 일단 그대로.
  // 추후 meta.files로 옮기는 걸 추천.
  const snippetUrl = "./packed/id2snippet_n10000_seed42.json";

  // ---- search state (A-3a: 매칭만 하이라이트/표시) ----
  let searchActive = 0;     // 0/1
  let searchOnly = 1;       // 1=only matched, 0=dim others
  let searchDimAlpha = 0.05;

  // mask is u32(0/1) per point
  let searchMask = new Uint32Array(n);

  hud.textContent = "loading buffers…";

  const [pointsBuf, idsBuf, clusterBuf, labelsJson, id2snippet] = await Promise.all([
    fetchArrayBuffer(pointsUrl),
    fetchArrayBuffer(idsUrl),
    fetchArrayBuffer(clusterUrl),
    fetch(labelUrl).then(r => r.json()),
    fetch(snippetUrl).then(r => r.json()),
  ]);

  const xy = new Float32Array(pointsBuf);
  const ids = new Uint32Array(idsBuf);
  const cluster16 = new Uint16Array(clusterBuf);

  if (xy.length !== n * 2) throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
  if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);
  if (cluster16.length !== n) throw new Error(`cluster length mismatch: ${cluster16.length} vs ${n}`);

  const numClusters = labelsJson.num_clusters ?? 0;
  const noiseBucket = labelsJson.noise_bucket ?? numClusters;

  const palette = makePalette(numClusters + 1);
  if (noiseBucket < numClusters + 1) {
    palette[noiseBucket * 4 + 0] = 160;
    palette[noiseBucket * 4 + 1] = 160;
    palette[noiseBucket * 4 + 2] = 160;
    palette[noiseBucket * 4 + 3] = 60;
  }

  // ---- cluster selection state ----
  let selectedCluster = -1; // -1 = none
  let selectMode = 1;       // 1=dim others, 2=only selected
  let dimAlpha = 0.08;

  renderLegend(legendEl, labelsJson, palette, (k) => {
    if (k === "__toggle_mode__") {
      selectMode = (selectMode === 1 ? 2 : 1);
      return;
    }
    selectedCluster = k;
  });

  // ---- pack cluster/palette to u32 ----
  const palCount = (numClusters ?? 0) + 1;

  const cluster32 = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const c = cluster16[i];
    cluster32[i] = (c < palCount) ? c : (palCount - 1);
  }

  const pal32 = new Uint32Array(palCount);
  for (let i = 0; i < palCount; i++) {
    const r = palette[i * 4 + 0], g = palette[i * 4 + 1], b = palette[i * 4 + 2], a = palette[i * 4 + 3];
    pal32[i] = (r) | (g << 8) | (b << 16) | (a << 24);
  }

  // ---- WebGPU init ----
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found.");
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  function configure() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      context.configure({ device, format, alphaMode: "premultiplied" });
    }
    return { w, h };
  }

  window.addEventListener("resize", () => configure());
  let { w: W, h: H } = configure();

  // ---- GPU buffers ----
  const xyGPU = device.createBuffer({
    size: xy.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(xyGPU, 0, xy);

  const clusterGPU = device.createBuffer({
    size: cluster32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(clusterGPU, 0, cluster32);

  const paletteGPU = device.createBuffer({
    size: pal32.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paletteGPU, 0, pal32);

  const searchGPU = device.createBuffer({
    size: searchMask.byteLength, // n * 4 bytes
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  searchMask.fill(0);
  device.queue.writeBuffer(searchGPU, 0, searchMask);

  // uniform: 64 bytes fixed
  const uniformGPU = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---- camera ----
  let camScale = 0.22;
  let camTx = 0.0;
  let camTy = 0.0;
  let pointSizePx = 3.0;

  // ---- search UI wiring ----
  const input = document.getElementById("searchInput");
  const btnClear = document.getElementById("searchClear");

  if (!input || !btnClear) {
    throw new Error("Missing #searchInput or #searchClear in HTML.");
  }

  function runSearch(q) {
    q = q.trim().toLowerCase();
    searchMask.fill(0);

    if (!q) {
      searchActive = 0;
      device.queue.writeBuffer(searchGPU, 0, searchMask);
      return;
    }

    searchActive = 1;

    for (let i = 0; i < n; i++) {
      const id = ids[i];
      const text = id2snippet[String(id)];
      if (text && text.toLowerCase().includes(q)) {
        searchMask[i] = 1;
      }
    }
    device.queue.writeBuffer(searchGPU, 0, searchMask);
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runSearch(input.value);
    if (e.key === "Escape") { input.value = ""; runSearch(""); }
  });

  btnClear.addEventListener("click", () => {
    input.value = "";
    runSearch("");
  });

  // ---- hover (CPU picking) ----
  let dragging = false;
  let lastX = 0, lastY = 0;

  function screenToWorld(px, py) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const xNdc = ((px - rect.left) * dpr / canvas.width) * 2 - 1;
    const yNdc = 1 - ((py - rect.top) * dpr / canvas.height) * 2;

    const wx = (xNdc - camTx) / camScale;
    const wy = (yNdc - camTy) / camScale;
    return [wx, wy];
  }

  function pickNearest(wx, wy) {
    let best = -1;
    let bestD2 = Infinity;
    for (let i = 0; i < n; i++) {
      const dx = xy[i * 2 + 0] - wx;
      const dy = xy[i * 2 + 1] - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = i; }
    }
    return [best, bestD2];
  }

  function hoverThresholdWorld() {
    const px = Math.max(2, pointSizePx);
    const ndcPerPx = 2.0 / W;
    const worldPerPx = ndcPerPx / camScale;
    return (px * worldPerPx) * 0.75;
  }

  const hudDefault = () =>
    `loaded: n=${n} | topics=${numClusters} (+noise) | drag: pan | wheel: zoom | +/-: size`;

  hud.textContent = hudDefault();

  canvas.addEventListener("mousemove", (e) => {
    if (dragging) return;

    const [wx, wy] = screenToWorld(e.clientX, e.clientY);
    const [idx, d2] = pickNearest(wx, wy);

    const th = hoverThresholdWorld();
    if (idx >= 0 && d2 < th * th) {
      const id = ids[idx];
      const c = cluster16[idx];
      const topic = labelsJson.labels?.[String(c)]?.label ?? `#${c}`;
      const snippet = id2snippet[String(id)] ?? "";
      hud.textContent = `id=${id} | topic=${topic}\n${snippet}`;
    } else {
      hud.textContent = hudDefault();
    }
  });

  // ---- pan/zoom ----
  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => dragging = false);
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;

    const dpr = Math.max(1, window.devicePixelRatio || 1);
    camTx += (dx * dpr) * (2 / canvas.width);
    camTy -= (dy * dpr) * (2 / canvas.height);
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.001);
    camScale = clamp(camScale * k, 0.01, 10.0);
  }, { passive: false });

  window.addEventListener("keydown", (e) => {
    if (e.key === "+" || e.key === "=") pointSizePx = clamp(pointSizePx + 0.5, 1.0, 20.0);
    if (e.key === "-" || e.key === "_") pointSizePx = clamp(pointSizePx - 0.5, 1.0, 20.0);
  });

  // ---- pipeline ----
  const shaderCode = await fetch("./shaders.wgsl").then(res => res.text());
  const shader = device.createShaderModule({ code: shaderCode });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shader, entryPoint: "vs" },
    fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformGPU } },
      { binding: 1, resource: { buffer: xyGPU } },
      { binding: 2, resource: { buffer: clusterGPU } },
      { binding: 3, resource: { buffer: paletteGPU } },
      { binding: 4, resource: { buffer: searchGPU } },
    ],
  });

  function writeUniform() {
    const buf = new ArrayBuffer(64);
    const dv = new DataView(buf);

    // f32
    dv.setFloat32(0, camScale, true);
    dv.setFloat32(4, camTx, true);
    dv.setFloat32(8, camTy, true);
    dv.setFloat32(12, pointSizePx, true);
    dv.setFloat32(16, 2.0 / W, true); // invW2
    dv.setFloat32(20, 2.0 / H, true); // invH2

    // i32
    dv.setInt32(24, selectedCluster, true);
    dv.setInt32(28, selectMode, true);
    dv.setInt32(32, searchActive, true);
    dv.setInt32(36, searchOnly, true);

    // f32
    dv.setFloat32(40, dimAlpha, true);
    dv.setFloat32(44, searchDimAlpha, true);

    // 48..63 padding zeros
    device.queue.writeBuffer(uniformGPU, 0, buf);
  }

  function frame() {
    ({ w: W, h: H } = configure());

    // 반드시 여기서 uniform을 매 프레임 업데이트
    writeUniform();

    const encoder = device.createCommandEncoder();
    const view = context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, n, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}