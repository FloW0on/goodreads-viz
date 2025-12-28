async function fetchArrayBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.arrayBuffer();
}
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.text();
}

function makeOrtho(scale, tx, ty) {
  return { scale, tx, ty };
}

function worldToScreen(x, y, cam, w, h) {
  const cx = x * cam.scale + cam.tx;
  const cy = y * cam.scale + cam.ty;
  const sx = (cx * 0.5 + 0.5) * w;
  const sy = (1.0 - (cy * 0.5 + 0.5)) * h;
  return { sx, sy };
}

function fmt(v, digits = 2) {
  if (v === null || v === undefined) return "—";
  if (Number.isFinite(v)) return v.toFixed(digits);
  return "—";
}

const PALETTE = [
  [0.121, 0.466, 0.705],
  [1.000, 0.498, 0.054],
  [0.172, 0.627, 0.172],
  [0.839, 0.153, 0.157],
  [0.580, 0.404, 0.741],
  [0.549, 0.337, 0.294],
  [0.890, 0.467, 0.761],
  [0.498, 0.498, 0.498],
  [0.737, 0.741, 0.133],
  [0.090, 0.745, 0.811],
];

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

function binIndex(v, bins) {
  for (let i = 0; i < bins.length; i++) if (v <= bins[i]) return i;
  return bins.length;
}

function buildColors(extra, mode) {
  const n = extra.length;
  const colors = new Float32Array(n * 3);

  if (mode === "rating") {
    for (let i = 0; i < n; i++) {
      const r = extra[i].average_rating;
      const t = r == null ? 0 : clamp01(r / 5.0);
      const idx = Math.min(PALETTE.length - 1, Math.floor(t * PALETTE.length));
      const [R, G, B] = PALETTE[idx];
      colors[i * 3 + 0] = R;
      colors[i * 3 + 1] = G;
      colors[i * 3 + 2] = B;
    }
    return colors;
  }

  if (mode === "pages") {
    for (let i = 0; i < n; i++) {
      const p = extra[i].num_pages;
      const t = p == null ? 0 : clamp01(Math.log1p(p) / Math.log1p(1200));
      const idx = Math.min(PALETTE.length - 1, Math.floor(t * PALETTE.length));
      const [R, G, B] = PALETTE[idx];
      colors[i * 3 + 0] = R;
      colors[i * 3 + 1] = G;
      colors[i * 3 + 2] = B;
    }
    return colors;
  }

  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const y = extra[i].publication_year;
    if (y != null && y > 0) {
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minY) || !Number.isFinite(maxY) || minY === maxY) {
    minY = 1900; maxY = 2025;
  }
  for (let i = 0; i < n; i++) {
    const y = extra[i].publication_year;
    const t = (y == null || y <= 0) ? 0 : clamp01((y - minY) / (maxY - minY));
    const idx = Math.min(PALETTE.length - 1, Math.floor(t * PALETTE.length));
    const [R, G, B] = PALETTE[idx];
    colors[i * 3 + 0] = R;
    colors[i * 3 + 1] = G;
    colors[i * 3 + 2] = B;
  }
  return colors;
}

export async function runWebGPUPoints({ canvas, hud, metaUrl, metaExtraUrl, colorMode = "year" }) {
  if (!("gpu" in navigator)) throw new Error("WebGPU not supported in this browser.");

  const meta = await (await fetch(metaUrl)).json();
  const pointsUrl = meta.files.points_xy.replace(/^.*\/packed\//, "./packed/");
  const idsUrl = meta.files.ids.replace(/^.*\/packed\//, "./packed/");

  hud.textContent = "loading points/ids…";
  const [pointsBuf, idsBuf] = await Promise.all([
    fetchArrayBuffer(pointsUrl),
    fetchArrayBuffer(idsUrl),
  ]);

  let extra = null;
  if (metaExtraUrl) {
    hud.textContent = "loading meta (jsonl)…";
    const txt = await fetchText(metaExtraUrl);
    const lines = txt.split("\n").filter(Boolean);
    extra = new Array(lines.length);
    for (let i = 0; i < lines.length; i++) extra[i] = JSON.parse(lines[i]);
  } else {
    throw new Error("metaExtraUrl is required for colored points.");
  }

  const n = meta.n;
  const xy = new Float32Array(pointsBuf);
  const ids = new Uint32Array(idsBuf);

  if (xy.length !== n * 2) throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
  if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);
  if (extra.length !== n) throw new Error(`meta jsonl lines mismatch: ${extra.length} vs ${n}`);

  const colorsCPU = buildColors(extra, colorMode);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found.");
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  function resize() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      context.configure({ device, format, alphaMode: "premultiplied" });
    }
  }
  resize();
  window.addEventListener("resize", resize);

  const pointsGPU = device.createBuffer({
    size: pointsBuf.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(pointsGPU, 0, pointsBuf);

  const colorsGPU = device.createBuffer({
    size: colorsCPU.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(colorsGPU, 0, colorsCPU);

  const uniformGPU = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shader = device.createShaderModule({
    code: `
      struct U {
        scale: f32,
        tx: f32,
        ty: f32,
        _pad: f32,
      };
      @group(0) @binding(0) var<uniform> u: U;

      struct VSOut {
        @builtin(position) pos: vec4<f32>,
        @location(0) col: vec3<f32>,
      };

      @vertex
      fn vs(
        @location(0) aPos: vec2<f32>,
        @location(1) aCol: vec3<f32>
      ) -> VSOut {
        var o: VSOut;
        let x = aPos.x * u.scale + u.tx;
        let y = aPos.y * u.scale + u.ty;
        o.pos = vec4<f32>(x, y, 0.0, 1.0);
        o.col = aCol;
        return o;
      }

      @fragment
      fn fs(i: VSOut) -> @location(0) vec4<f32> {
        return vec4<f32>(i.col, 1.0);
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vs",
      buffers: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
        {
          arrayStride: 12,
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
        },
      ],
    },
    fragment: {
      module: shader,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: { topology: "point-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformGPU } }],
  });

  let cam = makeOrtho(0.22, 0.0, 0.0);

  let dragging = false;
  let lastX = 0, lastY = 0;

  canvas.addEventListener("mousedown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => dragging = false);
  window.addEventListener("mousemove", (e) => {
    if (dragging) {
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;

      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = canvas.width, h = canvas.height;
      cam.tx += (dx * dpr) * (2 / w);
      cam.ty -= (dy * dpr) * (2 / h);
    }
    updateHover(e);
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.001);
    cam.scale *= k;
    cam.scale = Math.min(Math.max(cam.scale, 0.01), 10.0);
  }, { passive: false });

  let hoverIndex = -1;
  let pinnedIndex = -1;

  function describe(i) {
    const id = ids[i];
    const m = extra[i];
    const year = m.publication_year ?? "—";
    const pages = m.num_pages ?? "—";
    const rating = m.average_rating ?? null;
    return `#${i}  id=${id}
${m.title}
rating=${rating === null ? "—" : fmt(rating, 2)}  year=${year}  pages=${pages}`;
  }

  function updateHud() {
    const i = pinnedIndex >= 0 ? pinnedIndex : hoverIndex;
    if (i < 0) {
      hud.textContent = `loaded: n=${n}
color by: ${colorMode}
(hover a point; click to pin)
Drag: pan, Wheel: zoom`;
    } else {
      const mode = pinnedIndex >= 0 ? "PINNED" : "HOVER";
      hud.textContent = `${mode}  (color by: ${colorMode})
${describe(i)}

(Drag: pan, Wheel: zoom, Click: pin/unpin)`;
    }
  }

  function updateHover(e) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const mx = (e.clientX - rect.left) * dpr;
    const my = (e.clientY - rect.top) * dpr;
    const w = canvas.width, h = canvas.height;

    const R = 10 * dpr;
    const R2 = R * R;

    let best = -1;
    let bestD2 = R2;

    for (let i = 0; i < n; i++) {
      const x = xy[i * 2], y = xy[i * 2 + 1];
      const { sx, sy } = worldToScreen(x, y, cam, w, h);
      const dx = sx - mx;
      const dy = sy - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        best = i;
      }
    }
    hoverIndex = best;
    updateHud();
  }

  canvas.addEventListener("click", () => {
    if (hoverIndex >= 0) pinnedIndex = (pinnedIndex === hoverIndex) ? -1 : hoverIndex;
    else pinnedIndex = -1;
    updateHud();
  });

  function frame() {
    resize();

    const u = new Float32Array([cam.scale, cam.tx, cam.ty, 0.0]);
    device.queue.writeBuffer(uniformGPU, 0, u.buffer);

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
    pass.setVertexBuffer(0, pointsGPU);
    pass.setVertexBuffer(1, colorsGPU);
    pass.draw(n, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  updateHud();
  requestAnimationFrame(frame);
}