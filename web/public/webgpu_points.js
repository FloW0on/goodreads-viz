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

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

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

function makeCam(scale = 0.22, tx = 0, ty = 0) {
  return { scale, tx, ty };
}

function worldToScreen(x, y, cam, w, h) {
  const cx = x * cam.scale + cam.tx;
  const cy = y * cam.scale + cam.ty;
  const sx = (cx * 0.5 + 0.5) * w;
  const sy = (1.0 - (cy * 0.5 + 0.5)) * h;
  return { sx, sy };
}

export async function runWebGPUPoints({ canvas, hud, metaUrl, metaExtraUrl, colorMode = "year" }) {
  if (!("gpu" in navigator)) throw new Error("WebGPU not supported in this browser.");

  const meta = await (await fetch(metaUrl)).json();
  const pointsUrl = meta.files.points_xy.replace(/^.*\/packed\//, "./packed/");
  const idsUrl    = meta.files.ids.replace(/^.*\/packed\//, "./packed/");

  hud.textContent = "loading points/ids/meta…";
  const [pointsBuf, idsBuf, metaJsonlText] = await Promise.all([
    fetchArrayBuffer(pointsUrl),
    fetchArrayBuffer(idsUrl),
    fetchText(metaExtraUrl),
  ]);

  const n = meta.n;
  const xy = new Float32Array(pointsBuf);
  const ids = new Uint32Array(idsBuf);

  const lines = metaJsonlText.split("\n").filter(Boolean);
  if (lines.length !== n) throw new Error(`meta jsonl lines mismatch: ${lines.length} vs ${n}`);

  const extra = new Array(n);
  for (let i = 0; i < n; i++) extra[i] = JSON.parse(lines[i]);

  if (xy.length !== n * 2) throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
  if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);

  const colorsCPU = buildColors(extra, colorMode);

  const inst = new Float32Array(n * 5);
  for (let i = 0; i < n; i++) {
    inst[i * 5 + 0] = xy[i * 2 + 0];
    inst[i * 5 + 1] = xy[i * 2 + 1];
    inst[i * 5 + 2] = colorsCPU[i * 3 + 0];
    inst[i * 5 + 3] = colorsCPU[i * 3 + 1];
    inst[i * 5 + 4] = colorsCPU[i * 3 + 2];
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found.");
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  let cam = makeCam(0.22, 0, 0);
  let pointSizePx = 3.0;

  function configure() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      context.configure({ device, format, alphaMode: "premultiplied" });
    }
    return { w, h, dpr };
  }
  let vp = configure();
  window.addEventListener("resize", () => { vp = configure(); });

  const instGPU = device.createBuffer({
    size: inst.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(instGPU, 0, inst);

  const quad = new Float32Array([
    -1, -1,
     1, -1,
     1,  1,
    -1, -1,
     1,  1,
    -1,  1,
  ]);
  const quadGPU = device.createBuffer({
    size: quad.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadGPU, 0, quad);

  const uniformGPU = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shader = device.createShaderModule({
    code: `
      struct U {
        scale: f32,
        tx: f32,
        ty: f32,
        pointSizePx: f32,
        viewportW: f32,
        viewportH: f32,
        _pad0: f32,
        _pad1: f32,
      };
      @group(0) @binding(0) var<uniform> u: U;

      struct VSOut {
        @builtin(position) pos: vec4<f32>,
        @location(0) col: vec3<f32>,
        @location(1) local: vec2<f32>,
      };

      // per-vertex: quad corner (-1..1)
      // per-instance: center xy + color rgb
      @vertex
      fn vs(
        @location(0) aCorner: vec2<f32>,
        @location(1) aCenter: vec2<f32>,
        @location(2) aCol: vec3<f32>,
      ) -> VSOut {
        var o: VSOut;

        // world -> clip
        let cx = aCenter.x * u.scale + u.tx;
        let cy = aCenter.y * u.scale + u.ty;

        // corner offset in clip space (pixel -> clip)
        let dx = (aCorner.x * u.pointSizePx) * (2.0 / u.viewportW);
        let dy = (aCorner.y * u.pointSizePx) * (2.0 / u.viewportH);

        o.pos = vec4<f32>(cx + dx, cy + dy, 0.0, 1.0);
        o.col = aCol;
        o.local = aCorner; // for round mask
        return o;
      }

      @fragment
      fn fs(i: VSOut) -> @location(0) vec4<f32> {
        let r2 = dot(i.local, i.local);
        if (r2 > 1.0) { discard; }
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
          stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
        {
          arrayStride: 20,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 1, offset: 0,  format: "float32x2" },
            { shaderLocation: 2, offset: 8,  format: "float32x3" },
          ],
        },
      ],
    },
    fragment: {
      module: shader,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformGPU } }],
  });

  let dragging = false;
  let lastX = 0, lastY = 0;

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => dragging = false);

  window.addEventListener("mousemove", (e) => {
    if (dragging) {
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;

      const { w, h, dpr } = vp;
      cam.tx += (dx * dpr) * (2 / w);
      cam.ty -= (dy * dpr) * (2 / h);
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.001);
    cam.scale *= k;
    cam.scale = Math.min(Math.max(cam.scale, 0.01), 10.0);
  }, { passive: false });

  window.addEventListener("keydown", (e) => {
    if (e.key === "+" || e.key === "=") pointSizePx = Math.min(20, pointSizePx + 1);
    if (e.key === "-" || e.key === "_") pointSizePx = Math.max(1, pointSizePx - 1);
  });

  hud.textContent =
`loaded: n=${n}
color by: ${colorMode}
drag: pan, wheel: zoom
+/- : point size (${pointSizePx}px)`;

  function frame() {
    vp = configure();
    const { w, h } = vp;

    const u = new Float32Array([cam.scale, cam.tx, cam.ty, pointSizePx, w, h, 0, 0]);
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

    pass.setVertexBuffer(0, quadGPU);
    pass.setVertexBuffer(1, instGPU);

    pass.draw(6, n, 0, 0);

    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}