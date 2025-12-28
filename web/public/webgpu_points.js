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

export async function runWebGPUPoints({ canvas, hud, metaUrl, metaExtraUrl }) {
  if (!("gpu" in navigator)) throw new Error("WebGPU not supported in this browser.");

  const meta = await (await fetch(metaUrl)).json();
  const pointsUrl = meta.files.points_xy.replace(/^.*\/packed\//, "./packed/");
  const idsUrl    = meta.files.ids.replace(/^.*\/packed\//, "./packed/");

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
    for (let i = 0; i < lines.length; i++) {
      extra[i] = JSON.parse(lines[i]);
    }
  }

  const n = meta.n;
  const xy = new Float32Array(pointsBuf);
  const ids = new Uint32Array(idsBuf);

  if (xy.length !== n * 2) throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
  if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);
  if (extra && extra.length !== n) throw new Error(`meta jsonl lines mismatch: ${extra.length} vs ${n}`);

  hud.textContent = `loaded: n=${n}, xy=float32, ids=uint32`;

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
      };

      @vertex
      fn vs(@location(0) aPos: vec2<f32>) -> VSOut {
        var o: VSOut;
        let x = aPos.x * u.scale + u.tx;
        let y = aPos.y * u.scale + u.ty;
        o.pos = vec4<f32>(x, y, 0.0, 1.0);
        return o;
      }

      @fragment
      fn fs() -> @location(0) vec4<f32> {
        return vec4<f32>(0.95, 0.95, 0.95, 1.0);
      }
    `,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vs",
      buffers: [{
        arrayStride: 8,
        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
      }],
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
    if (!extra) return `id=${id}`;
    const m = extra[i];
    const year = m.publication_year ?? "—";
    const pages = m.num_pages ?? "—";
    const rating = m.average_rating ?? null;
    return `#${i}  id=${id}\n${m.title}\nrating=${rating === null ? "—" : fmt(rating, 2)}  year=${year}  pages=${pages}`;
  }

  function updateHud() {
    const i = pinnedIndex >= 0 ? pinnedIndex : hoverIndex;
    if (i < 0) {
      hud.textContent = `loaded: n=${n}  (hover a point; click to pin)`;
    } else {
      const mode = pinnedIndex >= 0 ? "PINNED" : "HOVER";
      hud.textContent = `${mode}\n${describe(i)}\n\n(Drag: pan, Wheel: zoom, Click: pin/unpin)`;
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

  canvas.addEventListener("click", (e) => {
    if (hoverIndex >= 0) {
      pinnedIndex = (pinnedIndex === hoverIndex) ? -1 : hoverIndex;
    } else {
      pinnedIndex = -1;
    }
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
        clearValue: { r: 0.03, g: 0.03, b: 0.03, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, pointsGPU);
    pass.draw(n, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  updateHud();
  requestAnimationFrame(frame);
}