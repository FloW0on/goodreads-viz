async function fetchArrayBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.arrayBuffer();
}

function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

function toPackedRelative(p) {
  return String(p).replace(/^.*\/packed\//, "./packed/");
}

function makePalette(n) {
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const h = (i * 137.508) % 360; 
    const s = 0.70;
    const l = 0.50;
    const [r,g,b] = hslToRgb(h/360, s, l);
    out[i*4+0] = r;
    out[i*4+1] = g;
    out[i*4+2] = b;
    out[i*4+3] = 255;
  }
  return out;
}

function hslToRgb(h, s, l) {
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : (l + s - l*s);
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [Math.round(r*255), Math.round(g*255), Math.round(b*255)];
}

function rgbaToCss(r,g,b,a=255){ return `rgba(${r},${g},${b},${a/255})`; }

function renderLegend(legendEl, labelsJson, paletteRGBA) {
  const labels = labelsJson.labels || {};
  const keys = Object.keys(labels).map(k => parseInt(k,10)).sort((a,b)=>a-b);

  keys.sort((a,b) => (labels[b]?.size||0) - (labels[a]?.size||0));

  legendEl.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = "Topics (cluster keywords)";
  legendEl.appendChild(h);

  for (const k of keys.slice(0, 30)) { 
    const info = labels[String(k)];
    const r = paletteRGBA[k*4+0], g = paletteRGBA[k*4+1], b = paletteRGBA[k*4+2];
    const row = document.createElement("div");
    row.className = "item";

    const sw = document.createElement("div");
    sw.className = "swatch";
    sw.style.background = rgbaToCss(r,g,b);

    const t = document.createElement("div");
    t.className = "label";
    const title = document.createElement("div");
    title.textContent = `#${k} ${info?.label || ""}`;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `size=${info?.size ?? 0}  keywords=${(info?.keywords||[]).slice(0,6).join(", ")}`;

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

  const meta = await (await fetch(metaUrl)).json();
  const n = meta.n;

  const pointsUrl  = toPackedRelative(meta.files.points_xy);
  const idsUrl     = toPackedRelative(meta.files.ids);

  const clusterUrl = toPackedRelative(meta.files.cluster);
  const labelUrl   = toPackedRelative(meta.files.cluster_labels);

  hud.textContent = "loading buffers…";

  const [pointsBuf, idsBuf, clusterBuf, labelsJson] = await Promise.all([
    fetchArrayBuffer(pointsUrl),
    fetchArrayBuffer(idsUrl),
    fetchArrayBuffer(clusterUrl),
    (await fetch(labelUrl)).json(),
  ]);

  const xy = new Float32Array(pointsBuf);
  const ids = new Uint32Array(idsBuf);
  const cluster = new Uint16Array(clusterBuf);

  if (xy.length !== n*2) throw new Error(`xy length mismatch: ${xy.length} vs ${n*2}`);
  if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);
  if (cluster.length !== n) throw new Error(`cluster length mismatch: ${cluster.length} vs ${n}`);

  const numClusters = labelsJson.num_clusters ?? 0;
  const noiseBucket = labelsJson.noise_bucket ?? numClusters;

  const palette = makePalette(numClusters + 1);
  if (noiseBucket < numClusters + 1) {
    palette[noiseBucket*4+0] = 160;
    palette[noiseBucket*4+1] = 160;
    palette[noiseBucket*4+2] = 160;
    palette[noiseBucket*4+3] = 255;
  }

  renderLegend(legendEl, labelsJson, palette);

  hud.textContent = `loaded: n=${n} | topics=${numClusters} (+noise) | drag: pan | wheel: zoom | +/-: size`;

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

  const xyGPU = device.createBuffer({
    size: xy.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(xyGPU, 0, xy);

  const clusterGPU = device.createBuffer({
    size: cluster.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(clusterGPU, 0, cluster);

  const paletteGPU = device.createBuffer({
    size: palette.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paletteGPU, 0, palette);

  const uniformGPU = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let camScale = 0.22;
  let camTx = 0.0;
  let camTy = 0.0;
  let pointSizePx = 3.0;

  let dragging = false;
  let lastX = 0, lastY = 0;

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

  const shader = device.createShaderModule({
    code: `
      struct U {
        scale: f32,
        tx: f32,
        ty: f32,
        pointSizePx: f32,
        invW2: f32,   // 2/width
        invH2: f32,   // 2/height
        _pad0: f32,
        _pad1: f32,
      };
      @group(0) @binding(0) var<uniform> u: U;

      @group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
      @group(0) @binding(2) var<storage, read> cluster: array<u32>;
      @group(0) @binding(3) var<storage, read> palette: array<u32>; // RGBA8 packed as 4 bytes in u32? (we'll read as bytes via shifts)

      fn unpackRGBA8(p: u32) -> vec4<f32> {
        let r: f32 = f32((p      ) & 255u) / 255.0;
        let g: f32 = f32((p >>  8) & 255u) / 255.0;
        let b: f32 = f32((p >> 16) & 255u) / 255.0;
        let a: f32 = f32((p >> 24) & 255u) / 255.0;
        return vec4<f32>(r,g,b,a);
      }

      struct VSOut {
        @builtin(position) pos: vec4<f32>,
        @location(0) color: vec4<f32>,
      };

      // 6 vertices -> two triangles for a quad
      fn corner(vid: u32) -> vec2<f32> {
        // (x,y) in {-1,+1}
        // tri1: (-1,-1),(+1,-1),(+1,+1)
        // tri2: (-1,-1),(+1,+1),(-1,+1)
        if (vid == 0u) { return vec2<f32>(-1.0, -1.0); }
        if (vid == 1u) { return vec2<f32>( 1.0, -1.0); }
        if (vid == 2u) { return vec2<f32>( 1.0,  1.0); }
        if (vid == 3u) { return vec2<f32>(-1.0, -1.0); }
        if (vid == 4u) { return vec2<f32>( 1.0,  1.0); }
        return vec2<f32>(-1.0,  1.0);
      }

      @vertex
      fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VSOut {
        var o: VSOut;

        let p = xy[i];
        let x = p.x * u.scale + u.tx;
        let y = p.y * u.scale + u.ty;

        let c = corner(v % 6u);
        let dx = c.x * (u.pointSizePx * 0.5) * u.invW2;
        let dy = c.y * (u.pointSizePx * 0.5) * u.invH2;

        o.pos = vec4<f32>(x + dx, y + dy, 0.0, 1.0);

        let cid = cluster[i];
        o.color = unpackRGBA8(palette[cid]);
        return o;
      }

      @fragment
      fn fs(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
        return color;
      }
    `,
  });

  const cluster32 = new Uint32Array(n);
  for (let i=0; i<n; i++) cluster32[i] = cluster[i];
  device.queue.writeBuffer(clusterGPU, 0, cluster32);

  const pal32 = new Uint32Array((numClusters + 1));
  for (let i=0; i<(numClusters+1); i++) {
    const r = palette[i*4+0], g = palette[i*4+1], b = palette[i*4+2], a = palette[i*4+3];
    pal32[i] = (r) | (g<<8) | (b<<16) | (a<<24);
  }
  device.queue.writeBuffer(paletteGPU, 0, pal32);

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
    ],
  });

  function frame() {
    ({ w: W, h: H } = configure());
    const invW2 = 2.0 / W;
    const invH2 = 2.0 / H;

    const u = new Float32Array([camScale, camTx, camTy, pointSizePx, invW2, invH2, 0, 0]);
    device.queue.writeBuffer(uniformGPU, 0, u);

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