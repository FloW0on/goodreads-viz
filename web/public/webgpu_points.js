async function fetchArrayBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.arrayBuffer();
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

export async function runWebGPUPoints({ canvas, hud, metaUrl }) {
  if (!("gpu" in navigator)) throw new Error("WebGPU not supported in this browser.");

  const meta = await (await fetch(metaUrl)).json();
  const pointsUrl = meta.files.points_xy.replace(/^.*\/packed\//, "./packed/");
  const idsUrl = meta.files.ids.replace(/^.*\/packed\//, "./packed/");

  const [pointsBuf, idsBuf] = await Promise.all([
    fetchArrayBuffer(pointsUrl),
    fetchArrayBuffer(idsUrl),
  ]);

  const n = meta.n;
  const xy = new Float32Array(pointsBuf);
  const ids = new Uint32Array(idsBuf);

  if (xy.length !== n * 2) throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
  if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);

  let minX = +Infinity, maxX = -Infinity, minY = +Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xy[i * 2];
    const y = xy[i * 2 + 1];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }

  hud.textContent = `loaded: n=${n}, xy=${meta.xy.dtype}, ids=${meta.ids.dtype}`;

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
      canvas.width = w;
      canvas.height = h;
      context.configure({ device, format, alphaMode: "premultiplied" });
    }
    return { w, h, dpr };
  }
  resize();
  window.addEventListener("resize", resize);

  const pointsGPU = device.createBuffer({
    size: pointsBuf.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "points_xy",
  });
  device.queue.writeBuffer(pointsGPU, 0, pointsBuf);

  const uniformGPU = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "uniforms",
  });

  let camScale = 1.0;
  let camTx = 0.0;
  let camTy = 0.0;
  let pointPx = 2.0;

  camScale = 0.95;

  let dragging = false;
  let lastX = 0, lastY = 0;

  canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => (dragging = false));
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    const { w, h, dpr } = resize();
    camTx += (dx * dpr) * (2 / w);
    camTy -= (dy * dpr) * (2 / h);
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const k = Math.exp(-e.deltaY * 0.001);
      camScale = clamp(camScale * k, 0.02, 20.0);
    },
    { passive: false }
  );

  const shader = device.createShaderModule({
    code: `
      struct U {
        minX: f32,
        maxX: f32,
        minY: f32,
        maxY: f32,
        scale: f32,
        tx: f32,
        ty: f32,
        pointPx: f32,
        canvasW: f32,
        canvasH: f32,
        _pad0: f32,
        _pad1: f32,
      };

      @group(0) @binding(0) var<uniform> u: U;
      @group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;

      // 6-vertex quad pattern (two triangles)
      fn corner(vid: u32) -> vec2<f32> {
        // (-1,-1),(+1,-1),(-1,+1),(-1,+1),(+1,-1),(+1,+1)
        switch(vid) {
          case 0u: { return vec2<f32>(-1.0, -1.0); }
          case 1u: { return vec2<f32>( 1.0, -1.0); }
          case 2u: { return vec2<f32>(-1.0,  1.0); }
          case 3u: { return vec2<f32>(-1.0,  1.0); }
          case 4u: { return vec2<f32>( 1.0, -1.0); }
          default: { return vec2<f32>( 1.0,  1.0); }
        }
      }

      struct VSOut {
        @builtin(position) pos: vec4<f32>,
      };

      @vertex
      fn vs(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
        var o: VSOut;

        let pointIndex: u32 = vertexIndex / 6u;
        let vid: u32 = vertexIndex % 6u;

        let p: vec2<f32> = xy[pointIndex];

        // Normalize UMAP space to NDC around center
        let cx = 0.5 * (u.minX + u.maxX);
        let cy = 0.5 * (u.minY + u.maxY);
        let hx = 0.5 * (u.maxX - u.minX);
        let hy = 0.5 * (u.maxY - u.minY);
        let h  = max(hx, hy);
        let base = (p - vec2<f32>(cx, cy)) / max(h, 1e-6); // roughly in [-1,1]

        // camera in NDC
        let ndcCenter = base * u.scale + vec2<f32>(u.tx, u.ty);

        // point size in pixels -> NDC offset
        let pxToNdc = vec2<f32>(2.0 / u.canvasW, 2.0 / u.canvasH);
        let halfSize = u.pointPx * 0.5;
        let off = corner(vid) * halfSize * pxToNdc;

        let ndc = ndcCenter + off;

        o.pos = vec4<f32>(ndc, 0.0, 1.0);
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
    entries: [
      { binding: 0, resource: { buffer: uniformGPU } },
      { binding: 1, resource: { buffer: pointsGPU } },
    ],
  });

  function frame() {
    const { w, h } = resize();

    const u = new Float32Array([
      minX, maxX, minY, maxY,
      camScale, camTx, camTy, pointPx,
      w, h, 0, 0,
    ]);
    device.queue.writeBuffer(uniformGPU, 0, u);

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

    pass.draw(n * 6, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  window.addEventListener("keydown", (e) => {
    if (e.key === "=" || e.key === "+") pointPx = clamp(pointPx + 0.5, 0.5, 20.0);
    if (e.key === "-" || e.key === "_") pointPx = clamp(pointPx - 0.5, 0.5, 20.0);
  });

  console.log("[webgpu] ok", { n, minX, maxX, minY, maxY });
}