// core/renderer.js
// WebGPU 초기화 및 렌더링 (DDC 지원)

// WebGPU 기본 텍스처 크기 제한
const MAX_TEXTURE_SIZE = 8192;

let resizeTimeout = null;

/**
 * WebGPU 렌더러 클래스
 */
export class Renderer {
  constructor(canvas, ctx) {
    this.canvas = canvas;
    this.ctx = ctx;
    this.bus = ctx.bus;
    this.services = ctx.services;
    this.device = null;
    this.context = null;
    this.format = null;
    this.pipeline = null;
    this.bindGroup = null;

    // picking 리소스 슬롯
    this.data = null;
    this.pickPipeline = null;
    this.pickBindGroup = null;
    this.pickTexture = null;
    this.pickTextureView = null;
    this.pickReadback = null;
    this.pickReadbackSize = 256;
    this.pickW = 0;
    this.pickH = 0;

    // GPU 버퍼들
    this.buffers = {
      xy: null,
      ddc: null,           // DDC 분류 (색상용)
      ddcPalette: null,    // DDC 11색 팔레트
      cluster: null,       // 클러스터 (강조용)
      search: null,
      uniform: null,
    };

    // 데이터 정보
    this.pointCount = 0;
    this.drawCount = 0; 

    // 캔버스 크기
    this.width = 0;
    this.height = 0;

    this._pickBusy = false;
    this._pickSeq = 0;
    // ---- PICK PROFILING ----
    this._pickStats = {
      cap: 120,
      n: 0,
      head: 0,
      dt: new Float32Array(120),       // 전체 querySelection(ms)
      dtSubmit: new Float32Array(120), // onSubmittedWorkDone(ms)
      dtMap: new Float32Array(120),    // mapAsync~unmap(ms)
      hit: new Uint8Array(120),        // 1=hit, 0=miss
      skipped: 0,                      // _pickBusy로 스킵
      calls: 0,                        // querySelection 호출수
    };
  }

  /**
   * WebGPU 초기화
   */
  async init() {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU not supported in this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No GPU adapter found.");
    }

    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();

    // 초기 설정
    this.configure();

    // 리사이즈 이벤트
    window.addEventListener("resize", () => {clearTimeout(resizeTimeout); resizeTimeout = setTimeout(() => this.configure(), 100)});

    // 셰이더 로드 및 파이프라인 생성
    await this._createPipeline();
    await this._createPickPipeline();
  }

  /**
   * 캔버스 설정 (텍스처 크기 제한 포함)
   */
  configure() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    
    // 텍스처 크기 제한 적용
    let w = Math.floor(this.canvas.clientWidth * dpr);
    let h = Math.floor(this.canvas.clientHeight * dpr);
    
    // 최대 크기 제한 (WebGPU 기본 제한: 8192)
    if (w > MAX_TEXTURE_SIZE) {
      w = MAX_TEXTURE_SIZE;
    }
    if (h > MAX_TEXTURE_SIZE) {
      h = MAX_TEXTURE_SIZE;
    }
    
    // 최소 크기 보장
    w = Math.max(1, w);
    h = Math.max(1, h);

    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: "premultiplied",
      });
    }

    this.width = w;
    this.height = h;

    if(this.device){
      this._ensurePickTargets();
    }

    return { width: w, height: h };
  }

  /**
   * 데이터 설정
   * @param {Object} data - DataLoader에서 반환된 데이터
   */
  setData(data) {
    this.data = data;

      // 1) xy 정규화
    this.data.xy = (data.xy instanceof Float32Array) ? data.xy : new Float32Array(data.xy);

    // 2) ids 정규화 (querySelection에서 필요)
    if (!this.data.ids32) {
      const src = data.ids32 ?? data.ids; // dataLoader가 ids로 주는 경우 대비
      if (!src) throw new Error("setData: ids/ids32 missing in data");
      this.data.ids32 = (src instanceof Uint32Array) ? src : new Uint32Array(src);
    }

    // 3) ddc32 정규화 (shader/buffer에서 필요)
    if (!this.data.ddc32) {
      const src = data.ddc32 ?? data.ddc16 ?? data.ddc;
      if (!src) throw new Error("setData: ddc16/ddc32 missing in data");
      this.data.ddc32 = (src instanceof Uint32Array) ? src : new Uint32Array(src);
    }

    // 4) cluster32 정규화 (shader/buffer에서 필요)
    const clusterSrc =
       data.clusterView16 ?? data.cluster32 ?? data.cluster16 ?? data.cluster;
     if (!clusterSrc) throw new Error("setData: cluster missing in data");
     this.data.cluster32 =
       (clusterSrc instanceof Uint32Array) ? clusterSrc : new Uint32Array(clusterSrc);

    // 5) 팔레트도 안전하게 (이미 있을 가능성이 큼)
    if (this.data.ddcPalette32 && !(this.data.ddcPalette32 instanceof Uint32Array)) {
      this.data.ddcPalette32 = new Uint32Array(this.data.ddcPalette32);
    }

    this.pointCount = data.n;
    this.drawCount = this.pointCount; 

    // XY 버퍼
    this.buffers.xy = this.device.createBuffer({
      size: this.data.xy.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.xy, 0, this.data.xy);

    // DDC 버퍼 (색상용)
    this.buffers.ddc = this.device.createBuffer({
      size: this.data.ddc32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.ddc, 0, this.data.ddc32);

    // DDC 팔레트 버퍼 (11색)
    this.buffers.ddcPalette = this.device.createBuffer({
      size: this.data.ddcPalette32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.ddcPalette, 0, this.data.ddcPalette32);

    // 클러스터 버퍼 (강조용)
    this.buffers.cluster = this.device.createBuffer({
      size: this.data.cluster32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.cluster, 0, this.data.cluster32);

    // 검색 마스크 버퍼
    const searchMask = new Uint32Array(data.n);
    searchMask.fill(0);
    this.buffers.search = this.device.createBuffer({
      size: searchMask.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.search, 0, searchMask);

    // Uniform 버퍼 (64 bytes)
    this.buffers.uniform = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 바인드 그룹 생성
    this._createBindGroup();
    this._createPickBindGroup();

    this.services.querySelection = this.querySelection.bind(this);
    this.services.getPickProfile = this.getPickProfile.bind(this);
    this._ensurePickTargets();
    this._ensurePickReadbackBuffer();
  }

  setDrawCount(n) {
    if (!Number.isFinite(n)) return;
    const clamped = Math.max(0, Math.min(n | 0, this.pointCount));
    if (clamped === this.drawCount) return;
    this.drawCount = clamped;
  }

  /**
   * 검색 마스크 업데이트
   * @param {Uint32Array} mask - 검색 마스크
   */
  updateSearchMask(mask) {
    if (this.buffers.search) {
      this.device.queue.writeBuffer(this.buffers.search, 0, mask);
    }
  }

  /**
   * Uniform 업데이트
   * @param {Object} state - 렌더링 상태
   */
  updateUniforms(state) {

    const buf = new ArrayBuffer(64);
    const dv = new DataView(buf);

    // point size policy (zoom-aware)
    const basePx = state.pointSize ?? 3.0; // 리셋 시 크기
    const scaleRef = state.scaleRef ?? 0.22; // 리셋 기준 스케일에 맞춰 정규화
    const alpha = state.sizeAlpha ?? 0.7; // 감도

    let psRaw = basePx * Math.pow((state.camScale ?? 0.22) / scaleRef, alpha);

    const minPx = state.minPointPx ?? 2.0;
    const maxPx = state.maxPointPx ?? 20.0;
    let ps = Math.max(minPx, Math.min(psRaw, maxPx));

    // f32 값들
    dv.setFloat32(0, state.camScale, true);
    dv.setFloat32(4, state.camTx, true);
    dv.setFloat32(8, state.camTy, true);
    dv.setFloat32(12, ps, true); // pointSizePx 픽셀 단위 점 지름
    dv.setFloat32(16, 2.0 / this.width, true);  // invW2
    dv.setFloat32(20, 2.0 / this.height, true); // invH2

    // i32 값들
    dv.setInt32(24, state.selectedCluster, true);
    dv.setInt32(28, state.selectMode, true);
    dv.setInt32(32, state.searchActive, true);
    dv.setInt32(36, state.searchOnly, true);

    // f32 값들
    dv.setFloat32(40, state.dimAlpha, true);
    dv.setFloat32(44, state.searchDimAlpha, true);

    // i32: 선택된 DDC (-1 = 없음)
    dv.setInt32(48, state.selectedDdc ?? -1, true);

    // padding (52..55)
    dv.setFloat32(52, 0.0, true);

    // searchBoost (56..59)
    dv.setFloat32(56, state.searchBoost ?? 3.0, true);

    // tail padding (60..63)
    dv.setFloat32(60, 0.0, true);

    this.device.queue.writeBuffer(this.buffers.uniform, 0, buf);
  }

  /**
   * 렌더링 수행
   */
  render() {
    // this.configure();

    const encoder = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const inst = (this.drawCount > 0 ? this.drawCount : this.pointCount);
    pass.draw(6, inst, 0, 0);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * 캔버스 크기 반환
   */
  getSize() {
    return { width: this.width, height: this.height };
  }

  // ---- Private Methods ----

  async _createPipeline() {
    const shaderCode = await fetch("./shaders_ddc.wgsl").then(res => res.text());
    const shader = this.device.createShaderModule({ code: shaderCode });

    this.pipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module: shader, entryPoint: "vs" },
      fragment: {
        module: shader,
        entryPoint: "fs",
        targets: [{ format: this.format ,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  _createBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.uniform } },
        { binding: 1, resource: { buffer: this.buffers.xy } },
        { binding: 2, resource: { buffer: this.buffers.ddc } },
        { binding: 3, resource: { buffer: this.buffers.ddcPalette } },
        { binding: 4, resource: { buffer: this.buffers.cluster } },
        { binding: 5, resource: { buffer: this.buffers.search } },
      ],
    });
  }

    _createPickBindGroup() {
      if (!this.pickPipeline) return;
      if (!this.buffers.uniform) return; // buffers 준비 확인

      this.pickBindGroup = this.device.createBindGroup({
        layout: this.pickPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers.uniform } },
          { binding: 1, resource: { buffer: this.buffers.xy } },
          { binding: 2, resource: { buffer: this.buffers.ddc } },
          { binding: 3, resource: { buffer: this.buffers.ddcPalette } },
          { binding: 4, resource: { buffer: this.buffers.cluster } },
          { binding: 5, resource: { buffer: this.buffers.search } },
        ],
      });
    }

  _ensurePickTargets() {
    if (!this.device) return;
    if (this.pickTexture && this.pickW === this.width && this.pickH === this.height) return;

    this.pickW = this.width;
    this.pickH = this.height;

    this.pickTexture = this.device.createTexture({
      size: { width: this.width, height: this.height },
      format: "r32uint",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    this.pickTextureView = this.pickTexture.createView();
  }

  _ensurePickReadbackBuffer() {
    if (!this.device) return;
    if (this.pickReadback) return;

    this.pickReadback = this.device.createBuffer({
      size: this.pickReadbackSize, // 256 bytes
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  async _createPickPipeline() {
    const shaderCode = await fetch("./shaders_ddc.wgsl").then(res => res.text());
    const shader = this.device.createShaderModule({ code: shaderCode });

    this.pickPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module: shader, entryPoint: "vs" },
      fragment: {
        module: shader,
        entryPoint: "fs_pick",          
        targets: [{ format: "r32uint" }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.pickBindGroup = null;
  }

  _toCanvasPixel(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const x = Math.floor((clientX - rect.left) * dpr);
    const y = Math.floor((clientY - rect.top) * dpr);

    const cx = Math.max(0, Math.min(this.width - 1, x));
    const cy = Math.max(0, Math.min(this.height - 1, y));
    return { x: cx, y: cy };
  }

  _pushPickSample({ dt, dtSubmit, dtMap, hit }) {
    const s = this._pickStats;
    const i = s.head;
    s.dt[i] = dt;
    s.dtSubmit[i] = dtSubmit;
    s.dtMap[i] = dtMap;
    s.hit[i] = hit ? 1 : 0;

    s.head = (i + 1) % s.cap;
    s.n = Math.min(s.n + 1, s.cap);
  }

  _quantileFloat32(arr, n, q) {
    if (n <= 0) return 0;
    const tmp = new Array(n);
    for (let i = 0; i < n; i++) tmp[i] = arr[i];
    tmp.sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(n - 1, Math.floor((n - 1) * q)));
    return tmp[idx];
  }

  getPickProfile() {
    const s = this._pickStats;
    const n = s.n;

    // 최근 n개를 뽑아서 통계 계산
    const dt = new Float32Array(n);
    const dtSubmit = new Float32Array(n);
    const dtMap = new Float32Array(n);
    let hits = 0;

    for (let k = 0; k < n; k++) {
      const idx = (s.head - 1 - k + s.cap) % s.cap;
      dt[k] = s.dt[idx];
      dtSubmit[k] = s.dtSubmit[idx];
      dtMap[k] = s.dtMap[idx];
      hits += s.hit[idx];
    }

    const p50 = this._quantileFloat32(dt, n, 0.50);
    const p95 = this._quantileFloat32(dt, n, 0.95);
    const p50Submit = this._quantileFloat32(dtSubmit, n, 0.50);
    const p95Submit = this._quantileFloat32(dtSubmit, n, 0.95);
    const p50Map = this._quantileFloat32(dtMap, n, 0.50);
    const p95Map = this._quantileFloat32(dtMap, n, 0.95);

    const skipRate = s.skipped / Math.max(1, s.calls);

    return {
      n,
      calls: s.calls,
      skipped: s.skipped,
      skipRate,
      hitRate: (n > 0 ? hits / n : 0),
      p50, p95,
      p50Submit, p95Submit,
      p50Map, p95Map,
    };
  }


  async querySelection(clientX, clientY, meta = null) {
    if (!this.pickPipeline || !this.pickBindGroup) return [];
    if (!this.data) return [];

    // ---- PICK PROFILING: call/skip 카운트 ----
    this._pickStats.calls++;

    if (this._pickBusy) {
      this._pickStats.skipped++;
      return [];
    }
    this._pickBusy = true;

    const seq = ++this._pickSeq;

    // ---- PICK PROFILING: 타이머 ----
    const t0 = performance.now();
    let tSubmit0 = 0, tSubmit1 = 0;
    let tMap0 = 0, tMap1 = 0;
    const camScale = meta?.camScale; // 줌 스케일 기록

    try {
      
      this._ensurePickTargets();
      this._ensurePickReadbackBuffer();
      

      const { x, y } = this._toCanvasPixel(clientX, clientY);
      const encoder = this.device.createCommandEncoder();

      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.pickTextureView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      
      pass.setPipeline(this.pickPipeline);
      pass.setBindGroup(0, this.pickBindGroup);
      const inst = (this.drawCount > 0 ? this.drawCount : this.pointCount);
      pass.draw(6, inst, 0, 0);
      pass.end();

      encoder.copyTextureToBuffer(
        { texture: this.pickTexture, origin: { x, y } },
        { buffer: this.pickReadback, bytesPerRow: 256 },
        { width: 1, height: 1, depthOrArrayLayers: 1 }
      );

      this.device.queue.submit([encoder.finish()]);

      // ---- PICK PROFILING: GPU 제출 완료 대기 ----
      tSubmit0 = performance.now();
      await this.device.queue.onSubmittedWorkDone();
      tSubmit1 = performance.now();

      // ---- PICK PROFILING: map/readback ----
      tMap0 = performance.now();
      await this.pickReadback.mapAsync(GPUMapMode.READ);
      const arr = new Uint32Array(this.pickReadback.getMappedRange());
      const raw = arr[0];
      this.pickReadback.unmap();
      tMap1 = performance.now();

      if (seq != this._pickSeq) return [];

      const index = (raw === 0) ? -1 : (raw - 1);
      const hitOk = (index >= 0 && index < this.drawCount);

      // ---- PICK PROFILING: sample push ----
      const t1 = performance.now();
      this._pushPickSample({
        dt: (t1 - t0),
        dtSubmit: (tSubmit1 - tSubmit0),
        dtMap: (tMap1 - tMap0),
        hit: hitOk,
        scale: camScale,
      });

      if (!hitOk) return [];

      const xy = this.data.xy;
      const xw = xy[index * 2 + 0];
      const yw = xy[index * 2 + 1];
      const clusterArr = this.data.cluster32;

      return [{
        index,
        id: this.data.ids32[index],
        x: xw,
        y: yw,
        ddcId: this.data.ddc32[index],
        cluster: clusterArr[index]
      }];

    } catch (e) {
      console.log(e);
      try { this.pickReadback?.unmap?.(); } catch {}
      // 실패는 hit=false로 기록
      const t1 = performance.now();
      this._pushPickSample({
        dt: (t1 - t0),
        dtSubmit: (tSubmit1 - tSubmit0),
        dtMap: (tMap1 - tMap0),
        hit: false,
        scale: camScale,
      });
      return [];
    } finally {
      this._pickBusy = false;
    }
  }
}