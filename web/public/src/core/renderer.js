// core/renderer.js
// WebGPU 초기화 및 렌더링 (DDC 지원)

/**
 * WebGPU 렌더러 클래스
 */
export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.format = null;
    this.pipeline = null;
    this.bindGroup = null;

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

    // 캔버스 크기
    this.width = 0;
    this.height = 0;
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
    window.addEventListener("resize", () => this.configure());

    // 셰이더 로드 및 파이프라인 생성
    await this._createPipeline();
  }

  /**
   * 캔버스 설정
   */
  configure() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);

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

    return { width: w, height: h };
  }

  /**
   * 데이터 설정
   * @param {Object} data - DataLoader에서 반환된 데이터
   */
  setData(data) {
    this.pointCount = data.n;

    // XY 버퍼
    this.buffers.xy = this.device.createBuffer({
      size: data.xy.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.xy, 0, data.xy);

    // DDC 버퍼 (색상용)
    this.buffers.ddc = this.device.createBuffer({
      size: data.ddc32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.ddc, 0, data.ddc32);

    // DDC 팔레트 버퍼 (11색)
    this.buffers.ddcPalette = this.device.createBuffer({
      size: data.ddcPalette32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.ddcPalette, 0, data.ddcPalette32);

    // 클러스터 버퍼 (강조용)
    this.buffers.cluster = this.device.createBuffer({
      size: data.cluster32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.buffers.cluster, 0, data.cluster32);

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

    // f32 값들
    dv.setFloat32(0, state.camScale, true);
    dv.setFloat32(4, state.camTx, true);
    dv.setFloat32(8, state.camTy, true);
    dv.setFloat32(12, state.pointSize, true);
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

    // 52..63은 padding (zeros)

    this.device.queue.writeBuffer(this.buffers.uniform, 0, buf);
  }

  /**
   * 렌더링 수행
   */
  render() {
    this.configure();

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
    pass.draw(6, this.pointCount, 0, 0);
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
        targets: [{ format: this.format }],
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
}