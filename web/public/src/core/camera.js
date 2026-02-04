// core/camera.js
// 카메라 상태 및 뷰 변환 관리

import { clamp } from '../utils/math.js';

/**
 * 2D 카메라 클래스
 * 줌, 팬, 좌표 변환을 담당
 */
export class Camera {
  constructor(ctx, options = {}) {
    this.ctx = ctx;
    this.bus = ctx.bus;
    // 카메라 상태
    this.scale = options.scale ?? 0.22;
    this.tx = options.tx ?? 0.0;
    this.ty = options.ty ?? 0.0;

    // 제한값
    this.minScale = options.minScale ?? 0.01;
    this.maxScale = options.maxScale ?? 10.0;

    // 캔버스 참조
    this.canvas = null;
    this.canvasWidth = 0;
    this.canvasHeight = 0;

    // 드래그 상태
    this._dragging = false;
    this._lastX = 0;
    this._lastY = 0;

    // 바인딩
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onWheel = this._onWheel.bind(this);
  }

  /**
   * 캔버스에 이벤트 리스너 연결
   */
  attachTo(canvas) {
    this.canvas = canvas;
    this.updateCanvasSize();

    canvas.addEventListener("mousedown", this._onMouseDown);
    window.addEventListener("mouseup", this._onMouseUp);
    window.addEventListener("mousemove", this._onMouseMove);
    canvas.addEventListener("wheel", this._onWheel, { passive: false });
  }

  /**
   * 이벤트 리스너 해제
   */
  detach() {
    if (!this.canvas) return;

    this.canvas.removeEventListener("mousedown", this._onMouseDown);
    window.removeEventListener("mouseup", this._onMouseUp);
    window.removeEventListener("mousemove", this._onMouseMove);
    this.canvas.removeEventListener("wheel", this._onWheel);

    this.canvas = null;
  }

  /**
   * 캔버스 크기 업데이트
   */
  updateCanvasSize() {
    if (this.canvas) {
      this.canvasWidth = this.canvas.width;
      this.canvasHeight = this.canvas.height;
    }
  }

  /**
   * 캔버스 크기 설정 (외부에서 호출)
   */
  setCanvasSize(width, height) {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  /**
   * 화면 좌표를 월드 좌표로 변환
   */
  screenToWorld(screenX, screenY) {
    if (!this.canvas) return [0, 0];

    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const xNdc = ((screenX - rect.left) * dpr / this.canvasWidth) * 2 - 1;
    const yNdc = 1 - ((screenY - rect.top) * dpr / this.canvasHeight) * 2;

    const wx = (xNdc - this.tx) / this.scale;
    const wy = (yNdc - this.ty) / this.scale;

    return [wx, wy];
  }

  /**
   * 월드 좌표를 화면 좌표로 변환
   */
  worldToScreen(worldX, worldY) {
    if (!this.canvas) return [0, 0];

    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const xNdc = worldX * this.scale + this.tx;
    const yNdc = worldY * this.scale + this.ty;

    const screenX = ((xNdc + 1) / 2) * this.canvasWidth / dpr + rect.left;
    const screenY = ((1 - yNdc) / 2) * this.canvasHeight / dpr + rect.top;

    return [screenX, screenY];
  }

  /**
   * 팬 (이동)
   */
  pan(deltaX, deltaY) {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    this.tx += (deltaX * dpr) * (2 / this.canvasWidth);
    this.ty -= (deltaY * dpr) * (2 / this.canvasHeight);
  }

    /**
   * 줌 (커서 기준)
   * @param {number} delta - wheel deltaY (양수: 줌아웃 성향, 음수: 줌인 성향)
   * @param {number|null} centerX - 줌 중심 screenX (clientX)
   * @param {number|null} centerY - 줌 중심 screenY (clientY)
   */
  zoom(delta, centerX = null, centerY = null) {
    // wheel delta → zoom factor
    const k = Math.exp(-delta * 0.001);

    // 줌 기준점이 없으면 "화면 중앙"을 기준으로(기존 동작과 유사)
    if (centerX == null || centerY == null) {
      if (this.canvas) {
        const rect = this.canvas.getBoundingClientRect();
        centerX = rect.left + rect.width * 0.5;
        centerY = rect.top + rect.height * 0.5;
      } else {
        // 캔버스가 없으면 그냥 scale만 변경
        this.scale = clamp(this.scale * k, this.minScale, this.maxScale);
        return;
      }
    }

    // 1) 줌 전: 커서가 가리키는 world 좌표
    const [wx0, wy0] = this.screenToWorld(centerX, centerY);

    // 2) scale 업데이트 (clamp)
    const newScale = clamp(this.scale * k, this.minScale, this.maxScale);
    this.scale = newScale;

    // 3) 줌 후: 동일한 screen 좌표에서 world가 어떻게 보이는지 계산
    const [wx1, wy1] = this.screenToWorld(centerX, centerY);

    // 4) 커서가 가리키는 world가 유지되도록 tx/ty 보정
    // screenToWorld: wx = (xNdc - tx) / scale
    // => tx를 움직여서 wx1을 wx0으로 맞춘다.
    this.tx += (wx1 - wx0) * this.scale;
    this.ty += (wy1 - wy0) * this.scale;
  }

  /**
   * 특정 위치로 이동
   */
  moveTo(worldX, worldY, scale = null) {
    this.tx = -worldX * this.scale;
    this.ty = -worldY * this.scale;
    if (scale !== null) {
      this.scale = clamp(scale, this.minScale, this.maxScale);
    }
  }

  /**
   * 전체 데이터가 화면에 보이도록 카메라 조정
   * @param {Float32Array} xy - 포인트 좌표 배열 [x0, y0, x1, y1, ...]
   * @param {number} padding - 여백 비율 (0.1 = 10% 여백)
   */
  fitToBounds(xy, padding = 0.1) {
    if (!xy || xy.length < 2) return;

    // 바운딩 박스 계산
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < xy.length; i += 2) {
      const x = xy[i];
      const y = xy[i + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    // 중심점 계산
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // 데이터 범위
    const dataWidth = maxX - minX;
    const dataHeight = maxY - minY;

    // 여백 적용
    const paddedWidth = dataWidth * (1 + padding);
    const paddedHeight = dataHeight * (1 + padding);

    // 화면 비율 고려하여 스케일 계산
    // NDC는 -1 ~ 1 범위이므로 전체 범위는 2
    const scaleX = 2 / paddedWidth;
    const scaleY = 2 / paddedHeight;

    // 더 작은 스케일 선택 (전체가 보이도록)
    this.scale = clamp(Math.min(scaleX, scaleY), this.minScale, this.maxScale);

    // 중심을 화면 중앙에 위치
    this.tx = -centerX * this.scale;
    this.ty = -centerY * this.scale;

    console.log(`Camera fit: center=(${centerX.toFixed(2)}, ${centerY.toFixed(2)}), scale=${this.scale.toFixed(4)}`);
  }

  /**
   * 바운딩 박스로 카메라 조정 (bounds 객체 버전)
   * @param {Object} bounds - { minX, maxX, minY, maxY }
   * @param {number} padding - 여백 비율
   */
  fitToBoundsRect(bounds, padding = 0.1) {
    const { minX, maxX, minY, maxY } = bounds;

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    const dataWidth = maxX - minX;
    const dataHeight = maxY - minY;

    const paddedWidth = dataWidth * (1 + padding);
    const paddedHeight = dataHeight * (1 + padding);

    const scaleX = 2 / paddedWidth;
    const scaleY = 2 / paddedHeight;

    this.scale = clamp(Math.min(scaleX, scaleY), this.minScale, this.maxScale);
    this.tx = -centerX * this.scale;
    this.ty = -centerY * this.scale;
  }

  computeFitStateForBounds(bounds, padding = 0.1) {
    const { minX, maxX, minY, maxY } = bounds;

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    const dataWidth = maxX - minX;
    const dataHeight = maxY - minY;

    const paddedWidth = dataWidth * (1 + padding);
    const paddedHeight = dataHeight * (1 + padding);

    const scaleX = 2 / paddedWidth;
    const scaleY = 2 / paddedHeight;

    const scale = clamp(
      Math.min(scaleX, scaleY),
      this.minScale,
      this.maxScale
    );

    const tx = -centerX * scale;
    const ty = -centerY * scale;

    return { scale, tx, ty };
  }

  /**
   * 카메라 리셋
   */
  reset() {
    this.scale = 0.22;
    this.tx = 0.0;
    this.ty = 0.0;
  }

  /**
   * 현재 상태 반환
   */
  getState() {
    return {
      scale: this.scale,
      tx: this.tx,
      ty: this.ty,
    };
  }

  /**
   * 상태 복원
   */
  setState(state) {
    if (state.scale !== undefined) this.scale = state.scale;
    if (state.tx !== undefined) this.tx = state.tx;
    if (state.ty !== undefined) this.ty = state.ty;
  }

  /**
   * 드래그 중인지 여부
   */
  isDragging() {
    return this._dragging;
  }

  // ---- Private Event Handlers ----

  _onMouseDown(e) {
    this._dragging = true;
    this._lastX = e.clientX;
    this._lastY = e.clientY;
  }

  _onMouseUp() {
    this._dragging = false;
  }

  _onMouseMove(e) {
    if (!this._dragging) return;

    const dx = e.clientX - this._lastX;
    const dy = e.clientY - this._lastY;
    this._lastX = e.clientX;
    this._lastY = e.clientY;

    this.pan(dx, dy);
  }

  _onWheel(e) {
    e.preventDefault();
    this.zoom(e.deltaY, e.clientX, e.clientY);
  }

  animateTo(target, durationMs = 350) {
    const start = this.getState();
    const t0 = performance.now();

    // 기존 애니메이션이 있으면 중단
    this._animToken = (this._animToken ?? 0) + 1;
    const token = this._animToken;

    const easeInOut = (t) => (t < 0.5)
      ? 2 * t * t
      : 1 - Math.pow(-2 * t + 2, 2) / 2;

    const step = () => {
      if (token !== this._animToken) return;

      const now = performance.now();
      const u = Math.min(1, (now - t0) / durationMs);
      const k = easeInOut(u);

      this.scale = start.scale + (target.scale - start.scale) * k;
      this.tx    = start.tx    + (target.tx    - start.tx)    * k;
      this.ty    = start.ty    + (target.ty    - start.ty)    * k;

      if (u < 1) requestAnimationFrame(step);
    };

    requestAnimationFrame(step);
  }
}