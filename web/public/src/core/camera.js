// core/camera.js
// 카메라 상태 및 뷰 변환 관리

import { clamp } from '../utils/math.js';

/**
 * 2D 카메라 클래스
 * 줌, 팬, 좌표 변환을 담당
 */
export class Camera {
  constructor(options = {}) {
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
   * 줌
   * @param {number} delta - 줌 델타 (양수: 줌인, 음수: 줌아웃)
   * @param {number} centerX - 줌 중심 X (옵션)
   * @param {number} centerY - 줌 중심 Y (옵션)
   */
  zoom(delta, centerX = null, centerY = null) {
    const k = Math.exp(-delta * 0.001);
    this.scale = clamp(this.scale * k, this.minScale, this.maxScale);
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
    this.zoom(e.deltaY);
  }
}