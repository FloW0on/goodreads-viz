// features/hover.js
// 호버 및 포인트 피킹 기능

import { distanceSquared } from '../utils/math.js';

/**
 * 호버/피킹 기능 클래스
 */
export class HoverFeature {
  constructor(data, camera) {
    this.data = data;
    this.camera = camera;

    // 현재 호버 상태
    this.hoveredIndex = -1;
    this.hoveredId = null;
    this.hoveredCluster = null;

    // 설정
    this.pickRadiusMultiplier = 0.75;
    this.pointSize = 3.0;

    // 콜백
    this.onHoverChange = null;
  }

  /**
   * 포인트 크기 설정 (피킹 임계값에 영향)
   */
  setPointSize(size) {
    this.pointSize = size;
  }

  /**
   * 마우스 위치에서 가장 가까운 포인트 찾기
   * @param {number} screenX - 화면 X 좌표
   * @param {number} screenY - 화면 Y 좌표
   * @returns {Object|null} 찾은 포인트 정보 또는 null
   */
  pickPoint(screenX, screenY) {
    const [wx, wy] = this.camera.screenToWorld(screenX, screenY);
    const [index, d2] = this._findNearest(wx, wy);

    const threshold = this._getPickThreshold();
    if (index >= 0 && d2 < threshold * threshold) {
      return this._getPointInfo(index);
    }
    return null;
  }

  /**
   * 호버 업데이트 (mousemove에서 호출)
   * @param {number} screenX - 화면 X 좌표
   * @param {number} screenY - 화면 Y 좌표
   */
  updateHover(screenX, screenY) {
    // 카메라가 드래그 중이면 무시
    if (this.camera.isDragging()) return;

    const point = this.pickPoint(screenX, screenY);

    if (point) {
      this.hoveredIndex = point.index;
      this.hoveredId = point.id;
      this.hoveredCluster = point.cluster;
    } else {
      this.hoveredIndex = -1;
      this.hoveredId = null;
      this.hoveredCluster = null;
    }

    this._triggerCallback(point);
  }

  /**
   * 호버 상태 초기화
   */
  clearHover() {
    this.hoveredIndex = -1;
    this.hoveredId = null;
    this.hoveredCluster = null;
    this._triggerCallback(null);
  }

  /**
   * 현재 호버된 포인트 정보 반환
   */
  getHoveredPoint() {
    if (this.hoveredIndex < 0) return null;
    return this._getPointInfo(this.hoveredIndex);
  }

  /**
   * 특정 인덱스의 포인트 정보 반환
   */
  getPointInfo(index) {
    if (index < 0 || index >= this.data.n) return null;
    return this._getPointInfo(index);
  }

  /**
   * 콜백 설정
   */
  onHover(callback) {
    this.onHoverChange = callback;
  }

  // ---- Private Methods ----

  _findNearest(wx, wy) {
    const { xy, n } = this.data;
    let best = -1;
    let bestD2 = Infinity;

    for (let i = 0; i < n; i++) {
      const px = xy[i * 2 + 0];
      const py = xy[i * 2 + 1];
      const d2 = distanceSquared(wx, wy, px, py);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = i;
      }
    }

    return [best, bestD2];
  }

  _getPickThreshold() {
    const { width } = this.camera.canvas
      ? { width: this.camera.canvasWidth }
      : { width: 800 };

    const px = Math.max(2, this.pointSize);
    const ndcPerPx = 2.0 / width;
    const worldPerPx = ndcPerPx / this.camera.scale;
    return (px * worldPerPx) * this.pickRadiusMultiplier;
  }

  _getPointInfo(index) {
    const { xy, ids, cluster16, ddc16, labelsJson, ddcMeta, id2snippet } = this.data;

    const id = ids[index];
    const cluster = cluster16[index];
    const ddcId = ddc16[index];
    const x = xy[index * 2 + 0];
    const y = xy[index * 2 + 1];

    // 클러스터 라벨 정보
    const clusterInfo = labelsJson.labels?.[String(cluster)];
    const topicLabel = clusterInfo?.label ?? `#${cluster}`;
    const keywords = clusterInfo?.keywords ?? [];

    // DDC 정보
    const ddcInfo = ddcMeta?.classes?.[String(ddcId)];
    const ddcName = ddcInfo?.name_ko ?? ddcInfo?.name ?? `DDC ${ddcId}`;

    // 스니펫
    const snippet = id2snippet[String(id)] ?? '';

    return {
      index,
      id,
      x,
      y,
      // DDC
      ddcId,
      ddcName,
      // 클러스터
      cluster,
      topicLabel,
      keywords,
      clusterSize: clusterInfo?.size ?? 0,
      // 스니펫
      snippet,
    };
  }

  _triggerCallback(point) {
    if (this.onHoverChange) {
      this.onHoverChange(point);
    }
  }
}