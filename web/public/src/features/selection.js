// features/selection.js
// 클러스터 및 포인트 선택 기능

/**
 * 선택 기능 클래스
 */
export class SelectionFeature {
  constructor(data) {
    this.data = data;

    // 클러스터 선택 상태
    this.selectedCluster = -1;  // -1 = 선택 없음
    this.selectMode = 1;        // 1 = dim others, 2 = only selected

    // 개별 포인트 선택
    this.selectedPoint = -1;    // -1 = 선택 없음

    // 설정
    this.dimAlpha = 0.08;

    // 콜백
    this.onSelectionChange = null;
  }

  /**
   * 클러스터 선택
   * @param {number} clusterId - 클러스터 ID (-1이면 선택 해제)
   */
  selectCluster(clusterId) {
    this.selectedCluster = clusterId;
    this._triggerCallback();
  }

  /**
   * 클러스터 선택 해제
   */
  clearClusterSelection() {
    this.selectedCluster = -1;
    this._triggerCallback();
  }

  /**
   * 선택 모드 토글
   */
  toggleMode() {
    this.selectMode = (this.selectMode === 1) ? 2 : 1;
    this._triggerCallback();
  }

  /**
   * 선택 모드 설정
   * @param {number} mode - 1: dim others, 2: only selected
   */
  setMode(mode) {
    this.selectMode = mode;
    this._triggerCallback();
  }

  /**
   * 개별 포인트 선택
   * @param {number} pointIndex - 포인트 인덱스
   */
  selectPoint(pointIndex) {
    this.selectedPoint = pointIndex;
    this._triggerCallback();
  }

  /**
   * 포인트 선택 해제
   */
  clearPointSelection() {
    this.selectedPoint = -1;
    this._triggerCallback();
  }

  /**
   * 모든 선택 해제
   */
  clearAll() {
    this.selectedCluster = -1;
    this.selectedPoint = -1;
    this._triggerCallback();
  }

  /**
   * 선택된 클러스터의 포인트 인덱스들 반환
   */
  getSelectedClusterPoints() {
    if (this.selectedCluster < 0) return [];

    const { cluster16, n } = this.data;
    const indices = [];
    for (let i = 0; i < n; i++) {
      if (cluster16[i] === this.selectedCluster) {
        indices.push(i);
      }
    }
    return indices;
  }

  /**
   * 선택 상태 반환 (uniform용)
   */
  getState() {
    return {
      selectedCluster: this.selectedCluster,
      selectMode: this.selectMode,
      dimAlpha: this.dimAlpha,
    };
  }

  /**
   * 클러스터가 선택되었는지 확인
   */
  hasClusterSelection() {
    return this.selectedCluster >= 0;
  }

  /**
   * 포인트가 선택되었는지 확인
   */
  hasPointSelection() {
    return this.selectedPoint >= 0;
  }

  /**
   * 콜백 설정
   */
  onChange(callback) {
    this.onSelectionChange = callback;
  }

  // ---- Private Methods ----

  _triggerCallback() {
    if (this.onSelectionChange) {
      this.onSelectionChange({
        selectedCluster: this.selectedCluster,
        selectedPoint: this.selectedPoint,
        selectMode: this.selectMode,
      });
    }
  }
}
