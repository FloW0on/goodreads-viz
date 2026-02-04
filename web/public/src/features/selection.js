// features/selection.js
// 클러스터 및 포인트 선택 기능

export class SelectionFeature {
  constructor(data, ctx) {
    this.data = data;
    this.ctx = ctx;
    this.bus = ctx.bus;
    this.services = ctx.services; // 오타 수정

    // 선택 상태
    this.selectedCluster = -1; // -1 = 선택 없음
    this.selectedPoint = -1;   // -1 = 선택 없음

    // 선택 표현 모드
    this.selectMode = 1;       // 1 = dim others, 2 = only selected
    this.dimAlpha = 0.08;

    this.onSelectionChange = null;
  }

  setClusterFocus(clusterId, selectMode = 1) {
    const nextCluster = (clusterId ?? -1);
    const nextMode = (selectMode ?? 1);

    if (this.selectedCluster === nextCluster && this.selectMode === nextMode) return;

    this.selectedCluster = nextCluster;
    this.selectMode = nextMode;
    this._triggerCallback();
  }

  selectCluster(clusterId) {
    this.setClusterFocus(clusterId, this.selectMode ?? 1);
  }

  clearClusterSelection() {
    if (this.selectedCluster === -1) return;
    this.selectedCluster = -1;
    this._triggerCallback();
  }

  toggleMode() {
    this.selectMode = (this.selectMode === 1) ? 2 : 1;
    this._triggerCallback();
  }

  setMode(mode) {
    this.selectMode = mode;
    this._triggerCallback();
  }

  selectPoint(pointIndex) {
    if (this.selectedPoint === pointIndex) return;
    this.selectedPoint = pointIndex;
    this._triggerCallback();
  }

  clearPointSelection() {
    if (this.selectedPoint === -1) return;
    this.selectedPoint = -1;
    this._triggerCallback();
  }

  clearAll() {
    const changed = (this.selectedCluster !== -1 || this.selectedPoint !== -1);
    this.selectedCluster = -1;
    this.selectedPoint = -1;
    if (changed) this._triggerCallback();
  }

  getSelectedClusterPoints() {
    if (this.selectedCluster < 0) return [];
    const clusterArr = this.data.clusterView16 ?? this.data.cluster16; 
    const { n } = this.data;
    const indices = [];
    for (let i = 0; i < n; i++) {
      if (clusterArr[i] === this.selectedCluster) indices.push(i);
    }
    return indices;
  }

  getState() {
    return {
      selectedCluster: this.selectedCluster ?? -1,
      selectedPoint: this.selectedPoint ?? -1,
      selectMode: this.selectMode ?? 1,
      dimAlpha: this.dimAlpha ?? 0.15,
    };
  }

  hasClusterSelection() {
    return this.selectedCluster >= 0;
  }

  hasPointSelection() {
    return this.selectedPoint >= 0;
  }

  onChange(callback) {
    this.onSelectionChange = callback;
  }

  _triggerCallback() {
    if (!this.onSelectionChange) return;
    this.onSelectionChange({
      selectedCluster: this.selectedCluster,
      selectedPoint: this.selectedPoint,
      selectMode: this.selectMode,
    });
  }
}