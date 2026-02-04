// core/interactionState.js
export class InteractionState {
  constructor() {
    // Hover Source (ephemeral)
    this.hover = {
      pointIndex: -1,
      pointId: null,
      clusterId: -1,
      ddcId: -1,
      screen: null,  // {x,y}
      world: null,   // {x,y} optional
      view: null,    // point object (id/x/y/ddcName/topicLabel/keywords/snippet...)
    };

    // Click Source (persistent)
    this.click = {
      mode: "none",      // "none" | "point" | "cluster" | "ddc"
      pointIndex: -1,
      pointId: null,
      clusterId: -1,
      ddcId: -1,
    };

    // Legend Source (definition + filtering)
    this.legend = {
      scheme: "ddc",        // "ddc" | "cluster" (추가 가능)
      selectedDdc: -1,      // DDC 버튼 선택 (필터/강조의 기준)
      selectedCluster: -1,  // 클러스터 섹션 선택 (필터/강조)
    };

    this._listeners = new Set();
  }

  subscribe(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }
  _emit() {
    for (const fn of this._listeners) fn(this);
  }

  // Hover
  setHover(patch) { Object.assign(this.hover, patch); this._emit(); }
  clearHover() {
    this.hover.pointIndex = -1;
    this.hover.pointId = null;
    this.hover.clusterId = -1;
    this.hover.ddcId = -1;
    this.hover.screen = null;
    this.hover.world = null;
    this.hover.view = null;
    this._emit();
  }

  // Click
  setClick(patch) { Object.assign(this.click, patch); this._emit(); }
  clearClick() {
    this.click.mode = "none";
    this.click.pointIndex = -1;
    this.click.pointId = null;
    this.click.clusterId = -1;
    this.click.ddcId = -1;
    this._emit();
  }

  // Legend
  setLegend(patch) { Object.assign(this.legend, patch); this._emit(); }
}