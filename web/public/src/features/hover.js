// features/hover.js
// 호버 및 포인트 피킹 기능 (Shard 기반 snippet 지원)

import { distanceSquared } from '../utils/math.js';

/**
 * 호버/피킹 기능 클래스
 */
export class HoverFeature {
  constructor(data, camera, ctx, interaction) {
    this.data = data;
    this.camera = camera;
    this.ctx = ctx;
    this.bus = ctx.bus;
    this.services = ctx.services;

    this.interaction = interaction;
    // 현재 호버 상태
    this.hoveredIndex = -1;
    this.hoveredId = null;
    this.hoveredCluster = null;

    // 설정
    this.pointSizePx = 3.0;
    this.minHoverPx = 4.0;
    this.maxHoverPx = 48.0;
    this.hoverMul = 1.4;

    // 콜백
    this.onHoverChange = null;

    //스니펫 로드 컨트롤
    this._snippetTimer = null;
    this._snippetSeq = 0;
    this._snippetDebounceMs = 200;
    this._lastSnippetId = null;
  }

  /**
   * 포인트 크기 설정 (피킹 임계값에 영향)
   */
  setPointSizePx(px) {
    this.pointSizePx = px;
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

    const hoverRadiusPx = Math.max(
      this.minHoverPx,
      Math.min(this.pointSizePx * this.hoverMul, this.maxHoverPx)
    );

    const threshold = this._pxToWorldRadius(hoverRadiusPx);
    if (index >= 0 && d2 < threshold * threshold) {
      return this._getPointInfoSync(index);
    }
    return null;
  }

  /**
   * 호버 업데이트 (mousemove에서 호출)
   * @param {number} screenX - 화면 X 좌표
   * @param {number} screenY - 화면 Y 좌표
   */
  async updateHover(screenX, screenY) {
    // 카메라가 드래그 중이면 무시
    if (this.camera.isDragging()) return;

    // services가 아직 준비 안 됐으면 기존 동작 유지(또는 null 처리)
    if (!this.services?.querySelection) {
      // fallback
      this._cancelSnippet();
      this._triggerCallback(null);
      return;
    }

    const hoverRadiusPx = Math.max(
      this.minHoverPx,
      Math.min(this.pointSizePx * this.hoverMul, this.maxHoverPx)
    );

    // GPU picking: hits는 [{x,y,category,text,identifier}] 형태
    let hits = null;
    try{
      hits = await this.services.querySelection(screenX, screenY, hoverRadiusPx);
    }catch (e){
      hits = [];
    }
    if(hits == null){
      return;
    }
    const hit = hits && hits.length ? hits[0] : null;

    if (!hit) {
      this.hoveredIndex = -1;
      this.hoveredId = null;
      this.hoveredCluster = null;
      this._lastSnippetId = null;

      this._cancelSnippet();
      this._triggerCallback(null);

      if(this.interaction) this.interaction.clearHover();
      return;
    }
    // hit.identifier 는 데이터 id (ids[index])를 의미
    const idx = (typeof hit.index === "number") ? hit.index : -1;
    const id = hit.id ?? hit.identifier ?? null;

    // id가 없으면 hover로 처리하지 않음
    if (id == null) {
      this._cancelSnippet();
      this._triggerCallback(null);
      if (this.interaction) this.interaction.clearHover();
      return;
    }

    // 동일 id면 스니펫 재요청/업데이트 방지 (screen만 갱신)
    if (id === this._lastSnippetId) {
      if (this.interaction) {
        this.interaction.setHover({ screen: { x: screenX, y: screenY } });
      }
      return;
    }
    this._lastSnippetId = id;

    // GPU hit에 없으면 data에서 복원
    const clusterArr = this.data.clusterView16 ?? this.data.cluster16;
    const cluster =
      (hit.cluster ?? (idx >= 0 ? clusterArr?.[idx] : -1));

    const ddcId =
      (hit.ddcId ?? (idx >= 0 ? this.data.ddc16?.[idx] : -1));

    // point 구성
    const point = {
      index: idx,
      id,
      x: hit.x,
      y: hit.y,
      ddcId,
      cluster,
      snippet: "",
    };

    // 라벨/메타
    const clusterInfo = this.data.labelsJson?.labels?.[String(cluster)];
    point.topicLabel = clusterInfo?.label ?? `#${cluster}`;   // hit.cluster -> cluster
    point.keywords = clusterInfo?.keywords ?? [];
    point.clusterSize = clusterInfo?.size ?? 0;

    const ddcInfo = this.data.ddcMeta?.classes?.[String(ddcId)];
    point.ddcName = ddcInfo?.name_ko ?? ddcInfo?.name ?? "";

    // 기존 상태 업데이트
    this.hoveredIndex = point.index;
    this.hoveredId = point.id;
    this.hoveredCluster = point.cluster;

    // UI 즉시 업데이트 + 스니펫 디바운스 로드
    if (this.interaction) {
      this.interaction.setHover({
        pointIndex: point.index ?? -1,
        pointId: point.id ?? null,
        clusterId: (point.cluster ?? -1),
        ddcId: (point.ddcId ?? -1),
        screen: { x: screenX, y: screenY },
        view: point, // UI에서 snippet/라벨까지 즉시 사용 가능하도록
      });
    }
    this._triggerCallback(point);
    this._scheduleSnippet(point);
  }

  /**
   * 호버 상태 초기화
   */
  clearHover() {
    this.hoveredIndex = -1;
    this.hoveredId = null;
    this.hoveredCluster = null;
    this._cancelSnippet();
    this._triggerCallback(null);
    if(this.interaction) this.interaction.clearHover();
  }

  /**
   * 현재 호버된 포인트 정보 반환
   */
  getHoveredPoint() {
    if (this.hoveredIndex < 0) return null;
    return this._getPointInfoSync(this.hoveredIndex);
  }

  /**
   * 특정 인덱스의 포인트 정보 반환
   */
  getPointInfo(index) {
    if (index < 0 || index >= this.data.n) return null;
    return this._getPointInfoSync(index);
  }

  /**
   * 콜백 설정
   */
  onHover(callback) {
    this.onHoverChange = callback;
  }

  // ---- Private Methods ----

  _findNearest(wx, wy) { // 일단 유지
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

  _pxToWorldRadius(px) {
    const width = this.camera.canvasWidth ?? this.camera.canvas?.width ?? 800;
    const ndcPerPx = 2.0 / width;
    const worldPerPx = ndcPerPx / (this.camera.scale ?? 1);
    return px * worldPerPx;
  }

  /**
   * 동기적으로 기본 포인트 정보 반환 (snippet 제외)
   */
  _getPointInfoSync(index) { // 일단 유지
    const { xy, ids, cluster16, clusterView16, ddc16, labelsJson, ddcMeta } = this.data;

    const id = ids[index];
    const cluster = (clusterView16 ?? cluster16)[index];
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
      // snippet은 비동기로 로드됨 (초기값 빈 문자열)
      snippet: '',
    };
  }

  _cancelSnippet() {
    if (this._snippetTimer) clearTimeout(this._snippetTimer);
    this._snippetTimer = null;
    this._snippetSeq++; // 기존 비동기 응답 무효화
  }

  _scheduleSnippet(point) {
    if (!point || !this.data.snippetLoader) return;

    // 기존 타이머 취소 + seq 증가로 이전 요청 무효화
    this._cancelSnippet();
    const seq = this._snippetSeq; // cancel에서 증가된 값 이후의 현재 seq

    this._snippetTimer = setTimeout(async () => {
      // hover가 바꼈으면 중단
      if(seq !== this._snippetSeq) return;
      if (point.id !== this.hoveredId) return;

      try {
        const snippet = await this.data.snippetLoader.getSnippet(point.id);
        // 최신 요청인지 확인
        if (seq !== this._snippetSeq) return;

        // hovered가 바뀌었으면 UI 덮어쓰지 않음
        if (point.id !== this.hoveredId) return;

        point.snippet = (typeof snippet === "string") ? snippet : (snippet?.text ?? "");

        // snippet 로드 완료 후 한 번 더 콜백 (UI 업데이트)
        this._triggerCallback(point);
        if (this.interaction && point.id === this.hoveredId) {
          this.interaction.setHover({ view: point });
        }
      } catch (e) {
        if (seq !== this._snippetSeq) return;
        if (point.id !== this.hoveredId) return;

        point.snippet = "";
        this._triggerCallback(point);
        if (this.interaction && point.id === this.hoveredId) {
          this.interaction.setHover({ view: point });
        }
      }
    }, this._snippetDebounceMs);
  }

  _triggerCallback(point) {
    if (this.onHoverChange) {
      this.onHoverChange(point);
    }
  }
}