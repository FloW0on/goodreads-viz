// main.js
// 앱 진입점 - DDC 지원 버전

import { DataLoader } from './core/dataLoader.js';
import { Renderer } from './core/renderer.js';
import { Camera } from './core/camera.js';
import { SearchFeature } from './features/search.js';
import { SelectionFeature } from './features/selection.js';
import { HoverFeature } from './features/hover.js';
import { LegendUI } from './ui/legend.js';
import { HudUI } from './ui/hud.js';
import { clamp } from './utils/math.js';
import { PerfHUD } from "./core/perfHud.js";
import { createEmbeddingViewBus } from "./core/state/store.js";
import "./core/state/contracts.js"; 
import { InteractionState } from "./core/interactionState.js";
import { buildClusterBoundsCache } from './core/clusterBoundsCache.js';
import { BreadcrumbUI } from "./ui/breadcrumb.js";

const bus = createEmbeddingViewBus();
const services = {
  querySelection: async () => [], // 나중에 renderer(gpu picking)가 구현체로 덮어씀
};
const ctx = { bus, services };
window._ctx = ctx;
window._services = services;

// wiring이 됐는지 테스트
bus.on("tooltip", (t) => console.log("[bus] tooltip:", t));
bus.on("selection", (s) => console.log("[bus] selection:", s.length));
bus.on("viewport", (v) => console.log("[bus] viewport:", v));
bus.on("rangeSelection", (rs) => console.log("[bus] rangeSelection:", rs));

const perfHud = new PerfHUD();
/**
 * 앱 상태
 */
const state = {
  pointSize: 3.0,
  minPointSize: 1.0,
  maxPointSize: 20.0,
  selectedDdc: -1,
  selectedCluster: -1,
  selectMode: 1,
};

/**
 * 메인 함수
 */
async function main() {
  // DOM 요소
  const canvas = document.getElementById("gpuCanvas");
  const breadcrumbEl = document.getElementById("breadcrumb");
  const hudEl = document.getElementById("hud");
  const legendEl = document.getElementById("legend");
  const searchInput = document.getElementById("searchInput");
  const searchClear = document.getElementById("searchClear");

  if (!canvas) {
    throw new Error("Canvas element #gpuCanvas not found");
  }

  const interaction = new InteractionState();
  window._interaction = interaction;

  // UI 초기화
  const hudUI = new HudUI(hudEl);
  const legend = new LegendUI(legendEl, interaction);

  hudUI.showLoading("loading buffers…");

  try {
    // 1. 데이터 로드
    const t0 = performance.now();

    const loader = new DataLoader();
    const data = await loader.load("./packed/pack_meta_n500000_seed42.json");
    data.clusterView16 = data.clusterMerged?.pointGroup16 ?? data.cluster16;
    data.numClusterView = data.clusterMerged?.groups?.length ?? data.numClusters;
    window._data = data;

    const clusterBounds = buildClusterBoundsCache(data);
    window._clusterBounds = clusterBounds; // 디버그용

    const t1 = performance.now();
    perfHud.setLoadMs(data._perf?.total ?? (t1 - t0));
    perfHud.setLoadBreakdown(data._perf);
    perfHud.setPoints(data.n);

    // 2. 렌더러 초기화
    const renderer = new Renderer(canvas, ctx);
    await renderer.init();
    renderer.setData(data);

    // 3. 카메라 초기화
    const camera = new Camera(ctx);
    camera.attachTo(canvas);
    camera.fitToBounds(data.xy, 0.1);
    const homeCam = camera.getState();

    function zoomToClusterAnimated(clusterId, padding = 0.22) {
      if (clusterId == null || clusterId < 0) return;
      const bounds = clusterBounds.get(clusterId);
      if (!bounds) {
        console.warn("[zoomToClusterAnimated] no bounds for cluster", clusterId);
        return;
      }
      const target = camera.computeFitStateForBounds(bounds, padding);
      camera.animateTo(target, 350);
    }

    function boundsOfPointIndicesXYFlat(xy, indices, cap = 2000) {
      if (!xy || !indices || indices.length === 0) return null;

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      const m = Math.min(indices.length, cap);

      for (let k = 0; k < m; k++) {
        const i = indices[k];
        const x = xy[i * 2];
        const y = xy[i * 2 + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }

      if (!Number.isFinite(minX)) return null;
      return { minX, minY, maxX, maxY };
    }

    const breadcrumb = new BreadcrumbUI(breadcrumbEl);

    function updateBreadcrumb() {
      const ddcId = interaction.legend.selectedDdc ?? -1;
      const ddcName =
        (ddcId >= 0) ? (data.ddcMeta?.classes?.[String(ddcId)]?.name_ko ?? data.ddcMeta?.classes?.[String(ddcId)]?.name ?? "") : "";

      const nav = bus.getState().clusterNav;
      const clusterId = nav?.focusedCluster ?? -1;
      const clusterLabel =
        (clusterId >= 0) ? (data.labelsJson?.labels?.[String(clusterId)]?.label ?? `Cluster #${clusterId}`) : "";

      const pointId = interaction.click?.pointId ?? null;

      breadcrumb.setState({ ddcId, ddcName, clusterId, clusterLabel, pointId });
    }

    breadcrumb.onHome(() => resetViewAnimated());

    breadcrumb.onDdc((ddcId) => {
      selection.clearAll();
      interaction.clearClick();
      interaction.setLegend({ selectedDdc: ddcId, selectedCluster: -1 });
      bus.setClusterNav({ focusedCluster: -1, selectMode: 1, source: "breadcrumb" });
      bus.setSelection([]);
      updateBreadcrumb();
    });

    breadcrumb.onCluster((clusterId) => {
      bus.setClusterNav({ focusedCluster: clusterId, selectMode: 1, source: "breadcrumb" });
      zoomToClusterAnimated(clusterId, 0.22);
      updateBreadcrumb();
    });

    breadcrumb.onPoint((pointId) => {
      // 현재는 로그만
      console.log("[breadcrumb] pointId:", pointId);
    });

    // 상태 변화에 따라 breadcrumb 갱신
    interaction.subscribe(updateBreadcrumb);
    bus.on("clusterNav", updateBreadcrumb);
    bus.on("selection", updateBreadcrumb);

    // 초기 1회
    updateBreadcrumb();

    function resetViewAnimated() {
      camera.animateTo(homeCam, 350);

      selection.clearAll();
      interaction.clearClick();

      bus.setSelection([]);
      bus.setClusterNav({
        focusedCluster: -1,
        selectMode: 1,
        source: "reset",
      });
    }

    // 4. 기능 모듈 초기화
    const search = new SearchFeature(data, renderer);
    const selection = new SelectionFeature(data, ctx);
    const hover = new HoverFeature(data, camera, ctx, interaction);

    // 5. UI 설정
    hudUI.setData(data);
    hudUI.showDefault();

    legend.setData(data);
    legend.render();

    interaction.subscribe((s) => {
      legend.highlightDdc(s.legend.selectedDdc);
    });

    // 6. 검색 UI 연결
    search.attachUI(searchInput, searchClear);
    let _searchZoomTimer = 0;
    state.searchBoost = 10.0;      // 기본: no boost
    state.searchAutoZoom = true;  // 옵션
    search.onChange(({ active, query, matchCount, indices }) => {
      if (active && matchCount > 0) {
        state.pointSize = 8.0;
      } else {
        state.pointSize = 1.0;
      }
      const pickPs = (active && matchCount > 0) ? Math.max(state.pointSize, 6.0) : state.pointSize;
      hover?.setPointSizePx?.(pickPs);
      if (active) {
        perfHud?.setExtra?.(
          `search: "${query}"  matches=${matchCount}  zoom=${camera.getState().scale.toFixed(3)}`
        );
      } else {
        perfHud?.setExtra?.("");
      }
      if (!state.searchAutoZoom) return;
      if (!active || matchCount === 0) return;
      // 너무 자주 애니메이션 방지 (연속 엔터/연속 이벤트)
      clearTimeout(_searchZoomTimer);
      _searchZoomTimer = setTimeout(() => {
        const b = boundsOfPointIndicesXYFlat(data.xy, indices, 2000); // 너무 많으면 일부만
        if (!b) return;
        const target = camera.computeFitStateForBounds(b, 0.25); // padding
        camera.animateTo(target, 350);
      }, 80);
    });

    // clusterNav -> selection 상태로 브릿지 (GPU 강조용)
    bus.on("clusterNav", (nav) => {
      const cid = nav.focusedCluster;

    // bounds가 있는데도 점이 0개로 사라지면, 필터의 cluster 도메인이 불일치 가능성
     const b = cid >= 0 ? clusterBounds.get(cid) : null;
      if (cid >= 0 && !b) {
        console.warn("[clusterNav] bounds missing for cluster:", cid, "-> cluster ID domain mismatch suspected");
      }

      selection.setClusterFocus(nav.focusedCluster, nav.selectMode);
      if (nav?.source === "legend" || nav?.source === "click") {
        zoomToClusterAnimated(nav.focusedCluster, 0.22);
      }
      state.selectedCluster = nav.focusedCluster;
      state.selectMode = nav.selectMode;
    });
    const nav0 = bus.getState().clusterNav;
    selection.setClusterFocus(nav0.focusedCluster, nav0.selectMode);

    // 7. 이벤트 연결

    // DDC 레전드 클릭
    legend.onDdcClick((ddcId) => {
      state.selectedDdc = ddcId;
      interaction.setLegend({selectedDdc: ddcId});
    });

    // 클러스터 클릭 (강조용)
    legend.onClusterClick((clusterId) => {
      console.log("[legend click] id=", clusterId,
              "hasBounds?", !!data.clusterBoundsCache?.get(clusterId),
              "numClusters=", data.numClusters);
      interaction.setLegend({
        selectedDdc: -1,
        selectedCluster: clusterId,
      });
      legend.highlightDdc?.(-1);

      zoomToClusterAnimated(clusterId, 0.22);
      // 다음 프레임에 필터 적용 (깜빡임/증발 감소)
      requestAnimationFrame(() => {
        bus.setClusterNav({
          focusedCluster: clusterId,
          selectMode: 1, 
          source: "legend",
        });
      });
    });

    // Clear 버튼
    legend.onClearSelection(() => {
      state.selectedDdc = -1;
      interaction.setLegend({selectedDdc: -1, selectedCluster: -1});
      bus.setClusterNav({
        focusedCluster: -1,
        source: "legend",
      })
      bus.setSelection([]);
      selection.clearAll();
      interaction.clearClick();
    });

    // 호버 이벤트
    hover.onHover((point) => {
      if (point) {
        const hv = interaction.hover;
        console.log(
          "[SNIPPET-CHECK]",
          "viewId=", point.id,
          "snippetLen=", point.snippet?.length ?? 0
        );
      }
      // HUD 표현
      if (!point) {
        hudUI.showDefault();
      } else {
        // point.snippet은 처음엔 ""이고, HoverFeature에서 로드 완료 후 다시 콜백으로 갱신됨
        hudUI.showPointInfo(point);
      }

      // bus.tooltip (표현/렌더링용으로만 유지)
      // click source로 쓰지 않겠다
      const prev = bus.getState().tooltip;
      const next = point ? {
        x: point.x, y: point.y,
        ddcId: point.ddcId,
        clusterId: point.cluster,
        category: point.ddcId,
        identifier: point.id,
        text: point.snippet || point.topicLabel || point.ddcName || point.id,
      } : null;

      if (
        (prev === null && next === null) ||
        (prev && next && prev.identifier === next.identifier && prev.text === next.text && prev.category === next.category)
      ) {
        return;
      }
      bus.setTooltip(next);
    });

    hover.setPointSizePx(state.pointSize);

    // 마우스 이벤트
    let lastMove = 0;
    canvas.addEventListener("mousemove", (e) => {
      const now = performance.now();
      if (now - lastMove < 30) return; // 30ms(≈33Hz) 정도면 충분
      lastMove = now;
      hover.updateHover(e.clientX, e.clientY);
    });

    canvas.addEventListener("click", async(e) => {
      // 클릭 시점 hover 강제 갱신
      await hover.updateHover(e.clientX, e.clientY);  
      const hv = interaction.hover;
       // 빈 공간 클릭으로 reset은 현재 클러스터 포커스 상태일 때만 수행
      const focused = bus.getState().clusterNav?.focusedCluster ?? -1;
      if (!hv || hv.pointId == null) {
        if (focused >= 0) resetViewAnimated();
        return;
      }

      // click source 확정
      interaction.setClick({
        mode: "point",
        pointIndex: hv.pointIndex ?? -1,
        pointId: hv.pointId,
        clusterId: hv.clusterId ?? -1,
        ddcId: hv.ddcId ?? -1,
      });

      // SelectionFeature(렌더링 uniform용)에도 반영
      // pointIndex가 -1일 수 있으니 방어
      if ((hv.pointIndex ?? -1) >= 0) {
        selection.selectPoint(hv.pointIndex);
      } else {
        selection.clearPointSelection();
      }

      // 기존 bus.setSelection은 선택된 항목 목록(표현) 용도로 유지 가능
      // 기존 tooltip 구조를 재사용하되 identifier는 pointId로 명확히
      const sx = hv.view?.x;
      const sy = hv.view?.y;

      if (sx == null || sy == null) {
        bus.setSelection([]);
      } else {
        bus.setSelection([{
          x: sx,
          y: sy,
          ddcId: hv.ddcId ?? -1,
          clusterId: hv.clusterId ?? -1,
          category: hv.ddcId ?? -1,
          identifier: hv.pointId,
          text: hv.view?.snippet || hv.view?.topicLabel || hv.view?.ddcName || hv.pointId,
        }]);
      }
      if ((hv.clusterId ?? -1) >= 0) {
        zoomToClusterAnimated(hv.clusterId, 0.20);
      }
    });

    // 키보드 이벤트 (포인트 크기 조절)
    window.addEventListener("keydown", (e) => {
      if (e.key === "+" || e.key === "=") {
        state.pointSize = clamp(state.pointSize + 0.5, state.minPointSize, state.maxPointSize);
        hover.setPointSizePx(state.pointSize);
      }
      if (e.key === "-" || e.key === "_") {
        state.pointSize = clamp(state.pointSize - 0.5, state.minPointSize, state.maxPointSize);
        hover.setPointSizePx(state.pointSize);
      }
      if (e.key === "Escape" || e.key === "r" || e.key === "R") {
        resetViewAnimated();
      }
    });

    function computeDrawCountByZoom(scale, total) {
      if (scale < 0.018) return Math.min(total, 120_000);
      if (scale < 0.030) return Math.min(total, 220_000);
      if (scale < 0.050) return Math.min(total, 350_000);
      if (scale < 0.070) return Math.min(total, 450_000);
      return total;
    }

  // 8. 렌더 루프
  let lastPerfExtraTs = 0;

  function frame(now) {
    // ---- PerfHUD frame tick ----
    perfHud.frame(now);

    // ---- 캔버스/카메라 상태 ----
    const { width, height } = renderer.getSize();
    camera.setCanvasSize(width, height);

    const camState = camera.getState();
    const selState = selection.getState();
    const searchState = search.getState();

    // ---- viewport 상태 변화 감지 ----
    if (!frame._lastV) frame._lastV = { tx: NaN, ty: NaN, scale: NaN };
    const lv = frame._lastV;
    if (lv.tx !== camState.tx || lv.ty !== camState.ty || lv.scale !== camState.scale) {
      lv.tx = camState.tx;
      lv.ty = camState.ty;
      lv.scale = camState.scale;
      bus.setViewport({ panX: camState.tx, panY: camState.ty, zoom: camState.scale });
    }

    // ---- drawCount (LOD) ----
    const desiredDrawCount = computeDrawCountByZoom(
      camState.scale,
      data.n
    );
    if (renderer.drawCount !== desiredDrawCount) {
      renderer.setDrawCount(desiredDrawCount);
    }

    // ---- uniforms ----
    const uniformState = {
      camScale: camState.scale,
      camTx: camState.tx,
      camTy: camState.ty,
      pointSize: state.pointSize,
      ...selState,
      ...searchState,
      selectedDdc: interaction.legend.selectedDdc,
      searchBoost: state.searchBoost ?? 1.0,
    };
    renderer.updateUniforms(uniformState);

    // ---- render (딱 1번) ----
    renderer.render();

    // ---- PerfHUD extra (1초에 1번) ----
    if (now - lastPerfExtraTs > 1000) {
      lastPerfExtraTs = now;

      const p = ctx.services.getPickProfile?.();
      const hudMs = hudUI.getHudCommitMs?.();

      const line1 = p
        ? `pick: p50 ${p.p50.toFixed(1)}ms | p95 ${p.p95.toFixed(1)}ms | hit ${p.hitRate.toFixed(2)} | skip ${(p.skipped / Math.max(1, p.calls)).toFixed(2)}`
        : `pick: -`;

      const line2 = (hudMs != null)
        ? `hud: commit ${hudMs.toFixed(1)}ms`
        : null;

      const line3 = `zoom: ${camState.scale.toFixed(2)} | draw: ${renderer.drawCount}`;

      perfHud.setExtra([line1, line2, line3].filter(Boolean).join("\n"));
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

    // 로드 완료 메시지
    console.log(`Loaded: n=${data.n}, DDC classes=11, clusters=${data.numClusters}`);
    console.log(
      `Loaded: n=${data.n}, DDC classes=${data.ddcMeta?.num_classes ?? 11}, ` +
      `raw clusters=${data.labelsJson?.num_clusters ?? 0}, ` +
      `merged groups=${data.clusterMerged?.groups?.length ?? 0}`
    );

    console.log("selection", selection);
    console.log("typeof selectCluster", typeof selection.selectCluster);
    console.log("typeof setClusterFocus", typeof selection.setClusterFocus);

  } catch (error) {
    console.error("Initialization error:", error);
    hudUI.showError(error.message);
  }
}

// 앱 시작
main();