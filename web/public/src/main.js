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

/**
 * 앱 상태
 */
const state = {
  pointSize: 3.0,
  minPointSize: 1.0,
  maxPointSize: 20.0,
  selectedDdc: -1,  // DDC 필터 (-1 = 전체)
};

/**
 * 메인 함수
 */
async function main() {
  // DOM 요소
  const canvas = document.getElementById("gpuCanvas");
  const hudEl = document.getElementById("hud");
  const legendEl = document.getElementById("legend");
  const searchInput = document.getElementById("searchInput");
  const searchClear = document.getElementById("searchClear");

  if (!canvas) {
    throw new Error("Canvas element #gpuCanvas not found");
  }

  // UI 초기화
  const hud = new HudUI(hudEl);
  const legend = new LegendUI(legendEl);

  hud.showLoading("loading buffers…");

  try {
    // 1. 데이터 로드
    const loader = new DataLoader();
    const data = await loader.load("./packed/pack_meta_n10000_seed42.json");

    // 2. 렌더러 초기화
    const renderer = new Renderer(canvas);
    await renderer.init();
    renderer.setData(data);

    // 3. 카메라 초기화
    const camera = new Camera();
    camera.attachTo(canvas);

    // 4. 기능 모듈 초기화
    const search = new SearchFeature(data, renderer);
    const selection = new SelectionFeature(data);
    const hover = new HoverFeature(data, camera);

    // 5. UI 설정
    hud.setData(data);
    hud.showDefault();

    legend.setData(data);
    legend.render();

    // 6. 검색 UI 연결
    search.attachUI(searchInput, searchClear);

    // 7. 이벤트 연결

    // DDC 레전드 클릭
    legend.onDdcClick((ddcId) => {
      state.selectedDdc = ddcId;
    });

    // 클러스터 클릭 (강조용)
    legend.onClusterClick((clusterId) => {
      selection.selectCluster(clusterId);
    });

    // Clear 버튼
    legend.onClearSelection(() => {
      state.selectedDdc = -1;
      selection.clearClusterSelection();
    });

    // 호버 이벤트
    hover.onHover((point) => {
      if (point) {
        hud.showPointInfo(point);
      } else {
        hud.showDefault();
      }
    });

    hover.setPointSize(state.pointSize);

    // 마우스 이벤트
    canvas.addEventListener("mousemove", (e) => {
      hover.updateHover(e.clientX, e.clientY);
    });

    // 키보드 이벤트 (포인트 크기 조절)
    window.addEventListener("keydown", (e) => {
      if (e.key === "+" || e.key === "=") {
        state.pointSize = clamp(state.pointSize + 0.5, state.minPointSize, state.maxPointSize);
        hover.setPointSize(state.pointSize);
      }
      if (e.key === "-" || e.key === "_") {
        state.pointSize = clamp(state.pointSize - 0.5, state.minPointSize, state.maxPointSize);
        hover.setPointSize(state.pointSize);
      }
    });

    // 8. 렌더 루프
    function frame() {
      // 카메라 크기 업데이트
      const { width, height } = renderer.getSize();
      camera.setCanvasSize(width, height);

      // Uniform 상태 구성
      const camState = camera.getState();
      const selState = selection.getState();
      const searchState = search.getState();

      const uniformState = {
        camScale: camState.scale,
        camTx: camState.tx,
        camTy: camState.ty,
        pointSize: state.pointSize,
        ...selState,
        ...searchState,
        selectedDdc: state.selectedDdc,  // DDC 필터
      };

      // 렌더링
      renderer.updateUniforms(uniformState);
      renderer.render();

      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);

    // 로드 완료 메시지
    console.log(`Loaded: n=${data.n}, DDC classes=11, clusters=${data.numClusters}`);

  } catch (error) {
    console.error("Initialization error:", error);
    hud.showError(error.message);
  }
}

// 앱 시작
main();