// core/state/clusterNav.js
// 클러스터 포커스/네비게이션 상태 관리 (UI ↔ Renderer 중재)

export function createClusterNavState(bus) {
  const state = {
    focusedCluster: -1,   // -1 = none
    selectMode: 1,        // 1 = dim others, 2 = only selected
    source: null,         // 'hover' | 'click' | 'legend' | null
  };

  function emit() {
    bus.setState({ clusterNav: { ...state } });
  }

  return {
    getState() {
      return { ...state };
    },

    focus(clusterId, source = 'hover') {
      if (clusterId === state.focusedCluster) return;
      state.focusedCluster = clusterId;
      state.source = source;
      emit();
    },

    clear() {
      if (state.focusedCluster === -1) return;
      state.focusedCluster = -1;
      state.source = null;
      emit();
    },

    setMode(mode) {
      if (mode !== 1 && mode !== 2) return;
      if (state.selectMode === mode) return;
      state.selectMode = mode;
      emit();
    },
  };
}