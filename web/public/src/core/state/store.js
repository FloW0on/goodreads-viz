// store.js (런타임용 JS)
import { createClusterNavState } from "./clusterNav.js";

export function createEmbeddingViewBus(initial = {}) {
  const state = {
    viewport: initial.viewport ?? null,
    tooltip: initial.tooltip ?? null,
    selection: initial.selection ?? [],
    rangeSelection: initial.rangeSelection ?? null,
    clusterNav: {
      focusedCluster: -1,
      selectMode: 1,
      source: null,
    },
  };

  const listeners = {
    viewport: new Set(),
    tooltip: new Set(),
    selection: new Set(),
    rangeSelection: new Set(),
    clusterNav: new Set(),
  };

  function emit(event, payload) {
    listeners[event].forEach((fn) => fn(payload));
  }

  const bus = {
    getState() {
      return { ...state, selection: [...state.selection], clusterNav: {...state.clusterNav}, };
    },

    on(event, fn) {
      listeners[event].add(fn);
      return () => listeners[event].delete(fn);
    },

    setViewport(v) {
      state.viewport = v;
      emit("viewport", v);
    },

    setTooltip(dp) {
      state.tooltip = dp;
      emit("tooltip", dp);
    },

    setSelection(sel) {
      state.selection = sel;
      emit("selection", sel);
    },

    setRangeSelection(rs) {
      state.rangeSelection = rs;
      emit("rangeSelection", rs);
    },

    setClusterNav(next){
      state.clusterNav = {...state.clusterNav, ...next};
      emit("clusterNav", state.clusterNav);
    },

    clearAll() {
      state.tooltip = null;
      state.selection = [];
      state.rangeSelection = null;
      emit("tooltip", null);
      emit("selection", []);
      emit("rangeSelection", null);
    },
  };
  bus.clusterNav = createClusterNavState(bus);
  return bus;
}