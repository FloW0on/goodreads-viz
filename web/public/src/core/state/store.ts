// web/public/src/core/state/store.ts
import type { DataPoint, RangeSelection, ViewportState } from "./types";

type EventMap = {
  viewport: ViewportState | null;
  tooltip: DataPoint | null;
  selection: DataPoint[];            // 항상 배열(빈 배열이면 none)
  rangeSelection: RangeSelection;
  clusterNav: ClusterNavState;
};

// cluster navigation
export type ClusterNavSource =
  | "hover"
  | "click"
  | "legend"
  | "search"
  | null;

export type ClusterNavState = {
  focusedCluster: number; // -1 = none
  selectMode: 1 | 2;      // 1=dim others, 2=only selected
  source: ClusterNavSource;
};

/**
 * 구독 핸들
 */
export type Unsubscribe = () => void;

export type EmbeddingViewState = {
  viewport: ViewportState | null;
  tooltip: DataPoint | null;
  selection: DataPoint[];
  rangeSelection: RangeSelection;
  clusterNav: ClusterNavState;
};

export type EmbeddingViewBus = {
  // --- snapshot ---
  getState(): EmbeddingViewState;

  // --- event subscribe ---
  on<K extends keyof EventMap>(event: K, fn: (payload: EventMap[K]) => void): Unsubscribe;

  // --- setters (emit 포함) ---
  setViewport(v: ViewportState | null): void;
  setTooltip(dp: DataPoint | null): void;
  setSelection(sel: DataPoint[]): void;
  setRangeSelection(rs: RangeSelection): void;
  setClusterNav(next: Partial<ClusterNavState>): void;
  // --- convenience ---
  clearAll(): void;
};

export function createEmbeddingViewBus(initial?: Partial<EmbeddingViewState>): EmbeddingViewBus {
  const state: EmbeddingViewState = {
    viewport: initial?.viewport ?? null,
    tooltip: initial?.tooltip ?? null,
    selection: initial?.selection ?? [],
    rangeSelection: initial?.rangeSelection ?? null,
    clusterNav: {
      focusedCluster: -1,
      selectMode: 1,
      source: null,
    },
  };

  const listeners: { [K in keyof EventMap]: Set<(p: EventMap[K]) => void> } = {
    viewport: new Set(),
    tooltip: new Set(),
    selection: new Set(),
    rangeSelection: new Set(),
    clusterNav: new Set(),
  };

  function emit<K extends keyof EventMap>(event: K, payload: EventMap[K]) {
    listeners[event].forEach((fn) => fn(payload));
  }

  return {
    getState: () => ({ ...state, selection: [...state.selection] }),

    on(event, fn) {
      listeners[event].add(fn as any);
      return () => listeners[event].delete(fn as any);
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

    setClusterNav(next) {
      state.clusterNav = { ...state.clusterNav, ...next };
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
}
