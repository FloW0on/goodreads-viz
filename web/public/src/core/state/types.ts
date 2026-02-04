// web/public/src/core/state/types.ts

export type Vec2 = { x: number; y: number };

export type Rectangle = { x0: number; y0: number; x1: number; y1: number };

/**
 * Lasso selection: 폐곡선(화면 좌표 또는 데이터 좌표 중 하나로 통일)
 * 여기서는 "screen space(px)"로 통일하는 것을 권장.
 */
export type Lasso = Vec2[];

export type RangeSelection =
  | { kind: "rect"; rect: Rectangle }
  | { kind: "lasso"; lasso: Lasso }
  | null;

/**
 * Atlas tooltip/selection point contract.
 * - x,y: data space 좌표(UMAP 좌표)
 * - category: numeric class (e.g., DDC top-level or cluster id)
 * - text: tooltip text
 * - identifier: stable id (book id 등)
 */
export type DataPoint = {
  x: number;
  y: number;
  category: number;
  text?: string;
  identifier: number | string;
};

/**
 * Camera/viewport state (너 Camera가 serialize 가능한 형태로 정의)
 * 최소 요건: pan/zoom (또는 view matrix)
 */
export type ViewportState = {
  // data->clip 변환을 재구성할 수 있는 값들 중 하나로 고정
  // A) pan/zoom
  panX: number;
  panY: number;
  zoom: number;
};
