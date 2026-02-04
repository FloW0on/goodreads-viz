// web/public/src/core/state/contracts.ts
import type { DataPoint, Rectangle } from "./types";

/**
 * Atlas-style querySelection
 * - screenX, screenY: 화면 좌표(px) 기준
 * - unitDistance: 1px이 데이터좌표에서 얼마나 되는지(또는 반대로)
 */
export type QuerySelection = (screenX: number, screenY: number, unitDistance: number) => Promise<DataPoint[]>;

/**
 * Atlas-style queryClusterLabels
 * - clusters: 각 클러스터는 여러 Rectangle로 근사
 * - return: 클러스터 개수와 같은 길이의 라벨 배열
 */
export type QueryClusterLabels = (clusters: Rectangle[][]) => Promise<string[]>;

/**
 * “엔진이 제공해야 하는 서비스 계약”을 한 번에 묶은 인터페이스
 * - Feature들은 이 인터페이스만 받도록 설계
 */
export type EmbeddingViewServices = {
  querySelection: QuerySelection;
  queryClusterLabels?: QueryClusterLabels;
};