// utils/math.js
// 수학 유틸리티 및 좌표 변환

/**
 * 값을 범위 내로 제한
 */
export function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

/**
 * 두 점 사이의 거리 제곱 (성능 최적화용)
 */
export function distanceSquared(x1, y1, x2, y2) {
  const dx = x1 - x2;
  const dy = y1 - y2;
  return dx * dx + dy * dy;
}

/**
 * 두 점 사이의 유클리드 거리
 */
export function distance(x1, y1, x2, y2) {
  return Math.sqrt(distanceSquared(x1, y1, x2, y2));
}

/**
 * 선형 보간
 */
export function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * 값을 0-1 범위로 정규화
 */
export function normalize(value, min, max) {
  return (value - min) / (max - min);
}

/**
 * packed 파일 경로를 상대 경로로 변환
 */
export function toPackedRelative(p) {
  return String(p).replace(/^.*\/packed\//, "./packed/");
}
