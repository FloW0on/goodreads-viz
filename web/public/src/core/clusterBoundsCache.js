// core/clusterBoundsCache.js
// clusterId(= mergedGroupId) -> {minX, minY, maxX, maxY, count} bbox 캐시

export function buildClusterBoundsCache(data, opts = {}) {
  const { xy, n } = data;
  if (!xy || !n) {
    throw new Error("buildClusterBoundsCache: data.xy / data.n is required");
  }

  // merged cluster id를 공식 clusterId로 쓰기로 했으니, 기본은 clusterView16
  const clusterKey = opts.clusterKey ?? "clusterView16";

  const clusterArr =
    opts.clusterArr ??
    data[clusterKey] ??
    null;

  if (!clusterArr) {
    throw new Error(
      `buildClusterBoundsCache: data.${clusterKey} is required (merged cluster id array). ` +
      `If you intended to use raw clusters, set opts.clusterKey="cluster16".`
    );
  }

  if (clusterArr.length < n) {
    throw new Error(
      `buildClusterBoundsCache: cluster array length (${clusterArr.length}) < n (${n})`
    );
  }

  // Map: clusterId -> bbox
  const map = new Map();

  for (let i = 0; i < n; i++) {
    const c = clusterArr[i];
    const x = xy[i * 2 + 0];
    const y = xy[i * 2 + 1];

    let b = map.get(c);
    if (!b) {
      map.set(c, { minX: x, minY: y, maxX: x, maxY: y, count: 1 });
    } else {
      if (x < b.minX) b.minX = x;
      if (y < b.minY) b.minY = y;
      if (x > b.maxX) b.maxX = x;
      if (y > b.maxY) b.maxY = y;
      b.count++;
    }
  }

  return {
    get(clusterId) {
      if (clusterId == null || clusterId < 0) return null;
      return map.get(clusterId) ?? null;
    },
    has(clusterId) {
      if (clusterId == null || clusterId < 0) return false;
      return map.has(clusterId);
    },
    size() {
      return map.size;
    },
    map, // 디버그/통계용
    _meta: { clusterKey, n },
  };
}