// web/public/src/core/clusterMerge.js

function jaccard(aSet, bSet) {
  let inter = 0;
  for (const x of aSet) if (bSet.has(x)) inter++;
  const union = aSet.size + bSet.size - inter;
  return union === 0 ? 0 : inter / union;
}

function normalizeKw(kw) {
  // 가벼운 정규화: lower + trim
  return String(kw || "").toLowerCase().trim();
}

/**
 * clusterLabelsJson: { num_clusters, noise_bucket, labels: { "0": {size, keywords, label}, ... } }
 * returns:
 *  {
 *    groupOfCluster: Uint32Array(C+1)  // cluster k -> group id
 *    groups: [
 *      { id, label, size, clusters:[k...], keywords:[...] }
 *    ],
 *    numGroups
 *  }
 */
export function buildMergedClusterView(clusterLabelsJson, opts = {}) {
  const {
    jaccardThreshold = 0.60,
    minKeywords = 3,
    keepNoiseSeparate = true,
    noiseLabel = "noise/outliers",
  } = opts;

  const C = Number(clusterLabelsJson.num_clusters ?? 0);
  const noiseBucket = Number(clusterLabelsJson.noise_bucket ?? C);

  // labels object may include noiseBucket too
  const labels = clusterLabelsJson.labels || {};

  // prepare cluster entries 0..C-1 (exclude noise from merging unless chosen otherwise)
  const entries = [];
  for (let k = 0; k < C; k++) {
    const obj = labels[String(k)] || { size: 0, keywords: [], label: "" };
    const kws = (obj.keywords || []).map(normalizeKw).filter(Boolean);
    const kwSet = new Set(kws);
    entries.push({
      k,
      size: Number(obj.size || 0),
      label: String(obj.label || ""),
      kwSet,
      kws,
    });
  }

  // simple greedy union-find
  const parent = new Int32Array(C);
  for (let i = 0; i < C; i++) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const unite = (a, b) => {
    a = find(a); b = find(b);
    if (a === b) return;
    // union by size proxy: attach smaller root to larger root (by cluster size)
    const sa = entries[a]?.size ?? 0;
    const sb = entries[b]?.size ?? 0;
    if (sa >= sb) parent[b] = a;
    else parent[a] = b;
  };

  // Greedy pairwise (O(C^2))는 C=100~200에서만 OK.
  // 네 현재 C=108 정도라 안전. (C가 1000 넘어가면 다른 방식 필요)
  for (let i = 0; i < C; i++) {
    const A = entries[i];
    if (A.kwSet.size < minKeywords) continue;
    for (let j = i + 1; j < C; j++) {
      const B = entries[j];
      if (B.kwSet.size < minKeywords) continue;
      const sim = jaccard(A.kwSet, B.kwSet);
      if (sim >= jaccardThreshold) unite(i, j);
    }
  }

  // group mapping
  const rootToG = new Map();
  let gCount = 0;
  const groupOfCluster = new Uint32Array(C + 1);

  // build groups
  const tmpGroups = new Map(); // gid -> group object
  for (let k = 0; k < C; k++) {
    const r = find(k);
    let gid = rootToG.get(r);
    if (gid === undefined) {
      gid = gCount++;
      rootToG.set(r, gid);
      tmpGroups.set(gid, { id: gid, size: 0, clusters: [], keywordsCount: new Map() });
    }
    groupOfCluster[k] = gid;

    const e = entries[k];
    const g = tmpGroups.get(gid);
    g.size += e.size;
    g.clusters.push(k);
    for (const kw of e.kwSet) {
      g.keywordsCount.set(kw, (g.keywordsCount.get(kw) || 0) + 1);
    }
  }

  // noise bucket handling: keep separate group at the end
  if (keepNoiseSeparate) {
    groupOfCluster[noiseBucket] = gCount;
    const noiseObj = labels[String(noiseBucket)] || { size: 0, keywords: ["noise"], label: noiseLabel };
    tmpGroups.set(gCount, {
      id: gCount,
      size: Number(noiseObj.size || 0),
      clusters: [noiseBucket],
      keywordsCount: new Map([["noise", 999]]),
      forceLabel: String(noiseObj.label || noiseLabel),
    });
    gCount++;
  } else {
    // treat noise as its own "cluster" group already covered if noiseBucket < C (usually not)
    groupOfCluster[noiseBucket] = groupOfCluster[Math.min(noiseBucket, C - 1)] || 0;
  }

  // finalize group labels/keywords
  const groups = Array.from(tmpGroups.values()).map((g) => {
    // 대표 키워드: 등장 빈도 높은 순
    const sortedKw = Array.from(g.keywordsCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map((x) => x[0]);

    let label = g.forceLabel;
    if (!label) {
      // 가장 큰 클러스터 label을 대표로
      let bestK = g.clusters[0];
      let bestSize = -1;
      for (const k of g.clusters) {
        const s = (labels[String(k)]?.size) || 0;
        if (s > bestSize) { bestSize = s; bestK = k; }
      }
      label = labels[String(bestK)]?.label || sortedKw.slice(0, 4).join(" / ");
    }

    return {
      id: g.id,
      label,
      size: g.size,
      clusters: g.clusters,
      keywords: sortedKw,
    };
  });

  // size desc for UI convenience
  groups.sort((a, b) => b.size - a.size);

  return { groupOfCluster, groups, numGroups: gCount };
}