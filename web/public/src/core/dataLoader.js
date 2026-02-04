// core/dataLoader.js
// 데이터 파일 로딩 및 파싱 (Shard 기반 snippet 지원)
// 유연한 경로 처리: 절대 경로 → 상대 경로 자동 변환
import { buildMergedClusterView } from "./clusterMerge.js";
import { buildClusterBoundsCache } from "./clusterBoundsCache.js";
import { SearchIndexLoader } from "./searchIndexLoader.js";
/**
 * 절대 경로를 상대 경로로 변환
 * "C:\...\packed\file.f32" → "./packed/file.f32"
 * 이미 상대 경로면 그대로 반환
 */
function toRelativePath(path, baseDir = "./packed") {
  if (!path) return null;

  // 이미 상대 경로면 그대로
  if (path.startsWith("./") || path.startsWith("../")) {
    return path;
  }

  // 파일명만 추출 (Windows/Unix 모두 대응)
  const filename = path.split(/[/\\]/).pop();
  return `${baseDir}/${filename}`;
}

/**
 * ArrayBuffer를 fetch
 */
async function fetchArrayBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.arrayBuffer();
}

/**
 * JSON 파일을 fetch
 */
async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.json();
}

/**
 * Shard 기반 Snippet 로더 클래스
 */
class SnippetLoader {
  constructor(indexUrl, baseDir) {
    this.indexUrl = indexUrl;
    this.baseDir = baseDir;
    this.index = null;
    this.shardCache = new Map();
  }

  async init() {
    try {
      this.index = await fetchJson(this.indexUrl);
    } catch (e) {
      console.warn("SnippetLoader: index not found, snippets disabled", e);
      this.index = null;
    }
    return this;
  }

  async getSnippet(id) {
    if (!this.index) return null;

    const shardIdx = id % this.index.mod;

    if (!this.shardCache.has(shardIdx)) {
      const filename = `${this.baseDir}/snippets_${this.index.tag}_${String(shardIdx).padStart(3, "0")}.json`;
      try {
        const data = await fetchJson(filename);
        this.shardCache.set(shardIdx, data);
      } catch (e) {
        console.warn(`Failed to load shard ${shardIdx}:`, e);
        return null;
      }
    }

    return this.shardCache.get(shardIdx)[String(id)] || null;
  }

  async getSnippets(ids) {
    const results = {};
    for (const id of ids) {
      results[id] = await this.getSnippet(id);
    }
    return results;
  }
}

/**
 * 데이터 로더 클래스
 */
export class DataLoader {
  constructor() {
    this.data = null;
    this.snippetLoader = null;
  }

  /**
   * 메타 파일에서 모든 데이터 로드
   * @param {string} metaUrl - pack_meta.json 경로 (예: "./packed/pack_meta_n100000_seed42.json")
   * @returns {Promise<Object>} 로드된 데이터
   */
  async load(metaUrl) {
    // ---- PERF instrumentation ----
    const perf = {};
    const now = () => performance.now();
    const tAll0 = now();
    const markStart = (k) => (perf[`_${k}`] = now());
    const markEnd = (k) => {
      const t0 = perf[`_${k}`];
      if (t0 != null) perf[k] = now() - t0;
      delete perf[`_${k}`];
    };

    // 0) meta
    markStart("meta");
    const meta = await fetchJson(metaUrl);
    markEnd("meta");
    // meta.n = 10;
    const n = meta.n;

    // 태그 추출 (metaUrl에서)
    // "./packed/pack_meta_n100000_seed42.json" → "n100000_seed42"
    const tagMatch = metaUrl.match(/pack_meta_(.+)\.json/);
    const tag = tagMatch ? tagMatch[1] : "unknown";
    const baseDir = "./packed";

    // 1) file urls (abs->rel)
    const pointsUrl =
      toRelativePath(meta.files?.points_xy_layout, baseDir) ||
      toRelativePath(meta.files?.points_xy, baseDir) ||
      `${baseDir}/points_xy_${tag}.f32`;

    const idsUrl =
      toRelativePath(meta.files?.ids, baseDir) ||
      `${baseDir}/ids_${tag}.uint32`;

    const clusterUrl =
      toRelativePath(meta.files?.cluster, baseDir) ||
      `${baseDir}/cluster_${tag}.uint16`;

    const labelUrl =
      toRelativePath(meta.files?.cluster_labels, baseDir) ||
      `${baseDir}/cluster_labels_${tag}.json`;

    const ddcUrl =
      toRelativePath(meta.files?.ddc, baseDir) ||
      `${baseDir}/ddc_${tag}.uint16`;

    const ddcMetaUrl =
      toRelativePath(meta.files?.ddc_meta, baseDir) ||
      `${baseDir}/ddc_meta_${tag}.json`;

    console.log("Loading points:", pointsUrl);
    console.log("Loading files:", { pointsUrl, idsUrl, clusterUrl, labelUrl, ddcUrl, ddcMetaUrl });

    // 2) snippet loader (shard)
    markStart("snippets_index");
    this.snippetLoader = new SnippetLoader(
      `${baseDir}/snippets_index_${tag}.json`,
      `${baseDir}/snippets`
    );
    await this.snippetLoader.init();
    markEnd("snippets_index");

    // 3) parallel fetch
    markStart("fetch");
    const [pointsBuf, idsBuf, clusterBuf, ddcBuf, labelsJson, ddcMeta] = await Promise.all([
      fetchArrayBuffer(pointsUrl),
      fetchArrayBuffer(idsUrl),
      fetchArrayBuffer(clusterUrl),
      fetchArrayBuffer(ddcUrl),
      fetchJson(labelUrl),
      fetchJson(ddcMetaUrl),
    ]);
    markEnd("fetch");

    // --- normalize labelsJson schema (snake_case / camelCase 모두 지원) ---
    const rawNumClusters =
      (labelsJson.num_clusters ?? labelsJson.numClusters ?? labelsJson.num_clusters_count ?? 0);

    const rawNoiseBucket =
      (labelsJson.noise_bucket ?? labelsJson.noiseBucket ?? rawNumClusters);

    // labels dict도 혹시 다른 키면 보정
    const rawLabelsDict =
      (labelsJson.labels ?? labelsJson.cluster_labels ?? labelsJson.clusterLabels ?? {});

    // 표준 필드를 다시 주입 (이후 코드는 이 키만 사용)
    labelsJson.num_clusters = rawNumClusters;
    labelsJson.noise_bucket = rawNoiseBucket;
    labelsJson.labels = rawLabelsDict;

    // 4) typed arrays
    markStart("typed_view");
    const xy = new Float32Array(pointsBuf).slice(0, n*2);
    const ids = new Uint32Array(idsBuf).slice(0, n);
    const cluster16 = new Uint16Array(clusterBuf).slice(0, n);
    const ddc16 = new Uint16Array(ddcBuf).slice(0, n);
    markEnd("typed_view");
    
    // 5) validate
    if (xy.length !== n * 2) throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
    if (ids.length !== n) throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);
    if (cluster16.length !== n) throw new Error(`cluster length mismatch: ${cluster16.length} vs ${n}`);
    if (ddc16.length !== n) throw new Error(`ddc length mismatch: ${ddc16.length} vs ${n}`);

    // 7) merged cluster view (keyword-jaccard)
    markStart("cpu_post");
    const merged = buildMergedClusterView(labelsJson, {
      jaccardThreshold: 0.60,
      minKeywords: 3,
      keepNoiseSeparate: true,
    });

    const mergedGroups = merged?.groups ?? [];
    const mergedCount = mergedGroups.length;
    const mergedNoiseBucket = mergedCount; // merged에서는 마지막을 noise로

    console.log(
      "[merge] merged groups =",
      mergedCount,
      "raw clusters =",
      rawNumClusters,
      "raw noise_bucket =",
      rawNoiseBucket
    );

    // 8) raw cluster id -> merged group id
    // raw indices are 0..rawNumClusters, where rawNoiseBucket may be == rawNumClusters.
    const rawToMerged = new Int32Array(rawNumClusters + 1);
    rawToMerged.fill(-1);

    mergedGroups.forEach((g, gi) => {
      const members = g?.members ?? g?.clusters ?? g?.items;
      if (!members) return;
      for (const c of members) {
        const cc = (typeof c === "number") ? c : Number(c);
        if (Number.isFinite(cc) && cc >= 0 && cc <= rawNumClusters) {
          rawToMerged[cc] = gi;
        }
      }
    });

    // noise는 mergedNoiseBucket로
    if (rawNoiseBucket >= 0 && rawNoiseBucket <= rawNumClusters) {
      rawToMerged[rawNoiseBucket] = mergedNoiseBucket;
    }

    // 9) build merged cluster32 (for rendering)
    const mergedCluster32 = new Uint32Array(n);
    const mergedPalCount = mergedCount + 1;

    for (let i = 0; i < n; i++) {
      const raw = cluster16[i]; // uint16
      let mg = mergedNoiseBucket;

      if (raw <= rawNumClusters) {
        const mapped = rawToMerged[raw];
        mg = (mapped >= 0) ? mapped : mergedNoiseBucket;
      }
      // clamp
      mergedCluster32[i] = (mg < mergedPalCount) ? mg : (mergedPalCount - 1);
    }
    // 9.5) merged cluster id를 공식 clusterId로 쓰기 위한 16-bit view
    // mergedPalCount가 65535를 넘지 않는 한 uint16
    const clusterView16 = new Uint16Array(n);
    for (let i = 0; i < n; i++) {
      clusterView16[i] = mergedCluster32[i]; // mergedGroupId
    }

    // 10) palette for merged
    const mergedClusterPalette = this._makePalette(mergedPalCount);
    this._setNoiseColor(mergedClusterPalette, mergedNoiseBucket);
    const mergedClusterPalette32 = this._paletteToUint32(mergedClusterPalette, mergedPalCount);

    // 11) DDC -> uint32
    const ddc32 = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const d = ddc16[i];
      ddc32[i] = (d <= 10) ? d : 10;
    }
    const ddcPalette = this._makeDdcPalette();
    const ddcPalette32 = this._makeDdcPalette32();
    markEnd("cpu_post");

    // 12) store
    markStart("assemble");
    this.data = {
      n,
      tag,
      xy,
      ids,

      labelsJson,
      numClusters: mergedCount,

      // DDC
      ddc16,
      ddc32,
      ddcMeta,
      ddcPalette,
      ddcPalette32,

      // RAW cluster (원본도 보관: 디버깅/표시용)
      cluster16,
      rawNumClusters,
      rawNoiseBucket,

      // MERGED cluster (렌더링 기준)
      clusterMerged: merged,
      mergedCluster32,
      cluster32: mergedCluster32,
      clusterView16,
      mergedCount,
      mergedNoiseBucket,
      clusterPalette: mergedClusterPalette,
      clusterPalette32: mergedClusterPalette32,
      clusterPalCount: mergedPalCount,

      // Snippet
      snippetLoader: this.snippetLoader,
      searchIndex: new SearchIndexLoader(tag, "./packed/search_index", 256),
    };
    markEnd("assemble");
    markStart("bounds");
    this.data.clusterBoundsCache = buildClusterBoundsCache(this.data, {
      clusterKey: "clusterView16",
    });
    markEnd("bounds");

    perf.total = now() - tAll0;
    this.data._perf = perf;
    return this.data;
  }

  // ---- 팔레트 헬퍼 함수들 ----

  _makePalette(count) {
    const palette = [];
    for (let i = 0; i < count; i++) {
      const hue = (i * 137.508) % 360;
      palette.push(this._hslToRgb(hue / 360, 0.65, 0.55));
    }
    return palette;
  }

  _setNoiseColor(palette, noiseIdx) {
    if (noiseIdx < palette.length) {
      palette[noiseIdx] = [128, 128, 128];
    }
  }

  _paletteToUint32(palette, count) {
    const arr = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      const [r, g, b] = palette[i] || [128, 128, 128];
      arr[i] = (255 << 24) | (b << 16) | (g << 8) | r;
    }
    return arr;
  }

  _makeDdcPalette() {
    return [
      [141, 211, 199], // 0: 총류
      [255, 255, 179], // 1: 철학
      [190, 186, 218], // 2: 종교
      [251, 128, 114], // 3: 사회과학
      [128, 177, 211], // 4: 언어
      [253, 180, 98],  // 5: 자연과학
      [179, 222, 105], // 6: 기술
      [252, 205, 229], // 7: 예술
      [217, 217, 217], // 8: 문학
      [188, 128, 189], // 9: 역사
      [150, 150, 150], // 10: Unknown
    ];
  }

  _makeDdcPalette32() {
    const palette = this._makeDdcPalette();
    const arr = new Uint32Array(11);
    for (let i = 0; i < 11; i++) {
      const [r, g, b] = palette[i];
      arr[i] = (255 << 24) | (b << 16) | (g << 8) | r;
    }
    return arr;
  }

  _hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  async getSnippet(id) {
    if (!this.snippetLoader) return null;
    return await this.snippetLoader.getSnippet(id);
  }

  getData() {
    return this.data;
  }
}