// core/dataLoader.js
// 데이터 파일 로딩 및 파싱

import { toPackedRelative } from '../utils/math.js';
import { makePalette, setNoiseColor, paletteToUint32, makeDdcPalette, makeDdcPalette32, DDC_PALETTE } from '../utils/colors.js';

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
 * 데이터 로더 클래스
 */
export class DataLoader {
  constructor() {
    this.data = null;
  }

  /**
   * 메타 파일에서 모든 데이터 로드
   * @param {string} metaUrl - meta.json 경로
   * @returns {Promise<Object>} 로드된 데이터
   */
  async load(metaUrl) {
    // 메타 정보 로드
    const meta = await fetchJson(metaUrl);
    const n = meta.n;

    // 파일 경로 추출
    const pointsUrl = toPackedRelative(meta.files.points_xy);
    const idsUrl = toPackedRelative(meta.files.ids);
    const clusterUrl = toPackedRelative(meta.files.cluster);
    const labelUrl = toPackedRelative(meta.files.cluster_labels);
    const snippetUrl = "./packed/id2snippet_n10000_seed42.json";
    
    // DDC 파일 경로 (새로 추가)
    const ddcUrl = "./packed/ddc_n10000_seed42.uint16";
    const ddcMetaUrl = "./packed/ddc_meta_n10000_seed42.json";

    // 병렬 로드 (DDC 포함)
    const [pointsBuf, idsBuf, clusterBuf, ddcBuf, labelsJson, ddcMeta, id2snippet] = await Promise.all([
      fetchArrayBuffer(pointsUrl),
      fetchArrayBuffer(idsUrl),
      fetchArrayBuffer(clusterUrl),
      fetchArrayBuffer(ddcUrl),
      fetchJson(labelUrl),
      fetchJson(ddcMetaUrl),
      fetchJson(snippetUrl),
    ]);

    // 타입 변환
    const xy = new Float32Array(pointsBuf);
    const ids = new Uint32Array(idsBuf);
    const cluster16 = new Uint16Array(clusterBuf);
    const ddc16 = new Uint16Array(ddcBuf);

    // 검증
    if (xy.length !== n * 2) {
      throw new Error(`xy length mismatch: ${xy.length} vs ${n * 2}`);
    }
    if (ids.length !== n) {
      throw new Error(`ids length mismatch: ${ids.length} vs ${n}`);
    }
    if (cluster16.length !== n) {
      throw new Error(`cluster length mismatch: ${cluster16.length} vs ${n}`);
    }
    if (ddc16.length !== n) {
      throw new Error(`ddc length mismatch: ${ddc16.length} vs ${n}`);
    }

    // 클러스터 정보 (비지도 - 강조/선택용)
    const numClusters = labelsJson.num_clusters ?? 0;
    const noiseBucket = labelsJson.noise_bucket ?? numClusters;

    // 클러스터 팔레트 (선택 강조용)
    const clusterPalette = makePalette(numClusters + 1);
    if (noiseBucket < numClusters + 1) {
      setNoiseColor(clusterPalette, noiseBucket);
    }

    // 클러스터를 Uint32로 변환
    const clusterPalCount = numClusters + 1;
    const cluster32 = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const c = cluster16[i];
      cluster32[i] = (c < clusterPalCount) ? c : (clusterPalCount - 1);
    }

    // DDC를 Uint32로 변환 (0-10 범위 검증)
    const ddc32 = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const d = ddc16[i];
      ddc32[i] = (d <= 10) ? d : 10;  // 범위 밖이면 Unknown(10)
    }

    // DDC 팔레트 (11색 고정)
    const ddcPalette = makeDdcPalette();
    const ddcPalette32 = makeDdcPalette32();

    // 결과 저장 및 반환
    this.data = {
      n,
      xy,
      ids,
      
      // DDC (색상용)
      ddc16,
      ddc32,
      ddcMeta,
      ddcPalette,
      ddcPalette32,
      
      // 클러스터 (강조/선택용)
      cluster16,
      cluster32,
      labelsJson,
      numClusters,
      noiseBucket,
      clusterPalette,
      clusterPalette32: paletteToUint32(clusterPalette, clusterPalCount),
      clusterPalCount,
      
      // 스니펫
      id2snippet,
    };

    return this.data;
  }

  /**
   * 로드된 데이터 반환
   */
  getData() {
    return this.data;
  }
}