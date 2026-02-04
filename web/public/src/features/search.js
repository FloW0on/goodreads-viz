// features/search.js
// 검색 기능
import { makeTrigramHashes } from "../core/searchIndexLoader.js";
/**
 * 검색 기능 클래스
 */
export class SearchFeature {
  constructor(data, renderer) {
    this.data = data;
    this.renderer = renderer;

    // 검색 상태
    this.active = false;
    this.query = '';
    this.matchedIndices = [];
    this.mask = new Uint32Array(data.n);

    // 설정
    this.onlyMatched = true;  // true: 매칭만 표시, false: 나머지 dim
    this.dimAlpha = 0.05;

    // UI 요소
    this.inputEl = null;
    this.clearBtn = null;

    // 콜백
    this.onSearchChange = null;

    // 튜닝 파라미터
    this.maxCandidates = 8000; 
  }

  /**
   * UI 요소 연결
   */
  attachUI(inputEl, clearBtn) {
    this.inputEl = inputEl;
    this.clearBtn = clearBtn;

    if (inputEl) {
      inputEl.addEventListener("keydown", async (e) => {
        if (e.key === "Enter") await this.search(inputEl.value);
        if (e.key === "Escape") this.clear();
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener("click", () => this.clear());
    }
  }

  /**
   * 검색 실행
   * @param {string} query - 검색어
   */
  async search(query) {
    this.query = (query ?? "").trim().toLowerCase();
    this.mask.fill(0);
    this.matchedIndices = [];

    if (!this.query) {
      this.active = false;
      this._updateRenderer();
      this._triggerCallback();
      return;
    }

    const idx = this.data.searchIndex;
    const snippetLoader = this.data.snippetLoader;
    const ids = this.data.ids;
    const n = this.data.n;

    if (!idx || !snippetLoader || !ids || !n) {
      console.warn("[search] missing searchIndex/snippetLoader/ids/n");
      this.active = false;
      this._updateRenderer();
      this._triggerCallback();
      return;
    }

    this.active = true;

    // 1) query trigram hashes
    const hs = makeTrigramHashes(this.query);
    if (hs.length === 0) {
      // query가 너무 짧으면 (len<3) 단순 fallback: 후보 없이 종료하거나, prefix 방식 등을 별도 정의
      console.warn("[search] query too short (<3), no trigram search");
      this._updateRenderer();
      this._triggerCallback();
      return;
    }

    // 2) postings 로드 + 교집합
    // 작은 posting부터 intersect (성능)
    const postings = [];
    for (const h of hs) {
      const p = await idx.getPosting(h);
      if (!p || p.length === 0) {
        // 어떤 trigram도 없으면 결과 없음
        this._updateRenderer();
        this._triggerCallback();
        return;
      }
      postings.push(p);
    }
    postings.sort((a, b) => a.length - b.length);

    // intersect: Uint32Array끼리 투포인터
    let cand = postings[0];
    for (let k = 1; k < postings.length; k++) {
      cand = intersectSortedUint32(cand, postings[k], this.maxCandidates);
      if (!cand || cand.length === 0) break;
      if (cand.length >= this.maxCandidates) break;
    }
    if (!cand || cand.length === 0) {
      this._updateRenderer();
      this._triggerCallback();
      return;
    }

    // 3) 후보에 대해 snippet substring 검증
    // 후보가 많으면 shard별로 snippet shard를 미리 로드해서 lookup 비용 절감
    const q = this.query;

    // candidate pointIndex -> bookId
    // snippet shard = bookId % 256
    const shardToIds = new Map(); // shard -> [bookId...]
    const shardToPoints = new Map(); // shard -> [pointIndex...]

    for (let t = 0; t < cand.length; t++) {
      const pi = cand[t];
      const bid = ids[pi];
      const shard = bid % 256;
      if (!shardToIds.has(shard)) {
        shardToIds.set(shard, []);
        shardToPoints.set(shard, []);
      }
      shardToIds.get(shard).push(bid);
      shardToPoints.get(shard).push(pi);
    }

    // shard별로 snippet shard를 로드해두면 getSnippet이 cache hit
    // (SnippetLoader 내부가 shardCache를 채우기 때문에)
    const matches = [];
    console.log("[search] cand len =", cand.length);

    for (const [shard, idList] of shardToIds.entries()) {
      // shardCache warm-up: getSnippet을 한 번이라도 호출하면 로드됨
      // 여기서는 후보 id들을 직접 돌며 로드 + 검사
      const pointList = shardToPoints.get(shard);

      for (let j = 0; j < idList.length; j++) {
        const bid = idList[j];
        const pi = pointList[j];
        const text = await snippetLoader.getSnippet(bid);

        if (text && String(text).toLowerCase().includes(q)) {
          matches.push(pi);
          if (matches.length >= this.maxCandidates) break;
        }
      }
      if (matches.length >= this.maxCandidates) break;
    }

    // 4) mask 업데이트
    for (const pi of matches) {
      this.mask[pi] = 1;
    }
    this.matchedIndices = matches;

    this._updateRenderer();
    this._triggerCallback();
  }

  getXY(data, i) {
  const xy = data?.xy;
  if (!xy) return null;

  // case A: flat typed array (Float32Array 등) => [x0,y0,x1,y1,...]
  if (typeof xy[0] === "number" && typeof xy[1] === "number") {
    const x = xy[i * 2];
    const y = xy[i * 2 + 1];
    return (x == null || y == null) ? null : [x, y];
  }

  // case B: vec2 array => xy[i] = [x,y] or {0:x,1:y,length:2}
  const v = xy[i];
  if (v && v.length >= 2) return [v[0], v[1]];

  return null;
}

  /**
   * 검색 초기화
   */
  clear() {
    this.query = '';
    this.active = false;
    this.mask.fill(0);
    this.matchedIndices = [];

    if (this.inputEl) {
      this.inputEl.value = '';
    }

    this._updateRenderer();
    this._triggerCallback();
  }

  /**
   * 검색 결과 개수 반환
   */
  getMatchCount() {
    return this.matchedIndices.length;
  }

  /**
   * 검색 활성화 여부
   */
  isActive() {
    return this.active;
  }

  /**
   * 검색 상태 반환 (uniform용)
   */
  getState() {
    return {
      searchActive: this.active ? 1 : 0,
      searchOnly: this.onlyMatched ? 1 : 0,
      searchDimAlpha: this.dimAlpha,
    };
  }

  /**
   * 모드 토글 (only matched / dim others)
   */
  toggleMode() {
    this.onlyMatched = !this.onlyMatched;
    this._triggerCallback();
  }

  /**
   * 콜백 설정
   */
  onChange(callback) {
    this.onSearchChange = callback;
  }

  // ---- Private Methods ----

  _updateRenderer() {
    this.renderer?.updateSearchMask?.(this.mask);
  }

  _triggerCallback() {
    this.onSearchChange?.({
      active: this.active,
      query: this.query,
      matchCount: this.matchedIndices.length,
      indices: this.matchedIndices,
    });
  }
}
function intersectSortedUint32(a, b, cap = 1e9) {
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    const av = a[i], bv = b[j];
    if (av === bv) {
      out.push(av);
      if (out.length >= cap) break;
      i++; j++;
    } else if (av < bv) i++;
    else j++;
  }
  return new Uint32Array(out);
}