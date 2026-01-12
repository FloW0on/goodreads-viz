// features/search.js
// 검색 기능

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
  }

  /**
   * UI 요소 연결
   */
  attachUI(inputEl, clearBtn) {
    this.inputEl = inputEl;
    this.clearBtn = clearBtn;

    if (inputEl) {
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") this.search(inputEl.value);
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
  search(query) {
    this.query = query.trim().toLowerCase();
    this.mask.fill(0);
    this.matchedIndices = [];

    if (!this.query) {
      this.active = false;
      this._updateRenderer();
      this._triggerCallback();
      return;
    }

    this.active = true;

    // 검색 수행
    const { ids, id2snippet, n } = this.data;
    for (let i = 0; i < n; i++) {
      const id = ids[i];
      const text = id2snippet[String(id)];
      if (text && text.toLowerCase().includes(this.query)) {
        this.mask[i] = 1;
        this.matchedIndices.push(i);
      }
    }

    this._updateRenderer();
    this._triggerCallback();
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
    if (this.renderer) {
      this.renderer.updateSearchMask(this.mask);
    }
  }

  _triggerCallback() {
    if (this.onSearchChange) {
      this.onSearchChange({
        active: this.active,
        query: this.query,
        matchCount: this.matchedIndices.length,
        indices: this.matchedIndices,
      });
    }
  }
}
