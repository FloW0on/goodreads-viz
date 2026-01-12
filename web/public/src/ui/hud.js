// ui/hud.js
// HUD (Heads-Up Display) - 상단 정보 표시 UI

/**
 * HUD UI 클래스
 */
export class HudUI {
  constructor(hudEl) {
    this.el = hudEl;
    this.data = null;

    // 기본 메시지
    this.defaultMessage = '';
  }

  /**
   * 데이터 설정
   */
  setData(data) {
    this.data = data;
    this._updateDefaultMessage();
  }

  /**
   * 기본 메시지 표시
   */
  showDefault() {
    if (this.el) {
      this.el.textContent = this.defaultMessage;
    }
  }

  /**
   * 로딩 메시지 표시
   */
  showLoading(message = 'loading...') {
    if (this.el) {
      this.el.textContent = message;
    }
  }

  /**
   * 커스텀 메시지 표시
   */
  showMessage(message) {
    if (this.el) {
      this.el.textContent = message;
    }
  }

  /**
   * 포인트 정보 표시
   */
  showPointInfo(pointInfo) {
    if (!this.el || !pointInfo) {
      this.showDefault();
      return;
    }

    const { id, ddcName, topicLabel, snippet } = pointInfo;
    this.el.textContent = `id=${id} | DDC: ${ddcName} | cluster: ${topicLabel}\n${snippet}`;
  }

  /**
   * 검색 결과 정보 표시
   */
  showSearchInfo(query, matchCount, totalCount) {
    if (this.el) {
      this.el.textContent = `Search: "${query}" - ${matchCount} / ${totalCount} matches`;
    }
  }

  /**
   * 에러 메시지 표시
   */
  showError(error) {
    if (this.el) {
      this.el.textContent = `Error: ${error}`;
      this.el.style.color = '#ff4444';
    }
  }

  /**
   * 에러 스타일 초기화
   */
  clearError() {
    if (this.el) {
      this.el.style.color = '';
    }
  }

  // ---- Private Methods ----

  _updateDefaultMessage() {
    if (this.data) {
      const { n, numClusters } = this.data;
      this.defaultMessage = `loaded: n=${n} | topics=${numClusters} (+noise) | drag: pan | wheel: zoom | +/-: size`;
    } else {
      this.defaultMessage = 'drag: pan | wheel: zoom | +/-: size';
    }
  }
}