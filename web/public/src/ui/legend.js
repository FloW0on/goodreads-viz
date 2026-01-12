// ui/legend.js
// DDC 10대 분류 레전드 UI

import { DDC_PALETTE } from '../utils/colors.js';

/**
 * DDC 레전드 UI 클래스
 */
export class LegendUI {
  constructor(containerEl) {
    this.container = containerEl;
    this.data = null;
    this.selectedDdc = -1;

    // 콜백
    this.onDdcSelect = null;
    this.onClusterSelect = null;
    this.onClear = null;
  }

  /**
   * 데이터 설정
   */
  setData(data) {
    this.data = data;
  }

  /**
   * DDC 레전드 렌더링
   */
  render() {
    if (!this.container || !this.data) return;

    this.container.innerHTML = "";

    // DDC 통계 계산
    const ddcCounts = this._computeDdcCounts();

    // 제목
    const h = document.createElement("h3");
    h.textContent = "DDC Classification";
    h.style.marginBottom = "12px";
    this.container.appendChild(h);

    // 버튼 행
    this._renderButtonRow();

    // DDC 항목들 (0-10)
    for (let ddcId = 0; ddcId <= 10; ddcId++) {
      this._renderDdcItem(ddcId, ddcCounts[ddcId] || 0);
    }

    // 클러스터 섹션 (접이식)
    this._renderClusterSection();

    this.container.hidden = false;
  }

  /**
   * DDC 선택 하이라이트
   */
  highlightDdc(ddcId) {
    this.selectedDdc = ddcId;
    const items = this.container.querySelectorAll('.ddc-item');
    items.forEach(item => {
      const id = parseInt(item.dataset.ddcId, 10);
      if (id === ddcId) {
        item.classList.add('selected');
      } else {
        item.classList.remove('selected');
      }
    });
  }

  /**
   * 콜백 설정
   */
  onDdcClick(callback) {
    this.onDdcSelect = callback;
  }

  onClusterClick(callback) {
    this.onClusterSelect = callback;
  }

  onClearSelection(callback) {
    this.onClear = callback;
  }

  // ---- Private Methods ----

  _computeDdcCounts() {
    const counts = {};
    const { ddc16, n } = this.data;
    for (let i = 0; i < n; i++) {
      const d = ddc16[i];
      counts[d] = (counts[d] || 0) + 1;
    }
    return counts;
  }

  _renderButtonRow() {
    const btnRow = document.createElement("div");
    btnRow.className = "legend-buttons";
    btnRow.style.display = "flex";
    btnRow.style.gap = "8px";
    btnRow.style.marginBottom = "16px";

    // Clear 버튼
    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear";
    clearBtn.addEventListener("click", () => {
      this.selectedDdc = -1;
      this.highlightDdc(-1);
      if (this.onClear) this.onClear();
    });

    btnRow.appendChild(clearBtn);
    this.container.appendChild(btnRow);
  }

  _renderDdcItem(ddcId, count) {
    const info = DDC_PALETTE[ddcId];
    const pct = ((count / this.data.n) * 100).toFixed(1);

    const row = document.createElement("div");
    row.className = "ddc-item item";
    row.dataset.ddcId = ddcId;
    row.style.cursor = "pointer";
    row.addEventListener("click", () => {
      if (this.selectedDdc === ddcId) {
        // 이미 선택된 것 클릭하면 해제
        this.selectedDdc = -1;
        this.highlightDdc(-1);
        if (this.onDdcSelect) this.onDdcSelect(-1);
      } else {
        this.selectedDdc = ddcId;
        this.highlightDdc(ddcId);
        if (this.onDdcSelect) this.onDdcSelect(ddcId);
      }
    });

    // 색상 스와치
    const sw = document.createElement("div");
    sw.className = "swatch";
    sw.style.background = info.color;

    // 라벨 정보
    const t = document.createElement("div");
    t.className = "label";

    const title = document.createElement("div");
    title.textContent = `${ddcId}. ${info.name_ko}`;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${count.toLocaleString()} (${pct}%)`;

    t.appendChild(title);
    t.appendChild(meta);

    row.appendChild(sw);
    row.appendChild(t);
    this.container.appendChild(row);
  }

  _renderClusterSection() {
    const { labelsJson, numClusters } = this.data;
    if (!labelsJson || numClusters === 0) return;

    // 구분선
    const hr = document.createElement("hr");
    hr.style.margin = "16px 0";
    hr.style.border = "none";
    hr.style.borderTop = "1px solid #ddd";
    this.container.appendChild(hr);

    // 클러스터 섹션 제목
    const h = document.createElement("h4");
    h.textContent = `Clusters (${numClusters})`;
    h.style.marginBottom = "8px";
    h.style.fontSize = "13px";
    h.style.color = "#666";
    h.style.cursor = "pointer";
    this.container.appendChild(h);

    // 클러스터 목록 (접이식)
    const clusterList = document.createElement("div");
    clusterList.className = "cluster-list";
    clusterList.style.display = "none"; // 기본 접힘
    clusterList.style.maxHeight = "200px";
    clusterList.style.overflowY = "auto";
    clusterList.style.fontSize = "12px";

    // 토글
    h.addEventListener("click", () => {
      clusterList.style.display = clusterList.style.display === "none" ? "block" : "none";
      h.textContent = clusterList.style.display === "none" 
        ? `Clusters (${numClusters}) ▶` 
        : `Clusters (${numClusters}) ▼`;
    });
    h.textContent = `Clusters (${numClusters}) ▶`;

    // 클러스터 항목들
    const labels = labelsJson.labels || {};
    const keys = Object.keys(labels).map(k => parseInt(k, 10));
    keys.sort((a, b) => (labels[b]?.size || 0) - (labels[a]?.size || 0));

    for (const k of keys.slice(0, 20)) {
      const info = labels[String(k)];
      const row = document.createElement("div");
      row.style.padding = "4px";
      row.style.cursor = "pointer";
      row.style.borderRadius = "4px";
      row.addEventListener("click", () => {
        if (this.onClusterSelect) this.onClusterSelect(k);
      });
      row.addEventListener("mouseenter", () => {
        row.style.background = "#f0f0f0";
      });
      row.addEventListener("mouseleave", () => {
        row.style.background = "";
      });

      row.textContent = `#${k} ${info?.label || ""} (${info?.size || 0})`;
      clusterList.appendChild(row);
    }

    this.container.appendChild(clusterList);
  }
}