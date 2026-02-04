// web/public/src/ui/breadcrumb.js
export class BreadcrumbUI {
  constructor(containerEl) {
    this.el = containerEl;
    this.state = { ddcId: -1, ddcName: "", clusterId: -1, clusterLabel: "", pointId: null };
    this.handlers = {
      onHome: null,
      onDdc: null,
      onCluster: null,
      onPoint: null,
    };
  }

  onHome(cb) { this.handlers.onHome = cb; }
  onDdc(cb) { this.handlers.onDdc = cb; }
  onCluster(cb) { this.handlers.onCluster = cb; }
  onPoint(cb) { this.handlers.onPoint = cb; }

  setState(next) {
    this.state = { ...this.state, ...next };
    this.render();
  }

  render() {
    if (!this.el) return;
    const s = this.state;

    const items = [];
    items.push({ key: "home", label: "Home", enabled: true });

    if (s.ddcId >= 0) {
      items.push({ key: "ddc", label: s.ddcName || `DDC ${s.ddcId}`, enabled: true });
    }
    if (s.clusterId >= 0) {
      items.push({ key: "cluster", label: s.clusterLabel || `Cluster #${s.clusterId}`, enabled: true });
    }
    if (s.pointId != null) {
      items.push({ key: "point", label: `Point ${s.pointId}`, enabled: true });
    }

    this.el.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "breadcrumb";

    items.forEach((it, idx) => {
      if (idx > 0) {
        const sep = document.createElement("span");
        sep.className = "breadcrumb-sep";
        sep.textContent = " > ";
        wrap.appendChild(sep);
      }

      const a = document.createElement("button");
      a.type = "button";
      a.className = "breadcrumb-item";
      a.textContent = it.label;

      a.addEventListener("click", () => {
        if (it.key === "home") this.handlers.onHome?.();
        if (it.key === "ddc") this.handlers.onDdc?.(s.ddcId);
        if (it.key === "cluster") this.handlers.onCluster?.(s.clusterId);
        if (it.key === "point") this.handlers.onPoint?.(s.pointId);
      });

      wrap.appendChild(a);
    });

    this.el.appendChild(wrap);
  }
}