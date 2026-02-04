// perfHud.js
export class PerfHUD {
  constructor(parent = document.body) {
    this.el = document.createElement("div");
    Object.assign(this.el.style, {
      position: "fixed",
      left: "12px",
      bottom: "12px",
      top: "",
      zIndex: "9999",
      padding: "10px 12px",
      borderRadius: "10px",
      background: "rgba(0,0,0,0.65)",
      color: "#fff",
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
      fontSize: "12px",
      lineHeight: "1.35",
      pointerEvents: "none",
      whiteSpace: "pre",
    });
    parent.appendChild(this.el);

    this.lastTs = performance.now();
    this.lastFpsUpdate = this.lastTs;
    this.frameCount = 0;
    this.fps = 0;

    this.frameMsEMA = 0;
    this.alpha = 0.1;

    this.loadMs = undefined;
    this.loadBreakdown = null;
    this.points = undefined;
    this.extra = "";

    this._render();
  }

  setLoadMs(ms) {
    this.loadMs = ms;
    this._render();
  }

  setLoadBreakdown(obj) {
    this.loadBreakdown = obj || null;
    this._render();
  }

  setPoints(n) {
    this.points = n;
    this._render();
  }

  setExtra(s) {
    this.extra = s || "";
    this._render();
  }

  frame(now = performance.now()) {
    const dt = now - this.lastTs;
    this.lastTs = now;

    // frame ms EMA
    this.frameMsEMA = this.frameMsEMA === 0 ? dt : (1 - this.alpha) * this.frameMsEMA + this.alpha * dt;

    // FPS update (every ~1s)
    this.frameCount++;
    const elapsed = now - this.lastFpsUpdate;
    if (elapsed >= 1000) {
      this.fps = (this.frameCount * 1000) / elapsed;
      this.frameCount = 0;
      this.lastFpsUpdate = now;
      this._render();
    }
  }

  _render() {
    const lines = [];
    lines.push(`FPS: ${this.fps.toFixed(1)}`);
    lines.push(`frame: ${this.frameMsEMA.toFixed(2)} ms`);
    if (this.loadMs != null) lines.push(`load: ${this.loadMs.toFixed(1)} ms`);
    if (this.loadBreakdown) {
      const b = this.loadBreakdown;
      const fmt = (v) => (v == null ? "-" : v.toFixed(1));

      lines.push(
        `  meta ${fmt(b.meta)} | snipIdx ${fmt(b.snippets_index)} | fetch ${fmt(b.fetch)}`
      );
      lines.push(
        `  view ${fmt(b.typed_view)} | cpu ${fmt(b.cpu_post)} | asm ${fmt(b.assemble)} | bounds ${fmt(b.bounds)}`
      );
    }
    if (this.points != null) lines.push(`points: ${Number(this.points).toLocaleString("en-US")}`);
    if (this.extra) lines.push(this.extra);
    this.el.textContent = lines.join("\n");
  }
}