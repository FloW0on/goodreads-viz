import { runWebGPUPoints } from "./webgpu_points.js";

const TAG = "n10000_seed42";
const BASE = `./packed`;

const hud = document.getElementById("hud");
const canvas = document.getElementById("c");
const colorModeEl = document.getElementById("colorMode");

const start = () =>
  runWebGPUPoints({
    canvas,
    hud,
    metaUrl: `${BASE}/pack_meta_${TAG}.json`,
    metaExtraUrl: `${BASE}/meta_${TAG}.jsonl`,
    colorMode: colorModeEl?.value || "year",
  }).catch((e) => {
    console.error(e);
    hud.textContent = `ERROR: ${e?.message || e}`;
  });

start();

colorModeEl?.addEventListener("change", () => location.reload());