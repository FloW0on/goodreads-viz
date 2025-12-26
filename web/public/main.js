import { runWebGPUPoints } from "./webgpu_points.js";

const TAG = "n10000_seed42";   
const BASE = `./packed`;   
const hud = document.getElementById("hud");
const canvas = document.getElementById("c");

runWebGPUPoints({
  canvas,
  hud,
  metaUrl: `${BASE}/pack_meta_${TAG}.json`,
}).catch((e) => {
  console.error(e);
  hud.textContent = `ERROR: ${e?.message || e}`;
});