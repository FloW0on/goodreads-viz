// utils/colors.js
// 색상 변환 및 팔레트 생성 유틸리티

/**
 * DDC 10대 분류 + Unknown (11색) 고정 팔레트
 * 색상은 ColorBrewer Set3 기반
 */
export const DDC_PALETTE = {
  0: { color: "#8dd3c7", name: "Computer Science, Information", name_ko: "총류 (컴퓨터, 정보)" },
  1: { color: "#ffffb3", name: "Philosophy & Psychology", name_ko: "철학, 심리학" },
  2: { color: "#bebada", name: "Religion", name_ko: "종교" },
  3: { color: "#fb8072", name: "Social Sciences", name_ko: "사회과학" },
  4: { color: "#80b1d3", name: "Language", name_ko: "언어" },
  5: { color: "#fdb462", name: "Science", name_ko: "자연과학" },
  6: { color: "#b3de69", name: "Technology", name_ko: "기술, 응용과학" },
  7: { color: "#fccde5", name: "Arts & Recreation", name_ko: "예술, 오락" },
  8: { color: "#d9d9d9", name: "Literature", name_ko: "문학" },
  9: { color: "#bc80bd", name: "History & Geography", name_ko: "역사, 지리" },
  10: { color: "#969696", name: "Unknown", name_ko: "미분류" },
};

/**
 * Hex 색상을 RGB 배열로 변환
 */
export function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [
    parseInt(result[1], 16),
    parseInt(result[2], 16),
    parseInt(result[3], 16)
  ] : [128, 128, 128];
}

/**
 * DDC 팔레트를 Uint8Array RGBA로 변환
 */
export function makeDdcPalette() {
  const out = new Uint8Array(11 * 4);
  for (let i = 0; i <= 10; i++) {
    const [r, g, b] = hexToRgb(DDC_PALETTE[i].color);
    out[i * 4 + 0] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * DDC 팔레트를 GPU용 Uint32Array로 변환
 */
export function makeDdcPalette32() {
  const palette = makeDdcPalette();
  const pal32 = new Uint32Array(11);
  for (let i = 0; i < 11; i++) {
    const r = palette[i * 4 + 0];
    const g = palette[i * 4 + 1];
    const b = palette[i * 4 + 2];
    const a = palette[i * 4 + 3];
    pal32[i] = (r) | (g << 8) | (b << 16) | (a << 24);
  }
  return pal32;
}

/**
 * HSL to RGB 변환
 * @param {number} h - Hue (0-1)
 * @param {number} s - Saturation (0-1)
 * @param {number} l - Lightness (0-1)
 * @returns {[number, number, number]} RGB values (0-255)
 */
export function hslToRgb(h, s, l) {
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : (l + s - l * s);
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * RGBA를 CSS 문자열로 변환
 */
export function rgbaToCss(r, g, b, a = 255) {
  return `rgba(${r},${g},${b},${a / 255})`;
}

/**
 * 클러스터용 팔레트 생성
 * @param {number} n - 클러스터 수
 * @returns {Uint8Array} RGBA 팔레트 (n * 4 bytes)
 */
export function makePalette(n) {
  const out = new Uint8Array(n * 4);

  const s = 0.45;
  const l = 0.78;

  // 미리 정의된 기본 색상 (Hue in degrees)
  const baseHues = [
    205, 195, 215, 185, 225, 175, 235, 165, 245, 155, 255, 145,
    265, 135, 275, 125, 285, 115,
    295, 105, 305, 95, 315, 85
  ];

  for (let i = 0; i < n; i++) {
    let hDeg;
    if (i < baseHues.length) {
      hDeg = baseHues[i];
    } else {
      const t = i - baseHues.length;
      hDeg = 170 + (t * 13.0) % 60;
    }

    const [r, g, b] = hslToRgb(hDeg / 360, s, l);
    out[i * 4 + 0] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }

  return out;
}

/**
 * 팔레트를 GPU용 Uint32Array로 변환
 * @param {Uint8Array} palette - RGBA 팔레트
 * @param {number} count - 색상 수
 * @returns {Uint32Array}
 */
export function paletteToUint32(palette, count) {
  const pal32 = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const r = palette[i * 4 + 0];
    const g = palette[i * 4 + 1];
    const b = palette[i * 4 + 2];
    const a = palette[i * 4 + 3];
    pal32[i] = (r) | (g << 8) | (b << 16) | (a << 24);
  }
  return pal32;
}

/**
 * 노이즈 클러스터 색상 설정
 */
export function setNoiseColor(palette, noiseBucket) {
  palette[noiseBucket * 4 + 0] = 160;
  palette[noiseBucket * 4 + 1] = 160;
  palette[noiseBucket * 4 + 2] = 160;
  palette[noiseBucket * 4 + 3] = 60;
}