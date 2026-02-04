// shaders_ddc.wgsl
// 색상 = DDC (10대 분류)
// 강조 = 클러스터 (비지도)

struct U {
  scale: f32,
  tx: f32,
  ty: f32,
  pointSizePx: f32,
  invW2: f32,
  invH2: f32,

  // i32로 JS에서 write (DataView.setInt32)
  selectedCluster: i32, // -1 = none (클러스터 강조용)
  selectMode: i32,      // 1=dim others, 2=only selected
  searchActive: i32,    // 0/1
  searchOnly: i32,      // 1=only matched, 0=dim others

  dimAlpha: f32,      // non-selected alpha multiplier
  searchDimAlpha: f32, // non-matched alpha multiplier

  selectedDdc: i32,   // -1 = none (DDC 필터용)
  _pad1: f32,
  searchBoost: f32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> xy: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> ddc: array<u32>;           // DDC 분류 (0-10)
@group(0) @binding(3) var<storage, read> ddcPalette: array<u32>;    // DDC 11색 팔레트
@group(0) @binding(4) var<storage, read> cluster: array<u32>;       // 클러스터 (강조용)
@group(0) @binding(5) var<storage, read> searchMask: array<u32>;    // 0/1 per point

fn unpackRGBA8(p: u32) -> vec4<f32> {
  let r: f32 = f32((p      ) & 255u) / 255.0;
  let g: f32 = f32((p >>  8) & 255u) / 255.0;
  let b: f32 = f32((p >> 16) & 255u) / 255.0;
  let a: f32 = f32((p >> 24) & 255u) / 255.0;
  return vec4<f32>(r, g, b, a);
}

fn corner(vid: u32) -> vec2<f32> {
  if (vid == 0u) { return vec2<f32>(-1.0, -1.0); }
  if (vid == 1u) { return vec2<f32>( 1.0, -1.0); }
  if (vid == 2u) { return vec2<f32>( 1.0,  1.0); }
  if (vid == 3u) { return vec2<f32>(-1.0, -1.0); }
  if (vid == 4u) { return vec2<f32>( 1.0,  1.0); }
  return vec2<f32>(-1.0,  1.0);
}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) @interpolate(flat) ddcId: u32,     // DDC 분류
  @location(2) @interpolate(flat) clusterId: u32, // 클러스터 ID
  @location(3) @interpolate(flat) searchHit: u32, // 검색 매칭
  @location(4) local: vec2<f32>,
  @location(5) @interpolate(flat) iid: u32, // picking id (instance index)
};

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VSOut {
  var o: VSOut;

  let p = xy[i];
  let x = p.x * u.scale + u.tx;
  let y = p.y * u.scale + u.ty;

  let c = corner(v % 6u);
  // 검색 매칭이면 점 크기 boost (vertex 단계)
  let hit = searchMask[i];
  var ps = u.pointSizePx;
  if (u.searchActive != 0 && hit != 0u) {
    ps = ps * max(1.0, u.searchBoost);
  }

  let dx = c.x * (ps * 0.5) * u.invW2;
  let dy = c.y * (ps * 0.5) * u.invH2;

  o.pos = vec4<f32>(x + dx, y + dy, 0.0, 1.0);

  // 색상은 DDC에서 가져옴
  let ddcId = ddc[i];
  o.color = unpackRGBA8(ddcPalette[ddcId]);
  o.ddcId = ddcId;

  // 클러스터는 강조용으로 별도 전달
  o.clusterId = cluster[i];

  // 검색 매칭
  o.searchHit = searchMask[i];
  o.local = c;
  o.iid = i;

  return o;
}

@fragment
fn fs(
  @location(0) color: vec4<f32>,
  @location(1) @interpolate(flat) ddcId: u32,
  @location(2) @interpolate(flat) clusterId: u32,
  @location(3) @interpolate(flat) searchHit: u32,
  @location(4) local: vec2<f32>
) -> @location(0) vec4<f32> {

  var out = color;

  // 원형 마스크
  let r2 = dot(local, local);
  if (r2 > 1.0) {
    discard;
  }

  // DDC 필터 (특정 DDC만 보기) - dim 처리
  if (u.selectedDdc >= 0) {
    let selDdc = u32(u.selectedDdc);
    if (ddcId != selDdc) {
      out.a = out.a * u.dimAlpha;
    }
  }

  // 클러스터 선택 (강조)
  if (u.selectedCluster >= 0) {
    let sel = u32(u.selectedCluster);
    let isSel = (clusterId == sel);
    if (u.selectMode == 1) {
      if (!isSel) { out.a = out.a * u.dimAlpha; }
    } else {
      if (!isSel) { out.a = 0.0; }
    }
  }

  // 검색 하이라이트
  if (u.searchActive != 0) {  
    if (u.searchOnly != 0) {
      if (searchHit == 0u) { out.a = 0.5;}
    } else {
      if (searchHit == 0u) { out.a = out.a * u.searchDimAlpha; }
    }
  }

  return out;
}

@fragment
fn fs_pick(
  @location(1) @interpolate(flat) ddcId: u32,
  @location(2) @interpolate(flat) clusterId: u32,
  @location(3) @interpolate(flat) searchHit: u32,
  @location(4) local: vec2<f32>,
  @location(5) @interpolate(flat) iid: u32
) -> @location(0) u32 {

  // 원형 마스크
  let r2 = dot(local, local);
  if (r2 > 1.0) {
    discard;
  }

  // 클러스터 선택이 only selected 모드이고, 선택이 아닌 점은 완전 숨김
  if (u.selectedCluster >= 0) {
    let sel = u32(u.selectedCluster);
    let isSel = (clusterId == sel);
    if (u.selectMode != 1) {
      if (!isSel) { discard; }
    }
  }

  // 검색 하이라이트
  if (u.searchActive != 0) {
    let isMatch = (searchHit != 0u);
    if (u.searchOnly != 0) {
      if (!isMatch) { discard; }
    }
  }
  return iid+1u;
}