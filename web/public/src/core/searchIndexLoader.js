// core/searchIndexLoader.js
// trigram(FNV-1a 32-bit) inverted index loader (shard-based)

async function fetchArrayBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.arrayBuffer();
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.json();
}

// JS FNV-1a 32-bit (Python과 동일)
export function fnv1a32(str) {
  let h = 0x811c9dc5; // 2166136261
  for (let i = 0; i < str.length; i++) {
  }
  return h >>> 0;
}

export function fnv1a32_utf8(str) {
  const enc = new TextEncoder();
  const bytes = enc.encode(str);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0; // 16777619
  }
  return h >>> 0;
}

export function makeTrigramHashes(q) {
  const s = (q ?? "").toLowerCase().replace(/\s+/g, " ").trim();
  const out = [];
  if (s.length < 3) return out;
  for (let i = 0; i < s.length - 2; i++) {
    out.push(fnv1a32_utf8(s.slice(i, i + 3)));
  }
  // unique
  return Array.from(new Set(out));
}

export class SearchIndexLoader {
  constructor(tag, baseDir = "./packed/search_index", mod = 256) {
    this.tag = tag;
    this.baseDir = baseDir;
    this.mod = mod;

    this.vocabCache = new Map(); // shard -> vocab(json)
    this.binCache = new Map();   // shard -> Uint32Array(postings)
  }

  shardOfHash(h) {
    return (h % this.mod) >>> 0;
  }

  async _loadShard(shard) {
    if (!this.vocabCache.has(shard)) {
      const vocabUrl = `${this.baseDir}/search_tri_${this.tag}_${String(shard).padStart(3, "0")}.json`;
      const vocab = await fetchJson(vocabUrl);
      this.vocabCache.set(shard, vocab);
    }
    if (!this.binCache.has(shard)) {
      const binUrl = `${this.baseDir}/search_tri_${this.tag}_${String(shard).padStart(3, "0")}.u32`;
      const buf = await fetchArrayBuffer(binUrl);
      this.binCache.set(shard, new Uint32Array(buf));
    }
  }

  // hash -> Uint32Array view (copy 없이 subarray)
  async getPosting(h) {
    const shard = this.shardOfHash(h);
    await this._loadShard(shard);

    const vocab = this.vocabCache.get(shard);
    const bin = this.binCache.get(shard);

    const entry = vocab[String(h)];
    if (!entry) return null;

    const [off, cnt] = entry;
    return bin.subarray(off, off + cnt);
  }
}