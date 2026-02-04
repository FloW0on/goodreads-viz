// web/public/src/core/state/contracts.js

/**
 * @typedef {Object} DataPoint
 * @property {number} x            - data space x (e.g. UMAP x)
 * @property {number} y            - data space y (e.g. UMAP y)
 * @property {number} category     - numeric category (cluster id, DDC class, etc.)
 * @property {string} [text]       - tooltip text (optional)
 * @property {number|string} identifier - stable id (book id, row id, etc.)
 */

/**
 * @typedef {Object} Rectangle
 * @property {number} x0
 * @property {number} y0
 * @property {number} x1
 * @property {number} y1
 */

/**
 * Atlas-style querySelection contract.
 *
 * screenX, screenY:
 *   - screen space coordinates (pixels)
 *
 * unitDistance:
 *   - conversion factor between screen space and data space
 *   - exact meaning is renderer-defined
 *
 * @callback QuerySelection
 * @param {number} screenX
 * @param {number} screenY
 * @param {number} unitDistance
 * @returns {Promise<DataPoint[]>}
 */

/**
 * Atlas-style queryClusterLabels contract.
 *
 * clusters:
 *   - Array of clusters
 *   - each cluster is approximated by an array of rectangles
 *
 * @callback QueryClusterLabels
 * @param {Rectangle[][]} clusters
 * @returns {Promise<string[]>}
 */

/**
 * Services provided by the rendering / engine layer.
 *
 * Features (Hover, Selection, UI) must ONLY depend on this object,
 * never directly on Renderer / DataLoader / GPU internals.
 *
 * @typedef {Object} EmbeddingViewServices
 * @property {QuerySelection} querySelection
 * @property {QueryClusterLabels} [queryClusterLabels]
 */

// ------------------------------------------------------------
// Runtime exports (names must exist for JS imports)
// ------------------------------------------------------------

/**
 * This file intentionally exports no concrete implementation.
 *
 * It only defines the CONTRACT SHAPE.
 * Actual implementations are provided by:
 *   - Renderer (GPU picking)
 *   - Cluster label service
 *
 * Keeping this file explicit (instead of empty) is important:
 * - Documents architectural boundaries
 * - Prevents "mystery methods" from creeping in
 */
export {};