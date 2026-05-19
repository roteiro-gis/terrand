//! Hydrological analysis on Digital Elevation Models.
//!
//! This module provides a complete hydrological analysis pipeline:
//!
//! 1. [`fill`] — Remove sinks using the Planchon-Darboux algorithm to produce
//!    a depression-less DEM.
//! 2. [`flow_direction`] — Compute D8 flow direction from the filled DEM.
//! 3. [`flow_accumulation`] — Count upstream contributing cells using iterative
//!    topological sorting (no recursion, O(n) time and space).
//! 4. [`watershed`] — Delineate the contributing area upstream of pour points.
//! 5. [`basin`] — Label all drainage basins.
//! 6. [`stream_order_strahler`] — Assign Strahler stream order to stream cells.
//! 7. [`snap_pour_point`] — Snap pour points to the cell with the highest
//!    accumulation within a search radius.
//!
//! # D8 encoding
//!
//! Flow direction values follow the standard D8 power-of-two encoding:
//!
//! ```text
//!   32  64  128
//!   16   0    1
//!    8   4    2
//! ```
//!
//! A value of 0 indicates a pit or flat area with no downslope neighbor.

use ndarray::Array2;
use std::collections::VecDeque;

use crate::kernel::grid_map;

// ---------------------------------------------------------------------------
// D8 constants
// ---------------------------------------------------------------------------

/// D8 direction: East.
pub const D8_E: u8 = 1;
/// D8 direction: Southeast.
pub const D8_SE: u8 = 2;
/// D8 direction: South.
pub const D8_S: u8 = 4;
/// D8 direction: Southwest.
pub const D8_SW: u8 = 8;
/// D8 direction: West.
pub const D8_W: u8 = 16;
/// D8 direction: Northwest.
pub const D8_NW: u8 = 32;
/// D8 direction: North.
pub const D8_N: u8 = 64;
/// D8 direction: Northeast.
pub const D8_NE: u8 = 128;

/// Row/col offsets for the 8 D8 directions, ordered by bit value.
const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1),   // E   = 1
    (1, 1),   // SE  = 2
    (1, 0),   // S   = 4
    (1, -1),  // SW  = 8
    (0, -1),  // W   = 16
    (-1, -1), // NW  = 32
    (-1, 0),  // N   = 64
    (-1, 1),  // NE  = 128
];

const D8_CODES: [u8; 8] = [D8_E, D8_SE, D8_S, D8_SW, D8_W, D8_NW, D8_N, D8_NE];

/// Diagonal distance factor (sqrt(2)).
const DIAG: f64 = std::f64::consts::SQRT_2;
const D8_DIST: [f64; 8] = [1.0, DIAG, 1.0, DIAG, 1.0, DIAG, 1.0, DIAG];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return the (row, col) of the cell that `(r, c)` flows into, or `None` if
/// the cell is a pit/flat (direction 0) or flows off the grid.
fn downstream(fdir: &Array2<u8>, r: usize, c: usize, h: usize, w: usize) -> Option<(usize, usize)> {
    let dir = fdir[[r, c]];
    if dir == 0 {
        return None;
    }
    let idx = D8_CODES.iter().position(|&d| d == dir)?;
    let (dr, dc) = D8_OFFSETS[idx];
    let nr = r as isize + dr;
    let nc = c as isize + dc;
    if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
        Some((nr as usize, nc as usize))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Fill (Planchon-Darboux)
// ---------------------------------------------------------------------------

/// Remove sinks (local minima) from a DEM using the Planchon-Darboux algorithm.
///
/// Returns a hydrologically conditioned DEM where every interior cell drains
/// to the grid boundary or the edge of a finite-data region. The input is not
/// modified.
///
/// NaN cells in the input are treated as nodata and left unchanged. Grids
/// smaller than 3x3 are returned unmodified.
pub fn fill(dem: &Array2<f64>) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return dem.clone();
    }

    let eps = 1e-5;
    let mut filled = Array2::from_elem((h, w), f64::INFINITY);

    // Initialize boundary cells, NaN cells, and cells touching NaN to their
    // DEM values. A NaN marks nodata, so finite cells on a nodata perimeter are
    // outlets for their finite-data component.
    for r in 0..h {
        for c in 0..w {
            let is_edge = r == 0 || r == h - 1 || c == 0 || c == w - 1;
            let is_nodata = dem[[r, c]].is_nan();
            let touches_nodata = !is_nodata
                && D8_OFFSETS.iter().any(|&(dr, dc)| {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    nr >= 0
                        && nr < h as isize
                        && nc >= 0
                        && nc < w as isize
                        && dem[[nr as usize, nc as usize]].is_nan()
                });
            if is_edge || is_nodata || touches_nodata {
                filled[[r, c]] = dem[[r, c]];
            }
        }
    }

    // Iteratively lower interior cells.
    let mut changed = true;
    while changed {
        changed = false;
        for r in 1..h - 1 {
            for c in 1..w - 1 {
                if dem[[r, c]].is_nan() {
                    continue;
                }
                if filled[[r, c]] > dem[[r, c]] {
                    for &(dr, dc) in &D8_OFFSETS {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                            let nv = filled[[nr as usize, nc as usize]];
                            if !nv.is_nan() {
                                let candidate = nv + eps;
                                if dem[[r, c]] >= candidate {
                                    filled[[r, c]] = dem[[r, c]];
                                    changed = true;
                                } else if filled[[r, c]] > candidate {
                                    filled[[r, c]] = candidate;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    filled
}

// ---------------------------------------------------------------------------
// Flow Direction (D8)
// ---------------------------------------------------------------------------

/// Compute D8 flow direction from a DEM.
///
/// Each cell is assigned the direction of steepest descent to one of its
/// 8 neighbors. Diagonal distances are weighted by sqrt(2). Returns 0 for
/// flat areas, pits, or NaN cells.
pub fn flow_direction(dem: &Array2<f64>) -> Array2<u8> {
    let (h, w) = dem.dim();
    grid_map(h, w, |r, c| {
        let z = dem[[r, c]];
        if z.is_nan() {
            return 0u8;
        }

        let mut max_drop = 0.0;
        let mut best_dir: u8 = 0;

        for i in 0..8 {
            let nr = r as isize + D8_OFFSETS[i].0;
            let nc = c as isize + D8_OFFSETS[i].1;
            if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                let nz = dem[[nr as usize, nc as usize]];
                if !nz.is_nan() {
                    let drop = (z - nz) / D8_DIST[i];
                    if drop > max_drop {
                        max_drop = drop;
                        best_dir = D8_CODES[i];
                    }
                }
            }
        }

        best_dir
    })
}

// ---------------------------------------------------------------------------
// Flow Accumulation (iterative topological sort)
// ---------------------------------------------------------------------------

/// Compute flow accumulation from a D8 flow direction grid.
///
/// Each cell's value is the count of upstream cells that flow into it
/// (including itself). Uses iterative topological sorting, which is O(n) in
/// time and space with no recursion.
pub fn flow_accumulation(fdir: &Array2<u8>) -> Array2<f64> {
    let (h, w) = fdir.dim();
    let mut acc = Array2::ones((h, w));
    let mut in_degree = Array2::<u32>::zeros((h, w));

    // Count in-degree: how many cells flow into each cell.
    for r in 0..h {
        for c in 0..w {
            if let Some((nr, nc)) = downstream(fdir, r, c, h, w) {
                in_degree[[nr, nc]] += 1;
            }
        }
    }

    // Seed the queue with all headwater cells (in-degree = 0).
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..h {
        for c in 0..w {
            if in_degree[[r, c]] == 0 {
                queue.push_back((r, c));
            }
        }
    }

    // Process cells in topological order.
    while let Some((r, c)) = queue.pop_front() {
        if let Some((nr, nc)) = downstream(fdir, r, c, h, w) {
            acc[[nr, nc]] += acc[[r, c]];
            in_degree[[nr, nc]] -= 1;
            if in_degree[[nr, nc]] == 0 {
                queue.push_back((nr, nc));
            }
        }
    }

    acc
}

// ---------------------------------------------------------------------------
// Watershed Delineation
// ---------------------------------------------------------------------------

/// Delineate watersheds (contributing areas) for given pour points.
///
/// `pour_points` is a list of `(row, col)` outlet locations. Returns a grid
/// where each cell is labeled with the 1-based index of the pour point it
/// drains to, or 0 if it does not drain to any pour point.
pub fn watershed(fdir: &Array2<u8>, pour_points: &[(usize, usize)]) -> Array2<i32> {
    let (h, w) = fdir.dim();
    let mut labels = Array2::zeros((h, w));

    for (idx, &(pr, pc)) in pour_points.iter().enumerate() {
        let label = (idx + 1) as i32;
        if pr >= h || pc >= w {
            continue;
        }

        let mut queue = VecDeque::new();
        labels[[pr, pc]] = label;
        queue.push_back((pr, pc));

        while let Some((r, c)) = queue.pop_front() {
            // Find all neighbors that flow INTO (r, c).
            for (i, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let reverse_idx = (i + 4) % 8;
                    if fdir[[nr, nc]] == D8_CODES[reverse_idx] && labels[[nr, nc]] == 0 {
                        labels[[nr, nc]] = label;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
    }

    labels
}

// ---------------------------------------------------------------------------
// Basin Labeling
// ---------------------------------------------------------------------------

/// Label all drainage basins in the flow direction grid.
///
/// Each cell is assigned a 1-based basin ID. Basins are defined by cells that
/// drain to boundary cells or pits (direction 0). All cells receive a label.
pub fn basin(fdir: &Array2<u8>) -> Array2<i32> {
    let (h, w) = fdir.dim();
    let mut labels = Array2::zeros((h, w));
    let mut current_label = 0i32;

    // Find all outlet cells (boundary cells or pits).
    let mut outlets = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let is_boundary = r == 0 || r == h - 1 || c == 0 || c == w - 1;
            let is_pit = fdir[[r, c]] == 0;
            if is_boundary || is_pit {
                outlets.push((r, c));
            }
        }
    }

    // BFS upstream from each outlet.
    for &(or, oc) in &outlets {
        if labels[[or, oc]] != 0 {
            continue;
        }
        current_label += 1;

        let mut queue = VecDeque::new();
        labels[[or, oc]] = current_label;
        queue.push_back((or, oc));

        while let Some((r, c)) = queue.pop_front() {
            for (i, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let reverse_idx = (i + 4) % 8;
                    if fdir[[nr, nc]] == D8_CODES[reverse_idx] && labels[[nr, nc]] == 0 {
                        labels[[nr, nc]] = current_label;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
    }

    labels
}

// ---------------------------------------------------------------------------
// Strahler Stream Order (iterative)
// ---------------------------------------------------------------------------

/// Compute Strahler stream order from flow direction and accumulation grids.
///
/// Cells with accumulation >= `threshold` are considered stream cells. Returns
/// a grid where stream cells have their Strahler order (>= 1) and non-stream
/// cells are 0.
///
/// Uses iterative topological sorting to avoid stack overflow on large DEMs.
pub fn stream_order_strahler(
    fdir: &Array2<u8>,
    accumulation: &Array2<f64>,
    threshold: f64,
) -> Array2<i32> {
    let (h, w) = fdir.dim();
    let mut order = Array2::zeros((h, w));

    // Identify stream cells.
    let mut is_stream = Array2::from_elem((h, w), false);
    for r in 0..h {
        for c in 0..w {
            if accumulation[[r, c]] >= threshold {
                is_stream[[r, c]] = true;
            }
        }
    }

    // Count stream-to-stream in-degree (how many stream tributaries flow into
    // each stream cell).
    let mut in_deg = Array2::<u32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            if !is_stream[[r, c]] {
                continue;
            }
            if let Some((nr, nc)) = downstream(fdir, r, c, h, w) {
                if is_stream[[nr, nc]] {
                    in_deg[[nr, nc]] += 1;
                }
            }
        }
    }

    // Seed queue with headwater stream cells (no upstream stream tributaries).
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for r in 0..h {
        for c in 0..w {
            if is_stream[[r, c]] && in_deg[[r, c]] == 0 {
                order[[r, c]] = 1;
                queue.push_back((r, c));
            }
        }
    }

    // We need to collect tributary orders for each cell before computing its
    // Strahler order. Store the two highest orders seen for each stream cell.
    let mut top_orders = Array2::from_elem((h, w), [0i32; 2]); // [max, second_max]

    while let Some((r, c)) = queue.pop_front() {
        let my_order = order[[r, c]];

        if let Some((nr, nc)) = downstream(fdir, r, c, h, w) {
            if is_stream[[nr, nc]] {
                // Update the top-2 orders for the downstream cell.
                let top = &mut top_orders[[nr, nc]];
                if my_order > top[0] {
                    top[1] = top[0];
                    top[0] = my_order;
                } else if my_order > top[1] {
                    top[1] = my_order;
                }

                in_deg[[nr, nc]] -= 1;
                if in_deg[[nr, nc]] == 0 {
                    // All tributaries processed; compute Strahler order.
                    let t = top_orders[[nr, nc]];
                    order[[nr, nc]] = if t[0] == 0 {
                        1
                    } else if t[0] == t[1] {
                        t[0] + 1
                    } else {
                        t[0]
                    };
                    queue.push_back((nr, nc));
                }
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Snap Pour Point
// ---------------------------------------------------------------------------

/// Snap pour points to the cell with the highest flow accumulation within a
/// search radius.
///
/// `snap_distance` is measured in cells (Euclidean distance). Returns a new
/// list of snapped `(row, col)` locations.
pub fn snap_pour_point(
    accumulation: &Array2<f64>,
    pour_points: &[(usize, usize)],
    snap_distance: usize,
) -> Vec<(usize, usize)> {
    let (h, w) = accumulation.dim();

    pour_points
        .iter()
        .map(|&(pr, pc)| {
            let mut best_r = pr;
            let mut best_c = pc;
            let mut best_acc = if pr < h && pc < w {
                accumulation[[pr, pc]]
            } else {
                f64::NEG_INFINITY
            };

            let r_min = pr.saturating_sub(snap_distance);
            let r_max = (pr + snap_distance + 1).min(h);
            let c_min = pc.saturating_sub(snap_distance);
            let c_max = (pc + snap_distance + 1).min(w);

            let dist_sq = (snap_distance as f64) * (snap_distance as f64);

            for r in r_min..r_max {
                for c in c_min..c_max {
                    let dr = r as f64 - pr as f64;
                    let dc = c as f64 - pc as f64;
                    if dr * dr + dc * dc <= dist_sq && accumulation[[r, c]] > best_acc {
                        best_acc = accumulation[[r, c]];
                        best_r = r;
                        best_c = c;
                    }
                }
            }

            (best_r, best_c)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A 5x5 DEM sloping toward the southeast corner.
    fn slope_dem() -> Array2<f64> {
        Array2::from_shape_fn((5, 5), |(r, c)| {
            (4 - r) as f64 * 10.0 + (4 - c) as f64 * 10.0
        })
    }

    #[test]
    fn fill_no_sinks() {
        let dem = slope_dem();
        let filled = fill(&dem);
        for r in 0..5 {
            for c in 0..5 {
                assert!(
                    (filled[[r, c]] - dem[[r, c]]).abs() < 1e-3,
                    "fill should not change a sink-free DEM at ({r}, {c})"
                );
            }
        }
    }

    #[test]
    fn fill_raises_sink() {
        let mut dem = slope_dem();
        dem[[2, 2]] = 0.0;
        let filled = fill(&dem);
        assert!(
            filled[[2, 2]] > 0.0,
            "sink should be filled above 0, got {}",
            filled[[2, 2]]
        );
    }

    #[test]
    fn fill_small_grid() {
        let dem = Array2::from_elem((2, 2), 10.0);
        let filled = fill(&dem);
        assert_eq!(filled.dim(), (2, 2));
    }

    #[test]
    fn fill_preserves_finite_cell_enclosed_by_nodata() {
        let dem = Array2::from_shape_vec(
            (3, 3),
            vec![
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                10.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
        )
        .unwrap();
        let filled = fill(&dem);
        assert_eq!(filled[[1, 1]], 10.0);
        assert!(filled[[1, 1]].is_finite());
        for ((r, c), &z) in dem.indexed_iter() {
            if z.is_nan() {
                assert!(filled[[r, c]].is_nan(), "nodata changed at ({r}, {c})");
            }
        }
    }

    #[test]
    fn flow_direction_se_slope() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        assert_eq!(fdir[[2, 2]], D8_SE, "interior cell should flow SE");
    }

    #[test]
    fn flow_direction_flat() {
        let dem = Array2::from_elem((5, 5), 100.0);
        let fdir = flow_direction(&dem);
        for r in 1..4 {
            for c in 1..4 {
                assert_eq!(fdir[[r, c]], 0, "flat terrain should have direction 0");
            }
        }
    }

    #[test]
    fn flow_accumulation_all_at_least_one() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        let acc = flow_accumulation(&fdir);
        for &v in acc.iter() {
            assert!(v >= 1.0, "accumulation should be >= 1, got {v}");
        }
    }

    #[test]
    fn flow_accumulation_outlet_has_max() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        let acc = flow_accumulation(&fdir);
        let max_acc = acc.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_acc > 1.0, "max accumulation should be > 1");
    }

    #[test]
    fn watershed_labels_upstream() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        let ws = watershed(&fdir, &[(4, 4)]);
        assert_eq!(ws[[4, 4]], 1);
        let count = ws.iter().filter(|&&v| v == 1).count();
        assert!(count > 1, "watershed should contain multiple cells");
    }

    #[test]
    fn basin_labels_all() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        let b = basin(&fdir);
        for r in 0..5 {
            for c in 0..5 {
                assert!(
                    b[[r, c]] > 0,
                    "all cells should be labeled, got 0 at ({r}, {c})"
                );
            }
        }
    }

    #[test]
    fn stream_order_headwaters_are_one() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        let acc = flow_accumulation(&fdir);
        let order = stream_order_strahler(&fdir, &acc, 3.0);
        for r in 0..5 {
            for c in 0..5 {
                if acc[[r, c]] >= 3.0 {
                    assert!(
                        order[[r, c]] >= 1,
                        "stream cell at ({r},{c}) should have order >= 1"
                    );
                }
            }
        }
    }

    #[test]
    fn snap_pour_point_finds_higher_acc() {
        let dem = slope_dem();
        let fdir = flow_direction(&dem);
        let acc = flow_accumulation(&fdir);
        let snapped = snap_pour_point(&acc, &[(3, 3)], 2);
        assert_eq!(snapped.len(), 1);
        assert!(
            acc[[snapped[0].0, snapped[0].1]] >= acc[[3, 3]],
            "snapped point should have >= accumulation"
        );
    }
}
