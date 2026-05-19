//! Internal: Horn algorithm gradient kernel, neighborhood access, and parallel
//! dispatch helpers.
//!
//! Not part of the public API.

use ndarray::Array2;

use crate::CellSize;

// ---------------------------------------------------------------------------
// Parallel / sequential grid construction
// ---------------------------------------------------------------------------

/// Build an `Array2<T>` by evaluating `f(row, col)` for every cell.
///
/// When the `parallel` feature is enabled, cells are evaluated in parallel via
/// Rayon. Otherwise the standard `from_shape_fn` sequential path is used.
pub(crate) fn grid_map<T, F>(h: usize, w: usize, f: F) -> Array2<T>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let data: Vec<T> = (0..h * w)
            .into_par_iter()
            .map(|i| f(i / w, i % w))
            .collect();
        Array2::from_shape_vec((h, w), data).unwrap()
    }
    #[cfg(not(feature = "parallel"))]
    {
        Array2::from_shape_fn((h, w), |(y, x)| f(y, x))
    }
}

// ---------------------------------------------------------------------------
// 3x3 neighborhood access with GDAL-compatible edge extrapolation
// ---------------------------------------------------------------------------

/// Fetch a DEM value at `(row + dy, col + dx)`, using linear extrapolation
/// when the coordinate falls outside the grid boundary. This matches GDAL's
/// `--compute_edges` behaviour for the Horn algorithm.
///
/// If the target cell is in bounds, its value is returned directly. If it is
/// out of bounds, the value is linearly extrapolated from the two nearest
/// in-bounds cells along the out-of-bounds axis. NaN values propagate
/// naturally through the extrapolation arithmetic.
#[inline]
pub(crate) fn get_extrapolated(
    dem: &Array2<f64>,
    row: usize,
    col: usize,
    dy: isize,
    dx: isize,
) -> f64 {
    let (h, w) = dem.dim();
    let iy = row as isize + dy;
    let ix = col as isize + dx;
    let hi = h as isize;
    let wi = w as isize;

    // Extrapolate vertically if row is out of bounds.
    if iy < 0 {
        let cx = ix.clamp(0, wi - 1) as usize;
        return 2.0 * dem[[0, cx]] - dem[[1, cx]];
    } else if iy >= hi {
        let cx = ix.clamp(0, wi - 1) as usize;
        return 2.0 * dem[[h - 1, cx]] - dem[[h - 2, cx]];
    }

    let ny = iy as usize;

    // Extrapolate horizontally if column is out of bounds.
    if ix < 0 {
        return 2.0 * dem[[ny, 0]] - dem[[ny, 1]];
    } else if ix >= wi {
        return 2.0 * dem[[ny, w - 1]] - dem[[ny, w - 2]];
    }

    dem[[ny, ix as usize]]
}

/// Fetch a DEM value at `(row + dy, col + dx)`, clamping to grid boundaries.
#[inline]
pub(crate) fn get_clamped(dem: &Array2<f64>, row: usize, col: usize, dy: isize, dx: isize) -> f64 {
    let (h, w) = dem.dim();
    let ny = (row as isize + dy).clamp(0, h as isize - 1) as usize;
    let nx = (col as isize + dx).clamp(0, w as isize - 1) as usize;
    dem[[ny, nx]]
}

// ---------------------------------------------------------------------------
// Horn algorithm first-order gradients
// ---------------------------------------------------------------------------

/// Compute the Horn algorithm gradients (dz/dx, dz/dy) at a single pixel.
///
/// Uses the standard 3x3 weighted kernel with GDAL-compatible linear
/// extrapolation at grid edges.
///
/// # 3x3 kernel layout
///
/// ```text
///   a  b  c
///   d  e  f
///   g  h  i
/// ```
///
/// - `dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_x)`
/// - `dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_y)`
#[inline]
pub(crate) fn horn_gradients(
    dem: &Array2<f64>,
    row: usize,
    col: usize,
    cs: CellSize,
) -> (f64, f64) {
    let g = |dy, dx| get_extrapolated(dem, row, col, dy, dx);

    let a = g(-1, -1);
    let b = g(-1, 0);
    let c = g(-1, 1);
    let d = g(0, -1);
    let f = g(0, 1);
    let gv = g(1, -1);
    let h = g(1, 0);
    let i = g(1, 1);

    let dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + gv)) / (8.0 * cs.x());
    let dzdy = ((gv + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cs.y());

    (dzdx, dzdy)
}

// ---------------------------------------------------------------------------
// Horn algorithm second-order derivatives
// ---------------------------------------------------------------------------

/// Compute second-order partial derivatives at a single pixel using the 3x3
/// neighborhood (clamped at edges).
///
/// Returns `(d2z/dx2, d2z/dy2, d2z/dxdy)`.
#[inline]
pub(crate) fn horn_second_derivatives(
    dem: &Array2<f64>,
    row: usize,
    col: usize,
    cs: CellSize,
) -> (f64, f64, f64) {
    let g = |dy, dx| get_clamped(dem, row, col, dy, dx);

    let a = g(-1, -1);
    let b = g(-1, 0);
    let c = g(-1, 1);
    let d = g(0, -1);
    let e = g(0, 0);
    let f = g(0, 1);
    let gv = g(1, -1);
    let h = g(1, 0);
    let i = g(1, 1);

    let d2zdx2 = (d - 2.0 * e + f) / (cs.x() * cs.x());
    let d2zdy2 = (b - 2.0 * e + h) / (cs.y() * cs.y());
    let d2zdxdy = ((c + gv) - (a + i)) / (4.0 * cs.x() * cs.y());

    (d2zdx2, d2zdy2, d2zdxdy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_extrapolated_interior() {
        let dem = Array2::from_shape_fn((5, 5), |(r, c)| (r * 5 + c) as f64);
        assert_eq!(get_extrapolated(&dem, 2, 2, 0, 0), 12.0);
        assert_eq!(get_extrapolated(&dem, 2, 2, -1, 0), 7.0);
    }

    #[test]
    fn test_get_extrapolated_edge() {
        // Linear gradient: row 0 = 0, row 1 = 10, row 2 = 20, ...
        let dem = Array2::from_shape_fn((5, 5), |(r, _)| r as f64 * 10.0);
        // Extrapolating above row 0: 2*dem[0,c] - dem[1,c] = 2*0 - 10 = -10
        assert_eq!(get_extrapolated(&dem, 0, 2, -1, 0), -10.0);
        // Extrapolating below row 4: 2*dem[4,c] - dem[3,c] = 2*40 - 30 = 50
        assert_eq!(get_extrapolated(&dem, 4, 2, 1, 0), 50.0);
    }

    #[test]
    fn test_horn_gradients_flat() {
        let dem = Array2::from_elem((5, 5), 100.0);
        let (dzdx, dzdy) = horn_gradients(&dem, 2, 2, CellSize::square(1.0).unwrap());
        assert!(dzdx.abs() < 1e-10);
        assert!(dzdy.abs() < 1e-10);
    }

    #[test]
    fn test_horn_gradients_x_slope() {
        // Rising 10 units per column
        let dem = Array2::from_shape_fn((5, 5), |(_, c)| c as f64 * 10.0);
        let (dzdx, dzdy) = horn_gradients(&dem, 2, 2, CellSize::square(1.0).unwrap());
        assert!((dzdx - 10.0).abs() < 1e-6, "dzdx should be 10, got {dzdx}");
        assert!(dzdy.abs() < 1e-6, "dzdy should be ~0, got {dzdy}");
    }

    #[test]
    fn test_horn_second_derivatives_flat() {
        let dem = Array2::from_elem((5, 5), 100.0);
        let (d2x, d2y, d2xy) = horn_second_derivatives(&dem, 2, 2, CellSize::square(1.0).unwrap());
        assert!(d2x.abs() < 1e-10);
        assert!(d2y.abs() < 1e-10);
        assert!(d2xy.abs() < 1e-10);
    }

    #[test]
    fn test_horn_second_derivatives_parabolic() {
        // z = x^2 => d2z/dx2 = 2
        let dem = Array2::from_shape_fn((5, 5), |(_, c)| (c as f64).powi(2));
        let (d2x, _, _) = horn_second_derivatives(&dem, 2, 2, CellSize::square(1.0).unwrap());
        assert!(
            (d2x - 2.0).abs() < 1e-6,
            "d2z/dx2 of x^2 should be 2, got {d2x}"
        );
    }
}
