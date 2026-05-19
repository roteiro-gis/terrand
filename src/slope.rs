//! Slope computation from a Digital Elevation Model using the Horn algorithm.
//!
//! Slope measures the steepness of the terrain surface at each cell. Three
//! output unit conventions are provided:
//!
//! - **Degrees** ([`slope`]): range \[0, 90\], where 0 is flat and 90 is vertical.
//! - **Radians** ([`slope_radians`]): range \[0, pi/2\].
//! - **Percent** ([`slope_percent`]): rise/run * 100, where 100% = 45 degrees.
//!
//! All functions use the Horn (1981) 3x3 weighted gradient kernel with
//! GDAL-compatible linear extrapolation at grid edges.
//!
//! # NaN handling
//!
//! Input NaN cells propagate to NaN in the output. Cells whose 3x3
//! neighborhood contains NaN will also produce NaN.

use ndarray::Array2;

use crate::cell_size::CellSize;
use crate::kernel::{grid_map, horn_gradients};

/// Compute slope in **degrees** from a DEM using the Horn algorithm.
///
/// Returns values in the range \[0, 90\]. Grids smaller than 3x3 return all
/// zeros (flat).
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use terrand::{slope, CellSize};
///
/// let dem = Array2::from_elem((10, 10), 100.0);
/// let result = slope(&dem, CellSize::square(30.0).unwrap());
/// assert!(result[[5, 5]].abs() < 1e-6); // flat
/// ```
pub fn slope(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);
        (dzdx * dzdx + dzdy * dzdy).sqrt().atan().to_degrees()
    })
}

/// Compute slope in **radians** from a DEM using the Horn algorithm.
///
/// Returns values in the range \[0, pi/2\]. Grids smaller than 3x3 return all
/// zeros.
pub fn slope_radians(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);
        (dzdx * dzdx + dzdy * dzdy).sqrt().atan()
    })
}

/// Compute slope as a **percentage** (rise/run * 100) from a DEM.
///
/// A 45-degree slope corresponds to 100%. There is no upper bound; near-vertical
/// slopes approach infinity. Grids smaller than 3x3 return all zeros.
pub fn slope_percent(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);
        (dzdx * dzdx + dzdy * dzdy).sqrt() * 100.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_dem_is_zero() {
        let dem = Array2::from_elem((10, 10), 100.0);
        let s = slope(&dem, CellSize::square(1.0).unwrap());
        for &v in s.iter() {
            assert!(v.abs() < 1e-6, "flat DEM slope should be 0, got {v}");
        }
    }

    #[test]
    fn uniform_gradient() {
        let dem = Array2::from_shape_fn((10, 10), |(_, c)| c as f64 * 10.0);
        let s = slope(&dem, CellSize::square(1.0).unwrap());
        let interior = s[[5, 5]];
        assert!(interior > 0.0);
        for row in 2..8 {
            for col in 2..8 {
                assert!(
                    (s[[row, col]] - interior).abs() < 1e-6,
                    "uniform gradient should have uniform slope"
                );
            }
        }
    }

    #[test]
    fn range_check() {
        let dem =
            Array2::from_shape_fn((20, 20), |(r, c)| (r as f64 * 3.0 + c as f64).sin() * 100.0);
        let s = slope(&dem, CellSize::square(1.0).unwrap());
        for &v in s.iter() {
            assert!((0.0..=90.0).contains(&v), "slope {v} out of [0, 90]");
        }
    }

    #[test]
    fn small_grid_returns_zeros() {
        let dem = Array2::from_elem((2, 2), 50.0);
        let s = slope(&dem, CellSize::square(1.0).unwrap());
        for &v in s.iter() {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn radians_matches_degrees() {
        let dem = Array2::from_shape_fn((10, 10), |(r, c)| (r as f64 + c as f64) * 5.0);
        let deg = slope(&dem, CellSize::square(1.0).unwrap());
        let rad = slope_radians(&dem, CellSize::square(1.0).unwrap());
        for r in 0..10 {
            for c in 0..10 {
                let expected = deg[[r, c]].to_radians();
                assert!(
                    (rad[[r, c]] - expected).abs() < 1e-10,
                    "radians mismatch at ({r}, {c})"
                );
            }
        }
    }

    #[test]
    fn percent_at_45_degrees() {
        // 45-degree slope: rise/run = 1.0, so percent = 100.
        // dz/dx = 1.0 => atan(1) = 45 deg => percent = 100
        // With Horn kernel on a grid where z = x*1.0, dzdx = 1.0
        let dem = Array2::from_shape_fn((10, 10), |(_, c)| c as f64);
        let pct = slope_percent(&dem, CellSize::square(1.0).unwrap());
        let v = pct[[5, 5]];
        assert!(
            (v - 100.0).abs() < 1.0,
            "45-degree slope should be ~100%, got {v}"
        );
    }
}
