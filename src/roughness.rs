//! Terrain roughness indices computed from a 3x3 neighborhood.
//!
//! Three standard roughness metrics are provided:
//!
//! - **TRI** ([`tri`]): Terrain Ruggedness Index (Riley et al. 1999). The
//!   square root of the mean of squared elevation differences between the center
//!   cell and its 8 neighbors.
//!
//! - **TPI** ([`tpi`]): Topographic Position Index. The difference between the
//!   center cell elevation and the mean elevation of its 8 neighbors. Positive
//!   values indicate ridges/hilltops, negative values indicate valleys.
//!
//! - **Roughness** ([`roughness`]): The largest inter-cell elevation difference
//!   in the 3x3 neighborhood (max - min).
//!
//! # NaN handling
//!
//! On grids at least 3x3, if the center cell or any neighbor is NaN, the
//! output is NaN for that cell. Grids smaller than 3x3 use the documented flat
//! fallback and return zeros.

use ndarray::Array2;

use crate::kernel::{get_clamped, grid_map};

/// 3x3 neighborhood offsets (8 neighbors).
const OFFSETS: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// Compute the Terrain Ruggedness Index (Riley et al. 1999).
///
/// TRI = sqrt(sum((z_center - z_neighbor)^2)) over the 8 neighbors.
/// Grids smaller than 3x3 return all zeros.
pub fn tri(dem: &Array2<f64>) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let center = dem[[row, col]];
        let sum_sq: f64 = OFFSETS
            .iter()
            .map(|&(dy, dx)| {
                let n = get_clamped(dem, row, col, dy, dx);
                (center - n) * (center - n)
            })
            .sum();
        sum_sq.sqrt()
    })
}

/// Compute the Topographic Position Index.
///
/// TPI = z_center - mean(z_neighbors). Positive = ridge/hilltop, negative =
/// valley/depression, near-zero = slope or flat. Grids smaller than 3x3
/// return all zeros.
pub fn tpi(dem: &Array2<f64>) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let center = dem[[row, col]];
        let mean: f64 = OFFSETS
            .iter()
            .map(|&(dy, dx)| get_clamped(dem, row, col, dy, dx))
            .sum::<f64>()
            / 8.0;
        center - mean
    })
}

/// Compute surface roughness as the elevation range in the 3x3 neighborhood.
///
/// Roughness = max(z) - min(z) over the center cell and its 8 neighbors.
/// Grids smaller than 3x3 return all zeros.
pub fn roughness(dem: &Array2<f64>) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let center = dem[[row, col]];
        if center.is_nan() {
            return f64::NAN;
        }
        let mut min_val = center;
        let mut max_val = center;
        for &(dy, dx) in &OFFSETS {
            let n = get_clamped(dem, row, col, dy, dx);
            if n.is_nan() {
                return f64::NAN;
            }
            if n < min_val {
                min_val = n;
            }
            if n > max_val {
                max_val = n;
            }
        }
        max_val - min_val
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_dem_zero_roughness() {
        let dem = Array2::from_elem((10, 10), 100.0);
        for grid in [tri(&dem), tpi(&dem), roughness(&dem)] {
            for &v in grid.iter() {
                assert!(v.abs() < 1e-10, "flat DEM should have 0 roughness, got {v}");
            }
        }
    }

    #[test]
    fn tri_positive_for_varied_terrain() {
        let dem =
            Array2::from_shape_fn((10, 10), |(r, c)| (r as f64 * 3.0 + c as f64).sin() * 50.0);
        let t = tri(&dem);
        // Interior cells should have positive TRI
        assert!(t[[5, 5]] > 0.0, "varied terrain should have positive TRI");
    }

    #[test]
    fn tpi_ridge_positive() {
        // Center cell is a ridge (higher than neighbors)
        let mut dem = Array2::from_elem((5, 5), 10.0);
        dem[[2, 2]] = 100.0;
        let t = tpi(&dem);
        assert!(t[[2, 2]] > 0.0, "ridge should have positive TPI");
    }

    #[test]
    fn tpi_valley_negative() {
        let mut dem = Array2::from_elem((5, 5), 100.0);
        dem[[2, 2]] = 10.0;
        let t = tpi(&dem);
        assert!(t[[2, 2]] < 0.0, "valley should have negative TPI");
    }

    #[test]
    fn roughness_matches_range() {
        let mut dem = Array2::from_elem((5, 5), 50.0);
        dem[[2, 1]] = 10.0;
        dem[[2, 3]] = 90.0;
        let r = roughness(&dem);
        assert!(
            (r[[2, 2]] - 80.0).abs() < 1e-6,
            "roughness should be 90-10=80, got {}",
            r[[2, 2]]
        );
    }

    #[test]
    fn small_grid_returns_zeros() {
        let dem = Array2::from_elem((2, 2), 50.0);
        for grid in [tri(&dem), tpi(&dem), roughness(&dem)] {
            for &v in grid.iter() {
                assert!(v.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn nan_neighbor_propagates_on_normal_grid() {
        let mut dem = Array2::from_elem((5, 5), 50.0);
        dem[[2, 3]] = f64::NAN;
        assert!(tri(&dem)[[2, 2]].is_nan());
        assert!(tpi(&dem)[[2, 2]].is_nan());
        assert!(roughness(&dem)[[2, 2]].is_nan());
    }
}
