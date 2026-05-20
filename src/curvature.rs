//! Surface curvature computation from a Digital Elevation Model.
//!
//! Three curvature metrics are provided:
//!
//! - **Profile curvature** ([`profile_curvature`]): curvature in the direction
//!   of maximum slope. Positive = convex (accelerating flow), negative = concave
//!   (decelerating flow).
//!
//! - **Plan curvature** ([`plan_curvature`]): curvature perpendicular to the
//!   slope direction. Negative = convergent flow, positive = divergent flow.
//!
//! - **General curvature** ([`general_curvature`]): the Laplacian of the
//!   surface, equal to -(d2z/dx2 + d2z/dy2). Combines both profile and plan
//!   effects.
//!
//! All use the Horn (1981) kernel for first-order gradients and finite-difference
//! second derivatives over the 3x3 neighborhood. First-order gradients use
//! GDAL-compatible edge extrapolation; second derivatives clamp at grid edges.
//!
//! # NaN handling
//!
//! On grids at least 3x3, input NaN cells propagate to NaN in the output.
//! Grids smaller than 3x3 use the documented flat fallback and return zeros.

use ndarray::Array2;

use crate::cell_size::CellSize;
use crate::kernel::{grid_map, horn_gradients, horn_second_derivatives};

/// Compute profile curvature from a DEM.
///
/// Profile curvature is the rate of change of slope in the steepest-descent
/// direction. Flat cells (zero gradient) produce zero curvature. Grids smaller
/// than 3x3 return all zeros.
pub fn profile_curvature(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);
        let (d2x, d2y, d2xy) = horn_second_derivatives(dem, row, col, cell_size);

        let p = dzdx * dzdx + dzdy * dzdy;
        if p < f64::EPSILON {
            return 0.0;
        }
        -(dzdx * dzdx * d2x + 2.0 * dzdx * dzdy * d2xy + dzdy * dzdy * d2y) / (p * p.sqrt())
    })
}

/// Compute plan (planimetric) curvature from a DEM.
///
/// Plan curvature measures convergence (negative) or divergence (positive) of
/// flow perpendicular to the slope direction. Flat cells produce zero. Grids
/// smaller than 3x3 return all zeros.
pub fn plan_curvature(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);
        let (d2x, d2y, d2xy) = horn_second_derivatives(dem, row, col, cell_size);

        let p = dzdx * dzdx + dzdy * dzdy;
        if p < f64::EPSILON {
            return 0.0;
        }
        (dzdy * dzdy * d2x - 2.0 * dzdx * dzdy * d2xy + dzdx * dzdx * d2y) / (p * p.sqrt())
    })
}

/// Compute general (mean / total) curvature from a DEM.
///
/// General curvature is the negative Laplacian of the surface:
/// `-(d2z/dx2 + d2z/dy2)`. Grids smaller than 3x3 return all zeros.
pub fn general_curvature(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::zeros((h, w));
    }
    grid_map(h, w, |row, col| {
        let (d2x, d2y, _) = horn_second_derivatives(dem, row, col, cell_size);
        -(d2x + d2y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_dem() {
        let dem = Array2::from_elem((10, 10), 100.0);
        let cs = CellSize::square(1.0).unwrap();
        for grid in [
            profile_curvature(&dem, cs),
            plan_curvature(&dem, cs),
            general_curvature(&dem, cs),
        ] {
            for &v in grid.iter() {
                assert!(v.abs() < 1e-6, "flat DEM curvature should be 0, got {v}");
            }
        }
    }

    #[test]
    fn linear_slope_has_zero_curvature() {
        let dem = Array2::from_shape_fn((10, 10), |(_, c)| c as f64);
        let cs = CellSize::square(1.0).unwrap();
        for grid in [
            profile_curvature(&dem, cs),
            plan_curvature(&dem, cs),
            general_curvature(&dem, cs),
        ] {
            for r in 2..8 {
                for c in 2..8 {
                    assert!(
                        grid[[r, c]].abs() < 1e-6,
                        "linear slope curvature should be 0"
                    );
                }
            }
        }
    }

    #[test]
    fn parabolic_general_curvature() {
        // z = x^2 + y^2 => d2z/dx2 = 2, d2z/dy2 = 2 => general = -(2+2) = -4
        let dem = Array2::from_shape_fn((10, 10), |(r, c)| (c as f64).powi(2) + (r as f64).powi(2));
        let g = general_curvature(&dem, CellSize::square(1.0).unwrap());
        let v = g[[5, 5]];
        assert!(
            (v - (-4.0)).abs() < 0.1,
            "parabolic general curvature should be ~-4, got {v}"
        );
    }

    #[test]
    fn small_grid_returns_zeros() {
        let dem = Array2::from_elem((2, 2), 50.0);
        let cs = CellSize::square(1.0).unwrap();
        for grid in [
            profile_curvature(&dem, cs),
            plan_curvature(&dem, cs),
            general_curvature(&dem, cs),
        ] {
            for &v in grid.iter() {
                assert!(v.abs() < 1e-10);
            }
        }
    }
}
