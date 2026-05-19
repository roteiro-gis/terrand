//! Aspect (slope direction) computation from a Digital Elevation Model.
//!
//! Aspect is the compass direction that the **downhill** slope faces:
//!
//! - 0 / 360 = North
//! - 90 = East
//! - 180 = South
//! - 270 = West
//! - **-1** = Flat (no measurable slope)
//!
//! This convention matches GDAL's `gdaldem aspect` output.
//!
//! # NaN handling
//!
//! Input NaN cells propagate to NaN in the output.

use ndarray::Array2;

use crate::cell_size::CellSize;
use crate::kernel::{grid_map, horn_gradients};

/// Flat-surface sentinel value, matching the GDAL convention.
pub const ASPECT_FLAT: f64 = -1.0;

/// Compute aspect in degrees from a DEM using the Horn algorithm.
///
/// Returns compass bearings in \[0, 360) for sloped cells, or [`ASPECT_FLAT`]
/// (-1.0) for flat cells. Grids smaller than 3x3 return all -1.0.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use terrand::{aspect, CellSize, ASPECT_FLAT};
///
/// // Flat DEM: all aspects are -1
/// let dem = Array2::from_elem((10, 10), 100.0);
/// let a = aspect(&dem, CellSize::square(30.0));
/// assert_eq!(a[[5, 5]], ASPECT_FLAT);
/// ```
pub fn aspect(dem: &Array2<f64>, cell_size: CellSize) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        return Array2::from_elem((h, w), ASPECT_FLAT);
    }
    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);

        if dzdx.abs() < f64::EPSILON && dzdy.abs() < f64::EPSILON {
            return ASPECT_FLAT;
        }

        // Convert gradient to compass bearing of downhill direction.
        // GDAL convention: downslope east = -dzdx, north = dzdy.
        let mut deg = (-dzdx).atan2(dzdy).to_degrees();
        if deg < 0.0 {
            deg += 360.0;
        }
        if deg >= 360.0 {
            deg -= 360.0;
        }
        deg
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_dem_returns_sentinel() {
        let dem = Array2::from_elem((10, 10), 100.0);
        let a = aspect(&dem, CellSize::square(1.0));
        for &v in a.iter() {
            assert_eq!(v, ASPECT_FLAT, "flat DEM should have aspect -1");
        }
    }

    #[test]
    fn east_increasing_faces_west() {
        // DEM rising east => downhill faces west (270)
        let dem = Array2::from_shape_fn((10, 10), |(_, c)| c as f64 * 10.0);
        let a = aspect(&dem, CellSize::square(1.0));
        let v = a[[5, 5]];
        assert!(
            (v - 270.0).abs() < 1.0,
            "east-increasing slope should face ~270 (west), got {v}"
        );
    }

    #[test]
    fn north_increasing_faces_south() {
        // DEM rising north (row 0 = high)
        let dem = Array2::from_shape_fn((10, 10), |(r, _)| (9 - r) as f64 * 10.0);
        let a = aspect(&dem, CellSize::square(1.0));
        let v = a[[5, 5]];
        assert!(
            (v - 180.0).abs() < 1.0,
            "north-increasing slope should face ~180 (south), got {v}"
        );
    }

    #[test]
    fn range_check() {
        let dem = Array2::from_shape_fn((20, 20), |(r, c)| {
            (r as f64 * 2.0 + c as f64 * 3.0).sin() * 50.0
        });
        let a = aspect(&dem, CellSize::square(1.0));
        for &v in a.iter() {
            assert!(
                v == ASPECT_FLAT || (0.0..360.0).contains(&v),
                "aspect {v} out of range"
            );
        }
    }

    #[test]
    fn small_grid_returns_flat() {
        let dem = Array2::from_elem((2, 2), 50.0);
        let a = aspect(&dem, CellSize::square(1.0));
        for &v in a.iter() {
            assert_eq!(v, ASPECT_FLAT);
        }
    }

    #[test]
    fn consistency_with_slope() {
        let dem = Array2::from_elem((10, 10), 42.0);
        let s = crate::slope(&dem, CellSize::square(1.0));
        let a = aspect(&dem, CellSize::square(1.0));
        for r in 0..10 {
            for c in 0..10 {
                if s[[r, c]].abs() < 1e-6 {
                    assert_eq!(a[[r, c]], ASPECT_FLAT);
                }
            }
        }
    }
}
