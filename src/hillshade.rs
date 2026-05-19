//! Hillshade illumination from a Digital Elevation Model.
//!
//! Computes a shaded-relief raster by simulating illumination from a
//! directional light source defined by its azimuth and altitude angles.
//!
//! The algorithm uses the Horn (1981) 3x3 gradient kernel with GDAL-compatible
//! edge extrapolation.
//!
//! # Output
//!
//! Values are in the range \[0, 255\], where 0 is fully shadowed and 255 is
//! fully illuminated. This matches the GDAL `gdaldem hillshade` convention.
//!
//! # NaN handling
//!
//! Input NaN cells propagate to NaN in the output.

use ndarray::Array2;

use crate::cell_size::CellSize;
use crate::kernel::{grid_map, horn_gradients};

/// Compute hillshade illumination from a DEM.
///
/// # Arguments
///
/// * `dem` - 2D array of elevation values.
/// * `cell_size` - Cell dimensions in the same linear units as elevation.
/// * `azimuth` - Sun azimuth in degrees clockwise from north. Typical default:
///   315 (northwest).
/// * `altitude` - Sun altitude angle in degrees above the horizon. Typical
///   default: 45.
///
/// # Returns
///
/// A 2D array of illumination values in \[0, 255\]. Grids smaller than 3x3
/// return a uniform value based on the sun altitude.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use terrand::{hillshade, CellSize};
///
/// let dem = Array2::from_elem((100, 100), 500.0);
/// let hs = hillshade(&dem, CellSize::square(30.0), 315.0, 45.0);
/// // Flat surface: shade = cos(zenith) * 255 ≈ 180.3
/// assert!((hs[[50, 50]] - 180.3).abs() < 1.0);
/// ```
pub fn hillshade(
    dem: &Array2<f64>,
    cell_size: CellSize,
    azimuth: f64,
    altitude: f64,
) -> Array2<f64> {
    let (h, w) = dem.dim();
    if h < 3 || w < 3 {
        let flat_val = 255.0 * altitude.to_radians().sin();
        return Array2::from_elem((h, w), flat_val);
    }

    // Convert azimuth from compass bearing to math angle (radians).
    let azimuth_rad = (360.0 - azimuth + 90.0).to_radians();
    let zenith_rad = std::f64::consts::FRAC_PI_2 - altitude.to_radians();

    let cos_zen = zenith_rad.cos();
    let sin_zen = zenith_rad.sin();

    grid_map(h, w, |row, col| {
        let (dzdx, dzdy) = horn_gradients(dem, row, col, cell_size);

        let slope_rad = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
        let aspect_rad = if dzdx.abs() < f64::EPSILON && dzdy.abs() < f64::EPSILON {
            0.0
        } else {
            dzdy.atan2(-dzdx)
        };

        let shade = (cos_zen * slope_rad.cos()
            + sin_zen * slope_rad.sin() * (azimuth_rad - aspect_rad).cos())
        .clamp(0.0, 1.0);

        shade * 255.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_surface_uniform() {
        let dem = Array2::from_elem((10, 10), 100.0);
        let hs = hillshade(&dem, CellSize::square(1.0), 315.0, 45.0);
        let first = hs[[1, 1]];
        for r in 1..9 {
            for c in 1..9 {
                assert!(
                    (hs[[r, c]] - first).abs() < 1e-6,
                    "flat surface should have uniform illumination"
                );
            }
        }
        // cos(45 deg) ≈ 0.707
        assert!(
            (first / 255.0 - 0.707).abs() < 0.01,
            "flat illumination should be ~0.707, got {}",
            first / 255.0
        );
    }

    #[test]
    fn small_grid_returns_flat_illumination() {
        let dem = Array2::from_elem((2, 2), 100.0);
        let hs = hillshade(&dem, CellSize::square(1.0), 315.0, 45.0);
        assert!(hs[[0, 0]] > 0.0);
    }

    #[test]
    fn range_check() {
        let dem = Array2::from_shape_fn((20, 20), |(r, c)| {
            (r as f64 * 10.0 + c as f64).sin() * 100.0
        });
        let hs = hillshade(&dem, CellSize::square(30.0), 315.0, 45.0);
        for &v in hs.iter() {
            assert!((0.0..=255.0).contains(&v), "hillshade {v} out of [0, 255]");
        }
    }

    #[test]
    fn slope_changes_illumination() {
        let dem = Array2::from_shape_fn((10, 10), |(_, c)| c as f64 * 10.0);
        let hs = hillshade(&dem, CellSize::square(1.0), 315.0, 45.0);
        let v = hs[[5, 5]];
        assert!((0.0..=255.0).contains(&v));
    }
}
