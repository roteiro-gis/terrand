//! Viewshed analysis: determine which cells are visible from an observer point.
//!
//! Uses Bresenham line-of-sight ray casting from the observer to the perimeter
//! of the search area. Supports Earth curvature and atmospheric refraction
//! correction.
//!
//! # Output
//!
//! The result is a binary visibility grid: `1.0` for visible cells, `0.0` for
//! hidden cells.

use ndarray::Array2;

use crate::cell_size::CellSize;
use crate::error::{Error, Result};

/// Mean Earth radius in meters.
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Configuration for viewshed analysis.
#[derive(Clone, Debug)]
pub struct ViewshedConfig {
    /// Observer height above the DEM surface, in the same units as elevation.
    pub observer_height: f64,
    /// Target height above the DEM surface, in the same units as elevation.
    /// Applied only to the final perimeter cell of each ray.
    pub target_height: f64,
    /// Maximum analysis distance in ground units (same units as cell size).
    /// Use `f64::INFINITY` for unlimited range.
    pub max_distance: f64,
    /// Atmospheric refraction coefficient. Standard value is 0.13. Set to 0.0
    /// to apply curvature correction without refraction. The curvature
    /// adjustment is: `distance^2 / (2 * earth_radius) * (1 - refraction)`.
    pub refraction_coeff: f64,
    /// Earth radius in the same units as the cell size and elevation. Use
    /// [`EARTH_RADIUS_M`] for metric DEMs.
    pub earth_radius: f64,
}

impl Default for ViewshedConfig {
    fn default() -> Self {
        Self {
            observer_height: 1.7,
            target_height: 0.0,
            max_distance: f64::INFINITY,
            refraction_coeff: 0.13,
            earth_radius: EARTH_RADIUS_M,
        }
    }
}

/// Compute viewshed visibility from an observer position.
///
/// Returns a 2D grid of the same dimensions as `dem` where visible cells
/// are `1.0` and hidden cells are `0.0`. The observer cell is always visible.
///
/// # Errors
///
/// Returns [`Error::ObserverOutOfBounds`] if the observer position is outside
/// the DEM grid. Returns [`Error::InvalidViewshedMaxDistance`] if
/// `config.max_distance` is zero, negative, NaN, or negative infinity.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use terrand::{viewshed, CellSize, ViewshedConfig};
///
/// let dem = Array2::from_elem((50, 50), 100.0);
/// let config = ViewshedConfig { max_distance: 500.0, ..Default::default() };
/// let vis = viewshed(&dem, CellSize::square(10.0).unwrap(), 25, 25, &config).unwrap();
/// assert_eq!(vis[[25, 25]], 1.0); // observer is always visible
/// ```
pub fn viewshed(
    dem: &Array2<f64>,
    cell_size: CellSize,
    observer_row: usize,
    observer_col: usize,
    config: &ViewshedConfig,
) -> Result<Array2<f64>> {
    let (h, w) = dem.dim();

    if observer_row >= h || observer_col >= w {
        return Err(Error::ObserverOutOfBounds {
            row: observer_row,
            col: observer_col,
            height: h,
            width: w,
        });
    }
    if config.max_distance <= 0.0 || config.max_distance.is_nan() {
        return Err(Error::InvalidViewshedMaxDistance(config.max_distance));
    }

    let mut result = Array2::zeros((h, w));
    result[[observer_row, observer_col]] = 1.0;

    let observer_elev = dem[[observer_row, observer_col]] + config.observer_height;

    // Determine a complete search envelope in cells. The final distance check
    // in `cast_ray` still clips to the true Euclidean ground distance.
    let (row_radius, col_radius) = search_radii(
        h,
        w,
        observer_row,
        observer_col,
        cell_size,
        config.max_distance,
    );

    // Build perimeter of the search rectangle.
    let mut perimeter = Vec::new();
    for dc in -col_radius..=col_radius {
        perimeter.push((-row_radius, dc)); // top row
        if row_radius != 0 {
            perimeter.push((row_radius, dc)); // bottom row
        }
    }
    if col_radius != 0 {
        for dr in (-row_radius + 1)..row_radius {
            perimeter.push((dr, -col_radius)); // left column
            perimeter.push((dr, col_radius)); // right column
        }
    }

    let obs = Observer {
        row: observer_row,
        col: observer_col,
        elev: observer_elev,
    };

    // Cast rays — parallel when feature enabled.
    #[cfg(feature = "parallel")]
    let visible_cells = {
        use rayon::prelude::*;
        perimeter
            .par_iter()
            .flat_map(|&(dr, dc)| cast_ray(dem, cell_size, &obs, dr, dc, config))
            .collect::<Vec<_>>()
    };

    #[cfg(not(feature = "parallel"))]
    let visible_cells = {
        perimeter
            .iter()
            .flat_map(|&(dr, dc)| cast_ray(dem, cell_size, &obs, dr, dc, config))
            .collect::<Vec<_>>()
    };

    for (row, col) in visible_cells {
        result[[row, col]] = 1.0;
    }

    Ok(result)
}

fn search_radii(
    height: usize,
    width: usize,
    observer_row: usize,
    observer_col: usize,
    cell_size: CellSize,
    max_distance: f64,
) -> (isize, isize) {
    let max_row_radius = observer_row.max(height - 1 - observer_row) as isize;
    let max_col_radius = observer_col.max(width - 1 - observer_col) as isize;

    if max_distance.is_infinite() {
        return (max_row_radius, max_col_radius);
    }

    let row_radius = (max_distance / cell_size.y())
        .ceil()
        .min(max_row_radius as f64) as isize;
    let col_radius = (max_distance / cell_size.x())
        .ceil()
        .min(max_col_radius as f64) as isize;

    (row_radius, col_radius)
}

/// Observer state bundled for `cast_ray` to avoid too many arguments.
struct Observer {
    row: usize,
    col: usize,
    elev: f64,
}

/// Cast a single ray from the observer toward the perimeter cell at offset
/// (dr, dc) and return the list of visible cells along the ray.
fn cast_ray(
    dem: &Array2<f64>,
    cell_size: CellSize,
    obs: &Observer,
    dr: isize,
    dc: isize,
    config: &ViewshedConfig,
) -> Vec<(usize, usize)> {
    let (h, w) = dem.dim();

    let target_row = (obs.row as isize + dr).clamp(0, h as isize - 1) as usize;
    let target_col = (obs.col as isize + dc).clamp(0, w as isize - 1) as usize;

    let cells = bresenham(
        obs.row as isize,
        obs.col as isize,
        target_row as isize,
        target_col as isize,
    );

    let mut visible = Vec::new();
    let mut max_slope = f64::NEG_INFINITY;

    for &(row, col) in cells.iter().skip(1) {
        let dx = (col as f64 - obs.col as f64) * cell_size.x();
        let dy = (row as f64 - obs.row as f64) * cell_size.y();
        let distance = (dx * dx + dy * dy).sqrt();

        if config.max_distance.is_finite() && distance > config.max_distance {
            break;
        }

        let curvature_drop = if config.earth_radius.is_finite() && config.earth_radius > 0.0 {
            distance * distance / (2.0 * config.earth_radius) * (1.0 - config.refraction_coeff)
        } else {
            0.0
        };

        let mut target_elev = dem[[row, col]] - curvature_drop;
        if row == target_row && col == target_col {
            target_elev += config.target_height;
        }

        let slope = (target_elev - obs.elev) / distance.max(1e-9);
        if slope >= max_slope {
            max_slope = slope;
            visible.push((row, col));
        }
    }

    visible
}

/// Bresenham's line algorithm: returns cells from (r0, c0) to (r1, c1).
fn bresenham(r0: isize, c0: isize, r1: isize, c1: isize) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    let mut r = r0;
    let mut c = c0;
    let dr = (r1 - r0).abs();
    let dc = (c1 - c0).abs();
    let sr = if r0 < r1 { 1 } else { -1 };
    let sc = if c0 < c1 { 1 } else { -1 };
    let mut err = dr - dc;

    loop {
        if r >= 0 && c >= 0 {
            cells.push((r as usize, c as usize));
        }
        if r == r1 && c == c1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dc {
            err -= dc;
            r += sr;
        }
        if e2 < dr {
            err += dr;
            c += sc;
        }
    }

    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_surface_all_visible() {
        let dem = Array2::from_elem((16, 16), 100.0);
        let config = ViewshedConfig {
            observer_height: 2.0,
            max_distance: 10.0,
            ..Default::default()
        };
        let vis = viewshed(&dem, CellSize::square(1.0).unwrap(), 8, 8, &config).unwrap();
        assert_eq!(vis[[8, 8]], 1.0);
        // Nearby cells should be visible on a flat surface.
        assert_eq!(vis[[8, 9]], 1.0);
    }

    #[test]
    fn wall_blocks_visibility() {
        let mut dem = Array2::from_elem((16, 16), 0.0);
        for row in 0..16 {
            dem[[row, 10]] = 1000.0;
        }
        let config = ViewshedConfig {
            observer_height: 1.5,
            max_distance: 20.0,
            refraction_coeff: 0.0,
            ..Default::default()
        };
        let vis = viewshed(&dem, CellSize::square(1.0).unwrap(), 7, 5, &config).unwrap();
        assert_eq!(vis[[7, 5]], 1.0, "observer should be visible");
        assert_eq!(vis[[7, 12]], 0.0, "cell behind wall should not be visible");
        assert_eq!(vis[[7, 3]], 1.0, "cell on observer side should be visible");
    }

    #[test]
    fn observer_at_edge() {
        let dem = Array2::from_elem((8, 8), 10.0);
        let config = ViewshedConfig::default();
        let vis = viewshed(&dem, CellSize::square(1.0).unwrap(), 0, 0, &config).unwrap();
        assert_eq!(vis[[0, 0]], 1.0);
    }

    #[test]
    fn observer_out_of_bounds() {
        let dem = Array2::from_elem((8, 8), 10.0);
        let config = ViewshedConfig::default();
        assert!(viewshed(&dem, CellSize::square(1.0).unwrap(), 10, 10, &config).is_err());
    }

    #[test]
    fn invalid_max_distance_is_rejected() {
        let dem = Array2::from_elem((8, 8), 10.0);
        for max_distance in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY] {
            let config = ViewshedConfig {
                max_distance,
                ..Default::default()
            };
            assert!(
                matches!(
                    viewshed(&dem, CellSize::square(1.0).unwrap(), 4, 4, &config),
                    Err(Error::InvalidViewshedMaxDistance(v))
                        if v == max_distance || (v.is_nan() && max_distance.is_nan())
                ),
                "accepted max_distance {max_distance}"
            );
        }
    }

    #[test]
    fn finite_radius_is_capped_to_grid_extent() {
        let dem = Array2::from_elem((8, 8), 10.0);
        let config = ViewshedConfig {
            max_distance: 1.0,
            ..Default::default()
        };
        let vis = viewshed(
            &dem,
            CellSize::square(f64::MIN_POSITIVE).unwrap(),
            4,
            4,
            &config,
        )
        .unwrap();
        assert_eq!(vis[[4, 4]], 1.0);
    }

    #[test]
    fn small_max_distance() {
        let dem = Array2::from_elem((16, 16), 0.0);
        let config = ViewshedConfig {
            observer_height: 2.0,
            max_distance: 1.5,
            refraction_coeff: 0.0,
            ..Default::default()
        };
        let vis = viewshed(&dem, CellSize::square(1.0).unwrap(), 7, 8, &config).unwrap();
        assert_eq!(vis[[7, 8]], 1.0);
        assert_eq!(vis[[7, 13]], 0.0, "distant cell should not be visible");
    }

    #[test]
    fn max_distance_uses_independent_radii_for_non_square_cells() {
        let dem = Array2::from_elem((11, 11), 0.0);
        let config = ViewshedConfig {
            observer_height: 2.0,
            max_distance: 5.0,
            refraction_coeff: 0.0,
            ..Default::default()
        };

        let vis = viewshed(&dem, CellSize::new(1.0, 10.0).unwrap(), 5, 5, &config).unwrap();

        assert_eq!(
            vis[[5, 10]],
            1.0,
            "cell five columns away should be inside the search envelope"
        );
        assert_eq!(
            vis[[6, 5]],
            0.0,
            "cell one row away should still be clipped by ground distance"
        );
    }

    #[test]
    fn bresenham_diagonal() {
        let cells = bresenham(0, 0, 4, 4);
        assert_eq!(cells.len(), 5);
        assert_eq!(cells[0], (0, 0));
        assert_eq!(cells[4], (4, 4));
    }

    #[test]
    fn bresenham_horizontal() {
        let cells = bresenham(3, 0, 3, 5);
        assert_eq!(cells.len(), 6);
        for (i, &(r, c)) in cells.iter().enumerate() {
            assert_eq!(r, 3);
            assert_eq!(c, i);
        }
    }
}
