//! Contour line generation using the marching squares algorithm.
//!
//! Generates isolines at regular elevation intervals from a DEM. Contour
//! coordinates are in **grid space** (column, row), where `(0.0, 0.0)` is the
//! top-left corner of the top-left cell. To convert to geographic coordinates,
//! apply your raster's affine transform externally.
//!
//! # Algorithm
//!
//! Each 2x2 cell quad is classified into one of 16 marching-squares cases.
//! Contour crossings are linearly interpolated along cell edges, then connected
//! into polylines using an endpoint hash map for O(n) assembly.

use std::collections::HashMap;

use ndarray::Array2;

use crate::error::{Error, Result};

/// A single contour line at a given elevation.
#[derive(Clone, Debug)]
pub struct ContourLine {
    /// Elevation value of this contour.
    pub elevation: f64,
    /// Coordinates as `(col, row)` pairs in grid space.
    pub coordinates: Vec<(f64, f64)>,
}

/// Configuration for contour generation.
#[derive(Clone, Debug)]
pub struct ContourConfig {
    /// Spacing between contour lines. Must be positive and finite.
    pub interval: f64,
    /// Base elevation for contour alignment. Contour levels are generated at
    /// multiples of `interval` starting from `base`. Default: 0.0.
    pub base: f64,
    /// Optional minimum elevation filter. Contours below this are discarded.
    pub min_value: Option<f64>,
    /// Optional maximum elevation filter. Contours above this are discarded.
    pub max_value: Option<f64>,
}

impl ContourConfig {
    /// Create a config with the given interval and default settings.
    pub fn new(interval: f64) -> Self {
        Self {
            interval,
            base: 0.0,
            min_value: None,
            max_value: None,
        }
    }
}

/// Generate contour lines from a DEM.
///
/// # Errors
///
/// Returns [`Error::InvalidContourInterval`] if `interval` is not positive and
/// finite.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use terrand::contour::{contour, ContourConfig};
///
/// let dem = ndarray::Array2::from_shape_fn((20, 20), |(r, _)| r as f64 * 10.0);
/// let lines = contour(&dem, &ContourConfig::new(25.0)).unwrap();
/// assert!(!lines.is_empty());
/// ```
pub fn contour(dem: &Array2<f64>, config: &ContourConfig) -> Result<Vec<ContourLine>> {
    if !config.interval.is_finite() || config.interval <= 0.0 {
        return Err(Error::InvalidContourInterval(config.interval));
    }

    let (h, w) = dem.dim();
    if h < 2 || w < 2 {
        return Ok(Vec::new());
    }

    // Find elevation range.
    let mut min_elev = f64::INFINITY;
    let mut max_elev = f64::NEG_INFINITY;
    for &v in dem.iter() {
        if v.is_finite() {
            if v < min_elev {
                min_elev = v;
            }
            if v > max_elev {
                max_elev = v;
            }
        }
    }

    if !min_elev.is_finite() || !max_elev.is_finite() {
        return Ok(Vec::new());
    }

    // Generate contour levels aligned to `base`.
    let first_level =
        ((min_elev - config.base) / config.interval).ceil() * config.interval + config.base;

    let mut contours = Vec::new();
    let mut level = first_level;

    while level <= max_elev {
        // Apply min/max filters.
        if let Some(min) = config.min_value {
            if level < min {
                level += config.interval;
                continue;
            }
        }
        if let Some(max) = config.max_value {
            if level > max {
                break;
            }
        }

        let segments = march_squares(dem, level);
        let lines = connect_segments(segments);

        for coords in lines {
            if coords.len() >= 2 {
                contours.push(ContourLine {
                    elevation: level,
                    coordinates: coords,
                });
            }
        }

        level += config.interval;
    }

    Ok(contours)
}

// ---------------------------------------------------------------------------
// Marching squares
// ---------------------------------------------------------------------------

/// A line segment between two 2D points.
type Segment = ((f64, f64), (f64, f64));

/// Generate contour segments for a single level using marching squares.
fn march_squares(grid: &Array2<f64>, level: f64) -> Vec<Segment> {
    let (h, w) = grid.dim();
    let mut segments = Vec::new();

    for row in 0..h - 1 {
        for col in 0..w - 1 {
            let tl = grid[[row, col]];
            let tr = grid[[row, col + 1]];
            let br = grid[[row + 1, col + 1]];
            let bl = grid[[row + 1, col]];

            if tl.is_nan() || tr.is_nan() || br.is_nan() || bl.is_nan() {
                continue;
            }

            // Case index: bit 0 = TL, bit 1 = TR, bit 2 = BR, bit 3 = BL.
            let mut case = 0u8;
            if tl >= level {
                case |= 1;
            }
            if tr >= level {
                case |= 2;
            }
            if br >= level {
                case |= 4;
            }
            if bl >= level {
                case |= 8;
            }

            if case == 0 || case == 15 {
                continue;
            }

            let top = lerp_h(col as f64, tl, (col + 1) as f64, tr, level, row as f64);
            let right = lerp_v(
                row as f64,
                tr,
                (row + 1) as f64,
                br,
                level,
                (col + 1) as f64,
            );
            let bottom = lerp_h(
                col as f64,
                bl,
                (col + 1) as f64,
                br,
                level,
                (row + 1) as f64,
            );
            let left = lerp_v(row as f64, tl, (row + 1) as f64, bl, level, col as f64);

            match case {
                1 | 14 => segments.push((top, left)),
                2 | 13 => segments.push((top, right)),
                3 | 12 => segments.push((left, right)),
                4 | 11 => segments.push((right, bottom)),
                5 => {
                    segments.push((top, right));
                    segments.push((bottom, left));
                }
                6 | 9 => segments.push((top, bottom)),
                7 | 8 => segments.push((left, bottom)),
                10 => {
                    segments.push((top, left));
                    segments.push((right, bottom));
                }
                _ => {}
            }
        }
    }

    segments
}

/// Linearly interpolate along a horizontal cell edge. Returns `(col, row)`.
#[inline]
fn lerp_h(col0: f64, val0: f64, col1: f64, val1: f64, level: f64, fixed_row: f64) -> (f64, f64) {
    let t = interp_t(val0, val1, level);
    (col0 + t * (col1 - col0), fixed_row)
}

/// Linearly interpolate along a vertical cell edge. Returns `(col, row)`.
#[inline]
fn lerp_v(row0: f64, val0: f64, row1: f64, val1: f64, level: f64, fixed_col: f64) -> (f64, f64) {
    let t = interp_t(val0, val1, level);
    (fixed_col, row0 + t * (row1 - row0))
}

#[inline]
fn interp_t(val0: f64, val1: f64, level: f64) -> f64 {
    let denom = val1 - val0;
    if denom.abs() < f64::EPSILON {
        0.5
    } else {
        ((level - val0) / denom).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Segment connection (O(n) via endpoint hash map)
// ---------------------------------------------------------------------------

/// Quantize a point to an integer key for hash-map-based endpoint matching.
/// Multiplying by 1e6 gives sub-cell precision while avoiding FP comparison.
fn point_key(p: (f64, f64)) -> (i64, i64) {
    ((p.0 * 1e6).round() as i64, (p.1 * 1e6).round() as i64)
}

/// Connect segments into polylines using an endpoint hash map.
///
/// This is O(n) amortized, compared to the naive O(n^2) scan.
fn connect_segments(segments: Vec<Segment>) -> Vec<Vec<(f64, f64)>> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Map from quantized endpoint -> list of segment indices that have that endpoint.
    let mut endpoint_map: HashMap<(i64, i64), Vec<usize>> =
        HashMap::with_capacity(segments.len() * 2);

    for (i, seg) in segments.iter().enumerate() {
        endpoint_map.entry(point_key(seg.0)).or_default().push(i);
        endpoint_map.entry(point_key(seg.1)).or_default().push(i);
    }

    let mut used = vec![false; segments.len()];
    let mut lines: Vec<Vec<(f64, f64)>> = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;

        let mut line = vec![segments[start_idx].0, segments[start_idx].1];

        // Extend forward from the tail.
        loop {
            let end = *line.last().unwrap();
            let key = point_key(end);
            let mut found = false;

            if let Some(indices) = endpoint_map.get(&key) {
                for &j in indices {
                    if used[j] {
                        continue;
                    }
                    let seg = &segments[j];
                    if point_key(seg.0) == key {
                        line.push(seg.1);
                        used[j] = true;
                        found = true;
                        break;
                    } else if point_key(seg.1) == key {
                        line.push(seg.0);
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                break;
            }
        }

        lines.push(line);
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_produces_contours() {
        let dem = Array2::from_shape_fn((10, 10), |(r, _)| r as f64 * 10.0);
        let lines = contour(&dem, &ContourConfig::new(20.0)).unwrap();
        assert!(!lines.is_empty(), "gradient should produce contour lines");
        let levels: Vec<f64> = lines.iter().map(|l| l.elevation).collect();
        assert!(
            levels.contains(&20.0),
            "should have contour at 20, got {levels:?}"
        );
        assert!(
            levels.contains(&40.0),
            "should have contour at 40, got {levels:?}"
        );
    }

    #[test]
    fn flat_dem_no_contours() {
        let dem = Array2::from_elem((10, 10), 100.0);
        let lines = contour(&dem, &ContourConfig::new(10.0)).unwrap();
        assert!(
            lines.len() <= 1,
            "flat DEM should produce 0 or 1 contour lines"
        );
    }

    #[test]
    fn invalid_interval_errors() {
        let dem = Array2::from_elem((5, 5), 10.0);
        assert!(contour(&dem, &ContourConfig::new(0.0)).is_err());
        assert!(contour(&dem, &ContourConfig::new(-5.0)).is_err());
        assert!(contour(&dem, &ContourConfig::new(f64::NAN)).is_err());
    }

    #[test]
    fn small_grid_empty() {
        let dem = Array2::from_elem((1, 1), 100.0);
        let lines = contour(&dem, &ContourConfig::new(10.0)).unwrap();
        assert!(lines.is_empty());
    }

    #[test]
    fn min_max_filter() {
        let dem = Array2::from_shape_fn((10, 10), |(r, _)| r as f64 * 10.0);
        let config = ContourConfig {
            interval: 10.0,
            base: 0.0,
            min_value: Some(25.0),
            max_value: Some(55.0),
        };
        let lines = contour(&dem, &config).unwrap();
        for line in &lines {
            assert!(line.elevation >= 25.0);
            assert!(line.elevation <= 55.0);
        }
    }

    #[test]
    fn contours_have_coordinates() {
        let dem = Array2::from_shape_fn((5, 5), |(r, _)| r as f64 * 25.0);
        let lines = contour(&dem, &ContourConfig::new(25.0)).unwrap();
        for l in &lines {
            assert!(
                l.coordinates.len() >= 2,
                "contour should have >= 2 points, got {}",
                l.coordinates.len()
            );
        }
    }

    #[test]
    fn base_alignment() {
        // With base = 5 and interval = 10, levels should be 5, 15, 25, ...
        let dem = Array2::from_shape_fn((10, 10), |(r, _)| r as f64 * 10.0);
        let config = ContourConfig {
            interval: 10.0,
            base: 5.0,
            min_value: None,
            max_value: None,
        };
        let lines = contour(&dem, &config).unwrap();
        for l in &lines {
            let offset = (l.elevation - 5.0) % 10.0;
            assert!(
                offset.abs() < 1e-6 || (offset - 10.0).abs() < 1e-6,
                "contour at {} not aligned to base 5 + interval 10",
                l.elevation
            );
        }
    }
}
