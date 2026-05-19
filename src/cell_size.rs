use crate::error::{Error, Result};

/// Cell dimensions of a regular grid.
///
/// Both `x` and `y` are in the same linear units as the elevation values
/// (typically meters). For geographic DEMs whose native units are degrees,
/// convert to meters first or use approximate meter-equivalent values.
///
/// # Examples
///
/// ```
/// use terrand::CellSize;
///
/// // 30-meter SRTM grid
/// let cs = CellSize::square(30.0).unwrap();
/// assert_eq!(cs.x(), 30.0);
/// assert_eq!(cs.y(), 30.0);
///
/// // Non-square pixels
/// let cs = CellSize::new(25.0, 30.0).unwrap();
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellSize {
    /// Cell width in the x (column) direction.
    x: f64,
    /// Cell height in the y (row) direction.
    y: f64,
}

impl CellSize {
    /// Create a cell size with independent x and y dimensions.
    ///
    /// Both dimensions must be positive and finite.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCellSize`] if either dimension is zero,
    /// negative, NaN, or infinite.
    #[inline]
    pub fn new(x: f64, y: f64) -> Result<Self> {
        if x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0 {
            Ok(Self { x, y })
        } else {
            Err(Error::InvalidCellSize { x, y })
        }
    }

    /// Create a square cell size where x and y are equal.
    ///
    /// The dimension must be positive and finite.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCellSize`] if `size` is zero, negative, NaN, or
    /// infinite.
    #[inline]
    pub fn square(size: f64) -> Result<Self> {
        Self::new(size, size)
    }

    /// Cell width in the x (column) direction.
    #[inline]
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Cell height in the y (row) direction.
    #[inline]
    pub fn y(&self) -> f64 {
        self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square() {
        let cs = CellSize::square(30.0).unwrap();
        assert_eq!(cs.x(), 30.0);
        assert_eq!(cs.y(), 30.0);
    }

    #[test]
    fn test_non_square() {
        let cs = CellSize::new(25.0, 30.0).unwrap();
        assert_eq!(cs.x(), 25.0);
        assert_eq!(cs.y(), 30.0);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        for (x, y) in [
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::NEG_INFINITY),
        ] {
            assert!(CellSize::new(x, y).is_err(), "accepted ({x}, {y})");
        }
        assert!(CellSize::square(0.0).is_err());
    }
}
