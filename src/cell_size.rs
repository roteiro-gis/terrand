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
/// let cs = CellSize::square(30.0);
/// assert_eq!(cs.x, 30.0);
/// assert_eq!(cs.y, 30.0);
///
/// // Non-square pixels
/// let cs = CellSize::new(25.0, 30.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellSize {
    /// Cell width in the x (column) direction.
    pub x: f64,
    /// Cell height in the y (row) direction.
    pub y: f64,
}

impl CellSize {
    /// Create a cell size with independent x and y dimensions.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Create a square cell size where x and y are equal.
    #[inline]
    pub fn square(size: f64) -> Self {
        Self { x: size, y: size }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square() {
        let cs = CellSize::square(30.0);
        assert_eq!(cs.x, 30.0);
        assert_eq!(cs.y, 30.0);
    }

    #[test]
    fn test_non_square() {
        let cs = CellSize::new(25.0, 30.0);
        assert_eq!(cs.x, 25.0);
        assert_eq!(cs.y, 30.0);
    }
}
