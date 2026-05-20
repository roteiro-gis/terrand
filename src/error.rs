//! Error types returned by fallible terrand operations.

use thiserror::Error;

/// Errors returned by terrand operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Cell dimensions must be positive finite values.
    #[error("cell dimensions must be positive and finite, got x={x}, y={y}")]
    InvalidCellSize {
        /// Invalid x cell size.
        x: f64,
        /// Invalid y cell size.
        y: f64,
    },

    /// Observer position is outside the DEM grid.
    #[error("observer position ({row}, {col}) is outside grid of size ({height}, {width})")]
    ObserverOutOfBounds {
        /// Observer row index.
        row: usize,
        /// Observer column index.
        col: usize,
        /// Grid height in rows.
        height: usize,
        /// Grid width in columns.
        width: usize,
    },

    /// Viewshed maximum distance must be positive and finite, or positive infinity.
    #[error("viewshed max_distance must be positive and finite, or infinity, got {0}")]
    InvalidViewshedMaxDistance(f64),

    /// Contour interval is not a positive, finite number.
    #[error("contour interval must be positive and finite, got {0}")]
    InvalidContourInterval(f64),

    /// Two input grids have different shapes.
    #[error("grid shapes must match: {left} has shape {left_shape:?}, {right} has shape {right_shape:?}")]
    ShapeMismatch {
        /// Name of the first grid.
        left: &'static str,
        /// Shape of the first grid.
        left_shape: (usize, usize),
        /// Name of the second grid.
        right: &'static str,
        /// Shape of the second grid.
        right_shape: (usize, usize),
    },

    /// Pour point position is outside the input grid.
    #[error("pour point ({row}, {col}) is outside grid of size ({height}, {width})")]
    PourPointOutOfBounds {
        /// Pour point row index.
        row: usize,
        /// Pour point column index.
        col: usize,
        /// Grid height in rows.
        height: usize,
        /// Grid width in columns.
        width: usize,
    },
}

/// Convenience alias for `Result<T, terrand::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
