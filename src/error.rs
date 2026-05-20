use thiserror::Error;

/// Errors returned by terrand operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Cell dimensions must be positive finite values.
    #[error("cell dimensions must be positive and finite, got x={x}, y={y}")]
    InvalidCellSize { x: f64, y: f64 },

    /// Observer position is outside the DEM grid.
    #[error("observer position ({row}, {col}) is outside grid of size ({height}, {width})")]
    ObserverOutOfBounds {
        row: usize,
        col: usize,
        height: usize,
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
        left: &'static str,
        left_shape: (usize, usize),
        right: &'static str,
        right_shape: (usize, usize),
    },

    /// Pour point position is outside the input grid.
    #[error("pour point ({row}, {col}) is outside grid of size ({height}, {width})")]
    PourPointOutOfBounds {
        row: usize,
        col: usize,
        height: usize,
        width: usize,
    },
}

/// Convenience alias for `Result<T, terrand::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
