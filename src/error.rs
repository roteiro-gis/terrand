use thiserror::Error;

/// Errors returned by terrand operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Observer position is outside the DEM grid.
    #[error("observer position ({row}, {col}) is outside grid of size ({height}, {width})")]
    ObserverOutOfBounds {
        row: usize,
        col: usize,
        height: usize,
        width: usize,
    },

    /// Contour interval is not a positive, finite number.
    #[error("contour interval must be positive and finite, got {0}")]
    InvalidContourInterval(f64),
}

/// Convenience alias for `Result<T, terrand::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
