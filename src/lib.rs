#![forbid(unsafe_code)]

//! **terrand** — Pure-Rust terrain analysis on ndarray.
//!
//! Provides DEM (Digital Elevation Model) analysis without GDAL or any C
//! dependencies. All operations take `ndarray::Array2<f64>` as input and
//! return ndarray arrays as output, making them composable with the rest of
//! the Rust scientific computing ecosystem.
//!
//! # Modules
//!
//! | Module | Operations |
//! |--------|-----------|
//! | [`slope`](slope) | Slope in degrees, radians, or percent |
//! | [`aspect`](aspect) | Slope direction as compass bearing |
//! | [`curvature`](curvature) | Profile, plan, and general curvature |
//! | [`hillshade`](hillshade) | Shaded-relief illumination |
//! | [`roughness`](roughness) | TRI, TPI, and surface roughness |
//! | [`hydrology`](hydrology) | Fill, flow direction, accumulation, watershed, basin, stream order |
//! | [`viewshed`](viewshed) | Line-of-sight visibility analysis |
//! | [`contour`](contour) | Marching-squares contour generation |
//!
//! # Quick start
//!
//! ```
//! use ndarray::Array2;
//! use terrand::{slope, aspect, hillshade, CellSize};
//!
//! // A synthetic DEM with a uniform east-facing slope.
//! let dem = Array2::from_shape_fn((100, 100), |(_, c)| c as f64 * 10.0);
//! let cell = CellSize::square(30.0).unwrap();
//!
//! let s = slope(&dem, cell);
//! let a = aspect(&dem, cell);
//! let hs = hillshade(&dem, cell, 315.0, 45.0);
//! ```
//!
//! # Parallelism
//!
//! Enable the `parallel` feature to use Rayon for multi-threaded computation
//! on all per-cell operations:
//!
//! ```toml
//! [dependencies]
//! terrand = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! # NaN handling
//!
//! All operations propagate `NaN` naturally: if an input cell or any cell in
//! its computation neighborhood is `NaN`, the output is `NaN`. This allows
//! nodata values to flow through without explicit masking.
//!
//! # Algorithms
//!
//! Surface analysis (slope, aspect, curvature, hillshade) uses the Horn (1981)
//! 3x3 weighted gradient kernel with GDAL-compatible linear extrapolation at
//! grid edges.

// Internal modules (not part of public API).
mod kernel;

// Public modules.
pub mod aspect;
pub mod cell_size;
pub mod contour;
pub mod curvature;
pub mod error;
pub mod hillshade;
pub mod hydrology;
pub mod roughness;
pub mod slope;
pub mod viewshed;

// Re-export primary types and functions at crate root for ergonomic access.
pub use aspect::{aspect, ASPECT_FLAT};
pub use cell_size::CellSize;
pub use curvature::{general_curvature, plan_curvature, profile_curvature};
pub use error::{Error, Result};
pub use hillshade::hillshade;
pub use hydrology::{
    basin, fill, flow_accumulation, flow_direction, snap_pour_point, stream_order_strahler,
    watershed,
};
pub use roughness::{roughness, tpi, tri};
pub use slope::{slope, slope_percent, slope_radians};
pub use viewshed::{viewshed, ViewshedConfig, EARTH_RADIUS_M};
