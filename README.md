# terrand

Terrain analysis utilities for regular 2D DEM grids.

Top-level functions compute DEM-derived rasters from `ndarray::Array2<f64>`:
slope, aspect, hillshade, curvature, roughness metrics, D8 hydrology, viewshed,
and contours.

`CellSize` stores positive finite grid spacing. Raster I/O, GDAL integration,
reprojection, resampling, and vertical datum handling are outside the crate.

## Installation

```sh
cargo add terrand-rs --rename terrand
```

Or add the dependency directly:

```toml
[dependencies]
terrand = { package = "terrand-rs", version = "0.1" }
```

```rust
use ndarray::Array2;
use terrand::{hillshade, slope, CellSize};

let dem = Array2::from_shape_fn((50, 50), |(row, col)| {
    row as f64 * 2.0 + col as f64 * 5.0
});
let cell = CellSize::square(30.0).unwrap();

let slope_degrees = slope(&dem, cell);
let shaded_relief = hillshade(&dem, cell, 315.0, 45.0);

assert_eq!(slope_degrees.dim(), dem.dim());
assert_eq!(shaded_relief.dim(), dem.dim());
```

Use `terrand::hydrology` for fill, D8 flow direction, accumulation,
watersheds, basin labels, Strahler stream order, and pour-point snapping. Use
`terrand::contour` for marching-squares contour lines in grid coordinates
`(col, row)`.

DEM nodata policy: non-finite elevations represent nodata. Surface kernels
generally propagate `NaN`; small-grid fallbacks return documented flat values.
Hydrology uses numeric direction and label grids without a separate nodata mask.
