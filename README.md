# terrand

Pure-Rust terrain analysis for regular 2D DEM grids.

`terrand` is an ndarray-first layer for DEM-derived rasters and grid-space
contours. It takes `ndarray::Array2<f64>` plus validated cell spacing and
returns arrays or contour polylines; it does not read files or depend on GDAL.

Use it for slope, aspect, hillshade, curvature, roughness metrics, D8
hydrology, viewshed, and marching-squares contours.

## Installation

```sh
cargo add terrand-rs --rename terrand
```

Or add the dependency directly:

```toml
[dependencies]
terrand = { package = "terrand-rs", version = "0.1" }
```

Enable Rayon-backed per-cell loops with:

```toml
[dependencies]
terrand = { package = "terrand-rs", version = "0.1", features = ["parallel"] }
```

## Example

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

## Scope

`CellSize` stores positive finite grid spacing. Horizontal units must match DEM
elevation units for slope, curvature, hillshade, and viewshed. Raster I/O,
reprojection, resampling, CRS handling, and vertical datum corrections are
outside the crate.

DEM nodata policy: use `NaN` for nodata. Surface kernels generally propagate
`NaN`; small-grid fallbacks return documented flat values. Hydrology treats
`NaN` as DEM nodata where documented, but flow-direction, accumulation, and
label products are numeric grids without a separate nodata mask.

The intended pure-Rust geospatial pipeline is:

```text
geotiff-rust -> ndarray DEM -> terrand -> eikonal/geotiff-rust
```
