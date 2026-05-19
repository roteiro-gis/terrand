# terrand

Pure-Rust terrain analysis kernels for regular 2D DEM grids.

`terrand` is the `ndarray` terrain layer in a no-GDAL raster stack: read DEMs
with `geotiff-rust`, derive terrain rasters with `terrand`, and use those
rasters with `eikonal` for distance fields, routing, and cost surfaces.

It provides slope, aspect, hillshade, curvature, roughness metrics, D8
hydrology, viewshed, and marching-squares contours. Inputs are
`ndarray::Array2<f64>`; outputs are derived arrays or grid-space contour lines.

```rust
use ndarray::Array2;
use terrand::{aspect, hillshade, slope, CellSize};

fn main() -> terrand::Result<()> {
    let dem = Array2::from_shape_fn((100, 100), |(row, col)| {
        row as f64 * 2.0 + col as f64 * 5.0
    });
    let cell = CellSize::square(30.0)?;

    let slope_deg = slope(&dem, cell);
    let aspect_deg = aspect(&dem, cell);
    let shade = hillshade(&dem, cell, 315.0, 45.0);

    assert_eq!(slope_deg.dim(), dem.dim());
    assert_eq!(aspect_deg.dim(), dem.dim());
    assert_eq!(shade.dim(), dem.dim());

    Ok(())
}
```

Enable Rayon-backed per-cell loops with:

```toml
[dependencies]
terrand = { version = "0.1", features = ["parallel"] }
```

Scope and data policy:

- `CellSize` dimensions must be positive, finite, and in the same horizontal
  units as elevation for slope, curvature, hillshade, and viewshed.
- `terrand` does not read or write rasters, reproject data, resample grids, or
  apply vertical datum corrections.
- Surface kernels generally propagate `NaN` through their 3x3 arithmetic;
  small-grid fallbacks return documented flat values.
- Hydrology treats `NaN` as DEM nodata for elevation inputs. Flow-direction and
  label products are numeric grids without a separate nodata mask.
- Contour coordinates are in grid space `(col, row)`. Apply the raster affine
  transform externally to get map coordinates.

License: MIT OR Apache-2.0
