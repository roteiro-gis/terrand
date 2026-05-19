# terrand

Small, pure-Rust terrain analysis kernels for regular 2D DEM grids.

`terrand` is the ndarray-first terrain layer for a lightweight raster analysis
stack: `geotiff-rust` for GeoTIFF/COG I/O, `terrand` for DEM-derived rasters,
and `eikonal` for distance fields and routing. Give it an
`ndarray::Array2<f64>` plus a validated `CellSize`, and it returns derived
rasters or grid-space contour lines. It does not depend on GDAL, define a file
format, or perform CRS transformations.

Use it for:

- slope in degrees, radians, or percent
- aspect
- hillshade
- profile, plan, and general curvature
- terrain ruggedness, TPI, and roughness
- hydrology: fill, D8 flow direction, accumulation, watersheds, basins,
  Strahler stream order, and pour-point snapping
- line-of-sight viewshed
- marching-squares contours

## Install

```toml
[dependencies]
terrand = "0.1"
ndarray = "0.17"
```

Enable Rayon-backed per-cell loops with:

```toml
[dependencies]
terrand = { version = "0.1", features = ["parallel"] }
```

## Quick Start

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

## GeoTIFF Workflow

`terrand` intentionally keeps raster I/O out of scope. A typical GeoTIFF
pipeline is:

1. Read one DEM band with a raster crate.
2. Convert the raster nodata value to `NaN`.
3. Build a `CellSize` from the affine transform.
4. Run terrain kernels.
5. Convert output `NaN` to a file nodata value and write a new GeoTIFF.

This example uses the sibling pure-Rust GeoTIFF crates. The same array handoff
works with GDAL bindings or any reader that can produce `Array2<f64>`.

```toml
[dependencies]
terrand = "0.1"
ndarray = "0.17"
geotiff-reader = "0.5"
geotiff-writer = "0.5"
```

```rust
use geotiff_reader::GeoTiffFile;
use geotiff_writer::{Compression, GeoTiffBuilder};
use ndarray::{Array2, Ix2};
use std::{error::Error, io};
use terrand::{hillshade, slope, CellSize};

fn main() -> Result<(), Box<dyn Error>> {
    let input = GeoTiffFile::open("dem.tif")?;
    let transform = *input.transform().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "DEM GeoTIFF must have an affine transform",
        )
    })?;

    let dem: Array2<f64> = input.read_band::<f64>(0)?.into_dimensionality::<Ix2>()?;
    let nodata = input.nodata().and_then(|value| value.parse::<f64>().ok());
    let dem = dem.mapv(|z| {
        if nodata.is_some_and(|nd| z == nd) {
            f64::NAN
        } else {
            z
        }
    });

    // Horizontal cell units must match the elevation units. Reproject first if
    // the DEM is still in geographic degrees and elevations are meters.
    let cell = CellSize::new(transform.pixel_width.abs(), transform.pixel_height.abs())?;

    let slope_deg = slope(&dem, cell);
    let shade = hillshade(&dem, cell, 315.0, 45.0);

    let slope_out = slope_deg.mapv(|z| if z.is_nan() { -9999.0_f32 } else { z as f32 });
    let shade_out = shade.mapv(|z| if z.is_nan() { -9999.0_f32 } else { z as f32 });

    let mut builder = GeoTiffBuilder::new(input.width(), input.height())
        .transform(transform)
        .nodata("-9999")
        .compression(Compression::Deflate);

    if let Some(epsg) = input.epsg().and_then(|code| u16::try_from(code).ok()) {
        builder = builder.epsg(epsg);
    }

    builder.write_2d("slope-degrees.tif", slope_out.view())?;
    builder.write_2d("hillshade.tif", shade_out.view())?;

    Ok(())
}
```

## Constraints

- Inputs are regular 2D grids represented as `ndarray::Array2<f64>`.
- `CellSize` dimensions must be positive and finite. They must use the same
  horizontal units as elevation for slope, curvature, hillshade, and viewshed.
- `terrand` does not read/write rasters, reproject data, resample grids, or
  apply vertical datum corrections.
- Most kernels are full-raster, in-memory computations. They are not streaming
  tile processors.
- Surface-analysis kernels generally propagate `NaN` through their 3x3
  arithmetic on normal-size grids, but small-grid fallbacks return documented
  flat values.
- Hydrology treats `NaN` as DEM nodata for `fill`, but flow-direction and label
  products are numeric grids without a separate nodata mask.
- Contour coordinates are in grid space `(col, row)`. Apply your raster affine
  transform externally to get map coordinates.

## Positioning

Rust already has terrain options. `oxigdal-terrain` is a broader terrain
module in the OxiGDAL ecosystem, `surtgis-algorithms` exposes a large
collection of terrain, morphometric, visibility, smoothing, solar, and
streaming algorithms, and GDAL bindings give Rust access to GDAL's mature
raster toolchain.

`terrand` is intentionally narrower. Its niche is:

- small dependency surface: `ndarray`, `thiserror`, and optional `rayon`
- pure Rust, with no GDAL runtime or C build dependency
- direct `Array2<f64>` APIs instead of dataset, driver, or processing-framework
  abstractions
- kernels that are easy to compose inside the `geotiff-rust` + `terrand` +
  `eikonal` stack
- predictable building blocks for library code, tests, services, and tools
  that already own raster I/O and CRS handling elsewhere

`terrand` answers terrain-description questions: "what is the slope here?",
"which way does water flow?", "what cells are visible?", and "where do
contours cross this DEM?"

`eikonal` answers cost-propagation and routing questions: "how far or expensive
is every cell from this source?", "what is the weighted shortest path?", and
"what is the isochrone boundary?"

They compose cleanly. For example, use `terrand::slope_radians` to derive a
slope raster, then pass it into `eikonal::CostField::from_slope` to build a
terrain-aware travel-cost field. Use `terrand` when you need deterministic DEM
attributes; use `eikonal` when you need distance fields, paths, or reachability
over a cost surface.

Choose a broader terrain suite when you need many specialized geomorphometry
algorithms, streaming framework integration, or an all-in-one geospatial stack.
Choose GDAL bindings when you want GDAL's raster formats, warping, and command
parity. Choose `terrand` when the useful unit is a small Rust crate that takes
and returns ndarrays.

## Algorithms and Behavior

- Surface kernels use Horn-style 3x3 derivatives with GDAL-compatible edge
  extrapolation.
- Hydrology uses D8 flow directions with the common power-of-two encoding.
- Sink filling uses a Planchon-Darboux-style iterative fill.
- Viewshed uses Bresenham line-of-sight rays with optional Earth curvature and
  atmospheric refraction correction.
- Contours use marching squares and skip quads that contain `NaN`.

## License

MIT OR Apache-2.0
