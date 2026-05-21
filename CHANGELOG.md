# Changelog

## 0.1.0 - 2026-05-21

Initial public release.

- Added ndarray-first terrain analysis for slope, aspect, hillshade, curvature,
  TRI, TPI, and roughness.
- Added D8 hydrology operations: fill, flow direction, accumulation,
  watershed, basin labels, Strahler stream order, and pour-point snapping.
- Added viewshed analysis and marching-squares contour generation.
- Added optional Rayon-backed per-cell parallelism with the `parallel` feature.
- Published as `terrand-rs` while keeping the library import path `terrand`.
