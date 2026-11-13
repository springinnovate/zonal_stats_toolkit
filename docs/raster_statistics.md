# Raster statistics

Choose the metrics in each job's `operations` setting. For totals alone, use:

```ini
operations = sum
```

The runner computes only these metrics and their dependencies. For example,
`mean` requires a sum and valid-pixel count; `stdev` also requires squared sums.
Percentiles are computed only when requested. Output column names follow the
operations in the configuration.

## Block processing

The runner reads the polygon-ID block first. If it contains no polygon pixels,
it skips the corresponding service-raster read. An occupied block is still read
as a complete window; this is not a smaller-window or individual-pixel read.

Within an occupied block, the runner groups values by feature ID with a stable
sort. This preserves the pixel order within each feature and avoids repeatedly
scanning all occupied pixels to select each feature. It retains the existing
block boundaries, NoData rules, and feature-to-group aggregation order.

Sorting uses temporary arrays. When a conservative estimate of those arrays
exceeds 256 MiB, the runner uses per-feature selection for that block instead.
This guard limits additional sorting work space; it is not a limit on total
worker memory. Raster blocks and other working arrays still use memory, and
concurrent workers multiply that requirement. Dense blocks taking the fallback
can remain slower than blocks using grouped reductions.

## Reusing geometry

Raster jobs automatically share prepared polygons and polygon-ID rasters under
`<job working directory>/geometry_cache/`. A prepared polygon file can serve
rasters with the same target CRS and preparation settings. A polygon-ID raster
can serve rasters with the same complete grid and storage layout. Each service
still reads its own pixel values, applies its own NoData rules, and computes
fresh statistics.

This also applies to the shared worker pool (`--total-workers`): per-raster
temporary directories hold scratch files, while reusable geometry stays in
the job's persistent cache directory.

Cache keys include:

- SHA-256 hashes of the source vector and its sidecars, including geometry,
  attributes, feature order, and IDs; the selected layer is also part of the key.
- Target CRS, reprojection and simplification settings, toolkit code, spatial
  library versions, and relevant GDAL/PROJ configuration and local resources.
- For polygon-ID rasters, the exact affine transform, dimensions, CRS, storage
  block layout, rasterization options, tile size, and prepared-polygon identity.

Matching pixel size alone is insufficient. Different grids or polygon shards
use separate entries. Changing service-raster values or NoData does not require
new polygon geometry, but it always produces freshly calculated statistics.

Workers lock each cache entry while building or checking it. A completed entry
is published atomically with a manifest; a hit requires checksum and metadata
validation. Invalid or incomplete entries are rebuilt. Source changes detected
during a calculation cause it to fail instead of publishing results against a
stale geometry snapshot. Cache checks include file hashing, so a hit still has
an I/O cost.

ID-raster validation accepts equivalent coordinate-system WKT serializations
using OSR, including GeoTIFF's metadata normalization. Pixel alignment and
dimensions still require exact matches; band count, data type, and NoData are
also checked. Validation failures name the mismatched metadata and close GDAL
handles before cleanup. If cleanup fails too, the original error is retained
and the remaining work path is reported. Unpublished artifacts are never cache
hits.

Caching supports local GeoPackage, SQLite, Shapefile, File Geodatabase, GeoJSON,
and FlatGeobuf inputs. Unsupported or indirect inputs, active database journals,
and network-enabled or explicitly file-referenced coordinate transformations
bypass reuse and use the independent calculation path.

The cache persists when ordinary per-raster work files are cleaned. A later run
using the same job working directory can reuse it; a new working directory
starts a new cache. To reclaim space, remove that job's `geometry_cache`
directory only when no process is using it. Interrupted builds may leave
unpublished staging directories, which are never treated as cache hits.

## Reading progress

Each raster reports separate phases for input preparation, polygon tile
rasterization, stitching, raster-block statistics, and final grouping. Counts,
rates, and elapsed-time estimates restart for each phase. Rasterization reaching
100% means that phase has finished, not that the raster's statistics are ready.

The statistics counter advances after a block finishes. During long reductions,
the message also reports completed feature groups within the current block.
Empty blocks count as completed after the polygon-ID check. Block sizes and
feature counts differ, so the ETA is approximate; an individual raster read,
sort, or geometry operation may have no intermediate update. Failed or cancelled
work closes at its actual count rather than displaying successful completion.
Cache messages distinguish building, waiting for another worker, and reusing a
validated artifact.

## Python callers

`fast_zonal_statistics(..., operations=["sum"])` returns only the requested
metrics, plus any percentiles requested through `percentile_list`. Omitting
`operations` retains the full-statistics interface. The CLI supplies the job's
requested operations automatically.

Set `reuse_geometry=False` to compare against independently rebuilt geometry.
Set `geometry_cache_dir` to share a cache across different working directories;
the input and processing checks still apply. The default is
`working_dir/geometry_cache`.

## Benchmarking reuse

`benchmarks/benchmark_geometry_cache.py` compares uncached calculations, an empty
geometry cache shared by matching rasters, and a repeat with populated entries.
It checks exact sums and reports elapsed time and preparation/rasterization call
counts. Run it with `--help` for source-vector, raster-window, and repeat options.
Use `--service-workers` to measure concurrent rasters; `--rasterize-workers`
sets tile concurrency within each raster calculation. The benchmark includes
service-worker startup and shutdown when using more than one service worker.
The benchmark creates isolated temporary inputs; it does not modify the sources.
Its cold-cache case refers to geometry artifacts, not the operating-system file
cache. Subset results are not a forecast for a full parallel assessment.
