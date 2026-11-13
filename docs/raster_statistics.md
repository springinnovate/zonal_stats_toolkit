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

## Python callers

`fast_zonal_statistics(..., operations=["sum"])` returns only the requested
metrics, plus any percentiles requested through `percentile_list`. Omitting
`operations` retains the full-statistics interface. The CLI supplies the job's
requested operations automatically.
