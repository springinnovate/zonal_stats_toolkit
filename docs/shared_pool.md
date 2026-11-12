# Sharing a worker budget across feature groups

For raster-only configurations, `--total-workers` runs raster work from multiple
jobs (feature groups) through one process pool:

```cmd
python runner.py path\to\config.ini --total-workers 16
```

You can pass multiple configuration files before the option. Jobs are scheduled
in round-robin order, so another feature group can start before the first group
has finished all its rasters. A task processes one raster for one feature group;
individual features are still processed together within that group.

The pool has at most 16 compute worker processes in this example. Raster tile
creation runs inside its assigned worker, and output assembly uses the same
pool. Workers do not create nested process pools. The coordinator and the
progress queue's manager process also exist; they are not additional raster
compute workers. Numerical-library threads within each compute worker are
limited to one.

This is an opt-in mode. Omit `--total-workers` to retain the existing scheduling.
Do not combine it with `--job-workers`, its alias `--max-workers`, or
`--raster-workers`. Vector and vector-intersection jobs must use the existing
scheduling mode.

Each group's CSV and GeoPackage retain their existing fields, raster column
order, and values. Multiple groups can show active progress at once, with each
event labeled by its own group. Results are staged until every raster for a
group succeeds; a failed raster does not publish an incomplete group. Groups
that finished before a later failure can remain published.

The worker budget controls concurrency, not whether previous results are
reused. A calling pipeline remains responsible for selecting saved zonal
outputs and passing only jobs that need computation.
