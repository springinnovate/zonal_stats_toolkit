"""Compare uncached, cold-cache, and warm-cache zonals on native raster windows.

Run with the toolkit environment from the repository root. Inputs are read-only;
the benchmark copies a native window and its intersecting polygon subset into a
new temporary directory. Reported timings exclude fixture creation and include
all zonal work: cache validation, preparation, rasterization, reads, and sums.
"Cold" refers to the geometry cache; the operating-system file cache is not
flushed. Use --service-workers to run service rasters concurrently. Each mode
creates its own spawn process pool, including startup and shutdown in elapsed
time. Phase seconds are summed across tasks and can exceed elapsed wall time.
This subset benchmark does not estimate full-assessment runtime.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runner
import numpy as np
from osgeo import gdal, ogr, osr


class ProgressSink:
    def put(self, event):
        pass


def make_fixtures(args, destination):
    """Copy native pixels and intact source geometries, without resampling."""
    first = gdal.OpenEx(str(args.rasters[0]), gdal.OF_RASTER)
    grid = (first.RasterXSize, first.RasterYSize, first.GetGeoTransform(),
            first.GetProjection())
    x, y, width, height = args.window
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("Window offsets must be nonnegative and sizes positive")
    if x + width > grid[0] or y + height > grid[1]:
        raise ValueError("Window extends beyond the source raster")
    raster_srs = osr.SpatialReference(wkt=grid[3])
    raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    x0, y0 = gdal.ApplyGeoTransform(grid[2], x, y)
    x1, y1 = gdal.ApplyGeoTransform(grid[2], x + width, y + height)
    if grid[2][2] or grid[2][4]:
        raise ValueError("This benchmark fixture helper requires unrotated rasters")
    footprint = ogr.CreateGeometryFromWkt(
        f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, "
        f"{x0} {y1}, {x0} {y0}))"
    )
    footprint.Segmentize(min(abs(x1 - x0), abs(y1 - y0)) / 64)
    source = gdal.OpenEx(str(args.vector), gdal.OF_VECTOR)
    layer = source.GetLayerByName(args.layer) if args.layer else source.GetLayer(0)
    if layer is None or layer.GetSpatialRef() is None:
        raise ValueError("The selected vector layer needs a spatial reference")
    vector_srs = layer.GetSpatialRef().Clone()
    vector_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    footprint.Transform(osr.CoordinateTransformation(raster_srs, vector_srs))
    xmin, xmax, ymin, ymax = footprint.GetEnvelope()
    layer.SetSpatialFilterRect(xmin, ymin, xmax, ymax)
    vector = destination / "zones.gpkg"
    target = ogr.GetDriverByName("GPKG").CreateDataSource(str(vector))
    copied = target.CreateLayer(layer.GetName(), srs=vector_srs,
                                geom_type=layer.GetGeomType())
    definition = layer.GetLayerDefn()
    for index in range(definition.GetFieldCount()):
        copied.CreateField(definition.GetFieldDefn(index))
    copied.StartTransaction()
    count = 0
    for feature in layer:
        output = ogr.Feature(copied.GetLayerDefn())
        output.SetFrom(feature)
        # Keep source ordering, attributes, and whole geometries. Do not clip.
        copied.CreateFeature(output)
        count += 1
    copied.CommitTransaction()
    layer_name = copied.GetName()
    output = copied = target = layer = source = None
    if not count:
        raise ValueError("No source features intersect the benchmark window")
    rasters = []
    for index, path in enumerate(args.rasters):
        dataset = gdal.OpenEx(str(path), gdal.OF_RASTER)
        actual = (dataset.RasterXSize, dataset.RasterYSize,
                  dataset.GetGeoTransform(), dataset.GetProjection())
        if actual != grid:
            raise ValueError(f"Raster does not share the first raster's exact grid: {path}")
        raster = destination / f"{index:02d}_{path.name}"
        cropped = gdal.Translate(
            str(raster), dataset, srcWin=args.window,
            creationOptions=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256",
                             "COMPRESS=LZW"],
        )
        cropped = dataset = None
        rasters.append(raster)
    first = None
    return vector, layer_name, rasters, count


@contextmanager
def phase_measurements():
    measurements = {"prepare_calls": 0, "prepare_seconds": 0.0,
                    "rasterize_calls": 0, "rasterize_seconds": 0.0}
    saved = {}

    def wrap(function, label):
        def measured(*args, **kwargs):
            measurements[f"{label}_calls"] += 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                measurements[f"{label}_seconds"] += time.perf_counter() - started
        return measured

    for name, label in (("_prepare_aggregate_vector_for_rasterization", "prepare"),
                        ("_rasterize_aggregate_fids", "rasterize")):
        saved[name] = getattr(runner, name)
        setattr(runner, name, wrap(saved[name], label))
    try:
        yield measurements
    finally:
        for name, function in saved.items():
            setattr(runner, name, function)


def assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for raster, groups in actual.items():
        assert groups.keys() == expected[raster].keys()
        for group, values in groups.items():
            assert values.keys() == expected[raster][group].keys()
            for metric, value in values.items():
                np.testing.assert_equal(value, expected[raster][group][metric])


def calculate_raster(kwargs):
    """Run and instrument one raster inside its assigned service worker."""
    gdal.UseExceptions()
    runner._configure_gdal_cache()
    with phase_measurements() as measurements:
        result = runner.fast_zonal_statistics(progress_queue=ProgressSink(), **kwargs)
    return Path(kwargs["base_raster_path_band"][0]).name, result, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--layer")
    parser.add_argument("--fields", nargs="+", required=True)
    parser.add_argument("--rasters", type=Path, nargs="+", required=True)
    parser.add_argument("--window", nargs=4, type=int, required=True,
                        metavar=("XOFF", "YOFF", "WIDTH", "HEIGHT"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--simplify-meters", type=float, default=0)
    parser.add_argument("--rasterize-workers", type=int, default=1)
    parser.add_argument("--service-workers", type=int, default=1)
    parser.add_argument("--temporary-parent", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.service_workers < 1 or args.rasterize_workers < 1:
        parser.error("Worker counts must be positive")
    gdal.UseExceptions()
    runner._configure_gdal_cache()
    with tempfile.TemporaryDirectory(prefix="zonal-cache-benchmark-",
                                     dir=args.temporary_parent) as directory:
        work = Path(directory)
        fixtures = work / "fixtures"
        fixtures.mkdir()
        vector, layer, rasters, count = make_fixtures(args, fixtures)
        print(json.dumps({"fixture_features": count, "window": args.window,
                          "rasters": [str(p) for p in args.rasters],
                          "vector": str(args.vector), "fields": args.fields,
                          "repeats": args.repeats,
                          "service_workers": args.service_workers,
                          "rasterize_workers": args.rasterize_workers,
                          "runner_sha256": hashlib.sha256(
                              Path(runner.__file__).read_bytes()).hexdigest(),
                          "gdal_version": gdal.VersionInfo("RELEASE_NAME")}), flush=True)
        records = []
        reference = None
        for trial in range(args.repeats):
            cache = work / f"cache-{trial}"
            for mode in ("uncached", "cold_cache", "warm_cache"):
                results = {}
                calls = [dict(
                            base_raster_path_band=(raster, 1),
                            aggregate_vector_path=vector,
                            aggregate_vector_field=args.fields,
                            aggregate_layer_name=layer,
                            working_dir=work / f"{trial}-{mode}" / raster.stem,
                            max_simplify_tolerance_meters=args.simplify_meters,
                            rasterize_worker_count=args.rasterize_workers,
                            progress_id=raster.stem,
                            operations=["sum"], reuse_geometry=mode != "uncached",
                            geometry_cache_dir=cache,
                        ) for raster in rasters]
                started = time.perf_counter()
                if args.service_workers == 1:
                    completed = [calculate_raster(call) for call in calls]
                else:
                    with ProcessPoolExecutor(
                        max_workers=min(args.service_workers, len(calls)),
                        mp_context=multiprocessing.get_context("spawn"),
                    ) as executor:
                        completed = list(executor.map(calculate_raster, calls))
                elapsed = time.perf_counter() - started
                measurements = {"prepare_calls": 0, "prepare_seconds": 0.0,
                                "rasterize_calls": 0, "rasterize_seconds": 0.0}
                for name, result, timing in completed:
                    results[name] = result
                    for key, value in timing.items():
                        measurements[key] += value
                if reference is None:
                    reference = results
                else:
                    assert_equal(results, reference)
                record = {"trial": trial + 1, "mode": mode,
                          "elapsed_seconds": elapsed, **measurements}
                records.append(record)
                print(json.dumps(record), flush=True)
        medians = {mode: statistics.median(record["elapsed_seconds"] for record in records
                                           if record["mode"] == mode)
                   for mode in ("uncached", "cold_cache", "warm_cache")}
        print(json.dumps({"median_seconds": medians,
                          "cold_speedup": medians["uncached"] / medians["cold_cache"],
                          "warm_speedup": medians["uncached"] / medians["warm_cache"],
                          "nonzero_sum_groups": {
                              raster: int(sum(values["sum"] != 0 for values in groups.values()))
                              for raster, groups in reference.items()},
                          "exact_sum_equality": True}), flush=True)


if __name__ == "__main__":
    main()
