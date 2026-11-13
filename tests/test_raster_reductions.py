"""Independent references for grouped, operation-selective raster reductions."""

from pathlib import Path
import queue

import numpy as np
from osgeo import gdal, gdal_array, ogr, osr
import pytest

import runner


def _srs():
    result = osr.SpatialReference()
    result.ImportFromEPSG(6933)
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def _raster(path, values, nodata=None):
    values = np.asarray(values)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), values.shape[1], values.shape[0], 1,
        gdal_array.NumericTypeCodeToGDALTypeCode(values.dtype),
    )
    dataset.SetProjection(_srs().ExportToWkt())
    dataset.SetGeoTransform((0, 1, 0, values.shape[0], 0, -1))
    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(values)
    band = None
    dataset = None
    return path


def _zones(path, keys, width, height, *, strips=False):
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = dataset.CreateLayer("zones", srs=_srs(), geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("zone_key", ogr.OFTString))
    layer.StartTransaction()
    for index, key in enumerate(keys):
        xmin, xmax = ((index, index + 1) if strips else (0, width))
        shape = ogr.CreateGeometryFromWkt(
            f"POLYGON (({xmin} 0, {xmax} 0, {xmax} {height}, "
            f"{xmin} {height}, {xmin} 0))"
        )
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("zone_key", key)
        feature.SetGeometry(shape)
        layer.CreateFeature(feature)
    layer.CommitTransaction()
    feature = None
    layer = None
    dataset = None
    return path


@pytest.fixture
def controlled_case(tmp_path, monkeypatch):
    """Use real GDAL files with explicit membership labels, independent of geometry."""
    def make(values, ids, *, keys=None, nodata=None, row_blocks=False, after_labels=None):
        values = np.asarray(values)
        ids = np.asarray(ids, dtype=np.int32)
        assert values.shape == ids.shape
        if keys is None:
            keys = [f"zone-{index}" for index in range(1, max(1, int(ids.max())) + 1)]
        source = _raster(tmp_path / "values.tif", values, nodata)
        vector = _zones(tmp_path / "zones.gpkg", keys, values.shape[1], values.shape[0])

        def rasterize(base, aggregate, layer, target, target_nodata, *args, **kwargs):
            assert target_nodata == -1
            _raster(target, ids, -1)
            if after_labels is not None:
                after_labels()
            return 0

        monkeypatch.setattr(runner, "_rasterize_aggregate_fids", rasterize)
        if row_blocks:
            original_iterblocks = runner.geoprocessing.iterblocks

            def blocks(path_band, **kwargs):
                if Path(path_band[0]).name != "agg_fid.tif":
                    yield from original_iterblocks(path_band, **kwargs)
                    return
                assert kwargs["offset_only"]
                for y in range(values.shape[0]):
                    yield dict(xoff=0, yoff=y, win_xsize=values.shape[1], win_ysize=1)

            monkeypatch.setattr(runner.geoprocessing, "iterblocks", blocks)

        counter = 0

        def calculate(**kwargs):
            nonlocal counter
            counter += 1
            progress = kwargs.pop("progress_queue", queue.Queue())
            return runner.fast_zonal_statistics(
                (source, 1), vector, ["zone_key"], aggregate_layer_name="zones",
                working_dir=tmp_path / f"work-{counter}",
                max_simplify_tolerance_meters=0, progress_queue=progress,
                progress_id=f"case-{counter}", **kwargs,
            )

        return calculate, source

    return make


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float32, np.float64])
def test_group_values_preserves_sparse_ids_dtype_and_within_feature_order(dtype):
    ids = np.array([2**40, 7, 2**40, 7, 2**31 + 9, 7], dtype=np.int64)
    values = np.array([21, 4, 13, 8, 9, 2], dtype=dtype)
    groups = list(runner._group_raster_values(ids, values))
    assert [int(fid) for fid, _ in groups] == sorted(set(ids))
    for fid, grouped in groups:
        assert grouped.dtype == values.dtype
        np.testing.assert_array_equal(grouped, values[ids == fid])
    np.testing.assert_array_equal(values, np.array([21, 4, 13, 8, 9, 2], dtype=dtype))


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float32, np.float64])
def test_group_values_memory_fallback_preserves_ids_dtype_and_order(monkeypatch, dtype):
    ids = np.array([2**40, 7, 2**40, 7, 2**31 + 9, 7], dtype=np.int64)
    values = np.array([21, 4, 13, 8, 9, 2], dtype=dtype)
    original_ids, original_values = ids.copy(), values.copy()
    monkeypatch.setattr(runner, "_RASTER_GROUP_SORT_BUDGET_BYTES", 1)

    def forbidden_sort(*args, **kwargs):
        pytest.fail("the low-memory fallback allocated a sort permutation")

    monkeypatch.setattr(runner.np, "argsort", forbidden_sort)
    groups = list(runner._group_raster_values(ids, values))
    assert [int(fid) for fid, _ in groups] == sorted(set(ids))
    for fid, grouped in groups:
        assert grouped.dtype == values.dtype
        np.testing.assert_array_equal(grouped, values[ids == fid])
    np.testing.assert_array_equal(ids, original_ids)
    np.testing.assert_array_equal(values, original_values)


def test_group_values_handles_empty_input():
    assert list(runner._group_raster_values(
        np.array([], dtype=np.int64), np.array([], dtype=np.float32),
    )) == []


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float32, np.float64])
def test_selected_sum_matches_default_and_independent_numpy_sum(controlled_case, dtype):
    values = np.array([[11, 4, 13, 8], [7, 2, 5, 9]], dtype=dtype)
    ids = np.array([[2, 1, 2, 1], [1, 2, 2, 1]])
    calculate, _ = controlled_case(values, ids)
    full = calculate()
    selected = calculate(operations=["sum"])
    for fid in (1, 2):
        expected = np.sum(values[ids == fid])
        assert selected[f"zone-{fid}"] == {"sum": expected}
        assert selected[f"zone-{fid}"]["sum"] == full[f"zone-{fid}"]["sum"]


@pytest.mark.parametrize("operation", [
    "sum", "mean", "min", "max", "stdev", "total_count", "valid_count",
    "proportion_valid_nonzero", "area_ha_total", "area_ha_valid",
])
def test_each_operation_matches_independent_expected_metrics(controlled_case, operation):
    values = np.array([[0, 2, -9999, 4], [6, 8, 10, 12]], dtype=np.float64)
    ids = np.ones_like(values, dtype=np.int32)
    calculate, _ = controlled_case(values, ids, nodata=-9999)
    valid = values[values != -9999]
    expected = {
        "sum": np.sum(valid), "mean": np.mean(valid), "min": np.min(valid),
        "max": np.max(valid), "stdev": np.std(valid), "total_count": 8,
        "valid_count": 7, "proportion_valid_nonzero": 6 / 8,
        "area_ha_total": 8 / 10000, "area_ha_valid": 7 / 10000,
    }
    actual = calculate(operations=[operation], calculate_area_ha=operation.startswith("area_ha"))
    assert set(actual["zone-1"]) == {operation}
    assert actual["zone-1"][operation] == pytest.approx(expected[operation], rel=1e-12, abs=1e-12)


def test_percentiles_are_kept_with_selected_operations(controlled_case):
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    calculate, _ = controlled_case(values, np.ones_like(values, dtype=np.int32))
    actual = calculate(operations=["sum"], percentile_list=[0, 50, 100])["zone-1"]
    assert actual == {"sum": 21, "p0": 1, "p50": 4, "p100": 6}
    full = calculate(percentile_list=[0, 50, 100])["zone-1"]
    assert actual == {key: full[key] for key in actual}


@pytest.mark.parametrize("nonfinite", [False, True])
def test_selected_metrics_preserve_ignore_nodata_false_behavior(controlled_case, nonfinite):
    values = np.array([[0, 2, -9999, 4], [6, -9999, 10, 12]], dtype=np.float64)
    if nonfinite:
        values[1, :2] = [np.nan, np.inf]
    calculate, _ = controlled_case(values, np.ones_like(values, dtype=np.int32), nodata=-9999)
    operations = ["sum", "mean", "stdev", "min", "max", "total_count",
                  "valid_count", "proportion_valid_nonzero"]
    with np.errstate(invalid="ignore"):
        full = calculate(ignore_nodata=False)["zone-1"]
        selected = calculate(ignore_nodata=False, operations=operations)["zone-1"]
        sum_only = calculate(ignore_nodata=False, operations=["sum"])["zone-1"]
    assert set(selected) == set(operations)
    for operation in operations:
        np.testing.assert_equal(selected[operation], full[operation])
    np.testing.assert_equal(sum_only["sum"], full["sum"])
    if not nonfinite:
        assert sum_only["sum"] == np.sum(values)  # The finite sentinel participates.


def test_percentile_only_request_omits_core_metrics(controlled_case):
    values = np.array([[1, 3, 5, -9999], [-9999, -9999, -9999, -9999]], dtype=np.float64)
    ids = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
    calculate, _ = controlled_case(values, ids, nodata=-9999)
    actual = calculate(operations=[], percentile_list=[0, 50, 100])
    assert actual == {
        "zone-1": {"p0": 1, "p50": 3, "p100": 5},
        "zone-2": {"p0": None, "p50": None, "p100": None},
    }


def test_unknown_operation_rejected_before_calculation(controlled_case):
    values = np.ones((1, 1), dtype=np.float64)
    calculate, _ = controlled_case(values, [[1]])
    progress = queue.Queue()
    with pytest.raises(ValueError, match="Unsupported raster operations.*variance"):
        calculate(operations=["variance"], progress_queue=progress)
    assert progress.empty()


@pytest.mark.parametrize("nodata", [None, -9999.0, np.nan])
def test_nodata_nonfinite_and_zero_handling(controlled_case, nodata):
    values = np.array([[0, 2, np.nan, np.inf], [-np.inf, -9999, 4, 8]], dtype=np.float64)
    calculate, _ = controlled_case(values, np.ones_like(values, dtype=np.int32), nodata=nodata)
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= ~np.isclose(values, nodata)
    expected = values[valid]
    actual = calculate(operations=["sum", "valid_count", "total_count", "proportion_valid_nonzero"])["zone-1"]
    assert actual == {
        "sum": np.sum(expected), "valid_count": expected.size, "total_count": values.size,
        "proportion_valid_nonzero": np.count_nonzero(expected) / values.size,
    }


@pytest.mark.parametrize("ids", [np.full((2, 3), -1), np.ones((2, 3))])
def test_empty_and_all_nodata_features_keep_zero_sum_and_null_metrics(controlled_case, ids):
    values = np.full((2, 3), -9999, dtype=np.float32)
    calculate, _ = controlled_case(values, ids, keys=["empty"], nodata=-9999)
    actual = calculate(operations=["sum", "valid_count", "mean", "min", "max", "stdev"])["empty"]
    assert actual == {"sum": 0.0, "valid_count": 0, "mean": None, "min": None, "max": None, "stdev": None}


def test_multiple_features_with_shared_group_key(controlled_case):
    ids = np.tile(np.arange(1, 65, dtype=np.int32), 3).reshape(12, 16)
    np.random.default_rng(8).shuffle(ids, axis=1)
    values = np.arange(ids.size, dtype=np.float64).reshape(ids.shape)
    keys = [f"group-{fid % 5}" for fid in range(1, 65)]
    calculate, _ = controlled_case(values, ids, keys=keys, row_blocks=True)
    actual = calculate(operations=["sum", "valid_count"])
    assert set(actual) == set(keys)
    for group in set(keys):
        member_ids = [fid for fid, key in enumerate(keys, start=1) if key == group]
        selected = values[np.isin(ids, member_ids)]
        assert actual[group] == {"sum": np.sum(selected), "valid_count": selected.size}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiblock_sum_keeps_legacy_reduction_order(controlled_case, monkeypatch, dtype):
    values = np.array([
        [1e8, 1, -1e8, 3, 1e-3, 9],
        [1e5, 7, -1e5, 11, 1e-3, 13],
        [1e4, 17, -1e4, 19, 1e-3, 23],
    ], dtype=dtype)
    ids = np.array([[1, 2, 1, 2, 1, 2]] * 3)
    calculate, _ = controlled_case(values, ids, row_blocks=True)
    monkeypatch.setattr(runner, "_RASTER_GROUP_SORT_BUDGET_BYTES", 2**30)
    sorted_actual = calculate(operations=["sum"])
    monkeypatch.setattr(runner, "_RASTER_GROUP_SORT_BUDGET_BYTES", 1)
    fallback_actual = calculate(operations=["sum"])
    full_fallback = calculate()
    for fid in (1, 2):
        expected = 0.0
        for row_values, row_ids in zip(values, ids):
            expected += np.sum(row_values[row_ids == fid])
        # Identical feature ordering and block boundaries require exact equality;
        # a whole-array float64 sum would use a different rounding order.
        assert sorted_actual[f"zone-{fid}"]["sum"] == expected
        assert fallback_actual[f"zone-{fid}"]["sum"] == expected
        assert full_fallback[f"zone-{fid}"]["sum"] == expected


def test_sum_only_skips_unrequested_reductions(controlled_case, monkeypatch):
    calls = []
    real_sum = np.sum

    def unexpected(*args, **kwargs):
        pytest.fail("sum-only calculation evaluated an unrequested statistic")

    def observed_sum(values, *args, **kwargs):
        calls.append(np.asarray(values).copy())
        return real_sum(values, *args, **kwargs)

    def install_spies():
        # Install after geometry preparation: only inspect the numeric stage.
        for name in ("min", "max", "sqrt"):
            monkeypatch.setattr(runner.np, name, unexpected)
        monkeypatch.setattr(runner.np, "sum", observed_sum)

    monkeypatch.setattr(runner, "_raster_pixel_area_ha", unexpected)
    values = np.array([[2, 3, 5, 7]], dtype=np.float64)
    ids = np.array([[1, 2, 1, 2]])
    calculate, _ = controlled_case(values, ids, after_labels=install_spies)
    assert calculate(operations=["sum"]) == {"zone-1": {"sum": 7}, "zone-2": {"sum": 10}}
    assert len(calls) == 2  # No extra sum over squared values.
    np.testing.assert_array_equal(calls[0], [2, 5])
    np.testing.assert_array_equal(calls[1], [3, 7])


def test_empty_membership_blocks_do_not_read_service_values(controlled_case, monkeypatch):
    values = np.array([[3, 5], [7, 11]], dtype=np.float64)
    ids = np.array([[-1, -1], [1, 1]])
    calculate, source = controlled_case(values, ids, row_blocks=True)
    reads = []
    original_read = gdal.Band.ReadAsArray

    def observed_read(band, *args, **kwargs):
        if Path(band.GetDataset().GetDescription()) == source:
            reads.append((args, kwargs.copy()))
        return original_read(band, *args, **kwargs)

    monkeypatch.setattr(gdal.Band, "ReadAsArray", observed_read)
    assert calculate(operations=["sum"]) == {"zone-1": {"sum": 18}}
    assert reads == [((), dict(xoff=0, yoff=1, win_xsize=2, win_ysize=1))]


class _ProgressLog:
    """Record actual progress messages alongside read/reduction observations."""

    def __init__(self, timeline):
        self.timeline = timeline

    def put(self, event):
        self.timeline.append(("progress", dict(event)))


def test_statistics_progress_counts_completed_work_after_reduction(controlled_case, monkeypatch):
    values = np.array([[3, 5], [7, 11]], dtype=np.float64)
    calculate, source = controlled_case(values, [[-1, -1], [1, 1]], row_blocks=True)
    timeline = []
    real_read = gdal.Band.ReadAsArray
    real_group = runner._group_raster_values

    def read(band, *args, **kwargs):
        if Path(band.GetDataset().GetDescription()) == source:
            timeline.append(("read", kwargs["yoff"]))
        return real_read(band, *args, **kwargs)

    def groups(ids, raster_values):
        yield from real_group(ids, raster_values)
        timeline.append(("reduction-finished", None))

    monkeypatch.setattr(gdal.Band, "ReadAsArray", read)
    monkeypatch.setattr(runner, "_group_raster_values", groups)
    actual = calculate(operations=["sum"], progress_queue=_ProgressLog(timeline))
    assert actual == {"zone-1": {"sum": 18}}
    phase_index = next(index for index, (kind, event) in enumerate(timeline)
                       if kind == "progress" and event["event"] == "analysis_phase"
                       and event.get("unit") == "block")
    phase = timeline[phase_index][1]
    assert phase["total"] == 2
    assert phase["phase"] == "completed raster blocks 0/2"
    # A separate phase starts at zero rather than inheriting tile/preparation counts.
    assert "increment" not in phase
    completed = [(index, event) for index, (kind, event) in enumerate(timeline)
                 if kind == "progress" and event["event"] == "analysis_update"
                 and event.get("phase", "").startswith("completed raster blocks")]
    assert [event["phase"] for _, event in completed] == [
        "completed raster blocks 1/2", "completed raster blocks 2/2",
    ]
    assert [event["increment"] for _, event in completed] == [1, 1]
    read_index = timeline.index(("read", 1))
    reduce_index = timeline.index(("reduction-finished", None))
    assert phase_index < completed[0][0] < read_index < reduce_index < completed[1][0]
    assert [(kind, row) for kind, row in timeline if kind == "read"] == [("read", 1)]
    progress_events = [event for kind, event in timeline if kind == "progress"]
    assert progress_events[-1]["event"] == "analysis_close"
    assert progress_events[-1]["phase"] == "done"


@pytest.mark.parametrize("failure_stage", ["read", "reduction"])
def test_statistics_failure_never_reports_unfinished_block_complete(
    controlled_case, monkeypatch, failure_stage,
):
    values = np.array([[3, 5]], dtype=np.float64)
    calculate, source = controlled_case(values, [[1, 1]])
    timeline = []
    if failure_stage == "read":
        real_read = gdal.Band.ReadAsArray

        def failing_read(band, *args, **kwargs):
            if Path(band.GetDataset().GetDescription()) == source:
                raise RuntimeError("injected read failure")
            return real_read(band, *args, **kwargs)

        monkeypatch.setattr(gdal.Band, "ReadAsArray", failing_read)
    else:
        def failing_reduction(*args, **kwargs):
            raise RuntimeError("injected reduction failure")

        monkeypatch.setattr(runner, "_group_raster_values", failing_reduction)

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        calculate(operations=["sum"], progress_queue=_ProgressLog(timeline))
    events = [event for kind, event in timeline if kind == "progress"]
    assert any(event["event"] == "analysis_phase" and event.get("unit") == "block"
               and event["total"] == 1 for event in events)
    assert not any(event["event"] == "analysis_update"
                   and event.get("phase", "").startswith("completed raster blocks")
                   for event in events)
    assert events[-1]["event"] == "analysis_close"
    assert events[-1]["status"] == "failed"
    assert events[-1]["phase"] == "failed"


def test_real_polygon_membership_sum_only(tmp_path):
    """Exercise actual projection, pixel-center rasterization, and reduction."""
    values = np.array([[1, 2, 3], [4, -9999, 6]], dtype=np.float32)
    source = _raster(tmp_path / "values.tif", values, -9999)
    vector = _zones(tmp_path / "zones.gpkg", ["A", "B", "A"], 3, 2, strips=True)
    actual = runner.fast_zonal_statistics(
        (source, 1), vector, ["zone_key"], aggregate_layer_name="zones",
        working_dir=tmp_path / "work", operations=["sum"],
        max_simplify_tolerance_meters=0, rasterize_worker_count=1,
        progress_queue=queue.Queue(), progress_id="real-membership",
    )
    assert actual == {"A": {"sum": 14}, "B": {"sum": 2}}
