"""Exercise the shared budget with real spawned workers and tiny GIS inputs."""

from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import logging
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import numpy as np
from osgeo import gdal, ogr, osr
import pandas as pd
import pytest
from shapely.geometry import box

import runner


def _forbid_nested_pool(*args, **kwargs):
    raise AssertionError("a shared worker tried to create a nested process pool")


def _tracked_task(function, args, kwargs, state, lock, first_pair, events,
                  cache_events=None, fail_after_raster=None):
    """Run in a spawned process; instrument actual raster and publish work."""
    assert all(os.environ.get(name) == "1" for name in runner._SHARED_THREAD_ENV_VARS)
    assert gdal.GetConfigOption("GDAL_NUM_THREADS") == "1"
    runner.ProcessPoolExecutor = _forbid_nested_pool
    with lock:
        state["started"] += 1
        sequence = state["started"]
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
    events.append((function.__name__, args[0]["tag"]))
    prepare = runner._prepare_aggregate_vector_for_rasterization
    burn = runner._rasterize_aggregate_fids
    if cache_events is not None:
        # Install spies inside spawned workers; parent monkeypatches do not
        # cross the Windows spawn boundary. Record builds, not just manifests.
        def observed_prepare(*build_args, **build_kwargs):
            cache_events.append(("prepare", args[0]["tag"]))
            return prepare(*build_args, **build_kwargs)

        def observed_burn(*build_args, **build_kwargs):
            cache_events.append(("burn", args[0]["tag"]))
            return burn(*build_args, **build_kwargs)

        runner._prepare_aggregate_vector_for_rasterization = observed_prepare
        runner._rasterize_aggregate_fids = observed_burn
    try:
        # Both first workers must actually execute, not just be submitted.
        if sequence <= 2:
            first_pair.wait(timeout=40)
        result = function(*args, **kwargs)
        if (function.__name__ == "_shared_raster_task"
                and args[1] == fail_after_raster):
            raise RuntimeError("injected failure after geometry caching")
        return result
    finally:
        runner._prepare_aggregate_vector_for_rasterization = prepare
        runner._rasterize_aggregate_fids = burn
        with lock:
            state["active"] -= 1


class _RecordingExecutor:
    """Delegate to a real spawn pool while recording the scheduler's budget."""

    def __init__(self, state, lock, first_pair, events, *args,
                 cache_events=None, fail_after_raster=None, **kwargs):
        self.max_workers = kwargs["max_workers"]
        self.start_method = kwargs["mp_context"].get_start_method()
        self.pool = ProcessPoolExecutor(*args, **kwargs)
        self.state = state
        self.lock = lock
        self.first_pair = first_pair
        self.events = events
        self.cache_events = cache_events
        self.fail_after_raster = fail_after_raster
        self.submissions = []
        self.pending = 0
        self.peak_pending = 0
        self.pending_lock = threading.Lock()

    def submit(self, function, *args, **kwargs):
        self.submissions.append((function.__name__, args[0]["tag"]))
        with self.pending_lock:
            self.pending += 1
            self.peak_pending = max(self.peak_pending, self.pending)
        future = self.pool.submit(
            _tracked_task, function, args, kwargs,
            self.state, self.lock, self.first_pair, self.events,
            self.cache_events, self.fail_after_raster,
        )

        def completed(_future):
            with self.pending_lock:
                self.pending -= 1

        future.add_done_callback(completed)
        return future

    def shutdown(self, *args, **kwargs):
        return self.pool.shutdown(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)


@pytest.fixture
def progress_queue():
    with multiprocessing.get_context("spawn").Manager() as manager:
        yield manager.Queue()


@pytest.fixture
def raster_jobs(tmp_path):
    """Two feature groups and three rasters, with original vector attributes."""
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(6933)
    rasters = []
    for raster_index, name in enumerate(("z_value", "a_value", "m_value")):
        path = tmp_path / f"{name}.tif"
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(path), 4, 4, 1, gdal.GDT_Float32
        )
        dataset.SetGeoTransform((0, 1, 0, 4, 0, -1))
        dataset.SetProjection(spatial_reference.ExportToWkt())
        values = np.arange(1, 17, dtype=np.float32).reshape((4, 4))
        values *= raster_index + 1
        values[1, 1] = -1
        dataset.GetRasterBand(1).SetNoDataValue(-1)
        dataset.GetRasterBand(1).WriteArray(values)
        dataset = None
        rasters.append(path)

    jobs = []
    for group_index in range(2):
        tag = f"group_{group_index}"
        vector_path = tmp_path / f"{tag}.gpkg"
        # The second group's geometries overlap the first group's geometries.
        # Within each group the features are disjoint, as in the PAD workflow.
        geometries = (
            [box(0, 0, 2, 4), box(2, 0, 4, 4)]
            if group_index == 0 else [box(0, 0, 4, 2), box(0, 2, 4, 4)]
        )
        gpd.GeoDataFrame(
            {"feature_id": [f"{tag}_left", f"{tag}_right"],
             "original_name": ["First original name", "Second original name"],
             "GAP_Sts": ["1", "4"]},
            geometry=geometries, crs="EPSG:6933",
        ).to_file(vector_path, layer="zones", driver="GPKG", index=False)
        jobs.append({
            "base_raster_path_list": rasters,
            "base_vector_path_list": [],
            "base_vector_fields": [],
            "base_measure_vector": None,
            "base_measure_layer": None,
            "measure_crs": "auto",
            "agg_vector": vector_path,
            "agg_layer": "zones",
            "agg_field": ["feature_id"],
            "operations": ["sum", "mean", "valid_count", "total_count",
                           "area_ha_total", "area_ha_valid", "p50"],
            "output_csv": tmp_path / f"shared_{tag}.csv",
            "output_gpkg": tmp_path / f"shared_{tag}.gpkg",
            "workdir": tmp_path / f"shared_work_{tag}",
            "tag": tag,
            "task_graph": None,
            "progress_queue": None,
            "max_simplify_tolerance_meters": 0,
        })
    return jobs


def _job_run_list(jobs):
    return [(str(job["output_csv"]), job) for job in jobs]


def _drain_events(progress_queue):
    events = []
    while True:
        try:
            events.append(progress_queue.get_nowait())
        except queue.Empty:
            return events


def _assert_no_shared_scratch(job):
    # Completed cache artifacts persist; per-task/spooled scratch must not.
    assert {path.name for path in Path(job["workdir"]).iterdir()} <= {"geometry_cache"}


def _new_outputs(jobs, label):
    return [dict(job, output_csv=job["output_csv"].with_name(f"{label}_{job['tag']}.csv"),
                 output_gpkg=job["output_gpkg"].with_name(f"{label}_{job['tag']}.gpkg"))
            for job in jobs]


def _assert_matches_uncached_reference(job, tmp_path):
    """Compare each published statistic with independently run, uncached zonals."""
    actual = pd.read_csv(job["output_csv"], dtype={"GAP_Sts": str},
                         float_precision="round_trip").set_index("feature_id").sort_index()
    vector = gpd.read_file(job["output_gpkg"], layer="zones").set_index("feature_id").sort_index()
    source = gpd.read_file(job["agg_vector"], layer="zones").set_index("feature_id").sort_index()
    assert set(vector.columns) == set(actual.columns) | set(source.columns)
    stored = pd.DataFrame(vector[actual.columns])
    # CSV prints float32's shortest representation; restore its stored dtype
    # before requiring exact equality, rather than comparing widened decimals.
    restored = actual.copy()
    for column in actual:
        if stored[column].dtype == np.dtype("float32"):
            restored[column] = restored[column].astype(np.float32)
    pd.testing.assert_frame_equal(restored, stored, check_exact=True, check_dtype=False)
    assert_geodataframe_equal(vector[source.columns], source)
    for raster in job["base_raster_path_list"]:
        expected = runner.fast_zonal_statistics(
            (raster, 1), job["agg_vector"], job["agg_field"],
            aggregate_layer_name="zones", operations=[operation for operation in job["operations"]
                                                       if not operation.startswith("p")],
            percentile_list=[50], working_dir=tmp_path / "reference" / job["tag"] / raster.stem,
            rasterize_worker_count=1, max_simplify_tolerance_meters=0,
            reuse_geometry=False, progress_queue=queue.Queue(), progress_id="reference",
        )
        for metric in next(iter(expected.values())):
            expected_values = pd.Series({key: values[metric] for key, values in expected.items()}).sort_index()
            actual_values = actual[f"{metric}_{raster.stem}"]
            if expected_values.dtype == np.dtype("float32"):
                actual_values = actual_values.astype(np.float32)
            pd.testing.assert_series_equal(
                actual_values, expected_values,
                check_exact=True, check_dtype=False, check_names=False,
            )


@pytest.fixture
def cache_tracking_pool(monkeypatch):
    """Use the existing real-spawn recorder to observe cache work across runs."""
    with multiprocessing.get_context("spawn").Manager() as manager:
        cache_events = manager.list()

        def run(jobs, progress, *, fail_after_raster=None):
            state = manager.dict(started=0, active=0, peak=0)
            lock, first_pair, events = manager.Lock(), manager.Barrier(2), manager.list()

            def executor_factory(*args, **kwargs):
                return _RecordingExecutor(
                    state, lock, first_pair, events, *args, cache_events=cache_events,
                    fail_after_raster=fail_after_raster, **kwargs,
                )

            with monkeypatch.context() as patch:
                patch.setattr(runner, "ProcessPoolExecutor", executor_factory)
                runner.run_shared_raster_jobs(_job_run_list(jobs), 2, logging.WARNING, progress)
            assert state["peak"] <= 2

        yield run, cache_events


def test_shared_geometry_cache_survives_scratch_cleanup_and_recomputes_service_values(
    tmp_path, raster_jobs, progress_queue, cache_tracking_pool,
):
    run, builds = cache_tracking_pool
    run(raster_jobs, progress_queue)
    assert Counter(builds) == Counter({(kind, job["tag"]): 1
                                      for kind in ("prepare", "burn") for job in raster_jobs})
    cold = []
    for group_index, job in enumerate(raster_jobs):
        _assert_no_shared_scratch(job)
        _assert_matches_uncached_reference(job, tmp_path)
        cache = job["workdir"] / "geometry_cache"
        assert len(list((cache / "polygons").glob("*/manifest.json"))) == 1
        assert len(list((cache / "ids").glob("*/manifest.json"))) == 1
        actual = pd.read_csv(job["output_csv"])
        base_sums = np.array([54, 76] if group_index == 0 else [100, 30])
        for index, raster in enumerate(job["base_raster_path_list"], start=1):
            np.testing.assert_array_equal(actual[f"sum_{raster.stem}"], base_sums * index)
        cold.append(actual)

    # The same raster path/grid now has different values and a different NoData.
    # Its sums must be fresh, while both prepared geometry and ID pixels reuse.
    dataset = gdal.Open(str(raster_jobs[0]["base_raster_path_list"][0]), gdal.GA_Update)
    values = np.full((4, 4), 5, dtype=np.float32)
    values[0, 0] = -9999
    dataset.GetRasterBand(1).WriteArray(values)
    dataset.GetRasterBand(1).SetNoDataValue(-9999)
    dataset = None
    before = list(builds)
    warm = _new_outputs(raster_jobs, "warm")
    run(warm, progress_queue)
    assert list(builds) == before
    for group_index, (job, previous) in enumerate(zip(warm, cold)):
        _assert_no_shared_scratch(job)
        _assert_matches_uncached_reference(job, tmp_path)
        actual = pd.read_csv(job["output_csv"])
        np.testing.assert_array_equal(actual["sum_z_value"],
                                      [35, 40] if group_index == 0 else [40, 35])
        assert not actual["sum_z_value"].equals(previous["sum_z_value"])
        unchanged = [column for column in actual if not column.endswith("_z_value")]
        pd.testing.assert_frame_equal(actual[unchanged], previous[unchanged], check_exact=True)


@pytest.mark.parametrize("change", ["grid", "source"])
def test_shared_geometry_cache_invalidates_changed_grid_or_source(
    tmp_path, raster_jobs, progress_queue, cache_tracking_pool, change,
):
    run, builds = cache_tracking_pool
    job = raster_jobs[0]
    run([job], progress_queue)
    assert Counter(builds) == Counter({("prepare", job["tag"]): 1, ("burn", job["tag"]): 1})
    before = len(builds)
    if change == "grid":
        dataset = gdal.Open(str(job["base_raster_path_list"][0]), gdal.GA_Update)
        dataset.SetGeoTransform((1, 1, 0, 4, 0, -1))
        dataset = None
    else:
        dataset = ogr.Open(str(job["agg_vector"]), 1)
        layer = dataset.GetLayerByName("zones")
        feature = layer.GetFeature(1)
        feature.SetGeometry(ogr.CreateGeometryFromWkt(box(0, 0, 1, 4).wkt))
        feature.SetField("original_name", "Changed original name")
        assert layer.SetFeature(feature) == 0
        feature, layer, dataset = None, None, None
    updated, = _new_outputs([job], "updated")
    run([updated], progress_queue)
    expected_builds = Counter({("burn", job["tag"]): 1})
    if change == "source":
        expected_builds[("prepare", job["tag"])] = 1
    assert Counter(list(builds)[before:]) == expected_builds
    _assert_no_shared_scratch(updated)
    _assert_matches_uncached_reference(updated, tmp_path)


def test_shared_geometry_cache_survives_failed_task_and_is_reused_on_retry(
    tmp_path, raster_jobs, progress_queue, cache_tracking_pool,
):
    run, builds = cache_tracking_pool
    job = raster_jobs[0]
    with pytest.raises(RuntimeError, match="failed"):
        run([job], progress_queue, fail_after_raster=0)
    assert not job["output_csv"].exists()
    assert not job["output_gpkg"].exists()
    _assert_no_shared_scratch(job)
    assert Counter(builds) == Counter({("prepare", job["tag"]): 1, ("burn", job["tag"]): 1})
    before = list(builds)
    retried, = _new_outputs([job], "retry")
    run([retried], progress_queue)
    assert list(builds) == before
    _assert_no_shared_scratch(retried)
    _assert_matches_uncached_reference(retried, tmp_path)


def test_real_shared_pool_caps_workers_and_matches_legacy(
    monkeypatch, tmp_path, raster_jobs, progress_queue,
):
    """Prove cross-group activity, no nested pools, and CSV/GPKG equivalence."""
    expected = []
    for job in raster_jobs:
        legacy = dict(job)
        legacy.update(
            output_csv=tmp_path / f"legacy_{job['tag']}.csv",
            output_gpkg=tmp_path / f"legacy_{job['tag']}.gpkg",
            workdir=tmp_path / f"legacy_work_{job['tag']}",
            progress_queue=progress_queue,
            raster_workers=2,
        )
        runner.run_zonal_stats_job(**legacy)
        expected.append(legacy)
    _drain_events(progress_queue)

    executors = []
    with multiprocessing.get_context("spawn").Manager() as manager:
        state = manager.dict(started=0, active=0, peak=0)
        lock = manager.Lock()
        first_pair = manager.Barrier(2)
        events = manager.list()

        def executor_factory(*args, **kwargs):
            executor = _RecordingExecutor(
                state, lock, first_pair, events, *args, **kwargs
            )
            executors.append(executor)
            return executor

        monkeypatch.setattr(runner, "ProcessPoolExecutor", executor_factory)
        for name, value in zip(runner._SHARED_THREAD_ENV_VARS, ("9", "7", "5", "3", None)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        original_environment = {
            name: os.environ.get(name) for name in runner._SHARED_THREAD_ENV_VARS
        }
        runner.run_shared_raster_jobs(
            _job_run_list(raster_jobs), 2, logging.WARNING, progress_queue
        )
        assert {
            name: os.environ.get(name) for name in runner._SHARED_THREAD_ENV_VARS
        } == original_environment

        assert state["peak"] == 2
        assert state["active"] == 0
        assert {tag for _, tag in list(events)[:2]} == {"group_0", "group_1"}
        assert sum(name == "_shared_publish_task" for name, _ in events) == 2

    assert len(executors) == 1
    executor = executors[0]
    assert executor.max_workers == 2
    assert executor.start_method == "spawn"
    assert executor.peak_pending <= 4
    assert {tag for _, tag in executor.submissions[:2]} == {"group_0", "group_1"}
    progress_events = _drain_events(progress_queue)
    analysis_events = [event for event in progress_events if event["event"].startswith("analysis_")]
    assert analysis_events
    for event in analysis_events:
        assert event["tag"] in {"group_0", "group_1"}
        assert f":{event['tag']}:" in event["id"]
    assert sorted(
        event["tag"] for event in progress_events
        if event["event"] == "job_done" and event["status"] == "done"
    ) == ["group_0", "group_1"]

    for actual_job, expected_job in zip(raster_jobs, expected):
        actual_csv = pd.read_csv(actual_job["output_csv"])
        expected_csv = pd.read_csv(expected_job["output_csv"])
        pd.testing.assert_frame_equal(actual_csv, expected_csv, check_exact=True)
        assert [column for column in actual_csv if column.startswith("sum_")] == [
            "sum_z_value", "sum_a_value", "sum_m_value"
        ]
        actual_gpkg = gpd.read_file(actual_job["output_gpkg"], layer="zones")
        expected_gpkg = gpd.read_file(expected_job["output_gpkg"], layer="zones")
        assert_geodataframe_equal(actual_gpkg, expected_gpkg)
        assert actual_gpkg["GAP_Sts"].tolist() == ["1", "4"]
        assert actual_gpkg["original_name"].tolist() == [
            "First original name", "Second original name"
        ]
        _assert_no_shared_scratch(actual_job)


@pytest.mark.parametrize("extra_args", [
    ["--job-workers", "2"], ["--max-workers", "2"], ["--raster-workers", "2"],
])
def test_shared_cli_rejects_explicit_nested_worker_options(monkeypatch, extra_args):
    monkeypatch.setattr(
        sys, "argv", ["runner.py", "unused.ini", "--total-workers", "2"] + extra_args
    )
    with pytest.raises(ValueError, match="total-workers"):
        runner.main()


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_shared_cli_rejects_nonpositive_budget(monkeypatch, workers):
    monkeypatch.setattr(
        sys, "argv", ["runner.py", "unused.ini", "--total-workers", workers]
    )
    with pytest.raises(ValueError, match="total-workers"):
        runner.main()


@pytest.mark.parametrize("variant", ["mixed", "vector", "measure", "empty"])
def test_shared_pool_rejects_non_raster_jobs_before_starting_workers(
    monkeypatch, raster_jobs, variant,
):
    job = dict(raster_jobs[0])
    if variant in {"mixed", "vector"}:
        job["base_vector_path_list"] = [job["agg_vector"]]
        job["base_vector_fields"] = ["GAP_Sts"]
    if variant in {"vector", "measure", "empty"}:
        job["base_raster_path_list"] = []
    if variant == "measure":
        job["base_measure_vector"] = job["agg_vector"]
        job["operations"] = ["intersect_area_ha"]
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _forbid_nested_pool)
    with pytest.raises(ValueError, match="raster"):
        runner.run_shared_raster_jobs(_job_run_list([job]), 2, logging.WARNING, None)
    assert not job["output_csv"].exists()
    assert not job["output_gpkg"].exists()


def test_shared_raster_failure_does_not_publish_partial_group(
    monkeypatch, tmp_path, raster_jobs, progress_queue,
):
    job = dict(raster_jobs[0])
    job["base_raster_path_list"] = [
        job["base_raster_path_list"][0], tmp_path / "missing_raster.tif"
    ]
    monkeypatch.setenv("OMP_NUM_THREADS", "13")
    with pytest.raises(RuntimeError) as caught:
        runner.run_shared_raster_jobs(
            _job_run_list([job]), 2, logging.WARNING, progress_queue
        )
    assert job["tag"] in str(caught.value)
    assert "missing_raster" in str(caught.value)
    assert not job["output_csv"].exists()
    assert not job["output_gpkg"].exists()
    _assert_no_shared_scratch(job)
    assert os.environ["OMP_NUM_THREADS"] == "13"


def test_shared_pool_rejects_duplicate_output_paths(monkeypatch, raster_jobs):
    jobs = [dict(job) for job in raster_jobs]
    jobs[1]["output_csv"] = jobs[0]["output_csv"]
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _forbid_nested_pool)
    with pytest.raises(ValueError, match="same output path"):
        runner.run_shared_raster_jobs(_job_run_list(jobs), 2, logging.WARNING, None)


def test_shared_pool_preserves_existing_outputs(monkeypatch, raster_jobs):
    job = raster_jobs[0]
    job["output_csv"].write_text("previous completed run\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _forbid_nested_pool)
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_shared_raster_jobs(_job_run_list([job]), 2, logging.WARNING, None)
    assert job["output_csv"].read_text(encoding="utf-8") == "previous completed run\n"


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_shared_publication_rolls_back_incomplete_pair(
    monkeypatch, tmp_path, raster_jobs, failure,
):
    job = raster_jobs[0]
    result_path = tmp_path / "raster_result.pkl"
    pd.DataFrame({
        "feature_id": ["group_0_left", "group_0_right"],
        "sum_values": [60.0, 70.0],
    }).to_pickle(result_path)
    if failure == "write":
        def failing_writer(table, vector, layer, fields, csv_path, gpkg_path):
            table.to_csv(csv_path, index=False)
            raise RuntimeError("injected GeoPackage write failure")

        monkeypatch.setattr(runner, "_write_zonal_outputs", failing_writer)
        expected_error = RuntimeError
    else:
        # Simulate another run creating the second destination after preflight.
        job["output_gpkg"].write_bytes(b"another run's completed output")
        expected_error = FileExistsError

    with pytest.raises(expected_error):
        runner._shared_publish_task(job, [result_path])

    assert not job["output_csv"].exists()
    if failure == "publish":
        assert job["output_gpkg"].read_bytes() == b"another run's completed output"
    else:
        assert not job["output_gpkg"].exists()
    assert not list(tmp_path.glob(".zonal_publish_*"))


def test_parallel_progress_reports_the_event_group(monkeypatch):
    class FakeTqdm:
        instances = []

        def __init__(self, total, desc, unit, position, leave):
            self.total, self.desc, self.unit = total, desc, unit
            self.n = 0
            self.postfix_values = []
            self.instances.append(self)

        def update(self, count):
            self.n += count

        def set_postfix_str(self, value, refresh=True):
            self.postfix_values.append(value)

        def refresh(self):
            pass

        def close(self):
            pass

    events = queue.Queue()
    events.put({"event": "job_start", "tag": "group_a"})
    events.put({"event": "job_start", "tag": "group_b"})
    for tag in ("group_a", "group_b"):
        events.put({"event": "analysis_start", "id": f"raster:{tag}:values",
                    "tag": tag, "desc": "raster values", "total": 2,
                    "phase": "starting"})
    events.put({"event": "analysis_update", "id": "raster:group_a:values",
                "tag": "group_a", "increment": 1, "phase": "stitching tiles"})
    events.put({"event": "stop"})
    monkeypatch.setattr(runner, "tqdm", FakeTqdm)
    runner._progress_monitor(events, total_jobs=2)
    job_bar = FakeTqdm.instances[0]
    assert "group_a: raster values - stitching tiles" in job_bar.postfix_values
    assert "group_b: raster values - stitching tiles" not in job_bar.postfix_values
