"""Exercise the shared budget with real spawned workers and tiny GIS inputs."""

from concurrent.futures import ProcessPoolExecutor
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
from osgeo import gdal, osr
import pandas as pd
import pytest
from shapely.geometry import box

import runner


def _forbid_nested_pool(*args, **kwargs):
    raise AssertionError("a shared worker tried to create a nested process pool")


def _tracked_task(function, args, kwargs, state, lock, first_pair, events):
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
    try:
        # Both first workers must actually execute, not just be submitted.
        if sequence <= 2:
            first_pair.wait(timeout=40)
        return function(*args, **kwargs)
    finally:
        with lock:
            state["active"] -= 1


class _RecordingExecutor:
    """Delegate to a real spawn pool while recording the scheduler's budget."""

    def __init__(self, state, lock, first_pair, events, *args, **kwargs):
        self.max_workers = kwargs["max_workers"]
        self.start_method = kwargs["mp_context"].get_start_method()
        self.pool = ProcessPoolExecutor(*args, **kwargs)
        self.state = state
        self.lock = lock
        self.first_pair = first_pair
        self.events = events
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
        assert not list(actual_job["workdir"].iterdir())


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
    assert not list(job["workdir"].iterdir())
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
