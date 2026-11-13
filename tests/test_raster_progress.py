"""Keep raster phases independently timed and failures visibly incomplete."""

from concurrent.futures import Future
import queue

import pytest

import runner


class RecordingBar:
    """Record the public tqdm calls used by the progress monitor."""

    def __init__(self, total, desc, unit, position, leave):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.closed = False
        self.postfix = ""
        self.events = []

    def update(self, increment):
        self.n += increment
        self.events.append(("update", self.n, self.total, self.unit))

    def reset(self, total=None):
        # Like tqdm.reset, None by itself would retain the previous total.
        if total is not None:
            self.total = total
        self.n = 0
        self.events.append(("reset", self.n, self.total, self.unit))

    def set_postfix_str(self, value, refresh=True):
        self.postfix = value
        self.events.append(("phase", value, self.n, self.total, self.unit))

    def refresh(self):
        pass

    def close(self):
        self.closed = True


def monitor(monkeypatch, events, total_jobs=1):
    """Run queued events synchronously without terminal output."""
    bars = []

    def make_bar(**kwargs):
        bar = RecordingBar(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(runner, "tqdm", make_bar)
    updates = queue.Queue()
    for event in events:
        updates.put(event)
    updates.put({"event": "stop"})
    runner._progress_monitor(updates, total_jobs)
    return bars


def test_phase_changes_reset_counts_units_and_timing(monkeypatch):
    bars = monitor(monkeypatch, [
        {"event": "job_start", "tag": "pad"},
        {"event": "analysis_start", "id": "r", "desc": "raster test", "total": 3},
        {"event": "analysis_update", "id": "r", "increment": 3},
        {"event": "analysis_phase", "id": "r", "total": 16000,
         "unit": "tile", "phase": "rasterizing polygon tiles"},
        {"event": "analysis_update", "id": "r", "increment": 16000},
        {"event": "analysis_phase", "id": "r", "total": 1,
         "unit": "step", "phase": "stitching rasterized tiles"},
        {"event": "analysis_update", "id": "r", "increment": 1},
        {"event": "analysis_phase", "id": "r", "total": 69,
         "unit": "block", "phase": "processing raster blocks"},
        {"event": "analysis_update", "id": "r", "increment": 1},
        {"event": "analysis_update", "id": "r", "increment": 0,
         "phase": "processing features in block 2/69"},
    ])
    bar = bars[1]
    assert [event for event in bar.events if event[0] == "reset"] == [
        ("reset", 0, 16000, "tile"),
        ("reset", 0, 1, "step"),
        ("reset", 0, 69, "block"),
    ]
    assert bar.n == 1
    assert bar.total == 69
    assert bar.unit == "block"
    assert bar.postfix == "processing features in block 2/69"
    assert bars[0].postfix == "pad: raster test - processing features in block 2/69"
    assert bar.closed


def test_unknown_duration_phase_clears_previous_total(monkeypatch):
    bars = monitor(monkeypatch, [
        {"event": "analysis_start", "id": "r", "total": 100},
        {"event": "analysis_update", "id": "r", "increment": 100},
        {"event": "analysis_phase", "id": "r", "total": None,
         "unit": "step", "phase": "grouping feature statistics"},
        {"event": "analysis_close", "id": "r", "phase": "done"},
    ])
    assert bars[1].total is None
    assert bars[1].n == 0
    assert ("reset", 0, None, "step") in bars[1].events


@pytest.mark.parametrize("close_fields", [
    {"phase": "failed"},
    {"phase": "raster blocks failed: read error"},
    {"phase": "stopped", "status": "failed"},
    {"phase": "cancelled"},
    {"phase": "interrupted"},
    {"phase": "incomplete", "complete": False},
])
def test_unsuccessful_close_does_not_fill_bar(monkeypatch, close_fields):
    bars = monitor(monkeypatch, [
        {"event": "analysis_start", "id": "r", "total": 69, "unit": "block"},
        {"event": "analysis_update", "id": "r", "increment": 28},
        {"event": "analysis_close", "id": "r", **close_fields},
        {"event": "job_done", "tag": "pad", "status": "failed"},
    ])
    assert bars[1].n == 28
    assert bars[1].closed
    assert bars[0].n == 0
    assert bars[0].postfix == "pad failed"


def test_successful_legacy_vector_events_keep_completion_behavior(monkeypatch):
    bars = monitor(monkeypatch, [
        {"event": "analysis_start", "id": "vector", "total": 6},
        {"event": "analysis_update", "id": "vector", "increment": 2},
        {"event": "analysis_total", "id": "vector", "total": 8, "unit": "feature"},
        {"event": "analysis_set", "id": "vector", "value": 5},
        {"event": "analysis_close", "id": "vector", "phase": "done"},
        {"event": "job_done", "tag": "vector", "status": "done"},
    ])
    assert bars[1].n == bars[1].total == 8
    assert bars[1].unit == "feature"
    assert not any(event[0] == "reset" for event in bars[1].events)
    assert bars[0].n == 1


def test_phase_reset_is_scoped_to_one_raster(monkeypatch):
    bars = monitor(monkeypatch, [
        {"event": "analysis_start", "id": "a", "total": 10},
        {"event": "analysis_start", "id": "b", "total": 20},
        {"event": "analysis_update", "id": "a", "increment": 9},
        {"event": "analysis_update", "id": "b", "increment": 5},
        {"event": "analysis_phase", "id": "a", "total": 3,
         "unit": "block", "phase": "processing raster blocks"},
    ])
    assert (bars[1].n, bars[1].total) == (0, 3)
    assert (bars[2].n, bars[2].total) == (5, 20)


class InlineExecutor:
    """Run tile stubs immediately while preserving Future failure behavior."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as error:
            future.set_exception(error)
        return future


@pytest.fixture
def rasterization_stubs(monkeypatch):
    class BaseRaster:
        RasterXSize = 8
        RasterYSize = 4

        def GetGeoTransform(self):
            return (0, 1, 0, 4, 0, -1)

        def GetProjection(self):
            return "projection"

    monkeypatch.setattr(runner.gdal, "OpenEx", lambda *args: BaseRaster())
    monkeypatch.setattr(runner.geoprocessing, "new_raster_from_base", lambda *args: None)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(runner, "_rasterize_aggregate_fid_tile", lambda *args: None)
    monkeypatch.setattr(runner, "_stitch_raster_tiles", lambda *args: None)


def drain(updates):
    events = []
    while not updates.empty():
        events.append(updates.get_nowait())
    return events


def test_rasterization_reports_separate_tile_and_stitch_phases(tmp_path, rasterization_stubs):
    updates = queue.Queue()
    steps = runner._rasterize_aggregate_fids(
        tmp_path / "source.tif", tmp_path / "zones.gpkg", "zones",
        tmp_path / "ids.tif", -1, updates, "r", 500,
        tile_size=4, rasterize_worker_count=2,
    )
    events = drain(updates)
    phases = [event for event in events if event["event"] == "analysis_phase"]
    assert [(event["total"], event["unit"], event["phase"]) for event in phases] == [
        (2, "tile", "rasterizing polygon tiles"),
        (1, "step", "stitching rasterized tiles"),
    ]
    assert steps == 3  # Legacy cumulative return remains compatible.
    assert phases[1]["increment"] == 0
    completed = [event for event in events if event["event"] == "analysis_update"]
    assert [event["increment"] for event in completed] == [1, 1, 1]
    assert completed[-1]["phase"] == "stitched rasterized tiles"


def test_tile_failure_never_announces_stitching_or_completion(
    tmp_path, rasterization_stubs, monkeypatch,
):
    def fail(*args):
        raise RuntimeError("tile failed")

    monkeypatch.setattr(runner, "_rasterize_aggregate_fid_tile", fail)
    updates = queue.Queue()
    with pytest.raises(RuntimeError, match="tile failed"):
        runner._rasterize_aggregate_fids(
            tmp_path / "source.tif", tmp_path / "zones.gpkg", "zones",
            tmp_path / "ids.tif", -1, updates, "r", 500,
            tile_size=4, rasterize_worker_count=2,
        )
    events = drain(updates)
    assert len(events) == 1
    assert events[0]["phase"] == "rasterizing polygon tiles"
    assert not list(tmp_path.glob("ids_tiles_*"))
