"""Geometry-cache regressions using real GDAL data and independent expectations."""

from concurrent.futures import Future, ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np
from osgeo import gdal, ogr, osr
import pytest

import raster_cache
import runner


def _srs(epsg=6933):
    result = osr.SpatialReference()
    result.ImportFromEPSG(epsg)
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def _raster(path, values, *, nodata=-1, origin=(0, 4), epsg=6933, block=256):
    values = np.asarray(values, dtype=np.float32)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), values.shape[1], values.shape[0], 1, gdal.GDT_Float32,
        options=["TILED=YES", f"BLOCKXSIZE={block}", f"BLOCKYSIZE={block}"],
    )
    dataset.SetProjection(_srs(epsg).ExportToWkt())
    dataset.SetGeoTransform((origin[0], 1, 0, origin[1], 0, -1))
    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(values)
    band = None
    dataset = None
    return path


def _vector(path, records=None, *, layer_name="zones", append=False, driver="GPKG"):
    records = records or [("zone-a", 0, 2), ("zone-b", 2, 4)]
    dataset = (ogr.Open(str(path), 1) if append
               else ogr.GetDriverByName(driver).CreateDataSource(str(path)))
    layer = dataset.CreateLayer(layer_name, srs=_srs(), geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("zone_key", ogr.OFTString))
    for key, xmin, xmax in records:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("zone_key", key)
        feature.SetGeometry(ogr.CreateGeometryFromWkt(
            f"POLYGON (({xmin} 0, {xmax} 0, {xmax} 4, {xmin} 4, {xmin} 0))"
        ))
        assert layer.CreateFeature(feature) == 0
    feature = None
    layer = None
    dataset = None
    return path


class _InlineExecutor:
    """Run the real tile rasterizer without Windows process-spawn overhead."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


@pytest.fixture
def geometry_case(tmp_path, monkeypatch):
    vector = _vector(tmp_path / "zones.gpkg")
    values = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    source = _raster(tmp_path / "values.tif", values)
    calls = {"prepare": 0, "burn": 0}
    prepare = runner._prepare_aggregate_vector_for_rasterization
    burn = runner._rasterize_aggregate_fids

    def observed_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return prepare(*args, **kwargs)

    def observed_burn(*args, **kwargs):
        calls["burn"] += 1
        return burn(*args, **kwargs)

    monkeypatch.setattr(runner, "_prepare_aggregate_vector_for_rasterization", observed_prepare)
    monkeypatch.setattr(runner, "_rasterize_aggregate_fids", observed_burn)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", _InlineExecutor)
    invocation = 0

    def calculate(raster=source, zones=vector, **kwargs):
        nonlocal invocation
        invocation += 1
        options = dict(
            aggregate_layer_name="zones", operations=["sum"],
            working_dir=tmp_path / f"work-{invocation}",
            geometry_cache_dir=tmp_path / "geometry-cache",
            max_simplify_tolerance_meters=0, rasterize_worker_count=1,
            progress_queue=queue.Queue(), progress_id=f"call-{invocation}",
        )
        options.update(kwargs)
        return runner.fast_zonal_statistics((raster, 1), zones, ["zone_key"], **options)

    return calculate, source, vector, calls


@pytest.mark.parametrize("operations", [["sum"], None])
def test_cached_and_uncached_statistics_are_exactly_equal(geometry_case, operations):
    calculate, _, _, calls = geometry_case
    first = calculate(operations=operations)
    repeated = calculate(operations=operations)
    assert calls == {"prepare": 1, "burn": 1}
    uncached = calculate(operations=operations, reuse_geometry=False)
    np.testing.assert_equal(first, repeated)
    np.testing.assert_equal(first, uncached)
    assert first["zone-a"]["sum"] == 60
    assert first["zone-b"]["sum"] == 76
    assert calls == {"prepare": 2, "burn": 2}


def test_same_grid_services_share_geometry_but_not_nodata_or_values(geometry_case, tmp_path):
    calculate, _, _, calls = geometry_case
    original = calculate()
    values = np.full((4, 4), 10, dtype=np.float32)
    values[0, 0] = -9999
    values[0, 2] = np.nan
    second = _raster(tmp_path / "other-service.tif", values, nodata=-9999)
    actual = calculate(second, operations=["sum", "valid_count"])
    assert actual == {"zone-a": {"sum": 70, "valid_count": 7},
                      "zone-b": {"sum": 70, "valid_count": 7}}
    assert original["zone-a"]["sum"] != actual["zone-a"]["sum"]
    assert calls == {"prepare": 1, "burn": 1}


def test_service_pixel_changes_recompute_statistics_without_reburn(geometry_case):
    calculate, source, _, calls = geometry_case
    calculate()
    dataset = gdal.Open(str(source), gdal.GA_Update)
    dataset.GetRasterBand(1).WriteArray(np.full((4, 4), 3, dtype=np.float32))
    dataset = None
    assert calculate() == {"zone-a": {"sum": 24}, "zone-b": {"sum": 24}}
    assert calls == {"prepare": 1, "burn": 1}


@pytest.mark.parametrize("change", ["origin", "dimensions", "crs", "storage_blocks"])
def test_grid_changes_rebuild_only_the_required_artifacts(geometry_case, tmp_path, change):
    calculate, _, _, calls = geometry_case
    calculate()
    options = {"origin": {"origin": (0.25, 4)}, "dimensions": {},
               "crs": {"epsg": 3857}, "storage_blocks": {"block": 512}}[change]
    values = np.ones((4, 5 if change == "dimensions" else 4), dtype=np.float32)
    changed = _raster(tmp_path / "changed-grid.tif", values, **options)
    cached = calculate(changed)
    assert calls == {"prepare": 2 if change == "crs" else 1, "burn": 2}
    np.testing.assert_equal(cached, calculate(changed, reuse_geometry=False))


def test_simplification_tolerance_invalidates_both_artifacts(geometry_case):
    calculate, _, _, calls = geometry_case
    calculate()
    cached = calculate(max_simplify_tolerance_meters=0.25)
    assert calls == {"prepare": 2, "burn": 2}
    np.testing.assert_equal(cached, calculate(
        max_simplify_tolerance_meters=0.25, reuse_geometry=False))


def test_processing_version_invalidates_both_artifacts(geometry_case, monkeypatch):
    calculate, _, _, calls = geometry_case
    first = calculate()
    monkeypatch.setattr(runner, "_GEOMETRY_CACHE_VERSION", "changed-for-test")
    np.testing.assert_equal(first, calculate())
    assert calls == {"prepare": 2, "burn": 2}


def test_corrupted_id_raster_is_rebuilt_without_repreparing_geometry(geometry_case, tmp_path):
    calculate, _, _, calls = geometry_case
    expected = calculate()
    cached_ids, = (tmp_path / "geometry-cache").rglob("agg_fid.tif")
    cached_ids.write_bytes(b"interrupted TIFF write")
    np.testing.assert_equal(expected, calculate())
    assert calls == {"prepare": 1, "burn": 2}


def test_source_changed_during_preparation_is_not_published(geometry_case, monkeypatch, tmp_path):
    calculate, _, vector, calls = geometry_case
    prepare = runner._prepare_aggregate_vector_for_rasterization

    def changing_source(*args, **kwargs):
        result = prepare(*args, **kwargs)
        contents = vector.read_bytes()
        assert b"zone-a" in contents
        vector.write_bytes(contents.replace(b"zone-a", b"zone-c"))
        return result

    monkeypatch.setattr(runner, "_prepare_aggregate_vector_for_rasterization", changing_source)
    with pytest.raises(RuntimeError, match="changed during"):
        calculate()
    assert calls == {"prepare": 1, "burn": 0}
    assert not list((tmp_path / "geometry-cache").rglob("manifest.json"))
    monkeypatch.setattr(runner, "_prepare_aggregate_vector_for_rasterization", prepare)
    assert calculate() == {"zone-c": {"sum": 60}, "zone-b": {"sum": 76}}


@pytest.mark.parametrize("when", ["rasterization", "statistics_on_cache_hit"])
def test_source_changed_after_preparation_does_not_return_stale_statistics(
        geometry_case, monkeypatch, when):
    calculate, _, vector, _ = geometry_case

    def change_source():
        contents = vector.read_bytes()
        vector.write_bytes(contents.replace(b"zone-a", b"zone-c"))

    if when == "rasterization":
        burn = runner._rasterize_aggregate_fids

        def changing_burn(*args, **kwargs):
            result = burn(*args, **kwargs)
            change_source()
            return result

        monkeypatch.setattr(runner, "_rasterize_aggregate_fids", changing_burn)
    else:
        calculate()
        group_values = runner._group_raster_values

        def changing_statistics(*args, **kwargs):
            yield from group_values(*args, **kwargs)
            change_source()

        monkeypatch.setattr(runner, "_group_raster_values", changing_statistics)
    with pytest.raises(RuntimeError, match="changed"):
        calculate()


def test_actual_layer_is_part_of_key_and_default_layer_can_reuse_it(geometry_case):
    calculate, _, vector, calls = geometry_case
    _vector(vector, [("other-a", 0, 1)], layer_name="other", append=True)
    calculate(aggregate_layer_name=None)
    calculate(aggregate_layer_name="zones")
    assert calls == {"prepare": 1, "burn": 1}
    actual = calculate(aggregate_layer_name="other")
    assert actual == {"other-a": {"sum": 28}}
    assert calls == {"prepare": 2, "burn": 2}


@pytest.mark.parametrize("change", ["attribute", "geometry"])
def test_same_size_source_changes_with_restored_mtime_invalidate_cache(geometry_case, change):
    calculate, _, vector, calls = geometry_case
    before = calculate()
    stat = vector.stat()
    dataset = ogr.Open(str(vector), 1)
    layer = dataset.GetLayerByName("zones")
    feature = layer.GetFeature(1)
    if change == "attribute":
        feature.SetField("zone_key", "zone-c")
    else:
        feature.SetGeometry(ogr.CreateGeometryFromWkt(
            "POLYGON ((0 0, 1 0, 1 4, 0 4, 0 0))"))
    assert layer.SetFeature(feature) == 0
    feature = None
    layer = None
    dataset = None
    assert vector.stat().st_size == stat.st_size
    os.utime(vector, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = calculate()
    assert before != after
    assert calls == {"prepare": 2, "burn": 2}
    np.testing.assert_equal(after, calculate(reuse_geometry=False))


def test_overlap_burn_order_is_preserved_and_source_reordering_invalidates(geometry_case):
    calculate, _, vector, calls = geometry_case
    vector.unlink()
    _vector(vector, [("zone-a", 0, 4), ("zone-b", 0, 2)])
    first = calculate()
    assert first == {"zone-a": {"sum": 76}, "zone-b": {"sum": 60}}
    np.testing.assert_equal(first, calculate())
    vector.unlink()
    _vector(vector, [("zone-b", 0, 2), ("zone-a", 0, 4)])
    second = calculate()
    assert second == {"zone-a": {"sum": 136}, "zone-b": {"sum": 0}}
    assert calls == {"prepare": 2, "burn": 2}
    np.testing.assert_equal(second, calculate(reuse_geometry=False))


def test_fingerprint_uses_shapefile_sidecar_content_not_size_or_mtime(tmp_path):
    source = _vector(tmp_path / "zones.shp", driver="ESRI Shapefile")
    before = raster_cache.fingerprint_vector(source)
    attributes = source.with_suffix(".dbf")
    stat = attributes.stat()
    contents = attributes.read_bytes()
    assert b"zone-a" in contents
    attributes.write_bytes(contents.replace(b"zone-a", b"zone-c"))
    os.utime(attributes, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert attributes.stat().st_size == stat.st_size
    assert raster_cache.fingerprint_vector(source) != before


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_fingerprint_rejects_database_with_active_journal(tmp_path, suffix):
    source = _vector(tmp_path / "zones.gpkg")
    Path(str(source) + suffix).write_bytes(b"uncommitted database changes")
    with pytest.raises(ValueError, match="active journal"):
        raster_cache.fingerprint_vector(source)


def _text_artifact(root, build, validate=None, key=None):
    return raster_cache.artifact(
        root, "test", key or {"grid": [0, 1, 4]}, "value.txt", build,
        validate or (lambda path: path.read_text(encoding="utf-8").startswith("valid")),
    )


def test_artifact_key_is_canonical_and_different_keys_do_not_alias(tmp_path):
    builds = []

    def build(path):
        builds.append(path)
        path.write_text("valid-contents", encoding="utf-8")

    first, hit = _text_artifact(tmp_path, build, key={"grid": [0, 1], "version": 1})
    assert not hit
    assert _text_artifact(tmp_path, build, key={"version": 1, "grid": [0, 1]}) == (first, True)
    changed, hit = _text_artifact(tmp_path, build, key={"version": 2, "grid": [0, 1]})
    assert not hit and changed != first
    assert len(builds) == 2


@pytest.mark.parametrize("damage", ["missing_manifest", "invalid_manifest", "same_size_payload"])
def test_incomplete_or_corrupted_cache_is_rebuilt(tmp_path, damage):
    builds = []

    def build(path):
        builds.append(path)
        path.write_text("valid-contents", encoding="utf-8")

    first, hit = _text_artifact(tmp_path, build)
    assert not hit
    manifest = first.parent / "manifest.json"
    if damage == "missing_manifest":
        manifest.unlink()
    elif damage == "invalid_manifest":
        manifest.write_text("{invalid", encoding="utf-8")
    else:
        stat = first.stat()
        first.write_text("wrong-contents", encoding="utf-8")
        assert first.stat().st_size == stat.st_size
        os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second, hit = _text_artifact(tmp_path, build)
    assert not hit
    assert first == second
    assert len(builds) == 2
    assert second.read_text(encoding="utf-8") == "valid-contents"
    assert _text_artifact(tmp_path, build) == (second, True)
    assert len(builds) == 2


def test_cached_artifact_validation_failure_triggers_rebuild(tmp_path):
    builds = []
    reject_first = False

    def build(path):
        builds.append(path)
        path.write_text(f"valid-{len(builds)}", encoding="utf-8")

    def validate(path):
        if reject_first and path.read_text(encoding="utf-8") == "valid-1":
            raise ValueError("invalid cached format")
        return True

    _text_artifact(tmp_path, build, validate)
    reject_first = True
    path, hit = _text_artifact(tmp_path, build, validate)
    assert not hit
    assert len(builds) == 2
    assert path.read_text(encoding="utf-8") == "valid-2"


@pytest.mark.parametrize("failure", ["build", "validate"])
def test_failed_publication_never_becomes_a_cache_hit(tmp_path, failure):
    def broken_build(path):
        path.write_text("valid-partial", encoding="utf-8")
        if failure == "build":
            raise RuntimeError("interrupted build")

    with pytest.raises((RuntimeError, ValueError)):
        _text_artifact(tmp_path, broken_build, lambda path: False)
    assert not list(tmp_path.rglob("manifest.json"))
    path, hit = _text_artifact(
        tmp_path, lambda path: path.write_text("valid-complete", encoding="utf-8"))
    assert not hit
    assert path.read_text(encoding="utf-8") == "valid-complete"


@pytest.mark.parametrize("failure", ["build", "validate"])
def test_cleanup_failure_preserves_primary_error_and_retry_rebuilds(
        tmp_path, monkeypatch, caplog, failure):
    primary = (RuntimeError("primary build failure") if failure == "build"
               else ValueError("primary validation failure"))
    builds = []
    failed_cleanups = []
    rmtree = raster_cache.shutil.rmtree

    def broken_build(path):
        builds.append(path)
        path.write_text("valid-partial", encoding="utf-8")
        if failure == "build":
            raise primary

    def broken_validate(path):
        raise primary

    def locked_staging(path, *args, **kwargs):
        if ".staging-" in Path(path).name:
            failed_cleanups.append(Path(path))
            error = PermissionError(13, "[WinError 32] File is being used", str(path))
            error.winerror = 32
            raise error
        return rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(raster_cache.shutil, "rmtree", locked_staging)
        with pytest.raises(type(primary), match="primary .* failure") as raised:
            _text_artifact(tmp_path, broken_build, broken_validate)

    assert raised.value is primary
    assert len(failed_cleanups) == 1
    failed_stage = failed_cleanups[0]
    assert failed_stage == builds[0].parent
    assert builds[0].read_text(encoding="utf-8") == "valid-partial"
    assert not list(tmp_path.rglob("manifest.json"))
    assert str(failed_stage) in caplog.text
    assert "WinError 32" in caplog.text
    if hasattr(primary, "add_note"):
        assert any(str(failed_stage) in note and "WinError 32" in note
                   for note in primary.__notes__)

    def complete_build(path):
        builds.append(path)
        path.write_text("valid-complete", encoding="utf-8")

    path, hit = _text_artifact(tmp_path, complete_build)
    assert not hit
    assert len(builds) == 2
    assert builds[1].parent != failed_stage
    assert path.read_text(encoding="utf-8") == "valid-complete"
    assert _text_artifact(tmp_path, complete_build) == (path, True)
    assert len(builds) == 2
    assert failed_stage.exists()  # An unpublished leftover is never reused.
    assert len(list(tmp_path.rglob("manifest.json"))) == 1


def test_cleanup_failure_after_successful_publication_is_not_swallowed(
        tmp_path, monkeypatch):
    builds = []
    failed_cleanups = []
    cleanup_error = PermissionError(13, "[WinError 32] Old cache is being used")
    cleanup_error.winerror = 32
    rmtree = raster_cache.shutil.rmtree

    def build(path):
        builds.append(path)
        path.write_text(f"valid-{len(builds)}", encoding="utf-8")

    path, hit = _text_artifact(tmp_path, build)
    assert not hit
    path.write_text("corrupted", encoding="utf-8")

    def locked_replaced_entry(target, *args, **kwargs):
        if ".replaced-" in Path(target).name:
            failed_cleanups.append(Path(target))
            raise cleanup_error
        return rmtree(target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(raster_cache.shutil, "rmtree", locked_replaced_entry)
        with pytest.raises(PermissionError, match="WinError 32") as raised:
            _text_artifact(tmp_path, build)

    assert raised.value is cleanup_error
    assert len(failed_cleanups) == 1
    assert failed_cleanups[0].exists()
    assert len(builds) == 2
    assert path.read_text(encoding="utf-8") == "valid-2"
    assert (path.parent / "manifest.json").is_file()
    assert _text_artifact(tmp_path, build) == (path, True)
    assert len(builds) == 2


def test_same_key_threads_publish_once(tmp_path):
    start = threading.Barrier(4)
    builds = []

    def build(path):
        builds.append(path)
        time.sleep(0.05)
        path.write_text("valid-complete", encoding="utf-8")

    def calculate():
        start.wait(timeout=10)
        return _text_artifact(tmp_path, build)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: calculate(), range(4)))
    assert len(builds) == 1
    assert sum(not hit for _, hit in results) == 1
    assert len({path for path, _ in results}) == 1


def _process_artifact_worker(root, start, results):
    try:
        if not start.wait(20):
            raise TimeoutError("cache test start signal")

        def build(path):
            time.sleep(0.05)
            path.write_text("valid-complete", encoding="utf-8")

        path, hit = _text_artifact(Path(root), build)
        results.put((str(path), hit, None))
    except BaseException as error:
        results.put((None, None, repr(error)))


def _process_zonal_worker(raster, vector, working, cache, start, results):
    try:
        if not start.wait(30):
            raise TimeoutError("zonal cache test start signal")
        actual = runner.fast_zonal_statistics(
            (raster, 1), vector, ["zone_key"], aggregate_layer_name="zones",
            operations=["sum"], working_dir=working, geometry_cache_dir=cache,
            max_simplify_tolerance_meters=0, rasterize_worker_count=1,
            progress_queue=queue.Queue(), progress_id=Path(raster).stem,
        )
        results.put((Path(raster).name, actual, None))
    except BaseException as error:
        results.put((Path(raster).name, None, repr(error)))


def test_spawned_zonal_workers_share_geometry_and_keep_service_values_separate(tmp_path):
    """Exercise real worker startup and tile rasterization, without executor stubs."""
    vector = _vector(tmp_path / "zones.gpkg")
    first = _raster(tmp_path / "first.tif", np.arange(1, 17).reshape(4, 4))
    second = _raster(tmp_path / "second.tif", np.full((4, 4), 3))
    cache = tmp_path / "shared-geometry"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_process_zonal_worker, args=(
            str(raster), str(vector), str(tmp_path / f"worker-{index}"),
            str(cache), start, results,
        ))
        for index, raster in enumerate((first, second))
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=90) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert all(error is None for _, _, error in outcomes), outcomes
        actual = {name: values for name, values, _ in outcomes}
        assert actual == {
            "first.tif": {"zone-a": {"sum": 60}, "zone-b": {"sum": 76}},
            "second.tif": {"zone-a": {"sum": 24}, "zone-b": {"sum": 24}},
        }
        assert len(list((cache / "polygons").glob("*/manifest.json"))) == 1
        assert len(list((cache / "ids").glob("*/manifest.json"))) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=15)
        results.close()


def _crashing_artifact_worker(root, building):
    def build(path):
        path.write_text("valid-incomplete", encoding="utf-8")
        building.set()
        # Simulate abrupt worker loss: no context-manager cleanup executes.
        os._exit(23)

    _text_artifact(Path(root), build)


def test_crashed_worker_releases_lock_and_partial_artifact_is_not_reused(tmp_path):
    context = multiprocessing.get_context("spawn")
    building = context.Event()
    results = context.Queue()
    crashed = context.Process(target=_crashing_artifact_worker, args=(str(tmp_path), building))
    successor = context.Process(target=_process_artifact_worker,
                                args=(str(tmp_path), building, results))
    try:
        crashed.start()
        assert building.wait(timeout=45), "worker did not acquire the cache lock"
        crashed.join(timeout=10)
        assert crashed.exitcode == 23
        successor.start()
        path, hit, error = results.get(timeout=45)
        successor.join(timeout=10)
        assert successor.exitcode == 0
        assert error is None
        assert not hit
        assert Path(path).read_text(encoding="utf-8") == "valid-complete"
    finally:
        for process in (crashed, successor):
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()


def test_same_key_spawned_processes_publish_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [context.Process(target=_process_artifact_worker,
                                 args=(str(tmp_path), start, results)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        start.set()
        outcomes = [results.get(timeout=45) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert all(error is None for _, _, error in outcomes), outcomes
        assert sum(not hit for _, hit, _ in outcomes) == 1
        assert len({path for path, _, _ in outcomes}) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()
