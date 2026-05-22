import logging
import multiprocessing
import queue
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal, osr
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

import runner


def _projection_wkt(epsg):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs.ExportToWkt()


def _drain_queue(progress_queue):
    events = []
    while True:
        try:
            events.append(progress_queue.get_nowait())
        except queue.Empty:
            return events


def _write_raster(path, array, *, epsg=6933, nodata=-1, origin=(0, 4), pixel_size=1):
    path = Path(path)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        array.shape[1],
        array.shape[0],
        1,
        gdal.GDT_Float32,
        options=["TILED=YES"],
    )
    dataset.SetGeoTransform(
        (origin[0], pixel_size, 0.0, origin[1], 0.0, -pixel_size)
    )
    dataset.SetProjection(_projection_wkt(epsg))
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(array.astype(np.float32))
    band.FlushCache()
    dataset = None
    return path


def _write_vector(path, geodataframe, *, layer="data"):
    path = Path(path)
    if path.exists():
        path.unlink()
    geodataframe.to_file(path, layer=layer, driver="GPKG", index=False)
    return path


@pytest.fixture
def projected_zone_vector(tmp_path):
    geodataframe = gpd.GeoDataFrame(
        {
            "STATE": ["A", "A"],
            "COUNTY": ["001", "002"],
            "value": [10.0, 20.0],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]),
                Polygon([(2, 0), (4, 0), (4, 4), (2, 4)]),
            ],
        },
        crs="EPSG:6933",
    )
    return _write_vector(tmp_path / "zones.gpkg", geodataframe, layer="zones")


@pytest.fixture
def projected_raster(tmp_path):
    array = np.array(
        [
            [1, 2, 3, 4],
            [5, -1, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.float32,
    )
    return _write_raster(tmp_path / "values.tif", array, epsg=6933, nodata=-1)


@pytest.fixture
def multiprocessing_queue():
    manager = multiprocessing.Manager()
    try:
        yield manager.Queue()
    finally:
        manager.shutdown()


def test_safe_path_stem_sanitizes_and_truncates():
    assert runner._safe_path_stem("weird path/a b@c!.tif") == "a_b_c_"
    assert runner._safe_path_stem("!!!.tif") == "___"
    assert runner._safe_path_stem("abcdef.tif", max_length=3) == "abc"


def test_promote_polygon_to_multipolygon():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    promoted = runner._promote_polygon_to_multipolygon(polygon)
    assert isinstance(promoted, MultiPolygon)
    assert len(promoted.geoms) == 1
    empty_polygon = Polygon()
    assert runner._promote_polygon_to_multipolygon(empty_polygon) is empty_polygon


def test_gdal_error_handler_routes_messages(caplog):
    with caplog.at_level(logging.DEBUG, logger=runner.__name__):
        runner._gdal_error_handler(gdal.CE_Warning, 1, "warning text")
        runner._gdal_error_handler(gdal.CE_Failure, 2, "failure text")
    assert "GDAL warning 1: warning text" in caplog.text
    assert "GDAL error 2: failure text" in caplog.text


def test_configure_gdal_cache_only_lowers_large_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.gdal, "GetCacheMax", lambda: 1024)
    monkeypatch.setattr(runner.gdal, "SetCacheMax", calls.append)
    runner._configure_gdal_cache(max_cache_bytes=512)
    assert calls == [512]

    calls.clear()
    monkeypatch.setattr(runner.gdal, "GetCacheMax", lambda: 256)
    runner._configure_gdal_cache(max_cache_bytes=512)
    assert calls == []


def test_progress_callback_emits_integer_increments():
    progress_queue = queue.Queue()
    callback = runner._make_progress_callback(
        progress_queue, "analysis-id", "rasterizing", start_value=2
    )
    assert callback(0.0, None, None) == 1
    assert callback(0.125, None, None) == 1
    assert callback(1.0, None, None) == 1

    events = [progress_queue.get_nowait(), progress_queue.get_nowait()]
    assert events == [
        {
            "event": "analysis_set",
            "id": "analysis-id",
            "value": 14,
            "phase": "rasterizing 12%",
        },
        {
            "event": "analysis_set",
            "id": "analysis-id",
            "value": 102,
            "phase": "rasterizing 100%",
        },
    ]


def test_record_and_log_area_hectare_assumptions(caplog):
    runner._AREA_HECTARE_ASSUMPTIONS.clear()
    runner._record_area_hectare_assumption("assumption b")
    runner._record_area_hectare_assumption("assumption a")
    with caplog.at_level(logging.WARNING, logger=runner.__name__):
        runner._log_area_hectare_assumptions()
    assert "Area hectare assumption: assumption a" in caplog.text
    assert "Area hectare assumption: assumption b" in caplog.text


def test_raster_pixel_area_ha_projected_and_geographic(tmp_path):
    projected_srs = osr.SpatialReference()
    projected_srs.ImportFromEPSG(6933)
    projected_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    projected_info = {
        "pixel_size": (30.0, -30.0),
        "bounding_box": [0.0, 0.0, 30.0, 30.0],
        "projection_wkt": projected_srs.ExportToWkt(),
    }
    assert runner._raster_pixel_area_ha(
        "projected.tif", projected_info, projected_srs
    ) == pytest.approx(0.09)

    geographic_srs = osr.SpatialReference()
    geographic_srs.ImportFromEPSG(4326)
    geographic_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geographic_info = {
        "pixel_size": (1.0, -1.0),
        "bounding_box": [0.0, 0.0, 1.0, 1.0],
        "projection_wkt": geographic_srs.ExportToWkt(),
    }
    runner._AREA_HECTARE_ASSUMPTIONS.clear()
    area_ha = runner._raster_pixel_area_ha(
        tmp_path / "geographic.tif", geographic_info, geographic_srs
    )
    assert area_ha > 1_000_000
    assert runner._AREA_HECTARE_ASSUMPTIONS


def test_bounds_to_wgs84_and_linear_units():
    bounds = runner._bounds_to_wgs84((0, 0, 1000, 1000), CRS.from_epsg(6933))
    assert bounds[0] < bounds[2]
    assert bounds[1] < bounds[3]
    assert runner._linear_units_to_meters(CRS.from_epsg(6933)) == pytest.approx(1.0)


def test_select_measure_crs_explicit_and_auto(projected_zone_vector, tmp_path):
    agg_gdf = gpd.read_file(projected_zone_vector, layer="zones")
    measure_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0.5, 0.5)]},
        crs="EPSG:6933",
    )
    assert runner._select_measure_crs(
        agg_gdf, measure_gdf, "EPSG:6933", "intersect_area_ha", "tag"
    ) == CRS.from_epsg(6933)

    runner._MEASURE_CRS_ASSUMPTIONS.clear()
    assert runner._select_measure_crs(
        agg_gdf, measure_gdf, "auto", "intersect_area_ha", "tag"
    ) == CRS.from_epsg(6933)
    assert runner._MEASURE_CRS_ASSUMPTIONS

    with pytest.raises(ValueError, match="must be projected"):
        runner._select_measure_crs(
            agg_gdf, measure_gdf, "EPSG:4326", "intersect_area_ha", "tag"
        )


@pytest.mark.parametrize(
    ("operation", "geometry"),
    [
        ("intersect_area_ha", LineString([(0, 0), (1, 1)])),
        ("intersect_length_km", Point(0, 0)),
        ("intersect_count", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
    ],
)
def test_validate_measure_geometry_rejects_incompatible_types(operation, geometry):
    measure_gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:6933")
    with pytest.raises(ValueError, match=operation):
        runner._validate_measure_geometry(measure_gdf, operation, "measure.gpkg")


def test_parse_and_validate_config_resolves_paths(
    tmp_path, projected_zone_vector, projected_raster
):
    cfg_path = tmp_path / "project.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "[project]",
                "name = project",
                "global_work_dir = work",
                "log_level = DEBUG",
                "",
                "[job:raster_job]",
                f"agg_vector = {projected_zone_vector.name}",
                "agg_layer = zones",
                "agg_field = STATE, COUNTY",
                "operations = sum, mean, total_count, valid_count, proportion_valid_nonzero",
                f"base_raster_pattern = {projected_raster.name}",
                "output_csv = output/results.csv",
                "output_gpkg = output/results.gpkg",
            ]
        )
    )
    parsed = runner.parse_and_validate_config(cfg_path)
    job = parsed["job_list"][0]
    assert parsed["project"]["global_work_dir"] == tmp_path / "work"
    assert job["agg_vector"] == projected_zone_vector
    assert job["agg_field"] == ["STATE", "COUNTY"]
    assert "proportion_valid_nonzero" in job["operations"]
    assert job["base_raster_path_list"] == [projected_raster]
    assert job["output_csv"] == tmp_path / "output" / "results.csv"
    assert job["output_gpkg"] == tmp_path / "output" / "results.gpkg"
    assert job["workdir"] == tmp_path / "work" / "raster_job"


def test_parse_and_validate_config_requires_layer_for_multilayer_vector(tmp_path):
    first = gpd.GeoDataFrame(
        {"ZONE": ["a"], "geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    second = gpd.GeoDataFrame(
        {"ZONE": ["b"], "geometry": [Point(1, 1)]},
        crs="EPSG:4326",
    )
    vector_path = tmp_path / "multi.gpkg"
    first.to_file(vector_path, layer="first", driver="GPKG", index=False)
    second.to_file(vector_path, layer="second", driver="GPKG", index=False)
    cfg_path = tmp_path / "project.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "[project]",
                "name = project",
                "global_work_dir = work",
                "",
                "[job:bad]",
                "agg_vector = multi.gpkg",
                "agg_field = ZONE",
                "operations = sum",
                "base_measure_vector = multi.gpkg",
                "base_measure_layer = first",
                "output_csv = output.csv",
            ]
        )
    )
    with pytest.raises(ValueError, match="agg_layer is required"):
        runner.parse_and_validate_config(cfg_path)


def test_parse_and_validate_config_rejects_bad_measure_operation(
    tmp_path, projected_zone_vector
):
    cfg_path = tmp_path / "project.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "[project]",
                "name = project",
                "global_work_dir = work",
                "",
                "[job:bad]",
                f"agg_vector = {projected_zone_vector.name}",
                "agg_layer = zones",
                "agg_field = STATE",
                "operations = intersect_area_ha, sum",
                f"base_measure_vector = {projected_zone_vector.name}",
                "base_measure_layer = zones",
                "output_csv = output.csv",
            ]
        )
    )
    with pytest.raises(ValueError, match="must define exactly one operation"):
        runner.parse_and_validate_config(cfg_path)


def test_parse_and_validate_config_rejects_raster_only_operation_without_raster(
    tmp_path, projected_zone_vector
):
    cfg_path = tmp_path / "project.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "[project]",
                "name = project",
                "global_work_dir = work",
                "",
                "[job:bad]",
                f"agg_vector = {projected_zone_vector.name}",
                "agg_layer = zones",
                "agg_field = STATE",
                "operations = proportion_valid_nonzero",
                f"base_vector_pattern = {projected_zone_vector.name}[value]",
                "output_csv = output.csv",
            ]
        )
    )
    with pytest.raises(ValueError, match="raster-only operations"):
        runner.parse_and_validate_config(cfg_path)


def test_prepare_and_rasterize_aggregate_fids(
    tmp_path, projected_zone_vector, projected_raster
):
    prepared_vector = tmp_path / "prepared.gpkg"
    runner._prepare_aggregate_vector_for_rasterization(
        projected_zone_vector,
        "zones",
        prepared_vector,
        _projection_wkt(6933),
        0.0,
        False,
    )
    prepared_gdf = gpd.read_file(prepared_vector, layer="zones")
    assert "original_fid" in prepared_gdf.columns

    fid_raster = tmp_path / "fid.tif"
    progress_queue = queue.Queue()
    runner._rasterize_aggregate_fids(
        projected_raster,
        prepared_vector,
        "zones",
        fid_raster,
        -1,
        progress_queue,
        "id",
        0,
        tile_size=2,
        rasterize_worker_count=2,
    )
    dataset = gdal.OpenEx(str(fid_raster), gdal.OF_RASTER)
    array = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    assert set(np.unique(array)) == {1, 2}
    assert np.all(array[:, :2] == 1)
    assert np.all(array[:, 2:] == 2)

    events = _drain_queue(progress_queue)
    stitch_events = [
        event
        for event in events
        if event.get("phase") in {
            "stitching rasterized tiles",
            "stitched rasterized tiles",
        }
    ]
    assert [
        (event["phase"], event["increment"]) for event in stitch_events
    ] == [
        ("stitching rasterized tiles", 0),
        ("stitched rasterized tiles", 1),
    ]


def test_rasterize_aggregate_fids_can_spawn_from_job_process(
    tmp_path, projected_zone_vector, projected_raster, multiprocessing_queue
):
    prepared_vector = tmp_path / "prepared_nested.gpkg"
    runner._prepare_aggregate_vector_for_rasterization(
        projected_zone_vector,
        "zones",
        prepared_vector,
        _projection_wkt(6933),
        0.0,
        False,
    )
    fid_raster = tmp_path / "fid_nested.tif"
    with runner.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            runner._rasterize_aggregate_fids,
            projected_raster,
            prepared_vector,
            "zones",
            fid_raster,
            -1,
            multiprocessing_queue,
            "id",
            0,
            2,
            2,
        )
        future.result()

    dataset = gdal.OpenEx(str(fid_raster), gdal.OF_RASTER)
    array = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    assert np.all(array[:, :2] == 1)
    assert np.all(array[:, 2:] == 2)


def test_iter_raster_tiles_and_tile_bounds():
    tiles = list(runner._iter_raster_tiles(5, 4, 2))
    assert tiles == [
        {"xoff": 0, "yoff": 0, "win_xsize": 2, "win_ysize": 2},
        {"xoff": 2, "yoff": 0, "win_xsize": 2, "win_ysize": 2},
        {"xoff": 4, "yoff": 0, "win_xsize": 1, "win_ysize": 2},
        {"xoff": 0, "yoff": 2, "win_xsize": 2, "win_ysize": 2},
        {"xoff": 2, "yoff": 2, "win_xsize": 2, "win_ysize": 2},
        {"xoff": 4, "yoff": 2, "win_xsize": 1, "win_ysize": 2},
    ]
    assert runner._tile_bounds((0, 1, 0, 4, 0, -1), tiles[1]) == (2, 2, 4, 4)


def test_fast_zonal_statistics_computes_grouped_raster_stats(
    tmp_path, projected_zone_vector, projected_raster, multiprocessing_queue
):
    stats = runner.fast_zonal_statistics(
        (projected_raster, 1),
        projected_zone_vector,
        ["STATE", "COUNTY"],
        aggregate_layer_name="zones",
        working_dir=tmp_path / "work",
        clean_working_dir=True,
        percentile_list=[50],
        calculate_area_ha=True,
        progress_queue=multiprocessing_queue,
        progress_id="raster:test",
    )

    left = stats[("A", "001")]
    right = stats[("A", "002")]
    assert left["total_count"] == 8
    assert left["valid_count"] == 7
    assert left["sum"] == pytest.approx(54.0)
    assert left["mean"] == pytest.approx(54.0 / 7.0)
    assert left["min"] == pytest.approx(1.0)
    assert left["max"] == pytest.approx(14.0)
    assert left["p50"] == pytest.approx(9.0)
    assert left["area_ha_total"] == pytest.approx(0.0008)
    assert left["area_ha_valid"] == pytest.approx(0.0007)
    assert left["proportion_valid_nonzero"] == pytest.approx(7 / 8)
    assert right["total_count"] == 8
    assert right["valid_count"] == 8
    assert right["sum"] == pytest.approx(76.0)
    assert right["proportion_valid_nonzero"] == pytest.approx(1.0)


def test_run_vector_stats_job_writes_nearest_attribute_stats(
    tmp_path, projected_zone_vector
):
    base_gdf = gpd.GeoDataFrame(
        {
            "score": [1.0, 3.0, 10.0],
            "geometry": [Point(0.5, 0.5), Point(1.5, 1.5), Point(3.0, 1.0)],
        },
        crs="EPSG:6933",
    )
    base_path = _write_vector(tmp_path / "samples.gpkg", base_gdf, layer="samples")
    output_csv = tmp_path / "vector_stats.csv"

    runner.run_vector_stats_job(
        base_vector_path_list=[base_path],
        base_vector_fields=["score"],
        agg_vector=projected_zone_vector,
        agg_layer="zones",
        agg_field=["STATE", "COUNTY"],
        operations=["total_count", "valid_count", "sum", "mean", "min", "max", "p50"],
        output_csv=output_csv,
        workdir=tmp_path / "work",
        tag="vector",
        job_type="vector",
        progress_queue=queue.Queue(),
    )
    result = pd.read_csv(output_csv)
    left = result[result["COUNTY"].astype(str).str.zfill(3) == "001"].iloc[0]
    right = result[result["COUNTY"].astype(str).str.zfill(3) == "002"].iloc[0]
    assert left["total_count_samples"] == 2
    assert left["valid_count_score_samples"] == 2
    assert left["sum_score_samples"] == pytest.approx(4.0)
    assert left["mean_score_samples"] == pytest.approx(2.0)
    assert left["p50_score_samples"] == pytest.approx(2.0)
    assert right["total_count_samples"] == 1
    assert right["sum_score_samples"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("operation", "measure_geometries", "expected_column", "expected_values"),
    [
        (
            "intersect_area_ha",
            [Polygon([(0, 0), (1, 0), (1, 4), (0, 4)])],
            "intersect_area_ha_measure",
            [0.0004, 0.0],
        ),
        (
            "intersect_length_km",
            [LineString([(0, 1), (4, 1)])],
            "intersect_length_km_measure",
            [0.002, 0.002],
        ),
        (
            "intersect_count",
            [Point(0.5, 0.5), Point(2.5, 0.5), Point(3.5, 0.5)],
            "intersect_count_measure",
            [1, 2],
        ),
    ],
)
def test_run_vector_measure_job(
    tmp_path,
    projected_zone_vector,
    operation,
    measure_geometries,
    expected_column,
    expected_values,
):
    measure_gdf = gpd.GeoDataFrame(
        {"geometry": measure_geometries},
        crs="EPSG:6933",
    )
    measure_path = _write_vector(tmp_path / "measure.gpkg", measure_gdf, layer="measure")
    result = runner.run_vector_measure_job(
        agg_vector=projected_zone_vector,
        agg_layer="zones",
        agg_fields=["STATE", "COUNTY"],
        base_measure_vector=measure_path,
        base_measure_layer="measure",
        operation=operation,
        measure_crs="EPSG:6933",
        tag="measure",
        progress_queue=queue.Queue(),
    )
    assert list(result[expected_column]) == pytest.approx(expected_values)


def test_write_zonal_outputs_writes_csv_and_gpkg(tmp_path, projected_zone_vector):
    result_table = pd.DataFrame(
        {
            "STATE": ["A", "A"],
            "COUNTY": ["001", "002"],
            "sum_values": [1.5, 2.5],
        }
    )
    output_csv = tmp_path / "out" / "stats.csv"
    output_gpkg = tmp_path / "out" / "stats.gpkg"
    runner._write_zonal_outputs(
        result_table,
        projected_zone_vector,
        "zones",
        ["STATE", "COUNTY"],
        output_csv,
        output_gpkg,
    )
    csv_result = pd.read_csv(output_csv)
    gpkg_result = gpd.read_file(output_gpkg, layer="zones")
    assert list(csv_result["sum_values"]) == [1.5, 2.5]
    assert list(gpkg_result["sum_values"]) == [1.5, 2.5]


def test_run_zonal_stats_job_raster_integration(
    tmp_path, projected_zone_vector, projected_raster, multiprocessing_queue
):
    task_graph = runner.taskgraph.TaskGraph(tmp_path / "taskgraph", 1, None)
    output_csv = tmp_path / "summary.csv"
    try:
        runner.run_zonal_stats_job(
            base_raster_path_list=[projected_raster],
            base_vector_path_list=[],
            base_vector_fields=[],
            base_measure_vector=None,
            base_measure_layer=None,
            measure_crs="auto",
            agg_vector=projected_zone_vector,
            agg_layer="zones",
            agg_field=["STATE", "COUNTY"],
            operations=["sum", "valid_count", "proportion_valid_nonzero"],
            output_csv=output_csv,
            output_gpkg=None,
            workdir=tmp_path / "work",
            tag="raster_job",
            task_graph=task_graph,
            progress_queue=multiprocessing_queue,
        )
        task_graph.join()
    finally:
        task_graph.close()
    result = pd.read_csv(output_csv)
    assert set(result.columns) == {
        "STATE",
        "COUNTY",
        "sum_values",
        "valid_count_values",
        "proportion_valid_nonzero_values",
    }
    assert result["sum_values"].sum() == pytest.approx(130.0)
    left = result[result["COUNTY"].astype(str).str.zfill(3) == "001"].iloc[0]
    right = result[result["COUNTY"].astype(str).str.zfill(3) == "002"].iloc[0]
    assert left["proportion_valid_nonzero_values"] == pytest.approx(7 / 8)
    assert right["proportion_valid_nonzero_values"] == pytest.approx(1.0)


def test_run_zonal_stats_job_parallelizes_multiple_rasters(
    monkeypatch, tmp_path, projected_zone_vector
):
    class ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeProcessPoolExecutor:
        instances = []

        def __init__(self, max_workers, **kwargs):
            self.max_workers = max_workers
            self.submitted = []
            FakeProcessPoolExecutor.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, func, *args, **kwargs):
            self.submitted.append((func, args, kwargs))
            return ImmediateFuture(func(*args, **kwargs))

    calls = []

    def fake_fast_zonal_statistics(
        base_raster_path_band,
        aggregate_vector_path,
        aggregate_vector_field,
        *,
        rasterize_worker_count,
        **kwargs,
    ):
        raster_path, _ = base_raster_path_band
        calls.append((Path(raster_path).stem, rasterize_worker_count))
        return {"A": {"sum": 1 if Path(raster_path).stem == "first" else 2}}

    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(runner, "as_completed", lambda futures: futures)
    monkeypatch.setattr(runner, "fast_zonal_statistics", fake_fast_zonal_statistics)

    output_csv = tmp_path / "parallel.csv"
    runner.run_zonal_stats_job(
        base_raster_path_list=[tmp_path / "first.tif", tmp_path / "second.tif"],
        base_vector_path_list=[],
        base_vector_fields=[],
        base_measure_vector=None,
        base_measure_layer=None,
        measure_crs="auto",
        agg_vector=projected_zone_vector,
        agg_layer="zones",
        agg_field=["STATE"],
        operations=["sum"],
        output_csv=output_csv,
        output_gpkg=None,
        workdir=tmp_path / "work",
        tag="parallel_rasters",
        task_graph=None,
        progress_queue=queue.Queue(),
        raster_workers=2,
    )

    assert FakeProcessPoolExecutor.instances[0].max_workers == 2
    assert calls == [("first", 1), ("second", 1)]

    result = pd.read_csv(output_csv)
    assert list(result.columns) == ["STATE", "sum_first", "sum_second"]
    assert result.iloc[0].to_dict() == {
        "STATE": "A",
        "sum_first": 1,
        "sum_second": 2,
    }


def test_run_zonal_stats_job_measure_integration(tmp_path, projected_zone_vector):
    measure_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0.5, 0.5), Point(3.0, 0.5)]},
        crs="EPSG:6933",
    )
    measure_path = _write_vector(tmp_path / "measure.gpkg", measure_gdf, layer="measure")
    output_csv = tmp_path / "measure_summary.csv"
    runner.run_zonal_stats_job(
        base_raster_path_list=[],
        base_vector_path_list=[],
        base_vector_fields=[],
        base_measure_vector=measure_path,
        base_measure_layer="measure",
        measure_crs="EPSG:6933",
        agg_vector=projected_zone_vector,
        agg_layer="zones",
        agg_field=["STATE", "COUNTY"],
        operations=["intersect_count"],
        output_csv=output_csv,
        output_gpkg=None,
        workdir=tmp_path / "work",
        tag="measure_job",
        task_graph=None,
        progress_queue=queue.Queue(),
    )
    result = pd.read_csv(output_csv)
    assert list(result["intersect_count_measure"]) == [1, 1]


def test_run_zonal_stats_job_empty_outputs(tmp_path, projected_zone_vector):
    output_csv = tmp_path / "empty.csv"
    runner.run_zonal_stats_job(
        base_raster_path_list=[],
        base_vector_path_list=[],
        base_vector_fields=[],
        base_measure_vector=None,
        base_measure_layer=None,
        measure_crs="auto",
        agg_vector=projected_zone_vector,
        agg_layer="zones",
        agg_field=["STATE", "COUNTY"],
        operations=["sum"],
        output_csv=output_csv,
        output_gpkg=None,
        workdir=tmp_path / "work",
        tag="empty_job",
        task_graph=None,
        progress_queue=queue.Queue(),
    )
    assert pd.read_csv(output_csv).empty


def test_run_zonal_stats_job_process_uses_taskgraph(monkeypatch, tmp_path):
    class FakeTaskGraph:
        def __init__(self, path, workers, reporting_interval):
            self.path = path
            self.workers = workers
            self.reporting_interval = reporting_interval
            self.closed = False

        def join(self):
            return None

        def close(self):
            self.closed = True

    captured = {}

    def fake_run_zonal_stats_job(**kwargs):
        captured.update(kwargs)
        runner._AREA_HECTARE_ASSUMPTIONS.add("area assumption")
        runner._MEASURE_CRS_ASSUMPTIONS.add("measure assumption")

    monkeypatch.setattr(runner.taskgraph, "TaskGraph", FakeTaskGraph)
    monkeypatch.setattr(runner, "run_zonal_stats_job", fake_run_zonal_stats_job)

    result = runner._run_zonal_stats_job_process(
        "output.csv",
        {"workdir": tmp_path, "progress_queue": queue.Queue()},
        logging.INFO,
        2,
    )
    assert captured["task_graph"].workers == 2
    assert result["output_label"] == "output.csv"
    assert result["area_hectare_assumptions"] == ["area assumption"]
    assert result["measure_crs_assumptions"] == ["measure assumption"]


def test_progress_monitor_stops_cleanly():
    progress_queue = queue.Queue()
    progress_queue.put({"event": "stop"})
    runner._progress_monitor(progress_queue, total_jobs=1)


def test_progress_monitor_shows_active_job_phase(monkeypatch):
    class FakeTqdm:
        instances = []

        def __init__(self, total, desc, unit, position, leave):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.position = position
            self.leave = leave
            self.n = 0
            self.postfix_values = []
            self.closed = False
            FakeTqdm.instances.append(self)

        def update(self, increment):
            self.n += increment

        def set_postfix_str(self, value, refresh=True):
            self.postfix_values.append(value)

        def refresh(self):
            pass

        def close(self):
            self.closed = True

    progress_queue = queue.Queue()
    progress_queue.put({"event": "job_start", "tag": "counties_ecosystem_services"})
    progress_queue.put(
        {
            "event": "analysis_start",
            "id": "raster:counties_ecosystem_services:annual_value",
            "desc": "raster annual_value",
            "total": 4,
            "phase": "scanning aggregation vector",
        }
    )
    progress_queue.put(
        {
            "event": "analysis_update",
            "id": "raster:counties_ecosystem_services:annual_value",
            "increment": 0,
            "phase": "stitching rasterized tiles",
        }
    )
    progress_queue.put(
        {
            "event": "job_done",
            "tag": "counties_ecosystem_services",
            "status": "done",
        }
    )
    progress_queue.put({"event": "stop"})

    monkeypatch.setattr(runner, "tqdm", FakeTqdm)

    runner._progress_monitor(progress_queue, total_jobs=1)

    job_bar = FakeTqdm.instances[0]
    assert "counties_ecosystem_services running" in job_bar.postfix_values
    assert (
        "counties_ecosystem_services: raster annual_value - "
        "stitching rasterized tiles"
    ) in job_bar.postfix_values
    assert "counties_ecosystem_services done" in job_bar.postfix_values


def test_main_returns_when_no_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["runner.py", str(tmp_path / "config.yml")])
    monkeypatch.setattr(
        runner,
        "parse_and_validate_config",
        lambda path: {
            "project": {
                "name": "config",
                "global_work_dir": tmp_path / "work",
                "log_level": "INFO",
            },
            "job_list": [],
        },
    )
    runner.main()
