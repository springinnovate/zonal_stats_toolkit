"""GeoTIFF CRS normalization and deterministic cache-handle cleanup."""

import os
import queue

import numpy as np
from osgeo import gdal, ogr, osr
import pytest

import raster_cache
import runner


# Header from totalC_tCO2e_ha_2020.tif; tests need no assessment input files.
# GeoTIFF rewrites the inverse flattening and adds Greenwich's EPSG authority.
CARBON_WKT = (
    'PROJCS["NAD83 / Conus Albers",GEOGCS["NAD83",'
    'DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,'
    '298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],'
    'PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],'
    'PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],'
    'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","5070"]]'
)
CARBON_TRANSFORM = (-2364630.0, 30.0, 0.0, 3180570.0, 0.0, -30.0)


def _srs(wkt=CARBON_WKT):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _other_projection():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    return srs.ExportToWkt()


def _grid():
    return {"size": [4, 4], "transform": list(CARBON_TRANSFORM), "projection": CARBON_WKT}


def _raster(path, *, projection=CARBON_WKT, transform=CARBON_TRANSFORM,
            width=4, height=4, bands=1, dtype=gdal.GDT_Int32, nodata=-1):
    with gdal.GetDriverByName("GTiff").Create(str(path), width, height, bands, dtype) as dataset:
        if projection:
            dataset.SetProjection(projection)
        dataset.SetGeoTransform(transform)
        for index in range(1, bands + 1):
            band = dataset.GetRasterBand(index)
            if nodata is not None:
                band.SetNoDataValue(float(nodata))
            band.Fill(1)
            band = None
    return path


def _vector(path, *, burn_field=True, projection=CARBON_WKT):
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = dataset.CreateLayer("zones", srs=_srs(projection), geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("zone_key", ogr.OFTString))
    if burn_field:
        layer.CreateField(ogr.FieldDefn("original_fid", ogr.OFTInteger))
    for index in range(2):
        xmin = CARBON_TRANSFORM[0] + index * 60
        xmax = xmin + 60
        ymax, ymin = CARBON_TRANSFORM[3], CARBON_TRANSFORM[3] - 120
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("zone_key", f"zone-{index}")
        if burn_field:
            feature.SetField("original_fid", index + 1)
        feature.SetGeometry(ogr.CreateGeometryFromWkt(
            f"POLYGON (({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, "
            f"{xmin} {ymax}, {xmin} {ymin}))"
        ))
        assert layer.CreateFeature(feature) == 0
    feature, layer, dataset = None, None, None
    return path


def _rename_and_remove_with_exception_alive(path, error):
    assert error.__traceback__ is not None
    renamed = path.with_name(f"renamed_{path.name}")
    os.replace(path, renamed)
    renamed.unlink()
    # Keep the traceback alive through both Windows filesystem operations.
    assert error.__traceback__ is not None


def test_carbon_wkt_geotiff_roundtrip_is_accepted(tmp_path):
    path = _raster(tmp_path / "ids.tif")
    with gdal.OpenEx(str(path), gdal.OF_RASTER) as dataset:
        rewritten = dataset.GetProjection()
        assert rewritten != CARBON_WKT
        assert _srs(CARBON_WKT).IsSame(_srs(rewritten))
        assert list(dataset.GetGeoTransform()) == _grid()["transform"]
    runner._validate_id_raster(path, _grid())


def test_carbon_crs_cached_and_uncached_zonals_match_exactly(tmp_path, monkeypatch):
    source_pixels = _raster(tmp_path / "pixels.tif", dtype=gdal.GDT_Float32)
    values = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    with gdal.Open(str(source_pixels), gdal.GA_Update) as dataset:
        dataset.GetRasterBand(1).WriteArray(values)
    # A VRT retains the supplied noncanonical source WKT, whereas the ID TIFF
    # normalizes it. This exercises the original failure through the pipeline.
    source = tmp_path / "carbon.vrt"
    with gdal.Translate(str(source), str(source_pixels), format="VRT") as dataset:
        dataset.SetProjection(CARBON_WKT)
    with gdal.OpenEx(str(source), gdal.OF_RASTER) as dataset:
        supplied_wkt = dataset.GetProjection()
    with gdal.OpenEx(str(source_pixels), gdal.OF_RASTER) as dataset:
        assert supplied_wkt != dataset.GetProjection()
        assert _srs(supplied_wkt).IsSame(dataset.GetSpatialRef())
    vector = _vector(tmp_path / "zones.gpkg", burn_field=False)
    burns = []
    burn = runner._rasterize_aggregate_fids

    def observed_burn(*args, **kwargs):
        burns.append(args[3])
        return burn(*args, **kwargs)

    monkeypatch.setattr(runner, "_rasterize_aggregate_fids", observed_burn)

    def calculate(label, reuse):
        return runner.fast_zonal_statistics(
            (source, 1), vector, ["zone_key"], aggregate_layer_name="zones",
            operations=["sum", "mean", "valid_count", "total_count"],
            working_dir=tmp_path / label, geometry_cache_dir=tmp_path / "cache",
            reuse_geometry=reuse, max_simplify_tolerance_meters=0,
            rasterize_worker_count=1, progress_queue=queue.Queue(), progress_id=label,
        )

    cached = calculate("cold", True)
    np.testing.assert_equal(cached, calculate("warm", True))
    assert len(burns) == 1
    np.testing.assert_equal(cached, calculate("uncached", False))
    assert len(burns) == 2
    assert cached == {
        "zone-0": {"sum": 60, "mean": 7.5, "valid_count": 8, "total_count": 8},
        "zone-1": {"sum": 76, "mean": 9.5, "valid_count": 8, "total_count": 8},
    }


@pytest.mark.parametrize(("options", "diagnostic"), [
    ({"projection": _other_projection()}, "projection|CRS"),
    ({"projection": ""}, "projection|CRS"),
    ({"transform": (-2364600.0, 30, 0, 3180570, 0, -30)}, "transform"),
    ({"transform": (-2364630.0 + 1e-6, 30, 0, 3180570, 0, -30)}, "transform"),
    ({"transform": (-2364630.0, 30.01, 0, 3180570, 0, -30)}, "transform"),
    ({"width": 5}, "size|dimension"),
    ({"dtype": gdal.GDT_Float32}, "type"),
    ({"bands": 2}, "band|count"),
    ({"nodata": 0}, "nodata"),
    ({"nodata": None}, "nodata"),
])
def test_real_metadata_mismatches_still_fail_and_release_handles(tmp_path, options, diagnostic):
    path = _raster(tmp_path / "wrong.tif", **options)
    with pytest.raises(ValueError, match=f"(?i){diagnostic}") as caught:
        runner._validate_id_raster(path, _grid())
    _rename_and_remove_with_exception_alive(path, caught.value)


@pytest.mark.parametrize("problem", ["missing_field", "wrong_crs", "missing_layer"])
def test_prepared_vector_validation_releases_handles_when_exception_is_retained(tmp_path, problem):
    path = _vector(tmp_path / "prepared.gpkg", burn_field=problem != "missing_field",
                   projection=_other_projection() if problem == "wrong_crs" else CARBON_WKT)
    layer = "absent" if problem == "missing_layer" else "zones"
    with pytest.raises(ValueError) as caught:
        runner._validate_prepared_geometry(path, layer, CARBON_WKT)
    _rename_and_remove_with_exception_alive(path, caught.value)


def test_invalid_id_artifact_preserves_validation_error_and_removes_stage(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="(?i)projection|CRS") as caught:
        raster_cache.artifact(
            cache, "ids", _grid(), "agg_fid.tif",
            lambda path: _raster(path, projection=_other_projection()),
            lambda path: runner._validate_id_raster(path, _grid()),
        )
    assert caught.value.__traceback__ is not None
    assert not list(cache.rglob("*.staging-*"))
    assert not list(cache.rglob("manifest.json"))
    assert not list(cache.rglob("agg_fid.tif"))


def test_cleanup_error_does_not_replace_original_validation_error(tmp_path, monkeypatch):
    def cleanup_failure(*args, **kwargs):
        raise PermissionError("simulated Windows sharing violation")

    with monkeypatch.context() as patch:
        patch.setattr(raster_cache.shutil, "rmtree", cleanup_failure)
        with pytest.raises(ValueError, match="(?i)projection|CRS"):
            raster_cache.artifact(
                tmp_path / "cache", "ids", _grid(), "agg_fid.tif",
                lambda path: _raster(path, projection=_other_projection()),
                lambda path: runner._validate_id_raster(path, _grid()),
            )
