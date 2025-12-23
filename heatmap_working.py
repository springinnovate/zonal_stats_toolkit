from datasketches import kll_floats_sketch
from osgeo import gdal
from collections import defaultdict
import logging
from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS, Geod, Transformer

from ecoshard import geoprocessing

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
)


def make_linear_decay_kernel(base_raster_path, radius_m, kernel_path):
    Path(kernel_path).unlink(missing_ok=True)

    with rasterio.open(base_raster_path) as src:
        crs = CRS.from_wkt(src.crs.to_wkt())
        t = src.transform
        px = abs(t.a)
        py = abs(t.e)
        b = src.bounds

    def _meters_per_unit(crs_obj):
        ax0 = crs_obj.axis_info[0]
        if ax0.unit_name and ax0.unit_name.lower() in ("metre", "meter"):
            return 1.0
        if getattr(ax0, "unit_conversion_factor", None):
            return float(ax0.unit_conversion_factor)
        return 1.0

    if crs.is_geographic:
        geod = Geod(ellps="WGS84")
        lon0 = (b.left + b.right) / 2.0
        lat0 = (b.bottom + b.top) / 2.0
        ddeg = 0.01
        _, _, dx_m = geod.inv(lon0, lat0, lon0 + ddeg, lat0)
        _, _, dy_m = geod.inv(lon0, lat0, lon0, lat0 + ddeg)
        px_m = px * (abs(dx_m) / ddeg)
        py_m = py * (abs(dy_m) / ddeg)
    else:
        to_m = _meters_per_unit(crs)
        px_m = px * to_m
        py_m = py * to_m

    pix_m = (px_m + py_m) / 2.0
    n = int(math.ceil(radius_m / pix_m))
    size = 2 * n + 1

    ys, xs = np.mgrid[-n : n + 1, -n : n + 1]
    dist_m = np.sqrt((xs * px_m) ** 2 + (ys * py_m) ** 2)
    kernel = np.maximum(0.0, 1.0 - (dist_m / float(radius_m))).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": crs.to_wkt(),
        "transform": from_origin(0.0, 0.0, px, py),
        "nodata": 0.0,
        "compress": "lzw",
    }

    if size >= 16:
        block = (min(256, size) // 16) * 16
        if block == 0:
            profile["tiled"] = False
        else:
            profile.update({"tiled": True, "blockxsize": block, "blockysize": block})
    else:
        profile["tiled"] = False

    with rasterio.open(kernel_path, "w", **profile) as dst:
        dst.write(kernel, 1)


base_raster_path = "N_ret_ratio_2020_Colombia.tif"
kernel_path = "kernel.tif"
radius_m = 10_000
heatmap_raster_path = "heatmap.tif"
# make_linear_decay_kernel(base_raster_path, radius_m, kernel_path)

# geoprocessing.convolve_2d(
#     (base_raster_path, 1),
#     (kernel_path, 1),
#     "heatmap.tif",
#     ignore_nodata_and_edges=True,
#     mask_nodata=True,
#     normalize_kernel=False,
#     working_dir="./",
# )


sketch = kll_floats_sketch(k=200)
nodata = geoprocessing.get_raster_info(heatmap_raster_path)["nodata"][0]
for _, array_block in geoprocessing.iterblocks((heatmap_raster_path, 1)):
    if nodata is not None:
        array_block = array_block[array_block != nodata]
    sketch.update(array_block.astype(np.float32, copy=False).ravel())

threshold = sketch.get_quantile(0.95)
print(f"THRESHOLD: {threshold}")


def local_op(array):
    return array >= threshold


hotspot_raster_path = "hotspots.tif"

geoprocessing.raster_calculator(
    [(heatmap_raster_path, 1)],
    local_op,
    hotspot_raster_path,
    gdal.GDT_Byte,
    2,
)
