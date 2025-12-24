from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import os

import geopandas as gpd
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
)

logger = logging.getLogger(__name__)

os.environ["OGR_SQLITE_OPEN_OPTIONS"] = "READONLY=YES"
os.environ["OGR_GPKG_OPEN_OPTIONS"] = "READONLY=YES"


def main():
    fields_to_extract = ["Rt", "Rt_nohab_all"]
    base_vectors = {
        "1992": "./jeronimodata/Spring/Inspring/coastal_risk_tnc_esa1992_md5_96c41f.gpkg",
        "2020": "./jeronimodata/Spring/Inspring/coastal_risk_tnc_esa2020_md5_764c7d.gpkg",
    }

    target_vector_path = "coastal_risk_tnc_esa1992_2020.gpkg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(target_vector_path)
    target_vector_path = f"{base}_{timestamp}{ext}"

    logging.info("reading 1992 and 2020 vectors")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_1992 = ex.submit(
            lambda: (
                logger.info(f'reading {base_vectors["1992"]}'),
                gpd.read_file(
                    base_vectors["1992"],
                    columns=["Rt", "Rt_nohab_all", "geometry"],
                ),
            )[1]
        )
        fut_2020 = ex.submit(
            lambda: (
                logger.info(f'reading {base_vectors["2020"]}'),
                gpd.read_file(
                    base_vectors["2020"],
                    columns=["Rt", "Rt_nohab_all", "geometry"],
                ),
            )[1]
        )
        gdf_1992 = fut_1992.result()
        gdf_2020 = fut_2020.result()

    if gdf_1992.crs and gdf_2020.crs and gdf_1992.crs != gdf_2020.crs:
        raise ValueError(
            f'{base_vectors["1992"]} and {base_vectors["2020"]} are different projections'
        )

    # good tolerance for degrees
    match_tol = 1e-8

    logger.info("extract x and y coords")
    x92 = gdf_1992.geometry.x.to_numpy()
    y92 = gdf_1992.geometry.y.to_numpy()
    x20 = gdf_2020.geometry.x.to_numpy()
    y20 = gdf_2020.geometry.y.to_numpy()

    logger.info("convert x/y to 64 bit integer hashes")
    gdf_1992["_kx"] = np.round(x92 / match_tol).astype("int64")
    gdf_1992["_ky"] = np.round(y92 / match_tol).astype("int64")
    gdf_2020["_kx"] = np.round(x20 / match_tol).astype("int64")
    gdf_2020["_ky"] = np.round(y20 / match_tol).astype("int64")

    logger.info("extract relevant fields from 1992 into out")
    out = gdf_1992[["geometry", "_kx", "_ky"] + fields_to_extract]
    out = out.rename(columns={f: f"{f}_1992" for f in fields_to_extract})

    logger.info("process 2020 fields")
    gdf_2020_fields = (
        gdf_2020[["_kx", "_ky"] + fields_to_extract]
        .rename(columns={f: f"{f}_2020" for f in fields_to_extract})
        .drop_duplicates(subset=["_kx", "_ky"])
    )

    logger.info("do the geometric merge")
    out = out.merge(gdf_2020_fields, on=["_kx", "_ky"], how="left")
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf_1992.crs)

    missing = out["Rt_2020"].isna()
    if missing.any():
        raise ValueError(
            "points are missing, this should not happen if they are the same vectors"
        )

    logging.info("calculate Service_1992")
    out["Service_1992"] = out["Rt_nohab_all_1992"] - out["Rt_1992"]
    logging.info("calculate Service_2020")
    out["Service_2020"] = out["Rt_nohab_all_2020"] - out["Rt_2020"]
    logging.info("calculate Rt_change")
    out["Rt_change"] = out["Rt_2020"] - out["Rt_1992"]
    logging.info("calculate Service_change")
    out["Service_change"] = out["Service_2020"] - out["Service_1992"]

    logging.info("dropping x/y hashes")
    out = out.drop(columns=["_kx", "_ky"])

    layer_out = os.path.splitext(os.path.basename(target_vector_path))[0]
    logging.info(f"saving to {layer_out}")
    out.to_file(target_vector_path, layer=layer_out, driver="GPKG", index=False)


if __name__ == "__main__":
    main()
