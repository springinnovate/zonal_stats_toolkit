from __future__ import annotations
from threading import Thread
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import collections
import argparse
import configparser
import glob
import logging
import os
import shutil
import tempfile
import time

from tqdm import tqdm
from datasketches import kll_floats_sketch
from ecoshard import taskgraph, geoprocessing
from osgeo import gdal, ogr, osr
from pyproj import CRS, Geod, Transformer
from shapely.strtree import STRtree
import pandas as pd
import fiona
import numpy as np
import geopandas as gpd

logging.getLogger("ecoshard").setLevel(logging.INFO)
logging.getLogger("fiona").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_LOGGING_PERIOD = 10.0
AREA_HECTARE_OPERATIONS = {
    "area_ha_total",
    "area_ha_valid",
}
MEASURE_OPERATIONS = {
    "intersect_area_ha",
    "intersect_length_km",
    "intersect_count",
}
_AREA_HECTARE_ASSUMPTIONS = set()
_MEASURE_CRS_ASSUMPTIONS = set()
VALID_OPERATIONS = {
    *AREA_HECTARE_OPERATIONS,
    *MEASURE_OPERATIONS,
    "mean",
    "stdev",
    "min",
    "max",
    "sum",
    "total_count",
    "valid_count",
}


def _record_area_hectare_assumption(message):
    """Track a geographic-raster area approximation for end-of-run logging.

    Args:
        message: Human-readable description of the raster and approximation used.
    """
    _AREA_HECTARE_ASSUMPTIONS.add(message)


def _log_area_hectare_assumptions():
    """Log any geographic-raster area approximations after all jobs complete."""
    for message in sorted(_AREA_HECTARE_ASSUMPTIONS):
        logger.warning("Area hectare assumption: %s", message)


def _raster_pixel_area_ha(raster_path, raster_info, raster_srs):
    """Estimate one raster pixel's area in hectares.

    Projected rasters use the raster pixel size and CRS linear units directly.
    Geographic rasters use a representative pixel at the center of the raster
    extent and compute its ellipsoidal area with `pyproj.Geod`. That keeps area
    outputs count-based while making the geographic-CRS approximation explicit.

    Args:
        raster_path: Path to the raster, used only for logging assumptions.
        raster_info: Raster metadata from `geoprocessing.get_raster_info`.
        raster_srs: GDAL spatial reference for the raster.

    Returns:
        Estimated area of one raster pixel in hectares.
    """
    pixel_width, pixel_height = raster_info["pixel_size"]
    if raster_srs is not None and raster_srs.IsProjected():
        linear_units_to_meters = raster_srs.GetLinearUnits() or 1.0
        return (
            abs(pixel_width * pixel_height)
            * linear_units_to_meters
            * linear_units_to_meters
            / 10000.0
        )

    bounding_box = raster_info["bounding_box"]
    center_x = (bounding_box[0] + bounding_box[2]) * 0.5
    center_y = (bounding_box[1] + bounding_box[3]) * 0.5
    half_width = abs(pixel_width) * 0.5
    half_height = abs(pixel_height) * 0.5

    source_projection_wkt = raster_info.get("projection_wkt")
    if source_projection_wkt:
        source_crs = CRS.from_wkt(source_projection_wkt)
        source_description = source_crs.to_string()
    else:
        source_crs = CRS.from_epsg(4326)
        source_description = "missing CRS, assumed EPSG:4326"

    geod = source_crs.get_geod() if hasattr(source_crs, "get_geod") else None
    if geod is None:
        geod = Geod(ellps="WGS84")
        source_description = f"{source_description}; WGS84 ellipsoid assumed"

    x_values = [
        center_x - half_width,
        center_x + half_width,
        center_x + half_width,
        center_x - half_width,
    ]
    y_values = [
        center_y - half_height,
        center_y - half_height,
        center_y + half_height,
        center_y + half_height,
    ]
    area_m2, _ = geod.polygon_area_perimeter(x_values, y_values)
    _record_area_hectare_assumption(
        f"{raster_path}: raster CRS is not projected ({source_description}); "
        f"pixel area was estimated from a representative center pixel using "
        f"pyproj.Geod."
    )
    return abs(area_m2) / 10000.0


def _safe_path_stem(path, max_length=60):
    """Return a filesystem-safe abbreviated stem for cache directory names."""
    safe_stem = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in Path(path).stem
    )
    return (safe_stem or "unnamed")[:max_length]


def _prepare_aggregate_vector_for_rasterization(
    aggregate_vector_path,
    aggregate_layer_name,
    target_vector_path,
    raster_projection_wkt,
    simplify_tolerance,
    needs_reproject,
):
    """Project/simplify aggregation vector and add a FID burn field."""
    target_vector_path = Path(target_vector_path)
    target_vector_path.parent.mkdir(parents=True, exist_ok=True)
    target_vector_path.unlink(missing_ok=True)

    vector_translate_kwargs = {"format": "GPKG"}
    src_path = str(aggregate_vector_path)
    tmp_reprojected_path = None
    if needs_reproject:
        tmp_reprojected_path = target_vector_path.with_suffix(".reprojected.gpkg")
        tmp_reprojected_path.unlink(missing_ok=True)
        logger.info(
            "vector translate (reproject) start | output=%s | reproject=%s",
            tmp_reprojected_path,
            needs_reproject,
        )
        gdal.VectorTranslate(
            str(tmp_reprojected_path),
            src_path,
            dstSRS=raster_projection_wkt,
            **vector_translate_kwargs,
        )
        src_path = str(tmp_reprojected_path)

    logger.info(
        "vector translate (simplify) start | output=%s | simplifyTolerance=%s | reproject=%s",
        target_vector_path,
        simplify_tolerance,
        needs_reproject,
    )
    gdal.VectorTranslate(
        str(target_vector_path),
        src_path,
        simplifyTolerance=simplify_tolerance,
        **vector_translate_kwargs,
    )
    if tmp_reprojected_path:
        tmp_reprojected_path.unlink(missing_ok=True)

    aggregate_vector = gdal.OpenEx(
        str(target_vector_path), gdal.OF_VECTOR | gdal.OF_UPDATE
    )
    aggregate_layer = (
        aggregate_vector.GetLayerByName(aggregate_layer_name)
        if aggregate_layer_name is not None
        else aggregate_vector.GetLayer()
    )

    local_aggregate_field_name = "original_fid"
    # RasterizeLayer burns attribute values, so persist the OGR FID as a field.
    if aggregate_layer.FindFieldIndex(local_aggregate_field_name, 1) == -1:
        aggregate_layer.CreateField(
            ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger)
        )

    aggregate_layer.ResetReading()
    aggregate_layer.StartTransaction()
    for feature in aggregate_layer:
        feature.SetField(local_aggregate_field_name, feature.GetFID())
        aggregate_layer.SetFeature(feature)
    aggregate_layer.CommitTransaction()
    aggregate_layer = None
    aggregate_vector = None

    logger.info("vector translate done | output=%s", target_vector_path)


def _rasterize_aggregate_fids(
    base_raster_path,
    aggregate_vector_path,
    aggregate_layer_name,
    target_raster_path,
    target_nodata,
):
    """Rasterize prepared aggregation feature IDs onto the base raster grid."""
    target_raster_path = Path(target_raster_path)
    target_raster_path.parent.mkdir(parents=True, exist_ok=True)
    target_raster_path.unlink(missing_ok=True)

    logger.info("creating agg fid raster: %s", target_raster_path)
    geoprocessing.new_raster_from_base(
        str(base_raster_path),
        str(target_raster_path),
        gdal.GDT_Int32,
        [target_nodata],
    )

    aggregate_vector = gdal.OpenEx(str(aggregate_vector_path), gdal.OF_VECTOR)
    aggregate_layer = (
        aggregate_vector.GetLayerByName(aggregate_layer_name)
        if aggregate_layer_name is not None
        else aggregate_vector.GetLayer()
    )
    feature_id_raster_dataset = gdal.OpenEx(
        str(target_raster_path), gdal.GA_Update | gdal.OF_RASTER
    )
    if feature_id_raster_dataset is None:
        raise RuntimeError(f"Could not open target raster at {target_raster_path}")

    rasterize_callback = _make_logger_callback(
        "rasterizing polygons %.1f%% complete %s"
    )
    aggregate_layer.ResetReading()
    logger.info("rasterize start")
    error_code = gdal.RasterizeLayer(
        feature_id_raster_dataset,
        [1],
        aggregate_layer,
        callback=rasterize_callback,
        options=[
            "ALL_TOUCHED=FALSE",
            "ATTRIBUTE=original_fid",
        ],
    )
    feature_id_raster_dataset.FlushCache()
    feature_id_raster_dataset = None
    aggregate_layer = None
    aggregate_vector = None
    if error_code != 0:
        raise RuntimeError(
            f"RasterizeLayer failed with error code {error_code} for {target_raster_path}"
        )
    logger.info("rasterize done")


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Return:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """

    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if (current_time - logger_callback.last_time) > 5.0 or (
                df_complete == 1.0 and logger_callback.total_time >= 5.0
            ):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    logger.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    logger.info(message, df_complete * 100, "")
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0
        except Exception:
            logger.exception(
                "Unhandled error occurred while logging "
                "progress.  df_complete: %s, p_progress_arg: %s",
                df_complete,
                p_progress_arg,
            )

    return logger_callback


def parse_and_validate_config(cfg_path: Path) -> dict:
    """Parse and validate a project configuration file.

    Reads an INI-style configuration file describing a project and one or more
    jobs, validates its structure and contents, resolves relative paths, and
    returns a normalized configuration dictionary suitable for downstream
    processing.

    The configuration must contain a `[project]` section and one or more
    `[job:<tag>]` sections. Extensive validation is performed on required fields,
    file paths, layer names, attribute fields, and operation specifications. Most
    errors result in `ValueError`; missing files raise `FileNotFoundError`.

    Args:
        cfg_path: Path to the configuration file. Relative paths inside the config
            are resolved relative to the directory containing this file.

    Returns:
        A dictionary with two top-level keys:
            - `project`: A dict containing validated project-level settings
              (`name`, `global_work_dir`, `log_level`).
            - `job_list`: A list of dicts, one per job, containing validated and
              resolved job configuration, including paths, fields, operations, and
              output locations.

    Raises:
        ValueError: If the configuration structure is invalid, required fields are
            missing, values are malformed, or semantic validation fails.
        FileNotFoundError: If required files or glob patterns resolve to no files.
    """
    stem = cfg_path.stem
    cfg_dir = cfg_path.parent

    config = configparser.ConfigParser(interpolation=None)
    config.read(cfg_path)

    if "project" not in config:
        raise ValueError("Missing [project] section")

    project_name = config["project"].get("name", "").strip()
    if project_name != stem:
        raise ValueError(
            f"[project].name must equal config stem: expected {stem}, got {project_name}"
        )

    log_level_str = config["project"].get("log_level", "INFO").strip().upper()
    try:
        _ = getattr(logging, log_level_str)
    except AttributeError:
        raise ValueError(f"Invalid log_level: {log_level_str}")

    global_work_dir = Path(config["project"]["global_work_dir"].strip())
    if not global_work_dir.is_absolute():
        global_work_dir = cfg_dir / global_work_dir

    job_tags = []
    jobs_sections = []
    for section in config.sections():
        section_clean = section.strip()
        section_lower = section_clean.lower()
        if section_lower == "project":
            continue
        if section_lower.startswith("job:"):
            tag = section_clean.split(":", 1)[1].strip()
            if not tag:
                raise ValueError(f"Invalid job section name: [{section_clean}]")
            job_tags.append(tag)
            jobs_sections.append((tag, config[section]))
        else:
            raise ValueError(f"unknown section type: {section_lower}")

    if len(job_tags) != len(set(job_tags)):
        seen = set()
        dups = []
        for t in job_tags:
            if t in seen:
                dups.append(t)
            seen.add(t)
        raise ValueError(f"Duplicate job tags found: {sorted(set(dups))}")

    def _abs_from_cfg_dir(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (cfg_dir / path)

    def _split_top_level_commas(s: str) -> list[str]:
        parts = []
        buf = []
        depth = 0
        for ch in s:
            if ch == "[":
                depth += 1
                buf.append(ch)
            elif ch == "]":
                depth = max(depth - 1, 0)
                buf.append(ch)
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        part = "".join(buf).strip()
        if part:
            parts.append(part)
        return parts

    def _parse_vector_pattern_entry(entry: str, tag: str) -> tuple[str, list[str]]:
        i = entry.find("[")
        j = entry.rfind("]")
        if i == -1 or j == -1 or j < i:
            raise ValueError(
                f"[job:{tag}] base_vector_pattern entries must include fields as "
                f"path[field1,field2,...]. Bad entry: {entry}"
            )
        pattern_str = entry[:i].strip()
        fields_str = entry[i + 1 : j]
        fields = [f.strip() for f in fields_str.split(",") if f.strip()]
        if not pattern_str:
            raise ValueError(
                f"[job:{tag}] empty path in base_vector_pattern entry: {entry}"
            )
        if not fields:
            raise ValueError(
                f"[job:{tag}] empty field list in base_vector_pattern entry: {entry}"
            )
        return pattern_str, fields

    def _glob_patterns(pattern_csv: str) -> list[Path]:
        out = []
        for pattern in [p.strip() for p in pattern_csv.split(",") if p.strip()]:
            pat = pattern if Path(pattern).is_absolute() else str(cfg_dir / pattern)
            out.extend([Path(p) for p in glob.glob(pat)])
        return sorted({p for p in out})

    def _resolve_layer(
        vector_path: Path, layer_name: str, layer_config_key: str, tag: str
    ) -> str:
        layers = fiona.listlayers(str(vector_path))
        if not layers:
            raise ValueError(f"[job:{tag}] no layers found in {vector_path}")
        if layer_name:
            if layer_name not in layers:
                raise ValueError(
                    f'[job:{tag}] {layer_config_key} "{layer_name}" not found in '
                    f"{vector_path}. Available layers: {layers}"
                )
            return layer_name
        if len(layers) > 1:
            raise ValueError(
                f"[job:{tag}] {layer_config_key} is required for {vector_path} "
                f"because it has multiple layers: {layers}"
            )
        return layers[0]

    job_list = []
    for tag, job in jobs_sections:
        agg_vector_raw = job.get("agg_vector", "").strip()
        if not agg_vector_raw:
            raise ValueError(f"[job:{tag}] missing agg_vector")
        agg_vector = _abs_from_cfg_dir(agg_vector_raw)
        if not agg_vector.exists():
            raise FileNotFoundError(f"[job:{tag}] agg_vector not found: {agg_vector}")

        agg_field_raw = job.get("agg_field", "").strip()
        if not agg_field_raw:
            raise ValueError(f"[job:{tag}] missing agg_field")
        agg_fields = [
            field.strip() for field in agg_field_raw.split(",") if field.strip()
        ]
        if not agg_fields:
            raise ValueError(f"[job:{tag}] agg_field is empty")
        if len(agg_fields) != len(set(agg_fields)):
            raise ValueError(
                f"[job:{tag}] agg_field contains duplicate fields: {agg_field_raw}"
            )

        ops_raw = job.get("operations", "").strip()
        if not ops_raw:
            raise ValueError(f"[job:{tag}] missing operations")
        operations = [o.strip().lower() for o in ops_raw.split(",") if o.strip()]
        if not operations:
            raise ValueError(f"[job:{tag}] operations is empty")

        invalid_ops = sorted(set(operations) - VALID_OPERATIONS)
        if any(op for op in invalid_ops if not op.startswith("p")):
            raise ValueError(
                f"[job:{tag}] invalid operations: {invalid_ops}. "
                f"Valid operations: {sorted(VALID_OPERATIONS)}"
            )

        agg_layer = job.get("agg_layer", "").strip()
        agg_layer = _resolve_layer(agg_vector, agg_layer, "agg_layer", tag)

        with fiona.open(str(agg_vector), layer=agg_layer) as src:
            props = src.schema.get("properties", {})
            missing_agg_fields = [
                agg_field for agg_field in agg_fields if agg_field not in props
            ]
            if missing_agg_fields:
                raise ValueError(
                    f"[job:{tag}] agg_field entries {missing_agg_fields} not found in layer "
                    f'"{agg_layer}" of {agg_vector}. '
                    f"Available fields: {sorted(props.keys())}"
                )

        output_csv = job.get("output_csv", "").strip()
        output_gpkg = job.get("output_gpkg", "").strip()
        if not output_csv and not output_gpkg:
            raise ValueError(
                f"[job:{tag}] must define at least one of output_csv or output_gpkg"
            )

        workdir = global_work_dir / Path(tag)
        workdir.mkdir(parents=True, exist_ok=True)

        base_raster_path_list = []
        base_vector_path_list = []
        base_vector_fields = []
        base_measure_vector = None
        base_measure_layer = None
        measure_crs = job.get("measure_crs", "auto").strip() or "auto"

        base_raster_pattern = job.get("base_raster_pattern", "").strip()
        if base_raster_pattern:
            base_raster_path_list = _glob_patterns(base_raster_pattern)
            if not base_raster_path_list:
                raise FileNotFoundError(
                    f"[job:{tag}] no files found at {base_raster_pattern}"
                )

        base_vector_pattern = job.get("base_vector_pattern", "").strip()
        if base_vector_pattern:
            parts = _split_top_level_commas(base_vector_pattern)

            token_specs = []
            for part in parts:
                token_specs.append(_parse_vector_pattern_entry(part, tag))

            base_vector_fields = token_specs[0][1]
            for _, fields in token_specs[1:]:
                if fields != base_vector_fields:
                    raise ValueError(
                        f"[job:{tag}] base_vector_pattern uses inconsistent field lists"
                    )

            for pattern_str, _ in token_specs:
                pat = (
                    pattern_str
                    if Path(pattern_str).is_absolute()
                    else str(cfg_dir / pattern_str)
                )
                base_vector_path_list.extend([Path(p) for p in glob.glob(pat)])

            base_vector_path_list = sorted({p for p in base_vector_path_list})
            if not base_vector_path_list:
                raise FileNotFoundError(
                    f"[job:{tag}] no files found at {base_vector_pattern}"
                )

            for base_vector_path in base_vector_path_list:
                layer = _resolve_layer(
                    base_vector_path, "", "base_vector_pattern layer", tag
                )
                with fiona.open(str(base_vector_path), layer=layer) as src:
                    props = src.schema.get("properties", {})
                    missing = [f for f in base_vector_fields if f not in props]
                    if missing:
                        raise ValueError(
                            f'[job:{tag}] missing fields {missing} in layer "{layer}" of {base_vector_path}. '
                            f"Available fields: {sorted(props.keys())}"
                        )

        base_measure_vector_raw = job.get("base_measure_vector", "").strip()
        if base_measure_vector_raw:
            base_measure_vector = _abs_from_cfg_dir(base_measure_vector_raw)
            if not base_measure_vector.exists():
                raise FileNotFoundError(
                    f"[job:{tag}] base_measure_vector not found: {base_measure_vector}"
                )
            base_measure_layer = _resolve_layer(
                base_measure_vector,
                job.get("base_measure_layer", "").strip(),
                "base_measure_layer",
                tag,
            )

        is_measure_job = base_measure_vector is not None
        measure_operation_set = MEASURE_OPERATIONS.intersection(operations)
        if is_measure_job:
            if base_raster_path_list or base_vector_path_list:
                raise ValueError(
                    f"[job:{tag}] base_measure_vector jobs cannot also define "
                    "base_raster_pattern or base_vector_pattern"
                )
            if len(operations) != 1 or len(measure_operation_set) != 1:
                raise ValueError(
                    f"[job:{tag}] base_measure_vector jobs must define exactly "
                    f"one operation from {sorted(MEASURE_OPERATIONS)}"
                )
        elif measure_operation_set:
            raise ValueError(
                f"[job:{tag}] measure operations require base_measure_vector"
            )

        if (
            (not base_raster_path_list)
            and (not base_vector_path_list)
            and (not is_measure_job)
        ):
            raise ValueError(
                f"[job:{tag}] must define at least one of base_raster_pattern, "
                "base_vector_pattern, or base_measure_vector"
            )
        if (
            AREA_HECTARE_OPERATIONS.intersection(operations)
            and not base_raster_path_list
        ):
            raise ValueError(
                f"[job:{tag}] area_ha operations require base_raster_pattern"
            )

        job_list.append(
            {
                "tag": tag,
                "agg_vector": agg_vector,
                "agg_layer": agg_layer,
                "agg_field": agg_fields,
                "operations": operations,
                "workdir": workdir,
                "output_csv": output_csv or None,
                "output_gpkg": output_gpkg or None,
                "base_raster_path_list": base_raster_path_list,
                "base_vector_path_list": base_vector_path_list,
                "base_vector_fields": base_vector_fields,
                "base_measure_vector": base_measure_vector,
                "base_measure_layer": base_measure_layer,
                "measure_crs": measure_crs,
                "task_graph": None,
            }
        )

    return {
        "project": {
            "name": project_name,
            "global_work_dir": global_work_dir,
            "log_level": log_level_str,
        },
        "job_list": job_list,
    }


def fast_zonal_statistics(
    base_raster_path_band,
    aggregate_vector_path,
    aggregate_vector_field,
    aggregate_layer_name=None,
    ignore_nodata=True,
    working_dir=None,
    clean_working_dir=False,
    percentile_list=None,
    calculate_area_ha=False,
):
    raster_path, raster_band_index = base_raster_path_band
    aggregate_vector_fields = aggregate_vector_field
    aggregate_vector_field_label = ",".join(aggregate_vector_fields)

    logger.info(
        "fast_zonal_statistics start | raster=%s band=%s | vector=%s layer=%s "
        "fields=%s | ignore_nodata=%s | working_dir=%s clean=%s | "
        "percentiles=%s | calculate_area_ha=%s",
        raster_path,
        raster_band_index,
        str(aggregate_vector_path),
        aggregate_layer_name,
        aggregate_vector_field_label,
        ignore_nodata,
        working_dir,
        clean_working_dir,
        percentile_list,
        calculate_area_ha,
    )

    percentile_list = [] if percentile_list is None else list(percentile_list)
    percentile_list = sorted(
        {float(percentile_value) for percentile_value in percentile_list}
    )
    percentile_keys = [
        f"p{int(percentile_value) if percentile_value.is_integer() else percentile_value}"
        for percentile_value in percentile_list
    ]
    percentile_default_values = {
        percentile_key: None for percentile_key in percentile_keys
    }

    empty_group_stats_template = {
        "min": None,
        "max": None,
        "total_count": 0,
        "nodata_count": 0,
        "valid_count": 0,
        "area_ha_total": 0.0,
        "area_ha_valid": 0.0,
        "sum": 0.0,
        "stdev": None,
        **percentile_default_values,
    }
    grouped_stats_working_template = {
        **empty_group_stats_template,
        "sumsq": 0.0,
    }
    feature_stats_template = {
        "min": None,
        "max": None,
        "total_count": 0,
        "nodata_count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
    }

    raster_info = geoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info["nodata"][raster_band_index - 1]
    raster_pixel_width = abs(raster_info["pixel_size"][0])
    simplify_tolerance = raster_pixel_width * 0.5

    logger.info(
        "raster loaded | nodata=%s | pixel_size=%s | bbox=%s",
        raster_nodata,
        raster_info["pixel_size"],
        raster_info["bounding_box"],
    )

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_info["projection_wkt"])
    raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    raster_pixel_area_ha = None
    if calculate_area_ha:
        raster_pixel_area_ha = _raster_pixel_area_ha(
            raster_path, raster_info, raster_srs
        )
        logger.info("raster pixel area: %.12f ha", raster_pixel_area_ha)

    source_vector = gdal.OpenEx(str(aggregate_vector_path), gdal.OF_VECTOR)
    source_layer = (
        source_vector.GetLayerByName(aggregate_layer_name)
        if aggregate_layer_name is not None
        else source_vector.GetLayer()
    )

    source_srs = source_layer.GetSpatialRef()
    needs_reproject = True
    if source_srs is not None:
        source_srs = source_srs.Clone()
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        needs_reproject = not source_srs.IsSame(raster_srs)
        logger.info("vector SRS detected | needs_reproject=%s", needs_reproject)
    else:
        logger.info("vector SRS missing/unknown | forcing reprojection to raster SRS")

    source_layer = None
    source_vector = None

    working_dir = Path(working_dir) if working_dir else Path(tempfile.gettempdir())
    cache_working_dir = (
        working_dir
        / "fast_zonal_statistics_cache"
        / f"{_safe_path_stem(raster_path)}_band{raster_band_index}"
    )
    cache_working_dir.mkdir(parents=True, exist_ok=True)
    projected_vector_path = cache_working_dir / "projected_vector.gpkg"
    feature_id_raster_path = cache_working_dir / "agg_fid.tif"
    local_task_graph = taskgraph.TaskGraph(
        cache_working_dir / "taskgraph", 1, 15.0
    )
    logger.info("using zonal statistics cache dir: %s", cache_working_dir)

    def _raster_nodata_mask(value_array):
        finite_mask = np.isfinite(value_array)
        if raster_nodata is None:
            return ~finite_mask
        return np.isclose(value_array, raster_nodata) | ~finite_mask

    try:
        prepare_vector_task = local_task_graph.add_task(
            func=_prepare_aggregate_vector_for_rasterization,
            args=(
                aggregate_vector_path,
                aggregate_layer_name,
                projected_vector_path,
                raster_info["projection_wkt"],
                simplify_tolerance,
                needs_reproject,
            ),
            target_path_list=[projected_vector_path],
            task_name=f"prepare aggregate vector for {Path(aggregate_vector_path).stem}",
        )
        prepare_vector_task.join()

        aggregate_vector = gdal.OpenEx(str(projected_vector_path), gdal.OF_VECTOR)
        aggregate_layer = (
            aggregate_vector.GetLayerByName(aggregate_layer_name)
            if aggregate_layer_name is not None
            else aggregate_vector.GetLayer()
        )

        logger.info(
            "scanning vector for grouping field values: %s",
            aggregate_vector_field_label,
        )
        feature_id_set = set()
        feature_id_to_group_value = {}
        unique_group_values = set()

        aggregate_layer.ResetReading()
        for feature in aggregate_layer:
            feature_id = feature.GetFID()
            group_value = tuple(
                feature.GetField(field_name)
                for field_name in aggregate_vector_fields
            )
            if len(group_value) == 1:
                group_value = group_value[0]
            feature_id_set.add(feature_id)
            feature_id_to_group_value[feature_id] = group_value
            unique_group_values.add(group_value)
        aggregate_layer.ResetReading()

        logger.info(
            "vector scan done | features=%d | unique %s=%d",
            len(feature_id_set),
            aggregate_vector_field_label,
            len(unique_group_values),
        )

        raster_bounding_box = raster_info["bounding_box"]
        vector_extent = aggregate_layer.GetExtent()
        logger.info(
            "extent check | raster_bbox=%s | vector_extent=%s",
            raster_bounding_box,
            vector_extent,
        )

        vector_min_x, vector_max_x, vector_min_y, vector_max_y = vector_extent
        raster_min_x, raster_min_y, raster_max_x, raster_max_y = raster_bounding_box
        has_no_intersection = (
            vector_max_x < raster_min_x
            or vector_min_x > raster_max_x
            or vector_max_y < raster_min_y
            or vector_min_y > raster_max_y
        )

        if has_no_intersection:
            logger.error(
                "aggregate vector %s does not intersect with the raster %s: vector extent %s vs raster bounding box %s",
                str(aggregate_vector_path),
                raster_path,
                vector_extent,
                raster_bounding_box,
            )
            grouped_stats = {
                group_value: dict(empty_group_stats_template)
                for group_value in unique_group_values
            }
            logger.info(
                "returning empty stats for %d groups (no intersection)",
                len(unique_group_values),
            )
            aggregate_layer = None
            return grouped_stats

        raster_path_for_stats = raster_path
        logger.info("opening raster for read: %s", raster_path_for_stats)
        raster_dataset = gdal.OpenEx(raster_path_for_stats, gdal.OF_RASTER)
        raster_band = raster_dataset.GetRasterBand(raster_band_index)
        logger.info(
            "raster opened | size=%dx%d | band=%d",
            raster_band.XSize,
            raster_band.YSize,
            raster_band_index,
        )

        logger.info(
            "disjoint sets ready total_features=%d",
            len(feature_id_set),
        )

        feature_stats_by_id = collections.defaultdict(
            lambda: dict(feature_stats_template)
        )

        feature_id_raster_nodata = -1
        aggregate_layer = None
        aggregate_vector = None

        rasterize_task = local_task_graph.add_task(
            func=_rasterize_aggregate_fids,
            args=(
                raster_path_for_stats,
                projected_vector_path,
                aggregate_layer_name,
                feature_id_raster_path,
                feature_id_raster_nodata,
            ),
            dependent_task_list=[prepare_vector_task],
            target_path_list=[feature_id_raster_path],
            task_name=f"rasterize aggregate fids for {Path(raster_path).stem}",
        )
        rasterize_task.join()

        feature_id_raster_offsets = list(
            geoprocessing.iterblocks(
                (str(feature_id_raster_path), 1),
                offset_only=True,
                largest_block=2**28,
            )
        )
        logger.info(
            "iterblocks prepared | blocks=%d",
            len(feature_id_raster_offsets),
        )

        feature_id_raster_dataset = gdal.OpenEx(
            str(feature_id_raster_path), gdal.OF_RASTER
        )
        feature_id_raster_band = feature_id_raster_dataset.GetRasterBand(1)

        logger.info("gathering stats from raster blocks")
        block_log_time = time.time()
        group_sketch = None
        if percentile_list:
            group_sketch = defaultdict(lambda: kll_floats_sketch(k=200))
        for block_index, feature_id_offset in enumerate(feature_id_raster_offsets):
            block_log_time = _invoke_timed_callback(
                block_log_time,
                lambda block_index_value=block_index: logger.info(
                    "block processing | %.1f%% (%d/%d blocks)",
                    100.0
                    * float(block_index_value + 1)
                    / len(feature_id_raster_offsets),
                    block_index_value + 1,
                    len(feature_id_raster_offsets),
                ),
                _LOGGING_PERIOD,
            )

            feature_id_block = feature_id_raster_band.ReadAsArray(**feature_id_offset)
            raster_value_block = raster_band.ReadAsArray(**feature_id_offset)

            in_polygon_mask = feature_id_block != feature_id_raster_nodata
            if not np.any(in_polygon_mask):
                continue

            block_feature_ids = feature_id_block[in_polygon_mask]
            block_raster_values = raster_value_block[in_polygon_mask]

            for feature_id in np.unique(block_feature_ids):
                feature_values = block_raster_values[block_feature_ids == feature_id]
                total_count = feature_values.size
                if total_count == 0:
                    continue

                feature_nodata_mask = _raster_nodata_mask(feature_values)
                nodata_count = int(np.count_nonzero(feature_nodata_mask))

                feature_stats = feature_stats_by_id[feature_id]
                feature_stats["total_count"] += total_count
                feature_stats["nodata_count"] += nodata_count

                if ignore_nodata:
                    feature_values = feature_values[~feature_nodata_mask]
                if feature_values.size == 0:
                    continue

                if group_sketch is not None:
                    group_value = feature_id_to_group_value[feature_id]
                    sk = group_sketch[group_value]
                    sk.update(feature_values.astype(np.float32, copy=False).ravel())

                block_min_value = np.min(feature_values)
                block_max_value = np.max(feature_values)
                if feature_stats["min"] is None:
                    feature_stats["min"] = block_min_value
                    feature_stats["max"] = block_max_value
                else:
                    feature_stats["min"] = min(feature_stats["min"], block_min_value)
                    feature_stats["max"] = max(feature_stats["max"], block_max_value)

                feature_stats["sum"] += np.sum(feature_values)
                feature_stats["sumsq"] += np.sum(
                    feature_values * feature_values, dtype=np.float64
                )

        logger.info("aggregating done")

        feature_id_raster_band = None
        feature_id_raster_dataset = None

        remaining_unset_feature_ids = feature_id_set.difference(feature_stats_by_id)
        for missing_feature_id in remaining_unset_feature_ids:
            feature_stats_by_id[missing_feature_id]

        logger.info(
            "unset fid pass done | remaining_unset=%d | total_fids=%d",
            len(remaining_unset_feature_ids),
            len(feature_id_set),
        )

        raster_band = None
        raster_dataset = None
        aggregate_layer = None

        logger.info("grouping fid stats -> %s values", aggregate_vector_field_label)
        grouped_stats = collections.defaultdict(
            lambda: dict(grouped_stats_working_template)
        )

        for feature_id in feature_id_set:
            group_value = feature_id_to_group_value[feature_id]
            feature_stats = feature_stats_by_id[feature_id]
            group_stats = grouped_stats[group_value]

            group_stats["total_count"] += feature_stats["total_count"]
            group_stats["nodata_count"] += feature_stats["nodata_count"]
            group_stats["sum"] += feature_stats["sum"]
            group_stats["sumsq"] += feature_stats["sumsq"]

            feature_valid_count = (
                feature_stats["total_count"] - feature_stats["nodata_count"]
            )
            if feature_valid_count > 0:
                if group_stats["min"] is None:
                    group_stats["min"] = feature_stats["min"]
                    group_stats["max"] = feature_stats["max"]
                else:
                    group_stats["min"] = min(group_stats["min"], feature_stats["min"])
                    group_stats["max"] = max(group_stats["max"], feature_stats["max"])

        for group_value, group_stats in grouped_stats.items():
            valid_count = group_stats["total_count"] - group_stats["nodata_count"]
            group_stats["valid_count"] = valid_count
            if raster_pixel_area_ha is not None:
                group_stats["area_ha_total"] = (
                    group_stats["total_count"] * raster_pixel_area_ha
                )
                group_stats["area_ha_valid"] = valid_count * raster_pixel_area_ha
            group_stats["mean"] = (
                (group_stats["sum"] / valid_count) if valid_count > 0 else None
            )

        if group_sketch is not None:
            for group_value, sk in group_sketch.items():
                for p in percentile_list:
                    grouped_stats[group_value][
                        f"p{int(p) if float(p).is_integer() else p}"
                    ] = (None if sk.is_empty() else sk.get_quantile(p / 100.0))

        for group_value, group_stats in grouped_stats.items():
            logger.debug(
                "group=%r start total_count=%r nodata_count=%r sum=%r sumsq=%r keys=%r",
                group_value,
                group_stats.get("total_count"),
                group_stats.get("nodata_count"),
                group_stats.get("sum"),
                group_stats.get("sumsq"),
                sorted(group_stats.keys()),
            )

            valid_count = group_stats["total_count"] - group_stats["nodata_count"]
            group_stats["valid_count"] = valid_count
            if raster_pixel_area_ha is not None:
                group_stats["area_ha_total"] = (
                    group_stats["total_count"] * raster_pixel_area_ha
                )
                group_stats["area_ha_valid"] = valid_count * raster_pixel_area_ha
            logger.debug("group=%r computed valid_count=%r", group_value, valid_count)

            if valid_count > 0:
                mean_value = group_stats["sum"] / valid_count
                logger.debug(
                    "group=%r mean_value=%r (sum=%r / valid_count=%r)",
                    group_value,
                    mean_value,
                    group_stats["sum"],
                    valid_count,
                )

                variance_value = (group_stats["sumsq"] / valid_count) - (
                    mean_value * mean_value
                )
                logger.debug(
                    "group=%r raw variance_value=%r (sumsq/valid_count=%r - mean^2=%r)",
                    group_value,
                    variance_value,
                    group_stats["sumsq"] / valid_count,
                    mean_value * mean_value,
                )

                if variance_value < 0:
                    logger.debug(
                        "group=%r variance_value < 0, clamping to 0.0 (was %r)",
                        group_value,
                        variance_value,
                    )
                    variance_value = 0.0

                stdev_value = float(np.sqrt(variance_value))
                group_stats["stdev"] = stdev_value
                logger.debug(
                    "group=%r stdev=%r sqrt(variance_value=%r)",
                    group_value,
                    stdev_value,
                    variance_value,
                )
            else:
                group_stats["stdev"] = None
                logger.debug(
                    "group=%r stdev=None because valid_count <= 0 (total_count=%r nodata_count=%r)",
                    group_value,
                    group_stats["total_count"],
                    group_stats["nodata_count"],
                )

            logger.debug(
                "group=%r deleting sumsq (current sumsq=%r)",
                group_value,
                group_stats.get("sumsq"),
            )
            del group_stats["sumsq"]

            logger.debug(
                "group=%r end valid_count=%r stdev=%r keys_now=%r",
                group_value,
                group_stats.get("valid_count"),
                group_stats.get("stdev"),
                sorted(group_stats.keys()),
            )
        logger.info("grouping done | groups=%d", len(grouped_stats))
        logger.info("fast_zonal_statistics done")
        return dict(grouped_stats)
    finally:
        local_task_graph.close()
        if clean_working_dir:
            logger.info("cleaning zonal statistics cache dir: %s", cache_working_dir)
            shutil.rmtree(cache_working_dir)


def run_vector_stats_job(
    base_vector_path_list,
    base_vector_fields,
    agg_vector,
    agg_layer: str,
    agg_field,
    operations,
    output_csv: Path,
    workdir: Path,
    tag: str,
    job_type: str,
):
    """Run a vector-based statistics job and write aggregated results to CSV.

    For each feature in the base vector datasets, assigns it to the nearest
    aggregated geometry (after dissolving by `agg_field`) and computes summary
    statistics over specified attribute fields. Supported statistics include
    counts, sums, means, standard deviations, minima, maxima, and percentiles.
    Results are aggregated per dissolved geometry and written to a CSV file.

    All base vectors are reprojected to the aggregation CRS if necessary. Nearest
    geometry assignment is accelerated using a spatial index and processed in
    chunks with multithreading.

    Args:
        base_vector_path_list: List of paths to base vector datasets whose features
            will be assigned to aggregation geometries.
        base_vector_fields: List of attribute field names to aggregate from each
            base vector dataset.
        agg_vector: Path to the aggregation vector dataset.
        agg_layer: Name of the layer within `agg_vector` to use.
        agg_field: Attribute field or fields in the aggregation layer used to dissolve
            geometries and define aggregation groups.
        operations: List of operation specifiers (e.g. `"sum"`, `"mean"`,
            `"min"`, `"max"`, `"stdev"`, `"total_count"`, `"p50"`).
        output_csv: Path to the output CSV file to write.
        workdir: Working directory for intermediate job artifacts.
        tag: Job identifier used for logging and column name suffixes.
        job_type: Job type string; must be `"vector"`.

    Raises:
        ValueError: If `job_type` is not `"vector"` or if operation parsing fails.
        IOError: If vector datasets cannot be read or the output cannot be written.
    """
    if job_type != "vector":
        raise ValueError(f"unexpected job type for run_vector_stats_job: {job_type}")
    agg_fields = agg_field

    logger.info("parsing operations for tag=%s", tag)
    normalized_operations = [o.strip().lower() for o in operations if str(o).strip()]
    core_ops = []
    pct_list = []
    for operation in normalized_operations:
        if operation.startswith("p") and len(operation) > 1:
            pct_list.append(float(operation[1:]))
        else:
            core_ops.append(operation)
    core_ops = list(dict.fromkeys(core_ops))
    pct_list = sorted(set(pct_list))
    logger.info(
        "operations parsed for tag=%s core_ops=%s pct_list=%s",
        tag,
        core_ops,
        pct_list,
    )

    logger.info(
        "reading agg vector for tag=%s path=%s layer=%s",
        tag,
        agg_vector,
        agg_layer,
    )
    agg_gdf = gpd.read_file(agg_vector, layer=agg_layer)
    logger.info("agg vector read for tag=%s features=%d", tag, len(agg_gdf))

    agg_crs = CRS.from_user_input(agg_gdf.crs) if agg_gdf.crs else None
    logger.info("agg CRS for tag=%s crs=%s", tag, str(agg_crs) if agg_crs else None)

    logger.info("dissolving agg features for tag=%s by=%s", tag, agg_fields)
    agg_groups = agg_gdf.dissolve(by=agg_fields)
    logger.info("dissolve complete for tag=%s groups=%d", tag, len(agg_groups))

    group_geometries = list(agg_groups.geometry.values)
    group_keys = list(agg_groups.index)
    group_count = len(group_keys)

    logger.info("building STRtree for tag=%s groups=%d", tag, group_count)
    tree = STRtree(group_geometries)
    logger.info("STRtree built for tag=%s", tag)

    logger.info("building geometry-id index map for tag=%s groups=%d", tag, group_count)
    geom_id_to_idx = {
        id(geometry): index for index, geometry in enumerate(group_geometries)
    }
    logger.info("geometry-id index map built for tag=%s", tag)

    transformers_by_stem = {}
    assignments_by_stem = {}
    per_stem_frames = []

    chunk_size = 1_000
    logger.info("chunk_size set for tag=%s chunk_size=%d", tag, chunk_size)

    def _pct_to_suffix(percentile_value: float) -> str:
        return (
            str(int(percentile_value))
            if float(percentile_value).is_integer()
            else str(percentile_value)
        ).replace(".", "_")

    for base_vector_path in base_vector_path_list:
        base_vector_path = Path(base_vector_path)
        stem = base_vector_path.stem

        base_gdf = gpd.read_file(base_vector_path)
        keep_cols = [c for c in base_vector_fields if c in base_gdf.columns]
        base_gdf = base_gdf[keep_cols + ["geometry"]]

        base_crs = CRS.from_user_input(base_gdf.crs) if base_gdf.crs else None

        transformer = None
        if agg_crs and base_crs and agg_crs != base_crs:
            transformer = Transformer.from_crs(base_crs, agg_crs, always_xy=True)
            base_gdf = base_gdf.to_crs(agg_crs)

        transformers_by_stem[stem] = transformer

        feature_ids_all = base_gdf.index.to_numpy()
        geometries_all = base_gdf.geometry.values
        feature_count = len(feature_ids_all)

        nearest_group_index = np.empty(feature_count, dtype=np.int64)

        def _nearest_chunk_thread(args):
            start_index, geometries_chunk = args
            nearest_geometries = np.asarray(tree.nearest(geometries_chunk))
            if nearest_geometries.dtype == object:
                nearest_geometries = np.fromiter(
                    (geom_id_to_idx[id(geometry)] for geometry in nearest_geometries),
                    dtype=np.int64,
                    count=len(nearest_geometries),
                )
            return start_index, nearest_geometries.astype(np.int64, copy=False)

        nearest_tasks = [
            (
                start_index,
                geometries_all[
                    start_index : min(start_index + chunk_size, feature_count)
                ],
            )
            for start_index in range(0, feature_count, chunk_size)
        ]

        with ThreadPoolExecutor() as executor:
            for start_index, nearest_chunk in tqdm(
                executor.map(_nearest_chunk_thread, nearest_tasks, chunksize=1),
                total=len(nearest_tasks),
                desc=f"finding closest geom: {stem}",
            ):
                nearest_group_index[
                    start_index : start_index + len(nearest_chunk)
                ] = nearest_chunk

        order = np.argsort(nearest_group_index, kind="mergesort")
        groups_sorted = nearest_group_index[order]
        features_sorted = feature_ids_all[order]
        unique_groups, start_indices, counts = np.unique(
            groups_sorted, return_index=True, return_counts=True
        )
        assignments_by_stem[stem] = {
            group_keys[int(group_index)]: features_sorted[start : start + count]
            for group_index, start, count in zip(unique_groups, start_indices, counts)
        }

        stem_frame = pd.DataFrame(group_keys, columns=agg_fields)

        if "total_count" in core_ops:
            stem_frame[f"total_count_{stem}"] = np.bincount(
                nearest_group_index, minlength=group_count
            ).astype(np.int64)

        need_sort_per_field = (
            ("min" in core_ops) or ("max" in core_ops) or (len(pct_list) > 0)
        )

        for field in base_vector_fields:
            values_all = base_gdf[field].to_numpy()
            has_value_mask = ~pd.isna(values_all)

            groups_valid = nearest_group_index[has_value_mask]
            values_valid = values_all[has_value_mask].astype(float, copy=False)

            valid_count = None
            sum_values = None
            sum_values_sq = None

            if (
                ("valid_count" in core_ops)
                or ("mean" in core_ops)
                or ("stdev" in core_ops)
                or ("sum" in core_ops)
            ):
                valid_count = np.bincount(groups_valid, minlength=group_count).astype(
                    np.int64
                )

            if ("mean" in core_ops) or ("stdev" in core_ops) or ("sum" in core_ops):
                sum_values = np.bincount(
                    groups_valid, weights=values_valid, minlength=group_count
                ).astype(float, copy=False)

            if "stdev" in core_ops:
                sum_values_sq = np.bincount(
                    groups_valid,
                    weights=values_valid * values_valid,
                    minlength=group_count,
                ).astype(float, copy=False)

            if "valid_count" in core_ops:
                stem_frame[f"valid_count_{field}_{stem}"] = valid_count

            if "sum" in core_ops:
                out = np.full(group_count, np.nan, dtype=float)
                ok = valid_count > 0
                out[ok] = sum_values[ok]
                stem_frame[f"sum_{field}_{stem}"] = out

            if "mean" in core_ops:
                out = np.full(group_count, np.nan, dtype=float)
                ok = valid_count > 0
                out[ok] = sum_values[ok] / valid_count[ok]
                stem_frame[f"mean_{field}_{stem}"] = out

            if "stdev" in core_ops:
                mean = np.full(group_count, np.nan, dtype=float)
                ok = valid_count > 0
                mean[ok] = sum_values[ok] / valid_count[ok]
                mean_sq = np.full(group_count, np.nan, dtype=float)
                mean_sq[ok] = sum_values_sq[ok] / valid_count[ok]
                variance = mean_sq - mean * mean
                variance[variance < 0] = 0
                stem_frame[f"stdev_{field}_{stem}"] = np.sqrt(variance)

            if need_sort_per_field:
                sort_order = np.argsort(groups_valid, kind="mergesort")
                groups_sorted = groups_valid[sort_order]
                values_sorted = values_valid[sort_order]
                unique_groups, start_indices, counts = np.unique(
                    groups_sorted, return_index=True, return_counts=True
                )

                if "min" in core_ops:
                    out = np.full(group_count, np.nan, dtype=float)
                    out[unique_groups] = np.minimum.reduceat(
                        values_sorted, start_indices
                    )
                    stem_frame[f"min_{field}_{stem}"] = out

                if "max" in core_ops:
                    out = np.full(group_count, np.nan, dtype=float)
                    out[unique_groups] = np.maximum.reduceat(
                        values_sorted, start_indices
                    )
                    stem_frame[f"max_{field}_{stem}"] = out

                if pct_list:
                    for percentile_value in pct_list:
                        percentile_suffix = _pct_to_suffix(percentile_value)
                        column_name = f"p{percentile_suffix}_{field}_{stem}"
                        out = np.full(group_count, np.nan, dtype=float)
                        for group_index, start_index, count in zip(
                            unique_groups, start_indices, counts
                        ):
                            out[int(group_index)] = np.percentile(
                                values_sorted[start_index : start_index + count],
                                percentile_value,
                            )
                        stem_frame[column_name] = out

        per_stem_frames.append(stem_frame)

    if per_stem_frames:
        result_table = per_stem_frames[0]
        for stem_frame in per_stem_frames[1:]:
            result_table = result_table.merge(
                stem_frame, on=agg_fields, how="outer", sort=False
            )
    else:
        result_table = pd.DataFrame(columns=agg_fields)

    desired_columns = list(agg_fields)
    per_field_ops = [operation for operation in core_ops if operation != "total_count"]

    for base_vector_path in base_vector_path_list:
        stem = Path(base_vector_path).stem

        if "total_count" in core_ops:
            column_name = f"total_count_{stem}"
            if column_name in result_table.columns:
                desired_columns.append(column_name)

        for field in base_vector_fields:
            for operation in per_field_ops:
                column_name = f"{operation}_{field}_{stem}"
                if column_name in result_table.columns:
                    desired_columns.append(column_name)
            for percentile_value in pct_list:
                percentile_suffix = _pct_to_suffix(percentile_value)
                column_name = f"p{percentile_suffix}_{field}_{stem}"
                if column_name in result_table.columns:
                    desired_columns.append(column_name)

    remaining_columns = [c for c in result_table.columns if c not in desired_columns]
    result_table = result_table[desired_columns + remaining_columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_csv, index=False)


def _crs_label(crs):
    authority = crs.to_authority()
    return f"{authority[0]}:{authority[1]}" if authority else crs.to_string()


def _bounds_to_wgs84(bounds, source_crs):
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    min_x, min_y, max_x, max_y = bounds
    lon_values, lat_values = transformer.transform(
        [min_x, min_x, max_x, max_x],
        [min_y, max_y, min_y, max_y],
    )
    return (
        min(lon_values),
        min(lat_values),
        max(lon_values),
        max(lat_values),
    )


def _linear_units_to_meters(crs):
    if crs.axis_info:
        return crs.axis_info[0].unit_conversion_factor or 1.0
    return 1.0


def _select_measure_crs(agg_gdf, measure_gdf, measure_crs, operation, tag):
    agg_crs = CRS.from_user_input(agg_gdf.crs) if agg_gdf.crs else None
    base_crs = CRS.from_user_input(measure_gdf.crs) if measure_gdf.crs else None
    if agg_crs is None or base_crs is None:
        raise ValueError(
            f"[job:{tag}] agg_vector and base_measure_vector must both define a CRS"
        )

    if measure_crs.lower() != "auto":
        target_crs = CRS.from_user_input(measure_crs)
        if operation != "intersect_count" and not target_crs.is_projected:
            raise ValueError(
                f"[job:{tag}] measure_crs must be projected for {operation}: "
                f"{measure_crs}"
            )
        return target_crs

    if agg_crs == base_crs and agg_crs.is_projected:
        _MEASURE_CRS_ASSUMPTIONS.add(
            f"[job:{tag}] measure_crs=auto used shared projected CRS "
            f"{_crs_label(agg_crs)}."
        )
        return agg_crs

    agg_bounds = _bounds_to_wgs84(agg_gdf.total_bounds, agg_crs)
    base_bounds = _bounds_to_wgs84(measure_gdf.total_bounds, base_crs)
    min_lon = min(agg_bounds[0], base_bounds[0])
    min_lat = min(agg_bounds[1], base_bounds[1])
    max_lon = max(agg_bounds[2], base_bounds[2])
    max_lat = max(agg_bounds[3], base_bounds[3])
    center_lon = (min_lon + max_lon) * 0.5
    center_lat = (min_lat + max_lat) * 0.5
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    if -80.0 <= center_lat <= 84.0 and lon_span <= 6.0 and lat_span <= 10.0:
        zone = int((center_lon + 180.0) // 6.0) + 1
        zone = min(max(zone, 1), 60)
        epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
        target_crs = CRS.from_epsg(epsg)
        reason = (
            f"extent center ({center_lon:.4f}, {center_lat:.4f}) fits UTM zone {zone}"
        )
    else:
        target_crs = CRS.from_epsg(6933)
        reason = (
            "extent is too wide for a single UTM zone or outside UTM latitude bounds; "
            "using global equal-area EPSG:6933"
        )

    _MEASURE_CRS_ASSUMPTIONS.add(
        f"[job:{tag}] measure_crs=auto selected {_crs_label(target_crs)} for "
        f"{operation}; {reason}."
    )
    return target_crs


def _validate_measure_geometry(measure_gdf, operation, vector_path):
    geometry_types = set(
        measure_gdf.geometry.dropna().geom_type.str.replace("3D ", "", regex=False)
    )
    if not geometry_types:
        return
    if operation == "intersect_area_ha":
        allowed_types = {"Polygon", "MultiPolygon"}
    elif operation == "intersect_length_km":
        allowed_types = {"LineString", "MultiLineString", "LinearRing"}
    else:
        allowed_types = {"Point", "MultiPoint"}

    unexpected_types = sorted(geometry_types - allowed_types)
    if unexpected_types:
        raise ValueError(
            f"{operation} cannot measure geometry types {unexpected_types} in "
            f"{vector_path}. Expected one of {sorted(allowed_types)}."
        )


def run_vector_measure_job(
    agg_vector,
    agg_layer,
    agg_fields,
    base_measure_vector,
    base_measure_layer,
    operation,
    measure_crs,
    tag,
):
    """Run a vector-on-vector intersection measure job.

    Args:
        agg_vector: Aggregation vector path.
        agg_layer: Aggregation vector layer name.
        agg_fields: Field names that identify aggregation groups.
        base_measure_vector: Vector path to measure inside aggregation groups.
        base_measure_layer: Layer name in `base_measure_vector`.
        operation: One of `intersect_area_ha`, `intersect_length_km`, or
            `intersect_count`.
        measure_crs: CRS for planar measurement, or `auto`.
        tag: Job tag used in log and error messages.

    Returns:
        DataFrame containing aggregation fields and one measure column.
    """
    logger.info(
        "running vector measure job | tag=%s operation=%s base=%s layer=%s",
        tag,
        operation,
        base_measure_vector,
        base_measure_layer,
    )
    agg_gdf = gpd.read_file(agg_vector, layer=agg_layer)
    measure_gdf = gpd.read_file(base_measure_vector, layer=base_measure_layer)
    measure_gdf = measure_gdf[
        measure_gdf.geometry.notna() & (~measure_gdf.geometry.is_empty)
    ].copy()
    _validate_measure_geometry(measure_gdf, operation, base_measure_vector)

    agg_groups = agg_gdf.dissolve(by=agg_fields, as_index=False)
    agg_groups = agg_groups[agg_fields + ["geometry"]]
    column_name = f"{operation}_{Path(base_measure_vector).stem}"
    result_table = agg_groups[agg_fields].copy()
    result_table[column_name] = 0 if operation == "intersect_count" else 0.0

    if measure_gdf.empty:
        logger.info("base measure vector has no non-empty geometries")
        return result_table

    target_crs = _select_measure_crs(
        agg_groups, measure_gdf, measure_crs, operation, tag
    )
    agg_projected = agg_groups.to_crs(target_crs)
    measure_projected = measure_gdf[["geometry"]].to_crs(target_crs)
    measure_projected = measure_projected.explode(index_parts=False)

    if operation == "intersect_count":
        joined = gpd.sjoin(
            measure_projected,
            agg_projected[agg_fields + ["geometry"]],
            how="inner",
            predicate="intersects",
        )
        if joined.empty:
            return result_table
        measured = (
            joined.groupby(agg_fields, dropna=False)
            .size()
            .rename(column_name)
            .reset_index()
        )
    else:
        intersection_gdf = gpd.overlay(
            measure_projected,
            agg_projected[agg_fields + ["geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        if intersection_gdf.empty:
            return result_table
        linear_units_to_meters = _linear_units_to_meters(target_crs)
        if operation == "intersect_area_ha":
            intersection_gdf[column_name] = (
                intersection_gdf.geometry.area
                * linear_units_to_meters
                * linear_units_to_meters
                / 10000.0
            )
        else:
            intersection_gdf[column_name] = (
                intersection_gdf.geometry.length * linear_units_to_meters / 1000.0
            )
        measured = (
            intersection_gdf.groupby(agg_fields, dropna=False)[column_name]
            .sum()
            .reset_index()
        )

    result_table = result_table.drop(columns=[column_name]).merge(
        measured, on=agg_fields, how="left", sort=False
    )
    result_table[column_name] = result_table[column_name].fillna(0)
    if operation == "intersect_count":
        result_table[column_name] = result_table[column_name].astype(np.int64)
    return result_table


def _write_zonal_outputs(
    result_table: pd.DataFrame,
    agg_vector: Path,
    agg_layer: str,
    agg_fields: list[str],
    output_csv: Path | None,
    output_gpkg: Path | None,
):
    """Write zonal statistics to configured table and vector outputs."""
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_table.to_csv(output_csv, index=False)

    if output_gpkg is None:
        return

    output_gpkg = Path(output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    output_gdf = gpd.read_file(agg_vector, layer=agg_layer)
    result_columns = [
        column_name
        for column_name in result_table.columns
        if column_name not in agg_fields
    ]
    conflicting_columns = [
        column_name
        for column_name in result_columns
        if column_name in output_gdf.columns
    ]
    if conflicting_columns:
        output_gdf = output_gdf.drop(columns=conflicting_columns)

    output_gdf = output_gdf.merge(result_table, on=agg_fields, how="left", sort=False)
    output_gpkg.unlink(missing_ok=True)
    output_gdf.to_file(output_gpkg, layer=agg_layer, driver="GPKG")


def run_zonal_stats_job(
    base_raster_path_list: list[Path],
    base_vector_path_list: list[Path],
    base_vector_fields: list[str],
    base_measure_vector: Path | None,
    base_measure_layer: str | None,
    measure_crs: str,
    agg_vector: Path,
    agg_layer: str,
    agg_field,
    operations: list[str],
    output_csv: Path | None,
    output_gpkg: Path | None,
    workdir: Path,
    tag: str,
    task_graph,
):
    """Run a zonal statistics job over raster and/or vector base datasets.

    Computes statistics for one or more base rasters and/or base vector datasets
    using geometries from an aggregation vector layer. Raster zonal statistics are
    executed via a task graph using `fast_zonal_statistics`, while vector-based
    statistics are delegated to `run_vector_stats_job`. Results from all inputs
    are merged on the aggregation field and written to the configured outputs.

    Both raster- and vector-derived statistics support core operations
    (e.g. count, sum, mean) and percentile operations (e.g. `p50`). All paths,
    layers, and fields are assumed to be validated upstream.

    Args:
        base_raster_path_list: List of raster paths on which to compute zonal
            statistics. May be empty.
        base_vector_path_list: List of vector paths on which to compute nearest-
            geometry vector statistics. May be empty.
        base_vector_fields: Attribute field names to aggregate from base vectors.
        base_measure_vector: Optional vector path whose intersections are measured.
        base_measure_layer: Optional layer name in `base_measure_vector`.
        measure_crs: CRS for vector intersection measurement, or `auto`.
        agg_vector: Path to the aggregation vector dataset.
        agg_layer: Name of the layer within `agg_vector` to use for aggregation.
        agg_field: Attribute field or fields defining aggregation zones/groups.
        operations: List of operation specifiers (e.g. `"mean"`, `"sum"`, `"p90"`).
        output_csv: Optional path to the output CSV file to write.
        output_gpkg: Optional path to the output GeoPackage file to write.
        workdir: Working directory for intermediate files and task graph outputs.
        tag: Job identifier used for temporary filenames and task labeling.
        task_graph: Task graph instance used to schedule raster and vector jobs.

    Raises:
        ValueError: If operation parsing fails or required inputs are inconsistent.
        IOError: If intermediate or output files cannot be read or written.
    """
    agg_fields = agg_field
    ops = [o.strip().lower() for o in operations if str(o).strip()]
    core_ops = []
    pct_list = []
    for op in ops:
        if op.startswith("p") and len(op) > 1:
            pct_list.append(float(op[1:]))
        else:
            core_ops.append(op)
    core_ops = list(dict.fromkeys(core_ops))
    pct_list = sorted(set(pct_list))
    pct_keys = [f"p{int(p) if float(p).is_integer() else p}" for p in pct_list]

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if base_measure_vector is not None:
        combined_dataframe = run_vector_measure_job(
            agg_vector=agg_vector,
            agg_layer=agg_layer,
            agg_fields=agg_fields,
            base_measure_vector=base_measure_vector,
            base_measure_layer=base_measure_layer,
            operation=core_ops[0],
            measure_crs=measure_crs,
            tag=tag,
        )
        _write_zonal_outputs(
            combined_dataframe,
            agg_vector,
            agg_layer,
            agg_fields,
            output_csv,
            output_gpkg,
        )
        return

    grouped_stats_list = []
    if base_raster_path_list:
        raster_rows = []
        for base_raster_path in base_raster_path_list:
            base_raster_path = Path(base_raster_path)

            grouped_stats_task = task_graph.add_task(
                func=fast_zonal_statistics,
                args=((base_raster_path, 1), agg_vector, agg_fields),
                kwargs={
                    "aggregate_layer_name": agg_layer,
                    "ignore_nodata": True,
                    "working_dir": workdir,
                    "clean_working_dir": False,
                    "percentile_list": pct_list,
                    "calculate_area_ha": bool(
                        AREA_HECTARE_OPERATIONS.intersection(core_ops)
                    ),
                },
                store_result=True,
                task_name=(
                    f"fast_zonal_statistics for {Path(base_raster_path).stem}, "
                    f"{Path(agg_vector).stem} {','.join(agg_fields)}"
                ),
            )
            grouped_stats_list.append(
                (base_raster_path.stem, agg_fields, grouped_stats_task)
            )

    combined_dataframe = None

    if base_vector_path_list:
        vector_tmp_csv = workdir / f"{tag}__vector_stats.csv"

        vector_task = task_graph.add_task(
            func=run_vector_stats_job,
            kwargs={
                "base_vector_path_list": base_vector_path_list,
                "base_vector_fields": base_vector_fields,
                "agg_vector": agg_vector,
                "agg_layer": agg_layer,
                "agg_field": agg_fields,
                "operations": operations,
                "output_csv": vector_tmp_csv,
                "workdir": workdir,
                "tag": tag,
                "job_type": "vector",
            },
            task_name=f"vector stats for {tag}",
            target_path_list=[vector_tmp_csv],
        )
        vector_task.join()

        vector_dataframe = pd.read_csv(vector_tmp_csv)
        if "base_vector" in vector_dataframe.columns:
            vector_dataframe = vector_dataframe.rename(columns={"base_vector": "base"})

        combined_dataframe = vector_dataframe

    raster_dataframes = []

    for raster_stem, aggregation_field_names, group_task in grouped_stats_list:
        grouped_stats = group_task.get()

        raster_rows = []
        for group_value, statistics in grouped_stats.items():
            if len(aggregation_field_names) == 1:
                row = {aggregation_field_names[0]: group_value}
            else:
                row = dict(zip(aggregation_field_names, group_value))
            for operation in core_ops:
                row[f"{operation}_{raster_stem}"] = statistics.get(operation)
            for percentile_key in pct_keys:
                row[f"{percentile_key}_{raster_stem}"] = statistics.get(percentile_key)
            raster_rows.append(row)

        raster_dataframes.append(pd.DataFrame(raster_rows))

    raster_dataframe = None
    for raster_frame in raster_dataframes:
        raster_dataframe = (
            raster_frame
            if raster_dataframe is None
            else raster_dataframe.merge(raster_frame, on=agg_fields, how="outer")
        )

    if combined_dataframe is None:
        combined_dataframe = raster_dataframe
    elif raster_dataframe is not None:
        combined_dataframe = combined_dataframe.merge(
            raster_dataframe, on=agg_fields, how="outer"
        )

    if combined_dataframe is None:
        combined_dataframe = pd.DataFrame(columns=agg_fields)

    _write_zonal_outputs(
        combined_dataframe,
        agg_vector,
        agg_layer,
        agg_fields,
        output_csv,
        output_gpkg,
    )


def _invoke_timed_callback(reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Args:
        reference_time (float): time to base ``callback_period`` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and ``reference_time`` has exceeded
            ``callback_period``.
        callback_period (float): time in seconds to pass until
            ``callback_lambda`` is invoked.

    Return:
        ``reference_time`` if ``callback_lambda`` not invoked, otherwise the
        time when ``callback_lambda`` was invoked.

    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time


def main():
    """CLI entrypoint for validating zonal-stats runner configurations.

    Parses one or more positional arguments pointing to INI configuration files,
    validates each via `parse_and_validate_config`, configures logging based on
    each config's `[project].log_level` setting, and runs all jobs across all
    provided configs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs", nargs="+", help="Path(s) to INI configuration file(s)"
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
    )
    task_graph = taskgraph.TaskGraph(Path.cwd(), os.cpu_count() // 2 + 1, 15.0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread_list = []
    thread_error_list = []
    total_job_count = 0

    def _run_zonal_stats_job_thread(output_label, job_kwargs):
        try:
            run_zonal_stats_job(**job_kwargs)
        except Exception as error:
            logger.exception("job failed for output %s", output_label)
            thread_error_list.append((output_label, error))

    for config_path in args.configs:
        cfg_path = Path(config_path)
        cfg = parse_and_validate_config(cfg_path)

        for job in cfg["job_list"]:
            logger.info(
                "Validated job:%s (operations=%s)",
                job["tag"],
                ",".join(job["operations"]),
            )
            output_path_list = []
            for output_key in ("output_csv", "output_gpkg"):
                if job[output_key] is None:
                    continue
                output_path = Path(job[output_key])
                output_path_timestamped = output_path.with_name(
                    f"{output_path.stem}_{timestamp}{output_path.suffix}"
                )
                job[output_key] = output_path_timestamped
                output_path_list.append(output_path_timestamped)
            job["task_graph"] = task_graph
            output_label = ", ".join(
                str(output_path) for output_path in output_path_list
            )

            thread = Thread(
                target=_run_zonal_stats_job_thread,
                args=(output_label, job),
            )
            thread.start()
            thread_list.append((output_label, thread))
            total_job_count += 1
    logger.info(f"******** running {total_job_count} jobs")

    for output_label, thread in thread_list:
        thread.join()
        if not any(error_path == output_label for error_path, _ in thread_error_list):
            logger.info(f"********* {output_label} is complete!")

    if thread_error_list:
        task_graph.close()
        failed_outputs = ", ".join(str(path) for path, _ in thread_error_list)
        raise RuntimeError(f"zonal statistics jobs failed: {failed_outputs}") from (
            thread_error_list[0][1]
        )

    task_graph.join()
    task_graph.close()

    logging.getLogger(__name__).info("All %d jobs done", total_job_count)
    _log_area_hectare_assumptions()
    for message in sorted(_MEASURE_CRS_ASSUMPTIONS):
        logger.warning("Vector measure CRS choice: %s", message)


if __name__ == "__main__":
    main()
