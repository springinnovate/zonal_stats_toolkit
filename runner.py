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
from pyproj import CRS, Transformer
from shapely.strtree import STRtree
import pandas as pd
import fiona
import numpy as np
import geopandas as gpd

logging.getLogger("ecoshard").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_LOGGING_PERIOD = 10.0
VALID_OPERATIONS = {
    "mean",
    "stdev",
    "min",
    "max",
    "sum",
    "total_count",
    "valid_count",
}


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

    global_output_dir = Path(config["project"]["global_output_dir"].strip())
    if not global_output_dir.is_absolute():
        global_output_dir = cfg_dir / global_output_dir

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

    job_list = []
    for tag, job in jobs_sections:
        agg_vector_raw = job.get("agg_vector", "").strip()
        if not agg_vector_raw:
            raise ValueError(f"[job:{tag}] missing agg_vector")
        agg_vector = _abs_from_cfg_dir(agg_vector_raw)
        if not agg_vector.exists():
            raise FileNotFoundError(f"[job:{tag}] agg_vector not found: {agg_vector}")

        agg_field = job.get("agg_field", "").strip()
        if not agg_field:
            raise ValueError(f"[job:{tag}] missing agg_field")

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

        layers = fiona.listlayers(str(agg_vector))

        agg_layer = job.get("agg_layer", "").strip()
        if not agg_layer:
            if not layers:
                raise ValueError(f"[job:{tag}] no layers found in {agg_vector}")
            agg_layer = layers[0]

        if agg_layer not in layers:
            raise ValueError(
                f'[job:{tag}] agg_layer "{agg_layer}" not found in {agg_vector}. '
                f"Available layers: {layers}"
            )

        with fiona.open(str(agg_vector), layer=agg_layer) as src:
            props = src.schema.get("properties", {})
            if agg_field not in props:
                raise ValueError(
                    f'[job:{tag}] agg_field "{agg_field}" not found in layer "{agg_layer}" of {agg_vector}. '
                    f"Available fields: {sorted(props.keys())}"
                )

        row_col_order_raw = job.get("row_col_order", "").strip()
        if not row_col_order_raw:
            raise ValueError(f"[job:{tag}] missing row_col_order")
        row_col_order_parts = [
            p.strip() for p in row_col_order_raw.split(",") if p.strip()
        ]
        if len(row_col_order_parts) != 2:
            raise ValueError(f"[job:{tag}] row_col_order must have exactly 2 entries")
        other_tokens = {"base", "base_raster", "base_vector"}
        if "agg_field" not in row_col_order_parts:
            raise ValueError(f"[job:{tag}] row_col_order must include agg_field")
        other = [t for t in row_col_order_parts if t != "agg_field"]
        if len(other) != 1 or other[0] not in other_tokens:
            raise ValueError(
                f"[job:{tag}] row_col_order must be a permutation of agg_field and one of "
                f"{sorted(other_tokens)}. Got: {row_col_order_raw}"
            )
        row_col_order = ",".join(row_col_order_parts)

        outdir = global_output_dir
        workdir = global_work_dir / Path(tag)
        outdir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)

        base_raster_path_list = []
        base_vector_path_list = []
        base_vector_fields = []

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
                layers = fiona.listlayers(str(base_vector_path))
                if not layers:
                    raise ValueError(
                        f"[job:{tag}] no layers found in {base_vector_path}"
                    )
                layer = layers[0]
                with fiona.open(str(base_vector_path), layer=layer) as src:
                    props = src.schema.get("properties", {})
                    missing = [f for f in base_vector_fields if f not in props]
                    if missing:
                        raise ValueError(
                            f'[job:{tag}] missing fields {missing} in layer "{layer}" of {base_vector_path}. '
                            f"Available fields: {sorted(props.keys())}"
                        )

        if (not base_raster_path_list) and (not base_vector_path_list):
            raise ValueError(
                f"[job:{tag}] must define at least one of base_raster_pattern or base_vector_pattern"
            )

        job_list.append(
            {
                "tag": tag,
                "agg_vector": agg_vector,
                "agg_layer": agg_layer,
                "agg_field": agg_field,
                "operations": operations,
                "row_col_order": row_col_order,
                "workdir": workdir,
                "output_csv": outdir / f"{tag}.csv",
                "base_raster_path_list": base_raster_path_list,
                "base_vector_path_list": base_vector_path_list,
                "base_vector_fields": base_vector_fields,
                "task_graph": None,
            }
        )

    return {
        "project": {
            "name": project_name,
            "global_work_dir": global_work_dir,
            "global_output_dir": global_output_dir,
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
    clean_working_dir=True,
    percentile_list=None,
):
    raster_path, raster_band_index = base_raster_path_band

    logger.info(
        "fast_zonal_statistics start | raster=%s band=%s | vector=%s layer=%s field=%s | ignore_nodata=%s | working_dir=%s clean=%s | percentiles=%s",
        raster_path,
        raster_band_index,
        str(aggregate_vector_path),
        aggregate_layer_name,
        aggregate_vector_field,
        ignore_nodata,
        working_dir,
        clean_working_dir,
        percentile_list,
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

    def _open_vector_layer(vector_path, layer_name, vector_label, writable=False):
        open_flags = gdal.OF_VECTOR | (gdal.OF_UPDATE if writable else 0)
        vector_dataset = gdal.OpenEx(str(vector_path), open_flags)
        if vector_dataset is None:
            raise RuntimeError(f"Could not open {vector_label} vector at {vector_path}")

        if layer_name is not None:
            logger.info("selecting %s layer by name: %s", vector_label, layer_name)
            vector_layer = vector_dataset.GetLayerByName(layer_name)
        else:
            logger.info("selecting default %s layer", vector_label)
            vector_layer = vector_dataset.GetLayer()

        if vector_layer is None:
            raise RuntimeError(
                f"Could not open layer {layer_name} on {vector_label} vector {vector_path}"
            )

        return vector_dataset, vector_layer

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

    source_vector, source_layer = _open_vector_layer(
        aggregate_vector_path, aggregate_layer_name, "source"
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

    temp_working_dir = tempfile.mkdtemp(dir=working_dir)
    projected_vector_path = os.path.join(temp_working_dir, "projected_vector.gpkg")
    logger.info("created temp working dir: %s", temp_working_dir)

    def _raster_nodata_mask(value_array):
        if raster_nodata is None:
            return np.zeros(value_array.shape, dtype=bool)
        return np.isclose(value_array, raster_nodata)

    try:
        vector_translate_kwargs = {
            "simplifyTolerance": simplify_tolerance,
            "format": "GPKG",
        }
        if needs_reproject:
            vector_translate_kwargs["dstSRS"] = raster_info["projection_wkt"]

        logger.info(
            "vector translate start | output=%s | simplifyTolerance=%s | reproject=%s",
            projected_vector_path,
            simplify_tolerance,
            needs_reproject,
        )
        gdal.VectorTranslate(
            projected_vector_path,
            str(aggregate_vector_path),
            **vector_translate_kwargs,
        )
        logger.info("vector translate done | output=%s", projected_vector_path)

        source_layer = None

        aggregate_vector, aggregate_layer = _open_vector_layer(
            projected_vector_path,
            aggregate_layer_name,
            "projected",
            writable=True,
        )

        logger.info(
            "scanning vector for grouping field values: %s",
            aggregate_vector_field,
        )
        feature_id_set = set()
        feature_id_to_group_value = {}
        unique_group_values = set()

        aggregate_layer.ResetReading()
        for feature in aggregate_layer:
            feature_id = feature.GetFID()
            group_value = feature.GetField(aggregate_vector_field)
            feature_id_set.add(feature_id)
            feature_id_to_group_value[feature_id] = group_value
            unique_group_values.add(group_value)
        aggregate_layer.ResetReading()

        logger.info(
            "vector scan done | features=%d | unique %s=%d",
            len(feature_id_set),
            aggregate_vector_field,
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

        # we need to put an 'fid' code into the vector because otherwise we are
        # not guarnateed to have one
        local_aggregate_field_name = "original_fid"
        if aggregate_layer.FindFieldIndex(local_aggregate_field_name, 1) == -1:
            aggregate_layer.CreateField(
                ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger)
            )

        aggregate_layer.ResetReading()
        aggregate_layer.StartTransaction()
        for feat in aggregate_layer:
            feat.SetField(local_aggregate_field_name, feat.GetFID())
            aggregate_layer.SetFeature(feat)
        aggregate_layer.CommitTransaction()
        aggregate_layer.ResetReading()

        local_aggregate_field_name = "original_fid"
        rasterize_layer_args = {
            "options": [
                "ALL_TOUCHED=FALSE",
                f"ATTRIBUTE={local_aggregate_field_name}",
            ]
        }

        logger.info(
            "disjoint sets ready total_features=%d",
            len(feature_id_set),
        )

        feature_stats_by_id = collections.defaultdict(
            lambda: dict(feature_stats_template)
        )

        feature_id_raster_path = os.path.join(temp_working_dir, "agg_fid.tif")
        feature_id_raster_nodata = -1
        logger.info("creating agg fid raster: %s", feature_id_raster_path)
        geoprocessing.new_raster_from_base(
            raster_path_for_stats,
            feature_id_raster_path,
            gdal.GDT_Int32,
            [feature_id_raster_nodata],
        )

        feature_id_raster_offsets = list(
            geoprocessing.iterblocks(
                (feature_id_raster_path, 1),
                offset_only=True,
                largest_block=2**28,
            )
        )
        logger.info(
            "iterblocks prepared | blocks=%d",
            len(feature_id_raster_offsets),
        )

        feature_id_raster_dataset = gdal.OpenEx(
            feature_id_raster_path, gdal.GA_Update | gdal.OF_RASTER
        )
        feature_id_raster_band = feature_id_raster_dataset.GetRasterBand(1)

        logger.info("populating disjoint layer features (transaction start)")
        logger.info("populating disjoint layer features done (transaction commit)")

        rasterize_callback_message = "rasterizing polygons %.1f%% complete %s"
        rasterize_callback = _make_logger_callback(rasterize_callback_message)

        logger.info("rasterize start")
        aggregate_layer.ResetReading()
        gdal.RasterizeLayer(
            feature_id_raster_dataset,
            [1],
            aggregate_layer,
            callback=rasterize_callback,
            **rasterize_layer_args,
        )
        feature_id_raster_dataset.FlushCache()
        logger.info("rasterize done")

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

        logger.info("grouping fid stats -> %s values", aggregate_vector_field)
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
            valid_count = group_stats["total_count"] - group_stats["nodata_count"]
            group_stats["valid_count"] = valid_count
            if valid_count > 0:
                mean_value = group_stats["sum"] / valid_count
                variance_value = (
                    group_stats["sumsq"] / valid_count - mean_value * mean_value
                )
                if variance_value < 0:
                    variance_value = 0.0
                group_stats["stdev"] = float(np.sqrt(variance_value))
            else:
                group_stats["stdev"] = None
            del group_stats["sumsq"]

        logger.info("grouping done | groups=%d", len(grouped_stats))
        logger.info("fast_zonal_statistics done")
        return dict(grouped_stats)
    finally:
        if clean_working_dir:
            logger.info("cleaning temp working dir: %s", temp_working_dir)
            shutil.rmtree(temp_working_dir)


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
    row_col_order: str,
    job_type: str,
):
    if job_type != "vector":
        raise ValueError(f"unexpected job type for run_vector_stats_job: {job_type}")

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

    logger.info("dissolving agg features for tag=%s by=%s", tag, agg_field)
    agg_groups = agg_gdf.dissolve(by=agg_field)
    logger.info("dissolve complete for tag=%s groups=%d", tag, len(agg_groups))

    group_geometries = list(agg_groups.geometry.values)
    group_keys = list(agg_groups.index)
    group_keys_arr = np.asarray(group_keys, dtype=object)
    group_count = len(group_keys_arr)

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
            group_keys_arr[int(group_index)]: features_sorted[start : start + count]
            for group_index, start, count in zip(unique_groups, start_indices, counts)
        }

        stem_frame = pd.DataFrame({agg_field: group_keys_arr})

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
                stem_frame, on=agg_field, how="outer", sort=False
            )
    else:
        result_table = pd.DataFrame(columns=[agg_field])

    desired_columns = [agg_field]
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


def run_zonal_stats_job(
    base_raster_path_list: list[Path],
    base_vector_path_list: list[Path],
    base_vector_fields: list[str],
    agg_vector: Path,
    agg_layer: str,
    agg_field: str,
    operations: list[str],
    output_csv: Path,
    workdir: Path,
    tag: str,
    row_col_order: str,
    task_graph,
):
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

    grouped_stats_list = []
    if base_raster_path_list:
        raster_rows = []
        for base_raster_path in base_raster_path_list:
            base_raster_path = Path(base_raster_path)

            grouped_stats_task = task_graph.add_task(
                func=fast_zonal_statistics,
                args=((base_raster_path, 1), agg_vector, agg_field),
                kwargs={
                    "aggregate_layer_name": agg_layer,
                    "ignore_nodata": True,
                    "working_dir": workdir,
                    "clean_working_dir": True,
                    "percentile_list": pct_list,
                },
                store_result=True,
            )
            grouped_stats_list.append(
                (base_raster_path.stem, agg_field, grouped_stats_task)
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
                "agg_field": agg_field,
                "operations": operations,
                "output_csv": vector_tmp_csv,
                "workdir": workdir,
                "tag": tag,
                "row_col_order": "agg_field,base_vector",
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

    for raster_stem, aggregation_field_name, group_task in grouped_stats_list:
        grouped_stats = group_task.get()

        raster_rows = []
        for group_value, statistics in grouped_stats.items():
            row = {aggregation_field_name: group_value}
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
            else raster_dataframe.merge(raster_frame, on=agg_field, how="outer")
        )

    if combined_dataframe is None:
        combined_dataframe = raster_dataframe
    elif raster_dataframe is not None:
        combined_dataframe = combined_dataframe.merge(
            raster_dataframe, on=agg_field, how="outer"
        )

    if combined_dataframe is None:
        combined_dataframe = pd.DataFrame(columns=[agg_field])

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_dataframe.to_csv(output_csv, index=False)


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
    """CLI entrypoint for validating a zonal-stats runner configuration.

    Parses a single positional argument pointing to an INI configuration file,
    validates it via `parse_and_validate_config`, configures logging based on the
    `[project].log_level` setting, and logs a validation summary for each job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to INI configuration file")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
    )
    cfg = parse_and_validate_config(cfg_path)

    log_level = getattr(logging, cfg["project"]["log_level"])
    logger = logging.getLogger(cfg["project"]["name"])
    logger.setLevel(log_level)
    logger.info("Loaded config %s", str(cfg_path))
    task_graph = taskgraph.TaskGraph(
        cfg["project"]["global_work_dir"], os.cpu_count() // 2 + 1, 15.0
    )
    # task_graph = taskgraph.TaskGraph(cfg["project"]["global_work_dir"], -1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    thread_list = []

    for job in cfg["job_list"]:
        logger.info(
            "Validated job:%s (operations=%s)",
            job["tag"],
            ",".join(job["operations"]),
        )
        output_path = job["output_csv"]
        output_path_timestamped = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        job["output_csv"] = output_path_timestamped
        job["task_graph"] = task_graph

        thread = Thread(target=run_zonal_stats_job, kwargs=job)
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()
    logger.info(f"All {len(cfg['job_list'])} jobs done")
    task_graph.join()
    task_graph.close()


if __name__ == "__main__":
    main()
