from __future__ import annotations
from threading import Thread
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import collections
import argparse
import configparser
import csv
import logging
import math
import os
import shutil
import tempfile
import time

from ecoshard import taskgraph, geoprocessing
from osgeo import gdal, ogr, osr
import fiona
import numpy as np
from datasketches import kll_floats_sketch


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
    """Parse and validate a zonal-stats runner INI configuration file.

    The configuration file is expected to contain a `[project]` section and one or
    more `[job:<tag>]` sections. This function validates required keys, enforces
    naming constraints, checks file existence, verifies vector layer/field
    presence, and validates the `operations` list against `VALID_OPERATIONS`.

    Validation rules:
      1) `[project].name` must equal the configuration file stem.
      2) `[project].log_level` must be a valid `logging` level name.
      3) Job section tags (`job:<tag>`) must be unique.
      4) For each job, `agg_vector` must exist; `base_raster` must exist if set.
      5) `agg_layer` must exist in `agg_vector`, and `agg_field` must exist in
         that layer schema.
      6) `operations` must be present and all entries must be in
         `VALID_OPERATIONS`.

    The returned dictionary contains:
      - `project`: global configuration values.
      - `job_list`: a list of per-job dictionaries.

    Args:
        cfg_path: Path to the INI configuration file.

    Returns:
        A dictionary with keys:
          - `project`: Dict containing `name`, `global_work_dir`, and `log_level`.
          - `job_list`: List of dicts describing each job. Each job dict includes
            `tag`, `agg_vector`, `agg_layer`, `agg_field`, `base_raster`,
            `workdir`, `output_csv`, and `operations`.

    Raises:
        ValueError: If required sections/keys are missing, if values are invalid,
            if job tags are duplicated, if a layer/field is missing, or if any
            operation is not recognized.
        FileNotFoundError: If `agg_vector` does not exist, or if `base_raster` is
            provided but does not exist.
    """
    stem = cfg_path.stem

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
    global_output_dir = Path(config["project"]["global_output_dir"].strip())

    job_tags = []
    jobs_sections = []
    for section in config.sections():
        if section.startswith("job:"):
            tag = section.split(":", 1)[1].strip()
            if not tag:
                raise ValueError(f"Invalid job section name: [{section}]")
            job_tags.append(tag)
            jobs_sections.append((tag, config[section]))

    if len(job_tags) != len(set(job_tags)):
        seen = set()
        dups = []
        for t in job_tags:
            if t in seen:
                dups.append(t)
            seen.add(t)
        raise ValueError(f"Duplicate job tags found: {sorted(set(dups))}")

    job_list = []
    for tag, job in jobs_sections:
        agg_vector = Path(job.get("agg_vector", "").strip())
        if not agg_vector:
            raise ValueError(f"[job:{tag}] missing agg_vector")
        if not agg_vector.exists():
            raise FileNotFoundError(
                f"[job:{tag}] agg_vector not found: {agg_vector}"
            )

        base_raster_pattern = job.get("base_raster_pattern", "").strip()
        if base_raster_pattern in [None, ""]:
            raise FileNotFoundError(
                f"[job:{tag}] base_raster_pattern tag not found"
            )
        base_raster_path_list = [
            path
            for pattern in base_raster_pattern.split(",")
            if pattern.strip()
            for path in Path(".").glob(pattern.strip())
        ]
        if not base_raster_path_list:
            raise FileNotFoundError(
                f"[job:{tag}] no files found at {base_raster_pattern}"
            )

        agg_field = job.get("agg_field", "").strip()
        if not agg_field:
            raise ValueError(f"[job:{tag}] missing agg_field")

        ops_raw = job.get("operations", "").strip()
        if not ops_raw:
            raise ValueError(f"[job:{tag}] missing operations")
        operations = [
            o.strip().lower() for o in ops_raw.split(",") if o.strip()
        ]
        if not operations:
            raise ValueError(f"[job:{tag}] operations is empty")

        invalid_ops = sorted(set(operations) - VALID_OPERATIONS)
        # allow any p, but all others need to match
        if any([op for op in invalid_ops if not op.startswith("p")]):
            raise ValueError(
                f"[job:{tag}] invalid operations: {invalid_ops}. "
                f"Valid operations: {sorted(VALID_OPERATIONS)}"
            )

        layers = fiona.listlayers(str(agg_vector))

        agg_layer = job.get("agg_layer", "").strip()
        if agg_layer is None or not str(agg_layer).strip():
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
        outdir = global_output_dir
        workdir = global_work_dir / Path(tag)
        outdir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)
        job_list.append(
            {
                "tag": tag,
                "agg_vector": agg_vector,
                "agg_layer": agg_layer,
                "agg_field": agg_field,
                "base_raster_path_list": base_raster_path_list,
                "operations": operations,
                "row_col_order": job["row_col_order"],
                "workdir": workdir,
                "output_csv": outdir / f"{tag}.csv",
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

    def _open_vector_layer(
        vector_path, layer_name, vector_label, writable=False
    ):
        open_flags = gdal.OF_VECTOR | (gdal.OF_UPDATE if writable else 0)
        vector_dataset = gdal.OpenEx(str(vector_path), open_flags)
        if vector_dataset is None:
            raise RuntimeError(
                f"Could not open {vector_label} vector at {vector_path}"
            )

        if layer_name is not None:
            logger.info(
                "selecting %s layer by name: %s", vector_label, layer_name
            )
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
        logger.info(
            "vector SRS missing/unknown | forcing reprojection to raster SRS"
        )

    temp_working_dir = tempfile.mkdtemp(dir=working_dir)
    projected_vector_path = os.path.join(
        temp_working_dir, "projected_vector.gpkg"
    )
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
        raster_min_x, raster_min_y, raster_max_x, raster_max_y = (
            raster_bounding_box
        )
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
        logger.info(
            "populating disjoint layer features done (transaction commit)"
        )

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
        for block_index, feature_id_offset in enumerate(
            feature_id_raster_offsets
        ):
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

            feature_id_block = feature_id_raster_band.ReadAsArray(
                **feature_id_offset
            )
            raster_value_block = raster_band.ReadAsArray(**feature_id_offset)

            in_polygon_mask = feature_id_block != feature_id_raster_nodata
            if not np.any(in_polygon_mask):
                continue

            block_feature_ids = feature_id_block[in_polygon_mask]
            block_raster_values = raster_value_block[in_polygon_mask]

            for feature_id in np.unique(block_feature_ids):
                feature_values = block_raster_values[
                    block_feature_ids == feature_id
                ]
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
                    sk.update(
                        feature_values.astype(np.float32, copy=False).ravel()
                    )

                block_min_value = np.min(feature_values)
                block_max_value = np.max(feature_values)
                if feature_stats["min"] is None:
                    feature_stats["min"] = block_min_value
                    feature_stats["max"] = block_max_value
                else:
                    feature_stats["min"] = min(
                        feature_stats["min"], block_min_value
                    )
                    feature_stats["max"] = max(
                        feature_stats["max"], block_max_value
                    )

                feature_stats["sum"] += np.sum(feature_values)
                feature_stats["sumsq"] += np.sum(
                    feature_values * feature_values, dtype=np.float64
                )

        logger.info("aggregating done")

        feature_id_raster_band = None
        feature_id_raster_dataset = None

        remaining_unset_feature_ids = feature_id_set.difference(
            feature_stats_by_id
        )
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
                    group_stats["min"] = min(
                        group_stats["min"], feature_stats["min"]
                    )
                    group_stats["max"] = max(
                        group_stats["max"], feature_stats["max"]
                    )

        for group_value, group_stats in grouped_stats.items():
            valid_count = (
                group_stats["total_count"] - group_stats["nodata_count"]
            )
            group_stats["valid_count"] = valid_count
            group_stats["mean"] = (
                (group_stats["sum"] / valid_count) if valid_count > 0 else None
            )

        if group_sketch is not None:
            for group_value, sk in group_sketch.items():
                for p in percentile_list:
                    grouped_stats[group_value][
                        f"p{int(p) if float(p).is_integer() else p}"
                    ] = sk.get_quantile(p / 100.0)

        for group_value, group_stats in grouped_stats.items():
            valid_count = (
                group_stats["total_count"] - group_stats["nodata_count"]
            )
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


def run_zonal_stats_job(
    base_raster_path_list: list[Path],
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
    raster_stems = []
    raster_stats_by_stem = {}
    all_groups = set()

    percentile_list = [
        float(op[1:])
        for op in operations
        if op.startswith("p") and op[1:].replace(".", "", 1).isdigit()
    ]
    for raster_path in base_raster_path_list:
        stem = raster_path.stem
        raster_stems.append(stem)
        stats_task = task_graph.add_task(
            func=fast_zonal_statistics,
            args=(
                (str(raster_path), 1),
                str(agg_vector),
                agg_field,
            ),
            kwargs={
                "aggregate_layer_name": agg_layer,
                "ignore_nodata": True,
                "working_dir": str(workdir),
                "clean_working_dir": False,
                "percentile_list": percentile_list,
            },
            store_result=True,
            task_name=f"stats for {tag}",
        )
        stats = stats_task.get()
        raster_stats_by_stem[stem] = stats
        all_groups.update(stats.keys())

    parts = [p.strip() for p in row_col_order.split(",") if p.strip()]
    if parts == ["agg_field", "base_raster"]:
        first_col = agg_field
        columns = [
            f"{field}_{stem}" for stem in raster_stems for field in operations
        ]

        def row_iter():
            for group_value in sorted(
                all_groups, key=lambda v: (v is None, str(v))
            ):
                row = {
                    first_col: "" if group_value is None else str(group_value)
                }
                for stem in raster_stems:
                    s = raster_stats_by_stem[stem][group_value]
                    for field in operations:
                        row[f"{field}_{stem}"] = s[field]
                yield row

    elif parts == ["base_raster", "agg_field"]:
        first_col = "base_raster"
        columns = [
            f'{field}_{"" if gv is None else str(gv)}'
            for gv in sorted(all_groups, key=lambda v: (v is None, str(v)))
            for field in operations
        ]

        ordered_groups = sorted(all_groups, key=lambda v: (v is None, str(v)))

        def row_iter():
            for stem in raster_stems:
                row = {first_col: stem}
                stats = raster_stats_by_stem[stem]
                for group_value in ordered_groups:
                    s = stats[group_value]
                    group_label = (
                        "" if group_value is None else str(group_value)
                    )
                    for field in operations:
                        row[f"{field}_{group_label}"] = s[field]
                yield row

    else:
        raise ValueError(
            "row_col_order must be 'agg_field,base_raster' or 'base_raster,agg_field'"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[first_col] + columns)
        writer.writeheader()
        writer.writerows(row_iter())


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


def run_fast_zonal_statistics_test_case(
    *,
    fast_zonal_statistics_fn,
    raster_values,
    polygons,
    aggregate_vector_field="group_id",
    aggregate_layer_name="polys",
    ignore_nodata=True,
    percentile_list=None,
    raster_nodata=None,
    pixel_size=10.0,
    origin_x=1000.0,
    origin_y=2000.0,
    epsg=3857,
    allclose_rtol=1e-6,
    allclose_atol=1e-6,
    expect_grouped_stats=None,
    compare_keys=None,
    working_dir=None,
    clean_working_dir=True,
):
    raster_values = np.asarray(raster_values)
    if raster_values.ndim != 2:
        raise ValueError("raster_values must be 2D")

    percentile_list = [] if percentile_list is None else list(percentile_list)
    percentile_list = sorted(
        {float(percentile_value) for percentile_value in percentile_list}
    )
    percentile_keys = [
        f"p{int(percentile_value) if percentile_value.is_integer() else percentile_value}"
        for percentile_value in percentile_list
    ]

    if compare_keys is None:
        compare_keys = [
            "min",
            "max",
            "total_count",
            "nodata_count",
            "valid_count",
            "sum",
            "stdev",
        ] + percentile_keys

    temp_dir = tempfile.mkdtemp(dir=working_dir)
    raster_path = os.path.join(temp_dir, "test_raster.tif")
    vector_path = os.path.join(temp_dir, "test_vector.gpkg")

    def _close_enough(left_value, right_value):
        if left_value is None and right_value is None:
            return True
        if left_value is None or right_value is None:
            return False
        if isinstance(left_value, (int, np.integer)) and isinstance(
            right_value, (int, np.integer)
        ):
            return int(left_value) == int(right_value)
        if isinstance(left_value, (float, np.floating)) or isinstance(
            right_value, (float, np.floating)
        ):
            return bool(
                np.isclose(
                    float(left_value),
                    float(right_value),
                    rtol=allclose_rtol,
                    atol=allclose_atol,
                )
            )
        return left_value == right_value

    def _assert_group_stats_equal(
        actual_stats, expected_stats, group_value, key_name
    ):
        if not _close_enough(
            actual_stats.get(key_name), expected_stats.get(key_name)
        ):
            print(
                f"Group={group_value!r} key={key_name!r} actual={actual_stats.get(key_name)!r} expected={expected_stats.get(key_name)!r}"
            )

    def _pixel_window_to_polygon_wkt(window):
        x_offset, y_offset, width_pixels, height_pixels = window
        left = origin_x + x_offset * pixel_size
        right = origin_x + (x_offset + width_pixels) * pixel_size
        top = origin_y - y_offset * pixel_size
        bottom = origin_y - (y_offset + height_pixels) * pixel_size
        return f"POLYGON(({left} {top},{right} {top},{right} {bottom},{left} {bottom},{left} {top}))"

    def _masked_stats(values_1d, percentile_values):
        if values_1d.size == 0:
            return {
                "min": None,
                "max": None,
                "total_count": int(0),
                "nodata_count": int(0),
                "valid_count": int(0),
                "sum": 0.0,
                "stdev": None,
                **{k: None for k in percentile_values},
            }

        if raster_nodata is None:
            nodata_mask = np.zeros(values_1d.shape, dtype=bool)
        else:
            nodata_mask = np.isclose(values_1d, raster_nodata)

        count_value = int(values_1d.size)
        nodata_count_value = int(np.count_nonzero(nodata_mask))
        if ignore_nodata:
            valid_values = values_1d[~nodata_mask].astype(
                np.float64, copy=False
            )
        else:
            valid_values = values_1d.astype(np.float64, copy=False)

        valid_count_value = int(valid_values.size)
        if valid_count_value == 0:
            stats = {
                "min": 0.0,
                "max": 0.0,
                "total_count": count_value,
                "nodata_count": nodata_count_value,
                "valid_count": 0,
                "sum": 0.0,
                "stdev": None,
                **{k: None for k in percentile_values},
            }
            return stats

        sum_value = float(np.sum(valid_values))
        mean_value = sum_value / valid_count_value
        sumsq_value = float(np.sum(valid_values * valid_values))
        variance_value = (
            sumsq_value / valid_count_value - mean_value * mean_value
        )
        if variance_value < 0:
            variance_value = 0.0
        stdev_value = float(math.sqrt(variance_value))

        stats = {
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "total_count": count_value,
            "nodata_count": nodata_count_value,
            "valid_count": valid_count_value,
            "sum": sum_value,
            "stdev": stdev_value,
        }

        if percentile_list:
            pct_values = np.percentile(
                valid_values.astype(np.float64, copy=False), percentile_list
            ).tolist()
            for percentile_key, percentile_value in zip(
                percentile_values, pct_values
            ):
                stats[percentile_key] = float(percentile_value)
        else:
            for percentile_key in percentile_values:
                stats[percentile_key] = None

        return stats

    try:
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(int(epsg))
        spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        raster_driver = gdal.GetDriverByName("GTiff")
        raster_dataset = raster_driver.Create(
            raster_path,
            int(raster_values.shape[1]),
            int(raster_values.shape[0]),
            1,
            gdal.GDT_Float32,
        )
        raster_dataset.SetGeoTransform(
            (origin_x, pixel_size, 0.0, origin_y, 0.0, -pixel_size)
        )
        raster_dataset.SetProjection(spatial_reference.ExportToWkt())
        raster_band = raster_dataset.GetRasterBand(1)
        if raster_nodata is not None:
            raster_band.SetNoDataValue(float(raster_nodata))
        raster_band.WriteArray(raster_values.astype(np.float32, copy=False))
        raster_band.FlushCache()
        raster_band = None
        raster_dataset.FlushCache()
        raster_dataset = None

        vector_driver = ogr.GetDriverByName("GPKG")
        if os.path.exists(vector_path):
            vector_driver.DeleteDataSource(vector_path)
        vector_dataset = vector_driver.CreateDataSource(vector_path)
        vector_layer = vector_dataset.CreateLayer(
            aggregate_layer_name, spatial_reference, ogr.wkbPolygon
        )

        if polygons and isinstance(polygons[0].get("group_value"), str):
            group_field_type = ogr.OFTString
        else:
            group_field_type = ogr.OFTInteger

        vector_layer.CreateField(
            ogr.FieldDefn(aggregate_vector_field, group_field_type)
        )
        layer_definition = vector_layer.GetLayerDefn()

        for polygon_spec in polygons:
            group_value = polygon_spec["group_value"]
            window = polygon_spec["window"]
            polygon_wkt = polygon_spec.get(
                "wkt"
            ) or _pixel_window_to_polygon_wkt(window)
            polygon_geometry = ogr.CreateGeometryFromWkt(polygon_wkt)

            feature = ogr.Feature(layer_definition)
            feature.SetGeometry(polygon_geometry)
            feature.SetField(aggregate_vector_field, group_value)
            vector_layer.CreateFeature(feature)

        vector_layer = None
        vector_dataset = None

        actual_grouped_stats = fast_zonal_statistics_fn(
            (raster_path, 1),
            vector_path,
            aggregate_vector_field,
            aggregate_layer_name=aggregate_layer_name,
            ignore_nodata=ignore_nodata,
            working_dir=temp_dir,
            clean_working_dir=False,
            percentile_list=percentile_list,
        )

        if expect_grouped_stats is None:
            expected_values_by_group = collections.defaultdict(list)
            expected_counts_by_group = collections.defaultdict(
                lambda: {"total_count": 0, "nodata_count": 0}
            )

            for polygon_spec in polygons:
                group_value = polygon_spec["group_value"]
                x_offset, y_offset, width_pixels, height_pixels = polygon_spec[
                    "window"
                ]
                window_values = raster_values[
                    y_offset : y_offset + height_pixels,
                    x_offset : x_offset + width_pixels,
                ].ravel()

                expected_values_by_group[group_value].append(window_values)

                if raster_nodata is None:
                    nodata_mask = np.zeros(window_values.shape, dtype=bool)
                else:
                    nodata_mask = np.isclose(window_values, raster_nodata)

                expected_counts_by_group[group_value]["total_count"] += int(
                    window_values.size
                )
                expected_counts_by_group[group_value]["nodata_count"] += int(
                    np.count_nonzero(nodata_mask)
                )

            expect_grouped_stats = {}
            for group_value, chunk_list in expected_values_by_group.items():
                combined_values = (
                    np.concatenate(chunk_list)
                    if chunk_list
                    else np.array([], dtype=np.float64)
                )
                group_stats = _masked_stats(combined_values, percentile_keys)
                group_stats["total_count"] = int(
                    expected_counts_by_group[group_value]["total_count"]
                )
                group_stats["nodata_count"] = int(
                    expected_counts_by_group[group_value]["nodata_count"]
                )
                group_stats["valid_count"] = int(
                    group_stats["total_count"] - group_stats["nodata_count"]
                )
                expect_grouped_stats[group_value] = group_stats

        actual_group_keys = set(actual_grouped_stats.keys())
        expected_group_keys = set(expect_grouped_stats.keys())
        if actual_group_keys != expected_group_keys:
            raise AssertionError(
                f"Group keys mismatch actual={sorted(actual_group_keys)!r} expected={sorted(expected_group_keys)!r}"
            )

        for group_value, expected_stats in expect_grouped_stats.items():
            actual_stats = actual_grouped_stats[group_value]
            for key_name in compare_keys:
                _assert_group_stats_equal(
                    actual_stats, expected_stats, group_value, key_name
                )

        return {
            "temp_dir": temp_dir,
            "raster_path": raster_path,
            "vector_path": vector_path,
            "actual": actual_grouped_stats,
            "expected": expect_grouped_stats,
        }
    finally:
        if clean_working_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


def pretty_print_zonal_stats_result_with_polygon_values(
    polygons,
    result,
    *,
    raster_values,
    raster_nodata=None,
    ignore_nodata=True,
    title_key="group_id",
    stats_root_key="actual",
    sort_groups=True,
    sort_stats=True,
    preferred_stat_order=None,
    float_digits=6,
):
    import numpy as np

    raster_values = np.asarray(raster_values)
    if raster_values.ndim != 2:
        raise ValueError("raster_values must be a 2D array")

    if preferred_stat_order is None:
        preferred_stat_order = [
            "min",
            "max",
            "total_count",
            "nodata_count",
            "valid_count",
            "sum",
            "stdev",
        ]

    stats_by_group = result[stats_root_key]

    group_to_values_all = {}
    group_to_values_valid = {}

    def _format_value(value):
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value)

    def _format_array(array_values):
        array_values = np.asarray(array_values)
        if array_values.size == 0:
            return "[]"
        if np.issubdtype(array_values.dtype, np.integer):
            return (
                "["
                + ", ".join(str(int(v)) for v in array_values.tolist())
                + "]"
            )
        return (
            "["
            + ", ".join(
                f"{float(v):.{float_digits}f}".rstrip("0").rstrip(".")
                for v in array_values.tolist()
            )
            + "]"
        )

    def _nodata_mask(values_1d):
        if raster_nodata is None:
            return np.zeros(values_1d.shape, dtype=bool)
        return np.isclose(values_1d, raster_nodata)

    for polygon_id, polygon_spec in enumerate(polygons):
        group_value = polygon_spec["group_value"]
        x_offset, y_offset, width_pixels, height_pixels = polygon_spec["window"]

        window_values = raster_values[
            y_offset : y_offset + height_pixels,
            x_offset : x_offset + width_pixels,
        ].ravel()

        nodata_mask = _nodata_mask(window_values)
        valid_values = (
            window_values[~nodata_mask] if ignore_nodata else window_values
        )

        group_to_values_all.setdefault(group_value, []).append(
            window_values.astype(np.float64, copy=False)
        )
        group_to_values_valid.setdefault(group_value, []).append(
            valid_values.astype(np.float64, copy=False)
        )

        print(
            f'polygon_id={polygon_id} ({title_key}={group_value}, window={polygon_spec["window"]})'
        )
        print(
            f"  - selected_values_all: {_format_array(window_values.astype(np.float64, copy=False))}"
        )
        print(
            f"  - selected_values_valid: {_format_array(valid_values.astype(np.float64, copy=False))}"
        )
        print("")

    group_items = list(stats_by_group.items())
    if sort_groups:
        group_items.sort(key=lambda item: str(item[0]))

    for group_value, stats in group_items:
        all_group_values = (
            np.concatenate(group_to_values_all.get(group_value, []))
            if group_to_values_all.get(group_value)
            else np.array([], dtype=np.float64)
        )
        valid_group_values = (
            np.concatenate(group_to_values_valid.get(group_value, []))
            if group_to_values_valid.get(group_value)
            else np.array([], dtype=np.float64)
        )

        print(f"{title_key}={group_value}")
        print(f"  - selected_values_all: {_format_array(all_group_values)}")
        print(f"  - selected_values_valid: {_format_array(valid_group_values)}")

        stat_items = list(stats.items())
        if sort_stats:
            stat_items.sort(
                key=lambda item: (
                    (
                        preferred_stat_order.index(item[0])
                        if item[0] in preferred_stat_order
                        else 10_000
                    ),
                    item[0],
                )
            )

        for stat_key, stat_value in stat_items:
            print(f"  - {stat_key}: {_format_value(stat_value)}")
        print("")


def example_fast_zonal_statistics_test_case(*, fast_zonal_statistics_fn):
    import numpy as np

    raster_values = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, -9999, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        dtype=np.float32,
    )

    polygons = [
        {"group_value": 10, "window": (0, 0, 2, 2)},
        {"group_value": 10, "window": (2, 0, 3, 1)},
        {"group_value": 20, "window": (1, 2, 3, 2)},
    ]

    result = run_fast_zonal_statistics_test_case(
        fast_zonal_statistics_fn=fast_zonal_statistics_fn,
        raster_values=raster_values,
        polygons=polygons,
        aggregate_vector_field="group_id",
        aggregate_layer_name="polys",
        ignore_nodata=True,
        percentile_list=[50, 90],
        raster_nodata=-9999,
        pixel_size=10.0,
        origin_x=1000.0,
        origin_y=2000.0,
        epsg=3857,
        allclose_rtol=1e-6,
        allclose_atol=1e-6,
        clean_working_dir=True,
    )

    return polygons, raster_values, result


def run_test():
    polygons, raster_values, result = example_fast_zonal_statistics_test_case(
        fast_zonal_statistics_fn=fast_zonal_statistics
    )
    print(result["actual"][10])
    expected = {
        10: {
            "min": 1.0,
            "max": 7.0,
            "total_count": 7,
            "nodata_count": 0,
            "valid_count": 7,
            "sum": 28.0,
            "stdev": 2.000000,
            "p50": 4.000000,
            "p90": 7,
        },
        20: {
            "min": 12.0,
            "max": 19.0,
            "total_count": 6,
            "nodata_count": 1,
            "valid_count": 5,
            "sum": 80.0,
            "stdev": 2.607681,
            "p50": 17.000000,
            "p90": 19,
        },
    }

    def assert_actual_matches_expected(actual, expected, float_tol=1e-6):
        for gid, exp in expected.items():
            if gid not in actual:
                raise AssertionError(
                    f"missing group_id={gid} in actual results"
                )
            a = actual[gid]
            for k, v in exp.items():
                if k not in a:
                    raise AssertionError(
                        f"missing key={k} for group_id={gid} in actual results"
                    )
                av = a[k]
                if isinstance(v, float):
                    if av is None or not math.isclose(
                        float(av), float(v), rel_tol=0.0, abs_tol=float_tol
                    ):
                        raise AssertionError(
                            f"group_id={gid} key={k} expected={v} actual={av}"
                        )
                else:
                    if av != v:
                        raise AssertionError(
                            f"group_id={gid} key={k} expected={v} actual={av}"
                        )

    polygons, raster_values, result = example_fast_zonal_statistics_test_case(
        fast_zonal_statistics_fn=fast_zonal_statistics
    )

    assert_actual_matches_expected(result["actual"], expected)

    pretty_print_zonal_stats_result_with_polygon_values(
        polygons,
        result,
        raster_values=raster_values,
        raster_nodata=-9999,
        ignore_nodata=True,
        title_key="group_id",
        stats_root_key="actual",
        float_digits=6,
    )
    return


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

    if args.test:
        run_test()
        return

    cfg_path = Path(args.config)
    cfg = parse_and_validate_config(cfg_path)

    log_level = getattr(logging, cfg["project"]["log_level"])
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
    )
    logger = logging.getLogger(cfg["project"]["name"])
    logger.info("Loaded config %s", str(cfg_path))
    task_graph = taskgraph.TaskGraph(
        cfg["project"]["global_work_dir"], len(cfg["job_list"]) + 1, 15.0
    )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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
    logger.info("All jobs validated (%d)", len(cfg["job_list"]))
    task_graph.join()
    task_graph.close()


if __name__ == "__main__":
    main()
