from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import configparser
import csv
import logging
import math

from ecoshard import taskgraph
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from tqdm.auto import tqdm
import fiona
import numpy as np
import rasterio


VALID_OPERATIONS = {
    "avg",
    "stdev",
    "min",
    "max",
    "sum",
    "total_count",
    "valid_count",
    "median",
    "p5",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
}


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
            raise FileNotFoundError(f"[job:{tag}] agg_vector not found: {agg_vector}")

        base_raster_pattern = job.get("base_raster_pattern", "").strip()
        if base_raster_pattern in [None, ""]:
            raise FileNotFoundError(f"[job:{tag}] base_raster_pattern tag not found")
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
        operations = [o.strip().lower() for o in ops_raw.split(",") if o.strip()]
        if not operations:
            raise ValueError(f"[job:{tag}] operations is empty")

        invalid_ops = sorted(set(operations) - VALID_OPERATIONS)
        if invalid_ops:
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


def _collect_valid_zone_values_for_raster(
    raster_path: Path,
    agg_vector: Path,
    agg_layer: str,
    agg_field: str,
    zone_value_to_zone_id: dict[str, int],
    zone_id_to_zone_value: dict[int, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logging.getLogger(__name__)
    logger.info("opening raster %s", raster_path)
    with rasterio.open(raster_path) as raster_dataset:
        raster_band_array = raster_dataset.read(1)
        raster_transform = raster_dataset.transform
        raster_nodata_value = raster_dataset.nodata
        raster_crs = raster_dataset.crs

    pixel_width = abs(raster_transform.a)
    pixel_height = abs(raster_transform.e)
    geometry_simplify_tolerance = 0.5 * max(pixel_width, pixel_height)
    logger.info(
        "raster loaded (width=%d, height=%d, nodata=%s, simplify_tol=%f)",
        raster_band_array.shape[1],
        raster_band_array.shape[0],
        str(raster_nodata_value),
        geometry_simplify_tolerance,
    )

    shapes_with_zone_ids: list[tuple[dict, int]] = []

    logger.info("opening vector %s (layer=%s)", agg_vector, agg_layer or "<first>")
    with fiona.open(agg_vector, layer=agg_layer) as vector_layer:
        vector_layer_crs = None
        if vector_layer.crs_wkt:
            vector_layer_crs = CRS.from_wkt(vector_layer.crs_wkt)
        elif vector_layer.crs:
            vector_layer_crs = CRS.from_user_input(vector_layer.crs)

        raster_dataset_crs = CRS.from_user_input(raster_crs) if raster_crs else None

        geometry_transformer = None
        if (
            vector_layer_crs
            and raster_dataset_crs
            and vector_layer_crs != raster_dataset_crs
        ):
            logger.info(
                "reprojecting vector from %s to %s",
                vector_layer_crs.to_string(),
                raster_dataset_crs.to_string(),
            )
            geometry_transformer = Transformer.from_crs(
                vector_layer_crs,
                raster_dataset_crs,
                always_xy=True,
            )
        else:
            logger.info("no reprojection needed (vector and raster CRS compatible)")

        def reproject_xy(x, y, z=None):
            return geometry_transformer.transform(x, y)

        feature_count = 0
        used_feature_count = 0
        features = list(vector_layer)

        feature_records: list[tuple[int, dict, int]] = []

        feature_count = 0
        for feature in features:
            feature_count += 1

            feature_properties = feature.get("properties") or {}
            zone_value = feature_properties.get(agg_field)
            if zone_value is None:
                logger.debug(
                    'feature %d skipped: missing agg_field "%s"',
                    feature_count,
                    agg_field,
                )
                continue

            zone_value_string = str(zone_value)
            zone_id = zone_value_to_zone_id.get(zone_value_string)
            if zone_id is None:
                zone_id = len(zone_value_to_zone_id) + 1
                zone_value_to_zone_id[zone_value_string] = zone_id
                zone_id_to_zone_value[zone_id] = zone_value_string
                logger.debug(
                    'created new zone_id=%d for value="%s"',
                    zone_id,
                    zone_value_string,
                )

            feature_geometry = feature.get("geometry")
            if not feature_geometry:
                logger.debug(
                    "feature %d (zone_id=%d) skipped: no geometry",
                    feature_count,
                    zone_id,
                )
                continue

            feature_records.append((feature_count, feature_geometry, zone_id))

        logger.info(
            "dispatching %d features to thread pool for reprojection/simplify",
            len(feature_records),
        )

        shapes_with_zone_ids: list[tuple[dict, int]] = []
        used_feature_count = 0

        def process_record(
            record: tuple[int, dict, int],
        ) -> tuple[int, dict, int] | None:
            record_index, feature_geometry, zone_id = record
            shapely_geometry = shape(feature_geometry)

            if geometry_transformer is not None:
                shapely_geometry = shapely_transform(reproject_xy, shapely_geometry)

            shapely_geometry = shapely_geometry.simplify(
                geometry_simplify_tolerance,
                preserve_topology=True,
            )

            if shapely_geometry.is_empty:
                logger.debug(
                    "feature %d (zone_id=%d) skipped: geometry empty after simplify",
                    record_index,
                    zone_id,
                )
                return None

            return record_index, mapping(shapely_geometry), zone_id

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_record, record) for record in feature_records
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="simplifying geometries",
            ):
                result = future.result()
                if result is None:
                    continue
                record_index, mapped_geometry, zone_id = result
                shapes_with_zone_ids.append((mapped_geometry, zone_id))
                used_feature_count += 1

        logger.info(
            "vector processing complete: %d features read, %d geometries used after simplify",
            feature_count,
            used_feature_count,
        )

    logger.info(
        "vector processed: %d features read, %d geometries used after simplify",
        feature_count,
        used_feature_count,
    )

    if not shapes_with_zone_ids:
        logger.info("no geometries to rasterize; returning empty arrays")
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    logger.info(
        "rasterizing %d geometries to zone-id raster", len(shapes_with_zone_ids)
    )
    zone_id_raster = rasterize(
        shapes=shapes_with_zone_ids,
        out_shape=raster_band_array.shape,
        transform=raster_transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    if raster_nodata_value is None:
        valid_zone_and_data_mask = zone_id_raster > 0
    else:
        valid_zone_and_data_mask = (zone_id_raster > 0) & (
            raster_band_array != raster_nodata_value
        )

    zone_pixel_mask = zone_id_raster > 0

    all_zone_ids = zone_id_raster[zone_pixel_mask].astype(np.int64, copy=False)
    valid_zone_ids = zone_id_raster[valid_zone_and_data_mask].astype(
        np.int64, copy=False
    )
    valid_raster_values = raster_band_array[valid_zone_and_data_mask].astype(
        np.float64, copy=False
    )

    logger.info(
        "zonal mask stats: total_zoned_pixels=%d, valid_zoned_pixels=%d",
        int(all_zone_ids.size),
        int(valid_zone_ids.size),
    )

    return all_zone_ids, valid_zone_ids, valid_raster_values


import csv
import math
from pathlib import Path

import numpy as np
import logging

logger = logging.getLogger(__name__)


def run_zonal_stats_job(
    base_raster_path_list: list[Path],
    agg_vector: Path,
    agg_layer: str,
    agg_field: str,
    operations: list[str],
    output_csv: Path,
    workdir: Path,
    tag: str,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        'starting zonal stats job "%s": %d rasters, vector=%s, layer=%s, field=%s, operations=%s',
        tag,
        len(base_raster_path_list),
        agg_vector,
        agg_layer or "<first>",
        agg_field,
        ",".join(operations),
    )

    if not base_raster_path_list:
        logger.warning('no rasters provided for job "%s"; writing empty CSV', tag)

    zone_value_to_zone_id: dict[str, int] = {}
    zone_id_to_zone_value: dict[int, str] = {}

    per_zone_results: dict[int, dict[str, float]] = {}

    normalized_operations = [
        operation.strip().lower() for operation in operations if operation.strip()
    ]
    logger.info(
        'normalized operations for "%s" = %s',
        tag,
        ",".join(normalized_operations),
    )

    percentile_operations: dict[str, int] = {}
    for operation in normalized_operations:
        if operation.startswith("p") and operation[1:].isdigit():
            percentile_value = int(operation[1:])
            if 0 <= percentile_value <= 100:
                percentile_operations[operation] = percentile_value

    for index, raster_path in enumerate(base_raster_path_list, start=1):
        raster_stem = raster_path.stem
        logger.info(
            "processing raster %d/%d: %s",
            index,
            len(base_raster_path_list),
            raster_path,
        )

        (
            all_zone_ids,
            valid_zone_ids,
            valid_raster_values,
        ) = _collect_valid_zone_values_for_raster(
            raster_path,
            agg_vector,
            agg_layer,
            agg_field,
            zone_value_to_zone_id,
            zone_id_to_zone_value,
        )

        logger.debug(all_zone_ids)
        logger.debug(valid_zone_ids)
        logger.debug(valid_raster_values)

        logger.info(
            "raster %s: total_zoned_pixels=%d, valid_zoned_pixels=%d",
            raster_path,
            int(all_zone_ids.size),
            int(valid_zone_ids.size),
        )

        if not all_zone_ids.size:
            logger.info(
                "raster %s contributed no zoned pixels, skipping stats",
                raster_path,
            )
            continue

        current_max_zone_id = len(zone_value_to_zone_id)
        total_zone_pixel_counts = np.bincount(
            all_zone_ids,
            minlength=current_max_zone_id + 1,
        ).astype(np.int64, copy=False)

        valid_zone_pixel_counts = np.bincount(
            valid_zone_ids,
            minlength=current_max_zone_id + 1,
        ).astype(np.int64, copy=False)

        per_zone_value_sums = np.bincount(
            valid_zone_ids,
            weights=valid_raster_values,
            minlength=current_max_zone_id + 1,
        ).astype(np.float64, copy=False)

        for zone_id in range(1, current_max_zone_id + 1):
            per_zone_results.setdefault(zone_id, {})

        if "valid_count" in normalized_operations:
            col_name = f"valid_count_{raster_stem}"
            logger.info("computing %s", col_name)
            for zone_id in range(1, current_max_zone_id + 1):
                per_zone_results[zone_id][col_name] = float(
                    valid_zone_pixel_counts[zone_id]
                )

        if "total_count" in normalized_operations:
            col_name = f"total_count_{raster_stem}"
            logger.info("computing %s", col_name)
            for zone_id in range(1, current_max_zone_id + 1):
                per_zone_results[zone_id][col_name] = float(
                    total_zone_pixel_counts[zone_id]
                )

        if "sum" in normalized_operations:
            col_name = f"sum_{raster_stem}"
            logger.info("computing %s", col_name)
            for zone_id in range(1, current_max_zone_id + 1):
                per_zone_results[zone_id][col_name] = float(
                    per_zone_value_sums[zone_id]
                )

        if "avg" in normalized_operations:
            col_name = f"avg_{raster_stem}"
            logger.info("computing %s", col_name)
            for zone_id in range(1, current_max_zone_id + 1):
                pixel_count = valid_zone_pixel_counts[zone_id]
                per_zone_results[zone_id][col_name] = (
                    float(per_zone_value_sums[zone_id] / pixel_count)
                    if pixel_count
                    else float("nan")
                )

        operations_requiring_per_zone_value_arrays = any(
            operation in normalized_operations
            for operation in ("min", "max", "stdev", "median")
        ) or bool(percentile_operations)

        if operations_requiring_per_zone_value_arrays and valid_zone_ids.size:
            logger.info(
                "building per-zone value arrays for high-order stats for %s",
                raster_stem,
            )
            per_zone_value_arrays: dict[int, np.ndarray] = {}
            for zone_id in range(1, current_max_zone_id + 1):
                per_zone_value_arrays[zone_id] = valid_raster_values[
                    valid_zone_ids == zone_id
                ]

            if "min" in normalized_operations:
                col_name = f"min_{raster_stem}"
                logger.info("computing %s", col_name)
                for zone_id, zone_values in per_zone_value_arrays.items():
                    per_zone_results[zone_id][col_name] = (
                        float(np.min(zone_values)) if zone_values.size else float("nan")
                    )

            if "max" in normalized_operations:
                col_name = f"max_{raster_stem}"
                logger.info("computing %s", col_name)
                for zone_id, zone_values in per_zone_value_arrays.items():
                    per_zone_results[zone_id][col_name] = (
                        float(np.max(zone_values)) if zone_values.size else float("nan")
                    )

            if "median" in normalized_operations:
                col_name = f"median_{raster_stem}"
                logger.info("computing %s", col_name)
                for zone_id, zone_values in per_zone_value_arrays.items():
                    per_zone_results[zone_id][col_name] = (
                        float(np.median(zone_values))
                        if zone_values.size
                        else float("nan")
                    )

            if "stdev" in normalized_operations:
                col_name = f"stdev_{raster_stem}"
                logger.info("computing %s", col_name)
                for zone_id, zone_values in per_zone_value_arrays.items():
                    per_zone_results[zone_id][col_name] = (
                        float(np.std(zone_values, ddof=0))
                        if zone_values.size
                        else float("nan")
                    )

            if percentile_operations:
                logger.info(
                    "computing percentile operations per zone for %s: %s",
                    raster_stem,
                    ",".join(
                        f"{name}={value}"
                        for name, value in percentile_operations.items()
                    ),
                )
                for zone_id, zone_values in per_zone_value_arrays.items():
                    if not zone_values.size:
                        for percentile_operation in percentile_operations:
                            col_name = f"{percentile_operation}_{raster_stem}"
                            per_zone_results[zone_id][col_name] = float("nan")
                        continue
                    for (
                        percentile_operation,
                        percentile_value,
                    ) in percentile_operations.items():
                        col_name = f"{percentile_operation}_{raster_stem}"
                        per_zone_results[zone_id][col_name] = float(
                            np.percentile(zone_values, percentile_value)
                        )

    max_zone_id = len(zone_value_to_zone_id)
    logger.debug(f"zone_value_to_zone_id: {zone_value_to_zone_id}: {max_zone_id}")
    if max_zone_id == 0:
        logger.warning(
            'no zones discovered for job "%s" (check agg_field="%s"); writing header-only CSV',
            tag,
            agg_field,
        )

    stat_columns: list[str] = sorted(
        {
            column
            for zone_dict in per_zone_results.values()
            for column in zone_dict.keys()
        }
    )

    output_group_field_name = agg_field
    output_csv_columns = [output_group_field_name] + stat_columns
    logger.info('writing CSV output for "%s" to %s', tag, output_csv)

    with open(output_csv, "w", newline="") as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=output_csv_columns)
        csv_writer.writeheader()
        for zone_id in range(1, max_zone_id + 1):
            output_row = {output_group_field_name: zone_id_to_zone_value[zone_id]}
            for column_name in stat_columns:
                operation_value = per_zone_results[zone_id].get(
                    column_name, float("nan")
                )
                if isinstance(operation_value, float) and (
                    math.isnan(operation_value) or math.isinf(operation_value)
                ):
                    output_row[column_name] = ""
                else:
                    output_row[column_name] = operation_value
            csv_writer.writerow(output_row)

    logger.info('finished zonal stats job "%s"; wrote %d zones', tag, max_zone_id)


def main():
    """CLI entrypoint for validating a zonal-stats runner configuration.

    Parses a single positional argument pointing to an INI configuration file,
    validates it via `parse_and_validate_config`, configures logging based on the
    `[project].log_level` setting, and logs a validation summary for each job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to INI configuration file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = parse_and_validate_config(cfg_path)

    log_level = getattr(logging, cfg["project"]["log_level"])
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
    )
    logger = logging.getLogger(cfg["project"]["name"])
    logger.info("Loaded config %s", str(cfg_path))
    task_graph = taskgraph.TaskGraph(cfg["project"]["global_work_dir"], -1)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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

        task_graph.add_task(
            func=run_zonal_stats_job(**job),
            target_path_list=[output_path_timestamped],
            task_name=f"zonal stats {job['tag']}",
        )
    logger.info("All jobs validated (%d)", len(cfg["job_list"]))


if __name__ == "__main__":
    main()
