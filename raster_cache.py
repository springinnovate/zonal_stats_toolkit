"""Content-addressed, process-safe caches for local raster geometry artifacts."""

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

from osgeo import gdal


_SCHEMA = 1
_MANIFEST = "manifest.json"
_LOCAL_VECTOR_DRIVERS = {
    "GPKG", "SQLite", "ESRI Shapefile", "OpenFileGDB", "FileGDB",
    "GeoJSON", "FlatGeobuf",
}


def _stat_marker(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino,
            "device": stat.st_dev}


def _hash_file(path):
    """Hash a regular file, rejecting a change observed during the read."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Cache inputs and artifacts must be regular files: {path}")
    before = _stat_marker(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = _stat_marker(path)
    if before != after:
        raise ValueError(f"File changed while hashing: {path}")
    return digest.hexdigest(), after


def _local_path(value):
    raw = os.fspath(value)
    if (not isinstance(raw, str) or raw.startswith(("/vsi", "\\\\", "//"))
            or "://" in raw):
        raise ValueError(f"Only local filesystem vectors can be cached: {value}")
    try:
        path = Path(raw).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot resolve local vector: {value}") from error
    if str(path).startswith(("\\\\", "//", "/vsi")):
        raise ValueError(f"Only local filesystem vectors can be cached: {value}")
    return path


def _reject_database_journals(path):
    for suffix in ("-wal", "-journal"):
        if Path(str(path) + suffix).exists():
            raise ValueError(f"Cannot cache a database with an active journal: {path}{suffix}")


def fingerprint_vector(path):
    """Fingerprint local vector contents, including the driver's sidecar files.

    Timestamps are used only to detect changes during hashing, never as input
    identity. VRT and other indirect/unsupported drivers are rejected because
    their file lists need not describe all underlying data.
    """
    path = _local_path(path)
    if path.suffix.lower() in {".gpkg", ".sqlite", ".db"}:
        _reject_database_journals(path)
    try:
        dataset = gdal.OpenEx(str(path), gdal.OF_VECTOR | gdal.OF_READONLY)
        if dataset is None:
            raise ValueError(f"Cannot open vector for caching: {path}")
        driver = dataset.GetDriver().ShortName
        files = dataset.GetFileList() or []
    except RuntimeError as error:
        raise ValueError(f"Cannot open vector for caching: {path}") from error
    finally:
        dataset = None
    if driver not in _LOCAL_VECTOR_DRIVERS:
        raise ValueError(f"Unsupported vector cache driver {driver!r}: {path}")
    if driver in {"GPKG", "SQLite"}:
        _reject_database_journals(path)
    is_geodatabase = path.is_dir() and path.suffix.lower() == ".gdb"
    if path.is_dir() and not is_geodatabase:
        raise ValueError(f"Unsupported vector cache directory: {path}")

    def source_files():
        if is_geodatabase:
            entries = list(path.rglob("*"))
            if any(entry.is_symlink() for entry in entries):
                raise ValueError(f"Geodatabase cache inputs cannot contain symlinks: {path}")
            return sorted((entry for entry in entries if entry.is_file()), key=str)
        found = set()
        for name in files:
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            found.add(_local_path(candidate))
        if path.is_file():
            found.add(path)
        if driver == "ESRI Shapefile":
            for suffix in (".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn",
                           ".sbx", ".idm", ".ind", ".shp.xml"):
                sidecar = path.with_suffix(suffix)
                if sidecar.exists():
                    found.add(_local_path(sidecar))
        return sorted(found, key=str)

    inputs = source_files()
    if not inputs:
        raise ValueError(f"Vector driver did not expose local source files: {path}")
    records = []
    markers = []
    for source in inputs:
        digest, marker = _hash_file(source)
        records.append({"path": str(source), "sha256": digest, "size": marker["size"]})
        markers.append(marker)
    if source_files() != inputs or any(
            _stat_marker(source) != marker for source, marker in zip(inputs, markers)):
        raise ValueError(f"Vector files changed while hashing: {path}")
    if driver in {"GPKG", "SQLite"}:
        _reject_database_journals(path)
    return {"path": str(path), "driver": driver, "files": records}


@contextmanager
def _exclusive_lock(path, notify):
    """Hold a kernel-backed lock; closing/crashing releases it automatically."""
    if path.is_symlink():
        raise ValueError(f"Cache lock cannot be a symlink: {path}")
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"\0")
            lock.flush()
        acquired = False
        next_notice = time.monotonic() + 2
        try:
            while not acquired:
                try:
                    lock.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    if notify is not None and time.monotonic() >= next_notice:
                        notify("Waiting for another process to prepare the shared geometry cache")
                        next_notice = time.monotonic() + 2
                    time.sleep(0.1)
            yield
        finally:
            if acquired:
                lock.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate(validate, target):
    if validate(target) is False:
        raise ValueError(f"Cache artifact validation failed: {target}")


def _valid_hit(entry, expected, validate):
    try:
        if entry.is_symlink():
            return False
        if {item.name for item in entry.iterdir()} != {expected["filename"], _MANIFEST}:
            return False
        if (entry / _MANIFEST).is_symlink():
            return False
        manifest = json.loads((entry / _MANIFEST).read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False
        target = entry / expected["filename"]
        digest, marker = _hash_file(target)
        if (manifest.get("sha256") != digest or manifest.get("size") != marker["size"]
                or manifest.get("stat") != marker):
            return False
        _validate(validate, target)
        return True
    except Exception:
        # A cached file that the format-specific validator cannot read is not
        # a hit. Fresh-build errors still propagate from artifact().
        return False


def artifact(cache_root, kind, key_data, filename, build, validate, notify=None):
    """Return ``(artifact_path, cache_hit)`` after validation or a locked build.

    ``build(target_path)`` writes one self-contained file and must close writers
    and check source immutability before returning. ``validate(target_path)``
    raises (or returns False) on invalid output. Only a validated file and its
    complete manifest are published. Cache hits verify the full content hash.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", kind):
        raise ValueError(f"Invalid cache kind: {kind!r}")
    if (not filename or Path(filename).name != filename or filename in {".", "..", _MANIFEST}
            or "/" in filename or "\\" in filename or ":" in filename):
        raise ValueError(f"Artifact filename must be a simple basename: {filename!r}")
    expected = {"schema": _SCHEMA, "key_data": key_data, "filename": filename}
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = json.loads(encoded)
    key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    parent = Path(cache_root).resolve() / kind
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError(f"Cache kind directory cannot be a symlink: {parent}")
    entry = parent / key
    with _exclusive_lock(parent / f"{key}.lock", notify):
        if _valid_hit(entry, expected, validate):
            return entry / filename, True
        if notify is not None:
            notify(f"Preparing shared {kind} cache")
        stage = Path(tempfile.mkdtemp(prefix=f"{key}.staging-", dir=parent))
        backup = None
        try:
            target = stage / filename
            build(target)
            _validate(validate, target)
            if set(stage.iterdir()) != {target}:
                raise ValueError("Cache artifacts must be self-contained; unexpected sidecar/work files remain")
            digest, marker = _hash_file(target)
            manifest = {**expected, "sha256": digest, "size": marker["size"], "stat": marker}
            with (stage / _MANIFEST).open("w", encoding="utf-8") as output:
                json.dump(manifest, output, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            if entry.exists() or entry.is_symlink():
                backup = parent / f"{key}.replaced-{uuid.uuid4().hex}"
                entry.rename(backup)
            try:
                stage.rename(entry)
            except BaseException:
                if backup is not None:
                    backup.rename(entry)
                    backup = None
                raise
            return entry / filename, False
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            if backup is not None:
                if backup.is_dir() and not backup.is_symlink():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
