import os
import tempfile
from pathlib import Path


_LOCAL_TEMP_PARENT = Path.cwd() / ".pytest_tmp"
_LOCAL_TEMP_PARENT.mkdir(exist_ok=True)
_LOCAL_TEMP_DIR = Path(
    tempfile.mkdtemp(prefix=f"run-{os.getpid()}-", dir=_LOCAL_TEMP_PARENT)
)

for _temp_env_var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_temp_env_var] = str(_LOCAL_TEMP_DIR)

tempfile.tempdir = str(_LOCAL_TEMP_DIR)
