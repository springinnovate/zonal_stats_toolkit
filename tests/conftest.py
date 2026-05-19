import os
import tempfile
from pathlib import Path


_LOCAL_TEMP_DIR = Path.cwd() / ".pytest_tmp"
_LOCAL_TEMP_DIR.mkdir(exist_ok=True)

for _temp_env_var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_temp_env_var] = str(_LOCAL_TEMP_DIR)

tempfile.tempdir = str(_LOCAL_TEMP_DIR)
