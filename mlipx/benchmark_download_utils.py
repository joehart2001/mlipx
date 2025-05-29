import pathlib
import zipfile
from urllib.parse import urlparse
import fsspec
from pathlib import Path
import zipfile
import requests

# GitHub API URL (for listing contents if needed)
BENCHMARK_DATA_URL = "https://api.github.com/repos/joehart2001/mlipx/contents/benchmark_data"

# Raw download URL (for direct downloading)
BENCHMARK_DATA_DOWNLOAD_URL = "https://raw.githubusercontent.com/joehart2001/mlipx/main/benchmark_data/"

# Local cache directory
BENCHMARK_DATA_DIR = pathlib.Path.home() / ".cache" / "my_benchmark"


def get_benchmark_data(name: str, force: bool = False) -> Path:
    """
    Retrieve benchmark data. If it's a .zip, download and extract it.
    """
    uri = f"{BENCHMARK_DATA_DOWNLOAD_URL}/{name}"
    local_path = Path(BENCHMARK_DATA_DIR) / name

    # Download file if not already cached or if force is True
    if force or not local_path.exists():
        print(f"[download] Downloading {name} from {uri}")
        response = requests.get(uri)
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f_out:
            f_out.write(response.content)
    else:
        print(f"[cache] Found cached file: {local_path.name}")

    # If it's a zip, extract it
    if local_path.suffix == ".zip":
        extract_dir = local_path.parent
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    else:
        raise ValueError(f"Unsupported file format: {local_path}")