import time
import warnings
from pathlib import Path

import requests

from stress_signal.config import DATA_CACHE

BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def fetch_cached(
    url: str,
    name: str,
    max_age_days: float | None = None,
    headers: dict[str, str] | None = None,
    cache_dir: Path = DATA_CACHE,
    timeout: float = 180,
) -> Path:
    """Download `url` to `cache_dir/name` once and reuse it afterwards.

    With `max_age_days=None` the cached copy never expires. If a refresh fails,
    a stale cached copy is returned with a warning so offline reruns still work.
    """
    path = cache_dir / name
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if max_age_days is None or age_days <= max_age_days:
            return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, headers=headers or {}, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        if path.exists():
            warnings.warn(f"refresh of {name} failed ({exc}); using stale cache")
            return path
        raise
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return path
