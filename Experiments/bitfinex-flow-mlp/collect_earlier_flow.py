"""Fetch early public Binance minute archives needed by the extra evaluation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"


def get(task):
    symbol, month = task
    file = f"{symbol}-1m-2023-{month:02d}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{file}"
    path = SOURCE / file
    if not path.exists():
        for attempt in range(4):
            try:
                with urlopen(url, timeout=35) as response:
                    body = response.read()
                temp = path.with_suffix(".part")
                temp.write_bytes(body)
                with ZipFile(temp) as archive:
                    if len(archive.namelist()) != 1 or archive.testzip() is not None:
                        raise ValueError(f"Invalid archive for {file}")
                temp.replace(path)
                break
            except (ValueError, OSError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    return {"file": file, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    plan = [(symbol, month) for symbol in ("ETHUSDT", "BTCUSDT") for month in range(2, 11)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(get, plan))
    (SOURCE / "earlier_files.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    print("Verified", len(files), "earlier spot archives", flush=True)


if __name__ == "__main__":
    main()
