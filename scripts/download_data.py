"""Download the datasets used by the homeworks into ./data.

Two Google Drive files are required:

  1. stocks_df_combined_2026_09_18.parquet.brotli
     Shared dataset for Homework 3 and Homework 4 (33 tickers, US/EU/India,
     technical indicators + macro variables). Linked from the Module 02/03/04
     Colab notebooks.

  2. data.parquet
     Precomputed technical/macro indicators for the RSI strategy in
     Homework 2, question 4.

Usage:  python scripts/download_data.py
"""

import re
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FILES = {
    "stocks_df_combined_2026_09_18.parquet.brotli": "1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5",
    "data_hw2.parquet": "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def download(file_id: str, target: Path) -> None:
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar()))
    opener.addheaders = [("User-Agent", USER_AGENT)]

    # Step 1 - request the file; Google answers with a "virus scan" confirmation page.
    page = opener.open(
        f"https://drive.google.com/uc?export=download&id={file_id}", timeout=120
    ).read().decode("utf-8", "ignore")
    match = re.search(r'name="uuid"\s+value="([^"]+)"', page)
    uuid = match.group(1) if match else ""

    # Step 2 - follow the confirmation with the uuid, which returns the real file.
    query = urllib.parse.urlencode(
        {"id": file_id, "export": "download", "confirm": "t", "uuid": uuid})
    request = urllib.request.Request(
        "https://drive.usercontent.google.com/download?" + query,
        headers={"Referer": "https://drive.google.com/"},
    )
    with opener.open(request, timeout=1800) as response, open(target, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    print(f"downloaded {target.name}: {target.stat().st_size:,} bytes")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        target = DATA_DIR / name
        if target.exists():
            print(f"skipping {name} (already present)")
            continue
        download(file_id, target)


if __name__ == "__main__":
    main()
