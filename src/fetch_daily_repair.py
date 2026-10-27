"""
E001c 1단계 — 결함 8개월을 '일별' zip 으로 재수집, CSV 로 압축 해제해 보관.

월별 zip 은 Binance 재포장 단계에서 버그(중복/누락일/미정렬)가 확인됐다.
일별 zip 은 그 재포장을 거치지 않은 원천이라 신뢰도가 더 높다.
디스크 용량 여유가 충분하므로 CSV 를 남겨 눈으로 확인 가능하게 한다
(스트리밍 zip->parquet 한 번에 처리하지 않는 이유).
"""

import calendar
import hashlib
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ZIP_DIR = ROOT / "data" / "raw_csv" / "daily_zip"
CSV_DIR = ROOT / "data" / "raw_csv" / "extracted"

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT"
SYMBOL = "BTCUSDT"

TARGET_MONTHS = [
    "2021-02", "2021-05", "2022-08", "2022-09",
    "2022-10", "2022-11", "2023-05", "2023-10",
]

WORKERS = 6


def days_in_month(month: str):
    y, m = map(int, month.split("-"))
    n = calendar.monthrange(y, m)[1]
    return [f"{month}-{d:02d}" for d in range(1, n + 1)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch_one(day: str, month: str):
    zdir = ZIP_DIR / month
    cdir = CSV_DIR / month
    zdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    zpath = zdir / f"{SYMBOL}-aggTrades-{day}.zip"
    cpath = cdir / f"{SYMBOL}-aggTrades-{day}.csv"

    if cpath.exists() and zpath.exists():
        return day, "skip", cpath.stat().st_size

    url = f"{BASE}/{SYMBOL}-aggTrades-{day}.zip"
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return day, "404", 0
    r.raise_for_status()
    zpath.write_bytes(r.content)

    chk_url = url + ".CHECKSUM"
    chk_status = "n/a"
    try:
        cr = requests.get(chk_url, timeout=60)
        if cr.status_code == 200:
            expected = cr.text.split()[0].lower()
            chk_status = "ok" if sha256(zpath) == expected else "FAIL"
    except Exception:
        pass
    if chk_status == "FAIL":
        raise RuntimeError(f"체크섬 불일치: {day}")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    name = zf.namelist()[0]
    csv_bytes = zf.read(name)
    cpath.write_bytes(csv_bytes)

    return day, chk_status, len(csv_bytes)


def main():
    days = []
    for m in TARGET_MONTHS:
        days += [(d, m) for d in days_in_month(m)]

    print(f"E001c 1단계 — 일별 재수집  |  {len(TARGET_MONTHS)}개월, {len(days)}일  "
          f"(동시 {WORKERS}개)\n", flush=True)

    t0 = time.time()
    done, missing, total_bytes = 0, [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, d, m): (d, m) for d, m in days}
        for fut in as_completed(futs):
            d, m = futs[fut]
            day, status, size = fut.result()
            done += 1
            total_bytes += size
            if status == "404":
                missing.append(day)
                print(f"[{done:3d}/{len(days)}] {day}  404 (일별파일 없음)", flush=True)
            elif done % 20 == 0 or status == "FAIL":
                print(f"[{done:3d}/{len(days)}] {day}  {status}  "
                      f"{size/1024**2:6.1f}MB  누적 {total_bytes/1024**3:5.2f}GB", flush=True)

    mins = (time.time() - t0) / 60
    print(f"\n완료  {done}일, {mins:.1f}분, 총 {total_bytes/1024**3:.2f}GB")
    if missing:
        print(f"일별파일 없는 날짜({len(missing)}): {missing}")
    print(f"zip: {ZIP_DIR}")
    print(f"csv: {CSV_DIR}")


if __name__ == "__main__":
    sys.exit(main())
