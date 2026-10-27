"""
E001v2 1단계 — 60개월(2021-01~2025-12) 전체를 '일별' zip 으로 수집, CSV 로 압축 해제해 보관.

월별 zip 은 Binance 재포장 단계 버그(중복/누락일/미정렬)가 실증됐다.
일별 zip 은 그 재포장을 거치지 않아 신뢰도가 더 높다. 디스크 용량 여유가
충분하므로(3.5TB) CSV 를 남겨 눈으로 확인 가능하게 보관한다.

재시작 가능: 이미 받은(csv+zip 모두 존재) 날짜는 건너뛴다.
"""

import calendar
import hashlib
import io
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# 심볼은 인자로 받는다. BTCUSDT 는 기존 경로를 그대로 쓴다(이미 받아둔 자료 보존).
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
_SUB = "raw_daily" if SYMBOL == "BTCUSDT" else f"raw_daily_{SYMBOL}"

ZIP_DIR = ROOT / "data" / _SUB / "zip"
CSV_DIR = ROOT / "data" / _SUB / "csv"
LOGS = ROOT / "results" / "logs"

BASE = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}"
START, END = "2021-01", "2025-12"
WORKERS = 10
RETRIES = 4

manifest_lock = threading.Lock()


def months(start, end):
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def days_in_month(month: str):
    y, m = map(int, month.split("-"))
    n = calendar.monthrange(y, m)[1]
    return [f"{month}-{d:02d}" for d in range(1, n + 1)]


def sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def fetch_one(day: str, month: str, manifest_path: Path):
    zdir = ZIP_DIR / month
    cdir = CSV_DIR / month
    zdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    zpath = zdir / f"{SYMBOL}-aggTrades-{day}.zip"
    cpath = cdir / f"{SYMBOL}-aggTrades-{day}.csv"

    if cpath.exists() and cpath.stat().st_size > 0 and zpath.exists():
        return day, "skip", cpath.stat().st_size

    url = f"{BASE}/{SYMBOL}-aggTrades-{day}.zip"
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return day, "404", 0
            r.raise_for_status()
            content = r.content

            chk_status = "n/a"
            try:
                cr = requests.get(url + ".CHECKSUM", timeout=60)
                if cr.status_code == 200:
                    expected = cr.text.split()[0].lower()
                    chk_status = "ok" if sha256(content) == expected else "FAIL"
            except Exception:
                pass
            if chk_status == "FAIL":
                raise RuntimeError("체크섬 불일치")

            zf = zipfile.ZipFile(io.BytesIO(content))
            name = zf.namelist()[0]
            csv_bytes = zf.read(name)

            zpath.with_suffix(".zip.part").write_bytes(content)
            zpath.with_suffix(".zip.part").replace(zpath)
            cpath.with_suffix(".csv.part").write_bytes(csv_bytes)
            cpath.with_suffix(".csv.part").replace(cpath)

            with manifest_lock:
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "day": day, "month": month, "status": chk_status,
                        "zip_bytes": len(content), "csv_bytes": len(csv_bytes),
                    }) + "\n")

            return day, chk_status, len(csv_bytes)
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"{day}: {RETRIES}회 재시도 실패 ({last_err})")


def main():
    tag = "" if SYMBOL == "BTCUSDT" else f"_{SYMBOL}"
    manifest_path = LOGS / f"E001v2_daily_manifest{tag}.jsonl"
    LOGS.mkdir(parents=True, exist_ok=True)

    ms = months(START, END)
    days = [(d, m) for m in ms for d in days_in_month(m)]
    print(f"E001v2 1단계 — 전체 일별 수집  |  {SYMBOL}  {ms[0]}~{ms[-1]}  "
          f"{len(ms)}개월 {len(days)}일  (동시 {WORKERS}개)\n", flush=True)

    t0 = time.time()
    done, missing, failed, total_bytes = 0, [], [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, d, m, manifest_path): (d, m) for d, m in days}
        for fut in as_completed(futs):
            d, m = futs[fut]
            done += 1
            try:
                day, status, size = fut.result()
            except Exception as e:
                failed.append(d)
                print(f"[{done:4d}/{len(days)}] {d}  실패: {e}", flush=True)
                continue

            total_bytes += size
            if status == "404":
                missing.append(day)
                print(f"[{done:4d}/{len(days)}] {day}  404 (없음)", flush=True)
            elif done % 40 == 0 or status == "FAIL":
                elapsed = time.time() - t0
                rate = total_bytes / elapsed / 1024**2 if elapsed > 0 else 0
                eta_min = (len(days) - done) / max(done, 1) * elapsed / 60
                print(f"[{done:4d}/{len(days)}] {day}  {status}  "
                      f"누적 {total_bytes/1024**3:6.2f}GB  {rate:5.1f}MB/s  "
                      f"남은예상 {eta_min:5.1f}분", flush=True)

    mins = (time.time() - t0) / 60
    print(f"\n=== 1단계 완료 ===  {done}/{len(days)}일  {mins:.1f}분  "
          f"총 {total_bytes/1024**3:.2f}GB")
    if missing:
        print(f"없는 날짜({len(missing)}): {missing}")
    if failed:
        print(f"실패한 날짜({len(failed)}): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
