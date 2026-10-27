"""
E001 — Binance BTCUSDT 무기한선물 aggTrades 수집

zip 다운로드 -> SHA256 대조 -> 메모리에서 청크 파싱 -> parquet 저장.
CSV는 디스크에 쓰지 않는다.

raw 계층은 원천에 충실하게 둔다. is_buyer_maker 를 테이커 부호로 뒤집는 변환은
여기서 하지 않고 바 생성 단계(E003) 한 곳에서만 수행한다.
"""

import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TRADES = ROOT / "data" / "trades"
LOGS = ROOT / "results" / "logs"

BASE = "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT"
SYMBOL = "BTCUSDT"

# aggTrades 원본 컬럼 순서 (헤더 없는 구버전 파일용)
ALL_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]
KEEP = ["agg_trade_id", "transact_time", "price", "quantity", "is_buyer_maker"]

SCHEMA = pa.schema([
    ("agg_trade_id", pa.int64()),
    ("transact_time", pa.int64()),      # UTC 밀리초 원본 그대로. datetime 변환 안 함
    ("price", pa.float64()),
    ("quantity", pa.float64()),
    ("is_buyer_maker", pa.bool_()),     # 원천 그대로. 테이커 부호 변환은 E003에서
])

CHUNK = 2_000_000       # 메모리 여유가 적어 청크로 흘려보낸다
COMPRESSION = "zstd"


def months(start="2021-01", end="2025-12"):
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


def download(url, dest, retries=3):
    """스트리밍 다운로드. 부분 파일이 남지 않도록 .part 로 받고 원자적 교체."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(part, "wb") as f:
                    for blk in r.iter_content(1 << 20):
                        f.write(blk)
            part.replace(dest)
            return True
        except Exception as e:
            print(f"    다운로드 실패 {attempt}/{retries}: {e}", flush=True)
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify(zip_path, month):
    """Binance 가 함께 올리는 .CHECKSUM 과 대조. 20억 행 작업이라
    한 달치가 조용히 손상되면 나중에 찾기 어렵다."""
    url = f"{BASE}/{SYMBOL}-aggTrades-{month}.zip.CHECKSUM"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None                      # 체크섬 파일 없음
        expected = r.text.split()[0].lower()
    except Exception:
        return None
    return sha256(zip_path) == expected


def has_header(raw_head: bytes) -> bool:
    return raw_head[:12].lower().startswith(b"agg_trade_id")


def to_parquet(zip_path, out_path):
    """zip 안의 CSV 를 청크로 읽어 parquet row group 으로 바로 흘려보낸다.
    CSV 를 파일로 쓰지 않고, 월 전체를 메모리에 올리지도 않는다."""
    zf = zipfile.ZipFile(zip_path)
    name = zf.namelist()[0]

    with zf.open(name) as fh:
        head = fh.read(64)
    hdr = has_header(head)

    kwargs = dict(usecols=KEEP, chunksize=CHUNK)
    if hdr:
        kwargs["header"] = 0
    else:
        kwargs["header"] = None
        kwargs["names"] = ALL_COLS

    tmp = out_path.with_suffix(".parquet.part")
    rows = 0
    t_first = t_last = None
    id_first = id_last = None
    prev_t = None
    monotonic = True

    with zf.open(name) as fh, pq.ParquetWriter(tmp, SCHEMA, compression=COMPRESSION) as w:
        for chunk in pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), **kwargs):
            if chunk["is_buyer_maker"].dtype != bool:
                chunk["is_buyer_maker"] = (
                    chunk["is_buyer_maker"].astype(str).str.strip().str.lower()
                    .map({"true": True, "false": False})
                )
                if chunk["is_buyer_maker"].isna().any():
                    raise ValueError("is_buyer_maker 파싱 실패 — true/false 아닌 값")

            chunk = chunk[KEEP]
            w.write_table(pa.Table.from_pandas(chunk, schema=SCHEMA, preserve_index=False))

            rows += len(chunk)
            ts = chunk["transact_time"]
            if t_first is None:
                t_first = int(ts.iloc[0])
                id_first = int(chunk["agg_trade_id"].iloc[0])
            t_last = int(ts.iloc[-1])
            id_last = int(chunk["agg_trade_id"].iloc[-1])
            if not ts.is_monotonic_increasing or (prev_t is not None and ts.iloc[0] < prev_t):
                monotonic = False
            prev_t = t_last

    tmp.replace(out_path)
    return dict(
        rows=rows, header=hdr, time_monotonic=monotonic,
        t_first=t_first, t_last=t_last,
        id_first=id_first, id_last=id_last,
    )


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main():
    for d in (RAW, TRADES, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = LOGS / "E001_ingest_manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["month"])

    ms = months()
    print(f"E001 aggTrades 수집  |  {ms[0]} ~ {ms[-1]}  ({len(ms)}개월)")
    print(f"raw     -> {RAW}")
    print(f"parquet -> {TRADES}\n", flush=True)

    total_rows = 0
    t_start = time.time()

    for i, month in enumerate(ms, 1):
        pqt = TRADES / f"aggTrades-{month}.parquet"
        if month in done and pqt.exists():
            print(f"[{i:2d}/{len(ms)}] {month}  건너뜀 (완료됨)", flush=True)
            continue

        zpath = RAW / f"{SYMBOL}-aggTrades-{month}.zip"
        url = f"{BASE}/{SYMBOL}-aggTrades-{month}.zip"
        t0 = time.time()
        print(f"[{i:2d}/{len(ms)}] {month}", end="  ", flush=True)

        if not zpath.exists():
            if not download(url, zpath):
                print("404 — 해당 월 없음", flush=True)
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"month": month, "status": "missing"}) + "\n")
                continue
        zmb = zpath.stat().st_size / 1024**2
        t_dl = time.time() - t0

        chk = verify(zpath, month)
        if chk is False:
            raise SystemExit(f"체크섬 불일치: {month} — 파일 삭제 후 재실행 필요")

        t1 = time.time()
        info = to_parquet(zpath, pqt)
        t_pq = time.time() - t1
        pmb = pqt.stat().st_size / 1024**2
        total_rows += info["rows"]

        flag = "" if info["time_monotonic"] else "  [!] 시간 역전 있음"
        chk_s = {True: "ok", False: "FAIL", None: "n/a"}[chk]
        print(
            f"zip {zmb:6.1f}MB  pq {pmb:6.1f}MB  {info['rows']:>10,}행  "
            f"sha {chk_s}  ({t_dl:.0f}s+{t_pq:.0f}s)  "
            f"{utc(info['t_first'])[:10]}~{utc(info['t_last'])[:10]}{flag}",
            flush=True,
        )

        rec = dict(
            month=month, status="ok", zip_mb=round(zmb, 2), parquet_mb=round(pmb, 2),
            checksum=chk_s, sec_download=round(t_dl, 1), sec_parquet=round(t_pq, 1),
            utc_first=utc(info["t_first"]), utc_last=utc(info["t_last"]), **info,
        )
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mins = (time.time() - t_start) / 60
    print(f"\n완료  총 {total_rows:,}행  {mins:.1f}분")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
