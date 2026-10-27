"""
E001v2 2단계 — 일별 CSV(전 기간)를 월 단위 parquet 으로 조립.

메모리 안전을 위해 월 전체를 한 번에 올리지 않고 '일별로 스트리밍' 처리한다.
근거: 어느 날짜의 agg_trade_id 구간도 다른 날짜와 겹치지 않는다(사전 확인 완료).
따라서 일별 파일을 각각 id 기준 정렬한 뒤 날짜순으로 이어 쓰면 전체가 정렬된다
-- 월 전체를 메모리에 올려 정렬할 필요가 없다.

일별 파일 내부에서도 완전 중복행은 제거한다. 전날 마지막 id 와 겹치는 부분이
있으면(파일 경계 중복) 그 부분도 제거한다.
"""

import calendar
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
_SUB = "raw_daily" if SYMBOL == "BTCUSDT" else f"raw_daily_{SYMBOL}"
_TRD = "trades" if SYMBOL == "BTCUSDT" else f"trades_{SYMBOL}"

CSV_DIR = ROOT / "data" / _SUB / "csv"
TRADES = ROOT / "data" / _TRD
LOGS = ROOT / "results" / "logs"

START, END = "2021-01", "2025-12"
KEEP = ["agg_trade_id", "transact_time", "price", "quantity", "is_buyer_maker"]
ALL_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]
SCHEMA = pa.schema([
    ("agg_trade_id", pa.int64()),
    ("transact_time", pa.int64()),
    ("price", pa.float64()),
    ("quantity", pa.float64()),
    ("is_buyer_maker", pa.bool_()),
])


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


def load_day(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        head = f.read(20)
    hdr = head.lower().startswith(b"agg_trade_id")
    if hdr:
        df = pd.read_csv(path, usecols=KEEP)
    else:
        df = pd.read_csv(path, header=None, names=ALL_COLS, usecols=KEEP)
    if df["is_buyer_maker"].dtype != bool:
        df["is_buyer_maker"] = (
            df["is_buyer_maker"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False})
        )
        if df["is_buyer_maker"].isna().any():
            raise ValueError(f"{path.name}: is_buyer_maker 파싱 실패")
    df = df[KEEP].sort_values("agg_trade_id", kind="stable").reset_index(drop=True)
    df = df.drop_duplicates(subset=KEEP, keep="first").reset_index(drop=True)
    return df


def build_month(month: str, last_id_prev: int | None) -> dict:
    cdir = CSV_DIR / month
    files = sorted(cdir.glob("*.csv"))
    if not files:
        return dict(month=month, status="no_csv", rows=0, last_id=last_id_prev)

    out = TRADES / f"aggTrades-{month}.parquet"
    tmp = out.with_suffix(".parquet.part")

    rows = 0
    id_gaps = []
    last_id = last_id_prev
    t_first = t_last = None

    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as w:
        for f in files:
            df = load_day(f)
            if len(df) == 0:
                continue

            if last_id is not None:
                before = len(df)
                df = df[df["agg_trade_id"] > last_id].reset_index(drop=True)
                cross_dup = before - len(df)
                if cross_dup:
                    pass  # 파일 경계 중복 — 조용히 제거 (전날과 겹치는 부분)
                if len(df) and df["agg_trade_id"].iloc[0] - last_id > 1:
                    gap = int(df["agg_trade_id"].iloc[0] - last_id - 1)
                    id_gaps.append({"file": f.name, "gap": gap,
                                     "after_id": int(last_id)})

            if len(df) == 0:
                continue

            w.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
            rows += len(df)
            if t_first is None:
                t_first = int(df["transact_time"].iloc[0])
            t_last = int(df["transact_time"].iloc[-1])
            last_id = int(df["agg_trade_id"].iloc[-1])

    tmp.replace(out)
    return dict(month=month, status="ok", rows=rows, last_id=last_id,
                t_first=t_first, t_last=t_last, id_gaps=id_gaps)


def main():
    TRADES.mkdir(parents=True, exist_ok=True)
    ms = months(START, END)
    print(f"E001v2 2단계 — 일별 CSV -> 월별 parquet 조립  |  {len(ms)}개월\n", flush=True)

    results = {}
    last_id = None
    for k, month in enumerate(ms, 1):
        r = build_month(month, last_id)
        results[month] = r
        if r["status"] != "ok":
            print(f"[{k:2d}/{len(ms)}] {month}  {r['status']}", flush=True)
            continue
        last_id = r["last_id"]
        gap_note = f"  id결손 {sum(g['gap'] for g in r['id_gaps']):,}개" if r["id_gaps"] else ""
        print(f"[{k:2d}/{len(ms)}] {month}  {r['rows']:>11,}행{gap_note}", flush=True)

    tag = "" if SYMBOL == "BTCUSDT" else f"_{SYMBOL}"
    out_path = LOGS / f"E001v2_build_report{tag}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    total_rows = sum(r["rows"] for r in results.values())
    print(f"\n=== 2단계 완료 ===  총 {total_rows:,}행")
    print(f"리포트: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
