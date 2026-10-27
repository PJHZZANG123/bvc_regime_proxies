"""
E001c 2단계 — 일별 CSV들을 합쳐 결함 월의 parquet 을 재생성.

절차:
  1. 기존(결함 있는) 월별 parquet 을 data/trades/superseded/ 로 백업 (덮어쓰기 전 보존)
  2. 해당 월의 일별 CSV 전부를 agg_trade_id 기준으로 정렬해 이어붙임
     (transact_time 이 아니라 agg_trade_id 를 정렬키로 쓴다 — 2023-05/2023-10 조사에서
      확인했듯 id 정렬 시에만 시간이 단조가 됨)
  3. 완전 중복행(agg_trade_id 포함 전 컬럼 동일) 제거
  4. 재생성 후 감사 재실행으로 확인 (별도 스크립트)

재조립 후에도 남아있는 id 결손은 "거래소 원천 결측"으로 간주하고 지우지 않는다.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "raw_csv" / "extracted"
TRADES = ROOT / "data" / "trades"
SUPERSEDED = TRADES / "superseded"

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

TARGET_MONTHS = [
    "2021-02", "2021-05", "2022-08", "2022-09",
    "2022-10", "2022-11", "2023-05", "2023-10",
]


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
    return df[KEEP]


def rebuild_month(month: str) -> dict:
    cdir = CSV_DIR / month
    files = sorted(cdir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{month}: 추출된 CSV 없음 ({cdir})")

    parts = [load_day(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)

    df = df.sort_values("agg_trade_id", kind="stable").reset_index(drop=True)

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=KEEP, keep="first").reset_index(drop=True)
    n_dup_removed = n_before_dedup - len(df)

    # 같은 id 인데 내용이 다른 경우가 있으면 중대한 이상 -> 에러로 세운다
    id_dup = df["agg_trade_id"].duplicated(keep=False)
    if id_dup.any():
        conflict = df[id_dup].groupby("agg_trade_id")[["price", "quantity", "is_buyer_maker"]].nunique()
        n_conflict = int((conflict > 1).any(axis=1).sum())
        if n_conflict:
            raise ValueError(f"{month}: 동일 id, 다른 내용 충돌 {n_conflict}건 — 수동 확인 필요")
        # 완전 동일 내용의 잔여 중복(예: 일별 파일 경계 중복) 추가 제거
        df = df.drop_duplicates(subset="agg_trade_id", keep="first").reset_index(drop=True)

    assert df["agg_trade_id"].is_monotonic_increasing, f"{month}: 정렬 후에도 비단조"

    out = TRADES / f"aggTrades-{month}.parquet"
    old = TRADES / f"aggTrades-{month}.parquet"
    SUPERSEDED.mkdir(parents=True, exist_ok=True)
    if old.exists():
        backup = SUPERSEDED / f"aggTrades-{month}.defective.parquet"
        if not backup.exists():
            old.replace(backup)
        else:
            old.unlink()

    tmp = out.with_suffix(".parquet.part")
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False),
                    tmp, compression="zstd")
    tmp.replace(out)

    return dict(
        month=month, days=len(files), rows_raw=n_raw,
        dup_removed=n_dup_removed, rows_final=len(df),
    )


def main():
    print(f"E001c 2단계 — 결함월 재조립  |  {len(TARGET_MONTHS)}개월\n", flush=True)
    for month in TARGET_MONTHS:
        r = rebuild_month(month)
        print(f"{r['month']}  일수 {r['days']:2d}  원행수 {r['rows_raw']:>11,}  "
              f"중복제거 {r['dup_removed']:>9,}  최종 {r['rows_final']:>11,}", flush=True)
    print(f"\n완료. 백업: {SUPERSEDED}")


if __name__ == "__main__":
    sys.exit(main())
