"""
E003 — aggTrades 로부터 4시간봉·일봉 생성.

원칙
  - 타임존 UTC 고정, 봉 경계 좌폐우개 [start, end)
  - OHLCV 와 CVD 를 **같은 루프에서 한 번에** 산출 (따로 계산하면 봉 경계가 어긋남)
  - 정렬 키는 agg_trade_id (파일 행 순서나 transact_time 이 아니다 — E001b 참조)
  - 거래량 V 는 USDT 명목가치 Σ(p·q). 기준선 확정은 E002c
  - 테이커 부호: isBuyerMaker==False -> 테이커 매수(+1), True -> 테이커 매도(-1)
  - 결측 구간이 걸친 봉에 플래그만 단다. 값은 보간·삭제하지 않는다 (E001d/E001e)

메모리: 월별 parquet 을 배치로 흘리면서 부분집계를 합친다. 월 전체를 올리지 않는다.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "data" / "trades"
BARS = ROOT / "data" / "bars"
LOGS = ROOT / "results" / "logs"

BATCH = 4_000_000
INTERVALS = {"4h": 4 * 3600 * 1000, "1d": 86_400_000}


def load_gaps():
    """E001d 에서 확정한 결손 구간 6건."""
    p = LOGS / "E001d_all_gaps.json"
    g = json.loads(p.read_text(encoding="utf-8"))["gaps"]
    return [(int(x["t_before"]), int(x["t_after"])) for x in g]


def agg_batch(df, bar_ms):
    """한 배치를 봉 단위로 부분집계."""
    bar = (df["transact_time"].to_numpy() // bar_ms) * bar_ms
    d = pd.DataFrame({
        "bar": bar,
        "price": df["price"].to_numpy(),
        "usdt": df["price"].to_numpy() * df["quantity"].to_numpy(),
        "signed": np.where(df["is_buyer_maker"].to_numpy(),
                           -(df["price"].to_numpy() * df["quantity"].to_numpy()),
                           df["price"].to_numpy() * df["quantity"].to_numpy()),
        "qty": df["quantity"].to_numpy(),
    })
    g = d.groupby("bar", sort=True)
    return pd.DataFrame({
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "V_usdt": g["usdt"].sum(),
        "cvd_usdt": g["signed"].sum(),
        "V_btc_raw": g["qty"].sum(),      # 검증(klines volume 대조)용. 분석엔 쓰지 않음
        "n_trades": g["price"].size(),
    })


def merge_partial(acc: dict, part: pd.DataFrame):
    """배치 부분집계를 누적. 배치는 agg_trade_id(=시간) 순으로 오므로
    open 은 처음 본 값을 유지하고 close 는 나중 값으로 갱신한다."""
    for bar, row in part.iterrows():
        cur = acc.get(bar)
        if cur is None:
            acc[bar] = row.to_dict()
        else:
            cur["high"] = max(cur["high"], row["high"])
            cur["low"] = min(cur["low"], row["low"])
            cur["close"] = row["close"]          # 갱신
            cur["V_usdt"] += row["V_usdt"]
            cur["cvd_usdt"] += row["cvd_usdt"]
            cur["V_btc_raw"] += row["V_btc_raw"]
            cur["n_trades"] += row["n_trades"]


def overlap_ms(bar_start, bar_ms, gaps):
    bar_end = bar_start + bar_ms
    return sum(max(0, min(bar_end, g1) - max(bar_start, g0)) for g0, g1 in gaps)


def u(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def main():
    BARS.mkdir(parents=True, exist_ok=True)
    gaps = load_gaps()
    files = sorted(TRADES.glob("aggTrades-*.parquet"))
    print(f"E003 봉 생성  |  {len(files)}개월  ->  {', '.join(INTERVALS)}\n", flush=True)

    acc = {k: {} for k in INTERVALS}
    total_rows = 0

    for i, path in enumerate(files, 1):
        f = pq.ParquetFile(path)
        n = 0
        for b in f.iter_batches(batch_size=BATCH,
                                 columns=["transact_time", "price", "quantity", "is_buyer_maker"]):
            df = b.to_pandas()
            n += len(df)
            for name, bar_ms in INTERVALS.items():
                merge_partial(acc[name], agg_batch(df, bar_ms))
        total_rows += n
        print(f"[{i:2d}/{len(files)}] {path.stem.replace('aggTrades-','')}  {n:>11,}행", flush=True)

    print(f"\n총 {total_rows:,}행 처리\n")

    report = {}
    for name, bar_ms in INTERVALS.items():
        d = pd.DataFrame.from_dict(acc[name], orient="index").sort_index()
        d.index.name = "bar_start"
        d = d.reset_index()

        # 거래가 한 건도 없는 봉이 있는지 — 규칙을 코드에 명시한다.
        full = np.arange(int(d.bar_start.min()), int(d.bar_start.max()) + bar_ms, bar_ms)
        missing_bars = np.setdiff1d(full, d.bar_start.to_numpy())
        # BTCUSDT 무기한선물은 24/7 거래되므로 빈 봉은 없을 것으로 예상한다.
        # 만약 존재하면 값을 지어내지 않고 n_trades=0 / OHLC=NaN 으로 남겨 드러낸다.
        if len(missing_bars):
            empty = pd.DataFrame({
                "bar_start": missing_bars, "open": np.nan, "high": np.nan,
                "low": np.nan, "close": np.nan, "V_usdt": 0.0, "cvd_usdt": 0.0,
                "V_btc_raw": 0.0, "n_trades": 0,
            })
            d = pd.concat([d, empty], ignore_index=True).sort_values("bar_start").reset_index(drop=True)

        d["gap_s"] = d.bar_start.map(lambda t: overlap_ms(int(t), bar_ms, gaps) / 1000.0)
        d["has_gap"] = d.gap_s > 0
        d["timestamp"] = pd.to_datetime(d.bar_start, unit="ms", utc=True)

        cols = ["bar_start", "timestamp", "open", "high", "low", "close",
                "V_usdt", "cvd_usdt", "V_btc_raw", "n_trades", "has_gap", "gap_s"]
        d = d[cols]

        out = BARS / f"bars_{name}.parquet"
        pq.write_table(pa.Table.from_pandas(d, preserve_index=False), out, compression="zstd")

        print(f"--- {name} ---")
        print(f"  봉 개수      {len(d):,}")
        print(f"  기간         {u(d.bar_start.iloc[0]):%Y-%m-%d %H:%M} ~ "
              f"{u(d.bar_start.iloc[-1]):%Y-%m-%d %H:%M} UTC")
        print(f"  빈 봉        {len(missing_bars)}개")
        print(f"  결측 플래그  {int(d.has_gap.sum())}개  "
              f"({', '.join(f'{u(t):%Y-%m-%d %H:%M}' for t in d.bar_start[d.has_gap])})")
        print(f"  저장         {out}  ({out.stat().st_size/1024:,.0f} KB)\n")

        report[name] = dict(
            n_bars=int(len(d)), n_empty=int(len(missing_bars)),
            n_flagged=int(d.has_gap.sum()),
            start=str(u(d.bar_start.iloc[0])), end=str(u(d.bar_start.iloc[-1])),
            flagged=[str(u(t)) for t in d.bar_start[d.has_gap]],
        )

    (LOGS / "E003_bars_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"리포트: {LOGS / 'E003_bars_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
