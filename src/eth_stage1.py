"""
ETH 검증 1단계 — 감사 + 바 생성.

목적은 BTC 결과의 재현 여부 확인이다. 특히 두 가지.
  (1) K 선택이 같은 양상인가 (통과 K, K*)
  (2) 기준 국면의 연도별 뭉침이 ETH 에서도 나타나는가

BTC 와 동일한 규칙을 쓴다 — UTC 좌폐우개, OHLCV·CVD 동시 산출,
정렬 키는 agg_trade_id, V 는 USDT 명목가치.
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
TRADES = ROOT / "data" / "trades_ETHUSDT"
BARS = ROOT / "data" / "bars"
LOGS = ROOT / "results" / "logs"

BATCH = 4_000_000
INTERVALS = {"4h": 4 * 3600 * 1000, "1d": 86_400_000}


def audit():
    """월별로 중복·id결손·정렬을 확인한다. 결손 구간은 봉 플래그에 쓴다."""
    print("=== 감사 ===", flush=True)
    files = sorted(TRADES.glob("aggTrades-*.parquet"))
    gaps, total, prev_id, prev_t = [], 0, None, None
    n_dup_all = 0
    for k, p in enumerate(files, 1):
        f = pq.ParquetFile(p)
        ids_all, ts_all = [], []
        for b in f.iter_batches(batch_size=BATCH, columns=["agg_trade_id", "transact_time"]):
            ids_all.append(b.column("agg_trade_id").to_numpy())
            ts_all.append(b.column("transact_time").to_numpy())
        ids = np.concatenate(ids_all); ts = np.concatenate(ts_all)
        del ids_all, ts_all
        total += len(ids)
        if not (np.diff(ids) >= 0).all():
            raise SystemExit(f"{p.name}: id 정렬 깨짐")
        n_dup = len(ids) - len(np.unique(ids))
        n_dup_all += n_dup
        if prev_id is not None and ids[0] - prev_id > 1:
            gaps.append(dict(missing=int(ids[0]-prev_id-1), t_before=int(prev_t),
                             t_after=int(ts[0]), month=p.stem[-7:]))
        d = np.diff(ids)
        for i in np.where(d > 1)[0]:
            gaps.append(dict(missing=int(d[i]-1), t_before=int(ts[i]),
                             t_after=int(ts[i+1]), month=p.stem[-7:]))
        prev_id, prev_t = int(ids[-1]), int(ts[-1])
        if k % 12 == 0:
            print(f"  {k}/{len(files)}  누적 {total:,}행  결손 {len(gaps)}건", flush=True)
    miss = sum(g["missing"] for g in gaps)
    print(f"  총 {total:,}행   중복 {n_dup_all:,}   결손 {len(gaps)}건 / {miss:,}개"
          f" ({miss/total*100:.4f}%)", flush=True)
    return gaps, total


def agg_batch(df, bar_ms):
    bar = (df["transact_time"].to_numpy() // bar_ms) * bar_ms
    pr = df["price"].to_numpy(); qt = df["quantity"].to_numpy()
    usd = pr * qt
    d = pd.DataFrame({"bar": bar, "p": pr, "u": usd, "q": qt,
                      "s": np.where(df["is_buyer_maker"].to_numpy(), -usd, usd)})
    g = d.groupby("bar", sort=True)
    return pd.DataFrame({"open": g.p.first(), "high": g.p.max(), "low": g.p.min(),
                         "close": g.p.last(), "V_usdt": g.u.sum(),
                         "cvd_usdt": g.s.sum(), "V_btc_raw": g.q.sum(),
                         "n_trades": g.p.size()})


def merge(acc, part):
    for bar, row in part.iterrows():
        c = acc.get(bar)
        if c is None:
            acc[bar] = row.to_dict()
        else:
            c["high"] = max(c["high"], row["high"]); c["low"] = min(c["low"], row["low"])
            c["close"] = row["close"]
            for f in ("V_usdt", "cvd_usdt", "V_btc_raw", "n_trades"):
                c[f] += row[f]


def main():
    BARS.mkdir(parents=True, exist_ok=True)
    gaps, total = audit()
    gp = [(g["t_before"], g["t_after"]) for g in gaps]

    print("\n=== 바 생성 ===", flush=True)
    acc = {k: {} for k in INTERVALS}
    files = sorted(TRADES.glob("aggTrades-*.parquet"))
    for i, p in enumerate(files, 1):
        f = pq.ParquetFile(p)
        for b in f.iter_batches(batch_size=BATCH,
                                 columns=["transact_time", "price", "quantity", "is_buyer_maker"]):
            df = b.to_pandas()
            for nm, ms in INTERVALS.items():
                merge(acc[nm], agg_batch(df, ms))
        if i % 12 == 0:
            print(f"  {i}/{len(files)}  4h {len(acc['4h']):,}봉  1d {len(acc['1d']):,}봉", flush=True)

    def ov(t0, ms):
        t1 = t0 + ms
        return sum(max(0, min(t1, b) - max(t0, a)) for a, b in gp)

    rep = {}
    for nm, ms in INTERVALS.items():
        d = pd.DataFrame.from_dict(acc[nm], orient="index").sort_index()
        d.index.name = "bar_start"; d = d.reset_index()
        full = np.arange(int(d.bar_start.min()), int(d.bar_start.max()) + ms, ms)
        empty = np.setdiff1d(full, d.bar_start.to_numpy())
        d["gap_s"] = d.bar_start.map(lambda t: ov(int(t), ms) / 1000)
        d["has_gap"] = d.gap_s > 0
        d["timestamp"] = pd.to_datetime(d.bar_start, unit="ms", utc=True)
        cols = ["bar_start", "timestamp", "open", "high", "low", "close",
                "V_usdt", "cvd_usdt", "V_btc_raw", "n_trades", "has_gap", "gap_s"]
        out = BARS / f"bars_{nm}_ETHUSDT.parquet"
        pq.write_table(pa.Table.from_pandas(d[cols], preserve_index=False), out, compression="zstd")
        u = lambda x: datetime.fromtimestamp(int(x)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"\n--- {nm} ---  {len(d):,}봉  {u(d.bar_start.iloc[0])} ~ {u(d.bar_start.iloc[-1])}")
        print(f"  빈 봉 {len(empty)}   결측플래그 {int(d.has_gap.sum())}")
        rep[nm] = dict(n_bars=int(len(d)), n_empty=int(len(empty)),
                       n_flagged=int(d.has_gap.sum()))

    rep["audit"] = dict(total_rows=int(total), n_gaps=len(gaps),
                        missing_ids=int(sum(g["missing"] for g in gaps)),
                        gaps=gaps[:20])
    (LOGS / "ETH_stage1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n저장: {LOGS / 'ETH_stage1.json'}")


if __name__ == "__main__":
    sys.exit(main())
