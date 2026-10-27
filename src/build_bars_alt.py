"""
E058/E060 — 1시간봉과 달러바 생성.

달러바를 넣는 이유 (2026-09-16 재정의)
  단순 로버스트니스 항목이 아니다. 시간바는 봉마다 담기는 거래대금이 달라
  `flow_cvd` 의 **크기 자체가 시기별로** 달라지고, HMM 이 그 수준 차이를 보고
  국면을 시기로 가르는 것으로 보인다(4시간봉 M1 의 상태가 2021~22 / 2023 / 2024~25
  로 갈렸고, 시간 3등분 분할이 κ 0.638 로 BVC 를 앞섰다).
  달러바는 봉마다 거래대금이 같으므로 **V 를 통해 들어오던 시기 정보가 사라진다.**
  López de Prado 가 정보 기반 표집을 주장한 논거와 같은 지점이다.

임계값은 4시간봉과 봉 수를 맞춰 정한다(총 거래대금 / 목표 봉 수).
비교가 공정하려면 표본 크기가 비슷해야 한다.

거래를 쪼개지 않는다 — 누적이 임계값을 넘는 체결에서 봉을 닫는다.
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
HOUR_MS = 3_600_000
TARGET_DOLLAR_BARS = 11_000     # 4시간봉(10,956)과 맞춤


def load_gaps():
    g = json.loads((LOGS / "E001d_all_gaps.json").read_text(encoding="utf-8"))["gaps"]
    return [(int(x["t_before"]), int(x["t_after"])) for x in g]


def total_dollar_volume():
    tot = 0.0
    for p in sorted(BARS.glob("bars_4h.parquet")):
        d = pq.read_table(p, columns=["V_usdt"]).to_pandas()
        tot += float(d.V_usdt.sum())
    return tot


def stream_trades():
    for p in sorted(TRADES.glob("aggTrades-*.parquet")):
        f = pq.ParquetFile(p)
        for b in f.iter_batches(batch_size=BATCH,
                                columns=["transact_time", "price", "quantity", "is_buyer_maker"]):
            # 불린은 zero-copy 변환이 안 된다
            yield (b.column("transact_time").to_numpy(),
                   b.column("price").to_numpy(),
                   b.column("quantity").to_numpy(),
                   b.column("is_buyer_maker").to_numpy(zero_copy_only=False))


def build():
    thr = total_dollar_volume() / TARGET_DOLLAR_BARS
    print(f"달러바 임계값 {thr/1e6:,.1f}M USDT  (목표 {TARGET_DOLLAR_BARS:,}봉)\n", flush=True)

    # ---- 1시간봉 누적기 ----
    h = {}
    # ---- 달러바 누적기 ----
    dollar_rows = []
    cur = None

    n_rows = 0
    for k, (ts, pr, qt, bm) in enumerate(stream_trades(), 1):
        usd = pr * qt
        sgn = np.where(bm, -usd, usd)
        n_rows += len(ts)

        # 1시간봉 — 배치별 부분집계 후 병합
        bar = (ts // HOUR_MS) * HOUR_MS
        df = pd.DataFrame({"bar": bar, "p": pr, "u": usd, "s": sgn, "q": qt})
        g = df.groupby("bar", sort=True)
        part = pd.DataFrame({"open": g.p.first(), "high": g.p.max(), "low": g.p.min(),
                             "close": g.p.last(), "V_usdt": g.u.sum(),
                             "cvd_usdt": g.s.sum(), "V_btc_raw": g.q.sum(),
                             "n_trades": g.p.size()})
        for b_, row in part.iterrows():
            c = h.get(b_)
            if c is None:
                h[b_] = row.to_dict()
            else:
                c["high"] = max(c["high"], row["high"]); c["low"] = min(c["low"], row["low"])
                c["close"] = row["close"]
                c["V_usdt"] += row["V_usdt"]; c["cvd_usdt"] += row["cvd_usdt"]
                c["V_btc_raw"] += row["V_btc_raw"]; c["n_trades"] += row["n_trades"]

        # 달러바 — 체결 단위로 누적, 임계값 넘으면 닫는다
        cs = np.cumsum(usd)
        base = 0.0
        start = 0
        while True:
            if cur is None:
                cur = dict(t0=int(ts[start]), o=float(pr[start]), hi=-np.inf, lo=np.inf,
                           V=0.0, cvd=0.0, q=0.0, n=0)
            need = thr - cur["V"]
            idx = np.searchsorted(cs[start:] - base, need)
            end = start + int(idx)
            if end >= len(ts):                       # 배치 끝까지 못 채움
                sl = slice(start, len(ts))
                cur["hi"] = max(cur["hi"], float(pr[sl].max()))
                cur["lo"] = min(cur["lo"], float(pr[sl].min()))
                cur["close"] = float(pr[len(ts)-1]); cur["t1"] = int(ts[len(ts)-1])
                cur["V"] += float(usd[sl].sum()); cur["cvd"] += float(sgn[sl].sum())
                cur["q"] += float(qt[sl].sum()); cur["n"] += len(ts) - start
                break
            sl = slice(start, end + 1)
            cur["hi"] = max(cur["hi"], float(pr[sl].max()))
            cur["lo"] = min(cur["lo"], float(pr[sl].min()))
            cur["close"] = float(pr[end]); cur["t1"] = int(ts[end])
            cur["V"] += float(usd[sl].sum()); cur["cvd"] += float(sgn[sl].sum())
            cur["q"] += float(qt[sl].sum()); cur["n"] += end - start + 1
            dollar_rows.append(cur)
            cur = None
            base = float(cs[end])
            start = end + 1
            if start >= len(ts):
                break

        if k % 50 == 0:
            print(f"  배치 {k}  누적 {n_rows:,}행  1h {len(h):,}봉  달러바 {len(dollar_rows):,}봉",
                  flush=True)

    if cur is not None and cur["n"] > 0:              # 마지막 미완성 봉은 버린다
        print(f"  마지막 미완성 달러바 {cur['n']:,}체결 버림", flush=True)

    gaps = load_gaps()

    def overlap(t0, t1):
        return sum(max(0, min(t1, g1) - max(t0, g0)) for g0, g1 in gaps)

    # ---- 1시간봉 저장 ----
    d1 = pd.DataFrame.from_dict(h, orient="index").sort_index()
    d1.index.name = "bar_start"; d1 = d1.reset_index()
    d1["gap_s"] = d1.bar_start.map(lambda t: overlap(int(t), int(t) + HOUR_MS) / 1000)
    d1["has_gap"] = d1.gap_s > 0
    d1["timestamp"] = pd.to_datetime(d1.bar_start, unit="ms", utc=True)
    cols = ["bar_start", "timestamp", "open", "high", "low", "close",
            "V_usdt", "cvd_usdt", "V_btc_raw", "n_trades", "has_gap", "gap_s"]
    pq.write_table(pa.Table.from_pandas(d1[cols], preserve_index=False),
                   BARS / "bars_1h.parquet", compression="zstd")

    # ---- 달러바 저장 ----
    d2 = pd.DataFrame(dollar_rows)
    d2 = d2.rename(columns={"t0": "bar_start", "o": "open", "hi": "high", "lo": "low",
                            "V": "V_usdt", "cvd": "cvd_usdt", "q": "V_btc_raw", "n": "n_trades"})
    d2["gap_s"] = [overlap(int(a), int(b)) / 1000 for a, b in zip(d2.bar_start, d2.t1)]
    d2["has_gap"] = d2.gap_s > 0
    d2["timestamp"] = pd.to_datetime(d2.bar_start, unit="ms", utc=True)
    d2["dur_min"] = (d2.t1 - d2.bar_start) / 60000
    cols2 = ["bar_start", "timestamp", "open", "high", "low", "close",
             "V_usdt", "cvd_usdt", "V_btc_raw", "n_trades", "has_gap", "gap_s", "dur_min"]
    pq.write_table(pa.Table.from_pandas(d2[cols2], preserve_index=False),
                   BARS / "bars_dollar.parquet", compression="zstd")

    def u(ms): return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"\n--- 1시간봉 ---  {len(d1):,}봉  {u(d1.bar_start.iloc[0])} ~ {u(d1.bar_start.iloc[-1])}")
    print(f"  결측플래그 {int(d1.has_gap.sum())}봉")
    print(f"\n--- 달러바 ---  {len(d2):,}봉  임계값 {thr/1e6:,.1f}M USDT")
    print(f"  봉당 거래대금  중앙값 {d2.V_usdt.median()/1e6:,.1f}M  "
          f"변동계수 {d2.V_usdt.std()/d2.V_usdt.mean():.4f}  (0 에 가까울수록 균일)")
    print(f"  봉 길이(분)   중앙값 {d2.dur_min.median():,.0f}  "
          f"최소 {d2.dur_min.min():,.1f}  최대 {d2.dur_min.max():,.0f}")
    print(f"  연도별 봉 수  " + "  ".join(
        f"{y}:{int((pd.to_datetime(d2.timestamp).dt.year == y).sum()):,}"
        for y in range(2021, 2026)))
    print(f"  결측플래그 {int(d2.has_gap.sum())}봉")

    (LOGS / "E058_E060_bars.json").write_text(json.dumps(dict(
        dollar_threshold=thr, n_1h=int(len(d1)), n_dollar=int(len(d2)),
        dollar_V_cv=float(d2.V_usdt.std()/d2.V_usdt.mean()),
        dollar_dur_min_median=float(d2.dur_min.median()),
    ), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(build())
