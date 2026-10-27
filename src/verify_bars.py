"""
E003 검증 — 생성한 봉을 Binance 공식 klines 와 전 기간 대조.

klines 는 정답지가 아니다(2025-01-14 사례에서 klines 쪽이 덜 담긴 것을 확인).
따라서 완전일치를 요구하지 않고 **차이의 분포**를 보고, 이상치만 추려낸다.

OHLC 는 정확히 일치해야 한다(가격은 집계가 아니라 선택이므로).
거래량은 aggTrade 묶음이 봉 경계를 걸칠 때 미세 차이가 생길 수 있다.
"""

import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "bars"
KDIR = ROOT / "data" / "klines"
LOGS = ROOT / "results" / "logs"

# 월별 klines 는 쓰지 않는다 — aggTrades 와 동일한 재포장 버그가 확인됐다
# (2023-11-10 12:00 봉이 일별 파일 대비 15% 부족). 일별 1분봉을 받아 직접 집계한다.
BASE = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
WORKERS = 10


def all_days(start="2021-01-01", end="2025-12-31"):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def _fetch_day(day):
    p = KDIR / "1m" / f"BTCUSDT-1m-{day}.csv"
    if p.exists() and p.stat().st_size > 0:
        return p
    r = requests.get(f"{BASE}/BTCUSDT-1m-{day}.zip", timeout=120)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    tmp = p.with_suffix(".csv.part")
    tmp.write_bytes(zf.read(zf.namelist()[0]))
    tmp.replace(p)
    return p


def fetch_1m():
    (KDIR / "1m").mkdir(parents=True, exist_ok=True)
    days = all_days()
    have = sum(1 for d in days if (KDIR / "1m" / f"BTCUSDT-1m-{d}.csv").exists())
    if have < len(days):
        print(f"일별 1분봉 klines 수집 중 ({have}/{len(days)} 보유)...", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, _ in enumerate(as_completed([ex.submit(_fetch_day, d) for d in days]), 1):
                if k % 300 == 0:
                    print(f"  {k}/{len(days)}", flush=True)
    frames = []
    for d in days:
        raw = (KDIR / "1m" / f"BTCUSDT-1m-{d}.csv").read_bytes()
        hdr = raw[:9].lower().startswith(b"open_time")
        frames.append(pd.read_csv(io.BytesIO(raw), header=0 if hdr else None,
                                   names=None if hdr else KCOLS))
    kl = pd.concat(frames, ignore_index=True)
    kl["open_time"] = kl["open_time"].astype("int64")
    return kl.sort_values("open_time").reset_index(drop=True)


def fetch_klines(interval, bar_ms, minute=None):
    """일별 1분봉을 받아 원하는 봉 길이로 직접 집계."""
    m = minute if minute is not None else fetch_1m()
    b = (m.open_time.to_numpy() // bar_ms) * bar_ms
    g = m.assign(bar=b).groupby("bar", sort=True)
    return pd.DataFrame({
        "open_time": g.size().index,
        "open": g["open"].first().to_numpy(),
        "high": g["high"].max().to_numpy(),
        "low": g["low"].min().to_numpy(),
        "close": g["close"].last().to_numpy(),
        "volume": g["volume"].sum().to_numpy(),
        "quote_volume": g["quote_volume"].sum().to_numpy(),
        "count": g["count"].sum().to_numpy(),
        "taker_buy_quote_volume": g["taker_buy_quote_volume"].sum().to_numpy(),
    })


def check(interval, bar_ms, minute):
    print(f"\n{'='*84}\n{interval} 검증\n{'='*84}")
    ours = pq.read_table(BARS / f"bars_{interval}.parquet").to_pandas()
    kl = fetch_klines(interval, bar_ms, minute)

    # klines 자체 무결성 먼저 (월별 zip 재포장 버그가 여기에도 있는지)
    dup = int(kl.open_time.duplicated().sum())
    expect = np.arange(int(ours.bar_start.min()), int(ours.bar_start.max()) + bar_ms, bar_ms)
    miss = np.setdiff1d(expect, kl.open_time.to_numpy())
    print(f"klines 무결성: {len(kl):,}봉, 중복 {dup}, 누락 {len(miss)}")
    if dup:
        kl = kl.drop_duplicates("open_time", keep="first")

    m = ours.merge(kl, left_on="bar_start", right_on="open_time",
                    how="inner", suffixes=("", "_k"))
    print(f"대조 가능한 봉: {len(m):,} / 우리 {len(ours):,}")

    res = {}
    for col, kcol in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close")]:
        d = (m[col] - m[kcol]).abs()
        rel = d / m[kcol]
        n_bad = int((rel > 1e-9).sum())
        res[col] = n_bad
        print(f"  {col:<6} 불일치 {n_bad:>5,}봉  (최대 상대오차 {rel.max():.2e})")

    for col, kcol, label in [("V_btc_raw", "volume", "거래량(BTC)"),
                              ("V_usdt", "quote_volume", "거래량(USDT)")]:
        rel = (m[col] - m[kcol]).abs() / m[kcol]
        res[label] = dict(
            exact=int((rel < 1e-12).sum()),
            over_1bp=int((rel > 1e-4).sum()),
            over_1pct=int((rel > 1e-2).sum()),
            max_rel=float(rel.max()),
        )
        print(f"  {label:<12} 완전일치 {int((rel<1e-12).sum()):>6,}봉 "
              f"({(rel<1e-12).mean()*100:5.1f}%) | "
              f">1bp {int((rel>1e-4).sum()):>4,} | >1% {int((rel>1e-2).sum()):>3,} | "
              f"최대 {rel.max():.4%}")

    # CVD 대조: klines 의 taker_buy_quote_volume 으로 역산
    kl_cvd = 2 * m.taker_buy_quote_volume - m.quote_volume
    rel = (m.cvd_usdt - kl_cvd).abs() / m.quote_volume     # 분모는 총거래량(부호 0 근처 폭주 방지)
    print(f"  {'CVD(USDT)':<12} 완전일치 {int((rel<1e-12).sum()):>6,}봉 "
          f"({(rel<1e-12).mean()*100:5.1f}%) | >1bp {int((rel>1e-4).sum()):>4,} | "
          f"최대 {rel.max():.4%}   (klines taker_buy 로 역산)")
    print(f"  {'CVD 부호':<12} 불일치 {int((np.sign(m.cvd_usdt)!=np.sign(kl_cvd)).sum()):,}봉")
    res["cvd_sign_mismatch"] = int((np.sign(m.cvd_usdt) != np.sign(kl_cvd)).sum())

    # 차이가 큰 봉 상위 (플래그 붙은 봉이 여기 나오는지 확인)
    m = m.assign(rel_v=(m.V_usdt - m.quote_volume).abs() / m.quote_volume)
    top = m.nlargest(8, "rel_v")[["timestamp", "V_usdt", "quote_volume", "rel_v", "has_gap", "gap_s"]]
    print(f"\n  거래량 차이 상위 8봉:")
    print(f"    {'봉':<26}{'우리(M)':>12}{'공식(M)':>12}{'상대차':>10}{'has_gap':>9}{'gap_s':>8}")
    for _, r in top.iterrows():
        print(f"    {str(r.timestamp)[:19]:<26}{r.V_usdt/1e6:>12,.1f}{r.quote_volume/1e6:>12,.1f}"
              f"{r.rel_v:>9.3%}{str(r.has_gap):>9}{r.gap_s:>8,.0f}")
    return res


def main():
    minute = fetch_1m()
    print(f"기준 데이터: 일별 1분봉 klines {len(minute):,}개 "
          f"(중복 {int(minute.open_time.duplicated().sum())})")
    out = {}
    for interval, bar_ms in [("4h", 4*3600*1000), ("1d", 86_400_000)]:
        out[interval] = check(interval, bar_ms, minute)
    (LOGS / "E003_verify_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n리포트: {LOGS / 'E003_verify_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
