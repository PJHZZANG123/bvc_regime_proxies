"""
E010~E013 — 관측 변수 생성.

모든 모형의 첫 성분:  r_t = ln(C_t / C_{t-1})

흐름 4종
  CVD   Σ(ε·q·p)                        <- 바 생성 단계에서 이미 산출됨 (검증 기준)
  BVC   V_t · [2·F_ν(r_t/σ_roll) − 1]   ν=0.25
  CLV   V_t · (2C−H−L)/(H−L)            H==L 이면 CLV=0
  잔차  V_t·CLV_t = α + β·r_t + e_t 의 e_t

회귀 순서
  CLV 흐름(= V·CLV) 자체를 회귀한다. 회귀가 마지막 연산이어야 잔차의 직교성이 유지된다.

윈도우 (E004 확정)
  σ_roll = 50봉,  회귀 W = 100봉(4h) / 50봉(1d)
  둘 다 **현재 봉을 제외한** t−W ~ t−1 구간에서만 추정한다.

거래량 V = USDT 명목가치 (E002c 확정)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "bars"
FEAT = ROOT / "data" / "features"
LOGS = ROOT / "results" / "logs"

NU = 0.25
SIGMA_W = 50
REG_W = {"4h": 100, "1d": 50}
EPS = 1e-12


def rolling_ols_residual(y: pd.Series, x: pd.Series, W: int):
    """
    각 t 에서 [t−W, t−1] 구간으로 y = α + βx 를 적합하고,
    현재 봉의 잔차 e_t = y_t − (α̂ + β̂·x_t) 를 돌려준다.
    (현재 봉은 계수 추정에 쓰지 않는다.)
    """
    r = lambda s: s.rolling(W, min_periods=W).sum().shift(1)
    n = W
    sx, sy = r(x), r(y)
    sxx, sxy = r(x * x), r(x * y)

    denom = n * sxx - sx * sx
    beta = np.where(np.abs(denom) < EPS, np.nan, (n * sxy - sx * sy) / denom)
    alpha = (sy - beta * sx) / n
    resid = y - (alpha + beta * x)
    return pd.Series(resid, index=y.index), pd.Series(alpha, index=y.index), pd.Series(beta, index=y.index)


def build(interval: str) -> dict:
    W = REG_W[interval]
    b = pq.read_table(BARS / f"bars_{interval}.parquet").to_pandas()
    b = b.sort_values("bar_start").reset_index(drop=True)

    d = pd.DataFrame({
        "bar_start": b.bar_start,
        "timestamp": b.timestamp,
        "close": b.close,
        "V": b.V_usdt,
        "has_gap": b.has_gap,
        "gap_s": b.gap_s,
    })

    # --- 1) 로그수익률 ---
    d["r"] = np.log(b.close / b.close.shift(1))

    # --- 2) CVD 흐름 (검증 기준) ---
    d["flow_cvd"] = b.cvd_usdt

    # --- 3) BVC 흐름 ---
    # σ_roll: [t−50, t−1] 의 수익률 표준편차 (현재 봉 제외)
    d["sigma"] = d.r.rolling(SIGMA_W, min_periods=SIGMA_W).std().shift(1)
    z = (d.r / d.sigma).replace([np.inf, -np.inf], np.nan)
    d["z"] = z
    mult = 2.0 * tdist.cdf(z.to_numpy(), df=NU) - 1.0
    mult = np.where(np.isnan(z.to_numpy()), np.nan, mult)
    d["bvc_mult"] = mult
    d["flow_bvc"] = d.V * mult

    # --- 4) CLV 흐름 ---
    rng = (b.high - b.low).to_numpy()
    clv = np.where(rng <= 0, 0.0, (2.0 * b.close - b.high - b.low).to_numpy() / np.where(rng <= 0, 1.0, rng))
    d["clv"] = clv
    d["flow_clv"] = d.V * clv
    n_flat = int((rng <= 0).sum())

    # --- 5) 직교 잔차: V·CLV 자체를 회귀한다 (회귀가 마지막 연산) ---
    res, alpha, beta = rolling_ols_residual(d.flow_clv, d.r, W)
    d["flow_res"] = res
    d["reg_alpha"] = alpha
    d["reg_beta"] = beta

    # --- 6) 공통 표본 절단: 가장 긴 웜업에 다섯 모형 시작점을 맞춘다 ---
    need = ["r", "flow_cvd", "flow_bvc", "flow_clv", "flow_res"]
    first_valid = {c: int(d[c].first_valid_index()) for c in need}
    start = max(first_valid.values())
    cut = d.iloc[start:].reset_index(drop=True)
    assert cut[need].notna().all().all(), "절단 후에도 결측이 남아 있음"

    # --- 진단 ---
    diag = {
        "interval": interval,
        "sigma_window": SIGMA_W,
        "reg_window": W,
        "nu": NU,
        "bars_total": int(len(d)),
        "first_valid_index": first_valid,
        "common_start_index": int(start),
        "bars_after_cut": int(len(cut)),
        "retained_pct": round(len(cut) / len(d) * 100, 2),
        "utc_start": str(cut.timestamp.iloc[0]),
        "utc_end": str(cut.timestamp.iloc[-1]),
        "high_eq_low_bars": n_flat,
        "flagged_bars_after_cut": int(cut.has_gap.sum()),
        # 직교성 점검 (구성이 의도대로 됐는지 확인하는 진단)
        "corr_res_r": round(float(cut.flow_res.corr(cut.r)), 6),
        # 기술통계 (측정 대상이지 검증 수단이 아님)
        "corr_flow_r": {
            "cvd": round(float(cut.flow_cvd.corr(cut.r)), 4),
            "bvc": round(float(cut.flow_bvc.corr(cut.r)), 4),
            "clv": round(float(cut.flow_clv.corr(cut.r)), 4),
            "res": round(float(cut.flow_res.corr(cut.r)), 4),
        },
        "corr_flow_cvd": {
            "bvc": round(float(cut.flow_bvc.corr(cut.flow_cvd)), 4),
            "clv": round(float(cut.flow_clv.corr(cut.flow_cvd)), 4),
            "res": round(float(cut.flow_res.corr(cut.flow_cvd)), 4),
        },
    }

    out_cols = ["bar_start", "timestamp", "r", "flow_cvd", "flow_bvc", "flow_clv", "flow_res",
                "V", "close", "sigma", "z", "bvc_mult", "clv", "reg_alpha", "reg_beta",
                "has_gap", "gap_s"]
    out = FEAT / f"features_{interval}.parquet"
    pq.write_table(pa.Table.from_pandas(cut[out_cols], preserve_index=False), out, compression="zstd")

    print(f"--- {interval} ---")
    print(f"  σ 윈도우 {SIGMA_W}  회귀 윈도우 W={W}  ν={NU}")
    print(f"  최초 유효 인덱스: {first_valid}")
    print(f"  공통 시작 {start}  ->  {len(cut):,}봉 남음 ({diag['retained_pct']}%)")
    print(f"  구간 {diag['utc_start'][:16]} ~ {diag['utc_end'][:16]} UTC")
    print(f"  H==L 봉 {n_flat}개   절단 후 결측플래그 봉 {diag['flagged_bars_after_cut']}개")
    print(f"  직교성 corr(잔차, r) = {diag['corr_res_r']:+.6f}")
    print(f"  corr(흐름, r):   " + "  ".join(f"{k} {v:+.4f}" for k, v in diag["corr_flow_r"].items()))
    print(f"  corr(흐름, CVD): " + "  ".join(f"{k} {v:+.4f}" for k, v in diag["corr_flow_cvd"].items()))
    print(f"  저장 {out}  ({out.stat().st_size/1024:,.0f} KB)\n")
    return diag


def main():
    FEAT.mkdir(parents=True, exist_ok=True)
    print("E010~E013 변수 생성\n")
    rep = {iv: build(iv) for iv in ("4h", "1d")}
    p = LOGS / "E010_features_report.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"리포트: {p}")


if __name__ == "__main__":
    sys.exit(main())
