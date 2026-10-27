"""
논문 3장 표 1 — 분석 표본의 기술통계.

봉 파일에서 직접 계산한다. 분석 표본은 변수 웜업(BVC σ 창 50, 직교 잔차 회귀 창
4h 100 / 1d 50)을 뺀 뒤의 구간이며, build_features.py 의 공통 시작 인덱스와 같다.
BTC 는 features 파일의 r 과 일치하는지 확인한다.

    python src/describe_sample.py

출력: results/tables/T1_descriptive.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "bars"
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "tables" / "T1_descriptive.json"

CUT = {"4h": 101, "1d": 51}          # 공통 시작 인덱스 (E010_features_report.json)
ASSETS = {"BTC": "", "ETH": "_ETHUSDT"}


def describe(b: pd.DataFrame) -> dict:
    r, V, c = b.r, b.V_usdt, b.cvd_usdt
    return {
        "n_bars": int(len(b)),
        "start_utc": str(b.timestamp.iloc[0]),
        "end_utc": str(b.timestamp.iloc[-1]),
        "r_mean_pct": float(r.mean() * 100),
        "r_sd_pct": float(r.std() * 100),
        "r_skew": float(stats.skew(r)),
        "r_excess_kurt": float(stats.kurtosis(r)),
        "V_median_eok_usdt": float(V.median() / 1e8),
        "rows_per_bar_median_man": float(b.n_trades.median() / 1e4),
        "cvd_over_V_mean": float((c / V).mean()),
        "cvd_over_V_sd": float((c / V).std()),
        "abs_cvd_over_V_median": float((c / V).abs().median()),
        "cvd_positive_pct": float((c > 0).mean() * 100),
        "sign_agree_with_r_pct": float((np.sign(c) == np.sign(r)).mean() * 100),
        "r_zero_bars": int((r == 0).sum()),
        "flagged_bars": int(b.has_gap.sum()),
    }


def main():
    out = {"settings": {"cut_index": CUT, "r": "ln(C_t / C_{t-1})",
                        "V": "sum(p*q) USDT", "cvd": "sum(eps*p*q) USDT"}}
    for asset, suf in ASSETS.items():
        for iv, cut in CUT.items():
            b = pd.read_parquet(BARS / f"bars_{iv}{suf}.parquet")
            b = b.sort_values("bar_start").reset_index(drop=True)
            b["r"] = np.log(b.close / b.close.shift(1))
            b = b.iloc[cut:].reset_index(drop=True)
            if asset == "BTC":
                f = pd.read_parquet(FEAT / f"features_{iv}.parquet")
                assert len(f) == len(b) and np.allclose(f.r.to_numpy(), b.r.to_numpy()), \
                    f"{asset} {iv}: features 의 r 과 불일치"
            out[f"{asset}_{iv}"] = describe(b)
            d = out[f"{asset}_{iv}"]
            print(f"{asset} {iv}  {d['n_bars']:>6,}봉  sd {d['r_sd_pct']:.3f}%  "
                  f"초과첨도 {d['r_excess_kurt']:.2f}  부호일치 {d['sign_agree_with_r_pct']:.1f}%")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
