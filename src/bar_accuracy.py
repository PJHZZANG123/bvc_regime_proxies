"""
봉 단위 거래량 분류 정확도 — BVC 와 매수·매도를 반반으로 나누는 단순 규칙.

정확도는 ELO(2012a, 식 2)와 CPS(2015)의 정의를 따른다.
    Ar = Σ_τ [ min(V^B, V̂^B) + min(V^S, V̂^S) ] / Σ_τ V
실제 매수·매도 거래대금은 CVD 와 V 에서 얻는다: V^B = (V + CVD)/2, V^S = (V − CVD)/2.
BVC 의 추정치는 V̂^B = (V + BVC)/2 이므로 Ar = 1 − Σ|CVD − BVC| / (2 Σ V) 와 같다.
반반 규칙은 V̂^B = V̂^S = V/2 로 둔 것으로, Ar = 1 − Σ|CVD| / (2 Σ V) 다.

분석 표본(공통 시작 시점 뒤)의 4시간봉·일봉에서 계산한다.

    python src/bar_accuracy.py
출력: results/tables/T_bar_accuracy.json
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
TABLES = ROOT / "results" / "tables"


def accuracy(V, vb_true, vb_hat):
    vs_true, vs_hat = V - vb_true, V - vb_hat
    return float((np.minimum(vb_true, vb_hat) + np.minimum(vs_true, vs_hat)).sum() / V.sum())


def main():
    out = {}
    for iv in ("4h", "1d"):
        d = pq.read_table(FEAT / f"features_{iv}.parquet").to_pandas()
        d = d.dropna(subset=["flow_bvc", "flow_cvd", "V"])
        V = d.V.to_numpy(float)
        C = d.flow_cvd.to_numpy(float)
        B = d.flow_bvc.to_numpy(float)
        vb_true = (V + C) / 2
        res = dict(
            n_bars=int(len(d)),
            acc_bvc=accuracy(V, vb_true, (V + B) / 2),
            acc_half=accuracy(V, vb_true, V / 2),
            sign_agree=float(np.mean(np.sign(C) == np.sign(B))),
            median_abs_bvc_over_v=float(np.median(np.abs(B) / V)),
            median_abs_cvd_over_v=float(np.median(np.abs(C) / V)),
        )
        out[iv] = res
        print(f"{iv}: 봉 {res['n_bars']:,} | 정확도 BVC {res['acc_bvc']:.4f}, 반반 {res['acc_half']:.4f} | "
              f"부호 일치 {res['sign_agree']:.4f} | |BVC|/V 중앙 {res['median_abs_bvc_over_v']:.4f}, "
              f"|CVD|/V 중앙 {res['median_abs_cvd_over_v']:.4f}")
    p = TABLES / "T_bar_accuracy.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
