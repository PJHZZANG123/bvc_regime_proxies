"""
논문 4장 표에 들어가는 값 가운데 기존 결과 파일에 없는 것을 계산한다.

  1) M1 기준으로 정렬한 각 모형의 국면 비중 (헝가리안 대응 적용 후)
  2) 각 모형의 가장 낮은 지속확률 (E030_models.json 에서 그대로)
  3) M1 국면별 특성 — 봉 수, |r| 중앙값, 거래대금 중앙값, CVD/V 평균, CVD 가 양수인 비율

나머지(Δκ, 신뢰구간, 영가설, 국면별 일치율, 로버스트니스, ETH)는 기존 결과 파일을 쓴다.

    python src/chapter4_tables.py

출력: results/tables/T_ch4.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
MODELS = ROOT / "results" / "models"
LOGS = ROOT / "results" / "logs"
OUT = ROOT / "results" / "tables" / "T_ch4.json"

CONFIGS = [("4h", 3), ("1d", 2)]
NAMES = ["M1", "M2", "M3", "M4", "M5"]


def load_states(interval, K):
    fit = json.loads((LOGS / "E030_models.json").read_text(encoding="utf-8"))[interval]["models"]
    st = {nm: np.load(MODELS / fit[nm]["file"])["states"] for nm in NAMES}
    return st, fit


def align(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def main():
    out = {}
    for iv, K in CONFIGS:
        st, fit = load_states(iv, K)
        ref = st["M1"]
        f = pd.read_parquet(FEAT / f"features_{iv}.parquet")
        assert len(f) == len(ref)
        res = {"shares_aligned_to_M1": {}, "min_persist": {}}
        for nm in NAMES:
            al = ref if nm == "M1" else align(ref, st[nm], K)
            res["shares_aligned_to_M1"][nm] = [round(float((al == j).mean()), 4) for j in range(K)]
            res["min_persist"][nm] = round(float(fit[nm]["min_persist"]), 4)
        g = f.assign(state=ref, absr=f.r.abs(), ratio=f.flow_cvd / f.V).groupby("state")
        res["regime_profile"] = {
            int(k): {
                "n_bars": int(len(v)),
                "share": round(len(v) / len(f), 4),
                "median_abs_r_bp": round(float(v.absr.median() * 1e4), 1),
                "median_V_eok_usdt": round(float(v.V.median() / 1e8), 1),
                "mean_cvd_over_V": round(float(v.ratio.mean()), 4),
                "cvd_positive_pct": round(float((v.flow_cvd > 0).mean() * 100), 1),
            } for k, v in g
        }
        out[iv] = res
        print(iv, json.dumps(res["shares_aligned_to_M1"], ensure_ascii=False))
        for k, v in res["regime_profile"].items():
            print("   상태", k, v)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
