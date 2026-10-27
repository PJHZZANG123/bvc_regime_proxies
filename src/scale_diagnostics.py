"""
논문 3.2.1절·3.3.2절의 척도 관련 수치.

  1) BVC 롤링 σ 의 변동 폭 — 전체 표본 σ 하나로 부족한 이유
  2) 부호보존 로그 변환의 척도로 표준편차 대신 MAD 를 쓴 이유
     - 표준편차 / (1.4826·MAD) : 정규분포면 1
     - 가장 극단적인 1% 봉을 뺄 때 표준편차와 MAD 가 얼마나 줄어드는가

변환 방식(E020b)을 고른 뒤에 확인한 기술 수치다. 선택 근거가 아니라 설명용이다.

    python src/scale_diagnostics.py

출력: results/tables/T_scale_diagnostics.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "tables" / "T_scale_diagnostics.json"

SERIES = ["r", "flow_cvd", "flow_bvc", "flow_clv", "flow_res"]
NORMAL_SD_PER_MAD = 1.4826


def mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def main():
    out = {}
    for iv in ("4h", "1d"):
        f = pd.read_parquet(FEAT / f"features_{iv}.parquet")
        s = f.sigma * 100
        res = {"sigma_roll_pct": {
            "p05": float(s.quantile(0.05)), "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "yearly_median": {int(k): float(v) for k, v in
                              (f.sigma * 100).groupby(f.timestamp.dt.year).median().items()}}}
        for c in SERIES:
            x = f[c].to_numpy()
            dev = np.abs(x - np.median(x))
            keep = dev <= np.quantile(dev, 0.99)
            sd, m = x.std(ddof=1), mad(x)
            res[c] = {
                "sd_over_normal_mad": float(sd / (NORMAL_SD_PER_MAD * m)),
                "sd_change_drop_top1_pct": float((x[keep].std(ddof=1) / sd - 1) * 100),
                "mad_change_drop_top1_pct": float((mad(x[keep]) / m - 1) * 100),
            }
        out[iv] = res
        print(f"{iv}  σ p05 {res['sigma_roll_pct']['p05']:.2f}% · p95 {res['sigma_roll_pct']['p95']:.2f}%  |  "
              + "  ".join(f"{c} {res[c]['sd_over_normal_mad']:.2f}" for c in SERIES))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
