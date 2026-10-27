"""
논문 부록 표의 값.

  표 A1  관측값 변환 방식 비교 — 변환 뒤 초과첨도 (4시간봉 수익률·CVD)
         국면 수 기준 통과 여부는 실험 로그 E020b 의 기록을 그대로 쓴다(여기서 재적합하지 않음).
  표 B1  로버스트니스 전체 — results/tables/E05x_robustness_core.json, E058_E060_alt_bars.json
  표 C1  블록 길이별 Δκ 95% 구간 — results/tables/E044_delta_ci.json

    python src/appendix_tables.py

출력: results/tables/T_appendix.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import transform

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "T_appendix.json"


def winsorize4(X: np.ndarray) -> np.ndarray:
    """평균에서 표준편차 4배를 넘는 값을 경계값으로 눌러 둔 뒤 표준화 (E020b 의 윈저화 ±4sd)."""
    Z = transform.zscore(X)
    return transform.zscore(np.clip(Z, -4, 4))


def main():
    f = pd.read_parquet(FEAT / "features_4h.parquet")
    X = np.column_stack([f.r.to_numpy(), f.flow_cvd.to_numpy()])
    methods = {"zscore": transform.zscore, "winsorize4": winsorize4,
               "signed_log": transform.signed_log, "rank_normal": transform.rank_normal}
    a1 = {nm: {"exkurt_r": float(stats.kurtosis(fn(X)[:, 0])),
               "exkurt_cvd": float(stats.kurtosis(fn(X)[:, 1]))} for nm, fn in methods.items()}

    rob = json.loads((TABLES / "E05x_robustness_core.json").read_text(encoding="utf-8"))
    b1 = {k: {m: {"kappa": v["deltas"][m]["kappa"], "ci": v["deltas"][m]["ci_kappa"]}
              for m in ("M2", "M4", "M5")} | {"n_bars": v["n_bars"]} for k, v in rob.items()}
    alt = json.loads((TABLES / "E058_E060_alt_bars.json").read_text(encoding="utf-8"))
    b1_alt = {k: {"n_bars": v["n_bars"], "k_star": v["k_star"],
                  "control_kappa": v["agreement"]["M3"]["kappa"],
                  **{m: v["deltas"][m]["delta_kappa"] for m in ("M2", "M4", "M5")}}
              for k, v in alt.items()}

    ci = json.loads((TABLES / "E044_delta_ci.json").read_text(encoding="utf-8"))
    c1 = {}
    for key, v in ci.items():
        c1[key] = {L: {m: {"kappa": b[m]["point_kappa"], "ci_kappa": b[m]["ci_kappa"],
                           "agree": b[m]["point_agree"], "ci_agree": b[m]["ci_agree"]}
                       for m in ("M2", "M4", "M5")}
                   for L, b in v["by_block"].items()}

    out = {"A1": a1, "B1": b1, "B1_alt_bars": b1_alt, "C1": c1}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for nm, v in a1.items():
        print(f"A1 {nm:12} 첨도 r {v['exkurt_r']:+.2f}  CVD {v['exkurt_cvd']:+.2f}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
