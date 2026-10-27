"""
논문 3.3.3절 — 기준 모형(M1) 분할의 성격을 재는 두 값 (E041d 의 계산을 재현).

  상태 평균의 분산   Σ_k π_k ‖μ_k − μ̄‖²,  μ̄ = Σ_k π_k μ_k
                     μ_k 는 변환(부호보존 로그 후 표준화)한 [수익률, CVD] 공간의 상태 평균,
                     π_k 는 비터비 상태열에서 상태 k 의 비중. 두 성분에 걸쳐 더한다.
                     → 국면들이 평균에서 얼마나 갈리는가
  수익률 표준편차의 비  max_k σ_k / min_k σ_k,  σ_k = √Σ_k[r, r]
                     → 국면들이 변동성에서 얼마나 갈리는가
  CVD 축의 몫        상태 평균의 분산 가운데 CVD 성분이 차지하는 비율

M1 의 적합 모수만 쓴다. Δ 도 프록시도 보지 않는다.

    python src/partition_character.py

출력: results/tables/T_partition_character.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"
OUT = ROOT / "results" / "tables" / "T_partition_character.json"

CASES = {"4h_K2": "M1_4h_K2_seed3.npz", "4h_K3": "M1_4h_K3_seed9.npz", "1d_K2": "M1_1d_K2_seed4.npz"}


def character(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    mu, cov, st = z["means"], z["covars"], z["states"]
    K = len(mu)
    pi = np.bincount(st, minlength=K) / len(st)
    center = (pi[:, None] * mu).sum(0)
    per_comp = (pi[:, None] * (mu - center) ** 2).sum(0)          # [수익률 성분, CVD 성분]
    sd_r = np.sqrt(cov[:, 0, 0])
    return {
        "K": int(K),
        "state_share": [round(float(x), 4) for x in pi],
        "mean_variance": float(per_comp.sum()),
        "mean_variance_by_component": {"r": float(per_comp[0]), "cvd": float(per_comp[1])},
        "cvd_axis_share": float(per_comp[1] / per_comp.sum()),
        "sd_ratio_r": float(sd_r.max() / sd_r.min()),
    }


def main():
    out = {k: character(MODELS / fn) for k, fn in CASES.items()}
    for k, v in out.items():
        print(f"{k}:  상태 평균의 분산 {v['mean_variance']:.4f}   수익률 표준편차 비 {v['sd_ratio_r']:.2f}   "
              f"CVD 축의 몫 {v['cvd_axis_share'] * 100:.1f}%")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
