"""
논문 3.2절 표 3 — 흐름 변수와 CVD·수익률의 상관.

흐름 자료는 꼬리가 두꺼워 피어슨 상관 하나만으로는 믿기 어렵다. 세 방식으로 잰다.
  피어슨 (원자료) / 스피어만 순위 상관 / 부호보존 로그 변환 후 피어슨 (HMM 이 실제로 보는 값)

이 상관은 측정 대상을 기술하는 값이지 부호 검증 수단이 아니다(3.1.3절 참조).

    python src/flow_correlations.py

출력: results/tables/T3_flow_corr.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transform import signed_log

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "tables" / "T3_flow_corr.json"

FLOWS = ["flow_bvc", "flow_clv", "flow_res"]


def corr_block(f: pd.DataFrame, target: str) -> dict:
    cols = FLOWS + (["r"] if target != "r" else [])
    X = f[cols + [target]]
    S = pd.DataFrame(signed_log(X.to_numpy()), columns=X.columns)
    return {
        "pearson": X.corr().loc[cols, target].round(4).to_dict(),
        "spearman": X.corr(method="spearman").loc[cols, target].round(4).to_dict(),
        "pearson_signed_log": S.corr().loc[cols, target].round(4).to_dict(),
    }


def main():
    out = {"settings": {"source": "data/features/features_{4h,1d}.parquet (공통 표본)",
                        "signed_log": "transform.signed_log (전체 표본 MAD, 이후 표준화)"}}
    for iv in ("4h", "1d"):
        f = pd.read_parquet(FEAT / f"features_{iv}.parquet")
        out[iv] = {"n_bars": int(len(f)),
                   "with_cvd": corr_block(f, "flow_cvd"),
                   "with_r": corr_block(f, "r")}
        c = out[iv]["with_cvd"]
        print(f"{iv} {len(f):,}봉  CVD와 순위 상관 — BVC {c['spearman']['flow_bvc']:.3f} / "
              f"수익률 {c['spearman']['r']:.3f} / CLV {c['spearman']['flow_clv']:.3f}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
