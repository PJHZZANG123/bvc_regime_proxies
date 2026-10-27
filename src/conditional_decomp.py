"""
E043 — 조건부 분해. 프록시의 정보가 어디서 새는가.

배경
  E042(순열)에서 M4·M5 는 자기 섞인 판본을 유의하게 이겼다 → 실제 정보가 있다.
  E044(구간)에서 Δ 는 0 과 구분되지 않았다 → 통제군은 못 이긴다.
  둘을 합치면 "정보는 있는데 수익률을 못 이긴다"가 된다. 그 정보가 어디서
  상쇄되는지를 본다.

두 가지로 본다
  1. 기준 국면별 일치율 — M1 의 어느 상태를 재현하고 어느 상태를 놓치는가
  2. 로짓 — 봉의 성격(변동성·거래량·흐름강도)에 따라 프록시의 우위가 달라지는가
     프록시와 통제군을 쌓아 `is_proxy` 더미와 교차항을 넣고, 같은 봉에서 나온 두 행이
     상관되므로 봉 단위 군집 표준오차를 쓴다.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"

CONFIGS = [("4h", 3), ("1d", 2)]
LABEL = {"M2": "BVC", "M3": "통제군", "M4": "CLV", "M5": "직교 잔차"}


def load_aligned(interval, K):
    st = {}
    for nm in ["M1", "M2", "M3", "M4", "M5"]:
        f = sorted(MODELS.glob(f"{nm}_{interval}_K{K}_seed*.npz"))[0]
        st[nm] = np.load(f, allow_pickle=True)["states"].astype(int)
    ref = st["M1"]
    out = {"M1": ref}
    for nm in ["M2", "M3", "M4", "M5"]:
        C = confusion_matrix(ref, st[nm], labels=list(range(K)))
        r, c = linear_sum_assignment(-C)
        mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
        out[nm] = np.array([mp[int(s)] for s in st[nm]])
    return out


def run(interval, K):
    d = pq.read_table(FEAT / f"features_{interval}.parquet").to_pandas()
    st = load_aligned(interval, K)
    ref = st["M1"]
    n = len(ref)
    print(f"\n{'='*104}\n{interval}  K={K}  {n:,}봉\n{'='*104}")

    # ---------- 1. 기준 국면별 일치율 ----------
    print(f"\n  [1] M1 국면별 일치율")
    hdr = f"    {'모형':<12}" + "".join(f"{'상태'+str(k):>12}" for k in range(K)) + f"{'전체':>12}"
    print(hdr)
    print("    " + "-" * (12 + 12 * (K + 1)))
    rates = {}
    for nm in ["M2", "M3", "M4", "M5"]:
        ag = (st[nm] == ref).astype(int)
        per = [float(ag[ref == k].mean()) for k in range(K)]
        rates[nm] = dict(by_state=per, overall=float(ag.mean()))
        print(f"    {LABEL[nm]:<12}" + "".join(f"{p:>12.4f}" for p in per)
              + f"{ag.mean():>12.4f}")
    counts = [int((ref == k).sum()) for k in range(K)]
    print(f"    {'(봉 수)':<12}" + "".join(f"{c:>12,}" for c in counts) + f"{n:>12,}")

    print(f"\n  [2] 국면별 Δ  (프록시 − 통제군)")
    print(f"    {'갈래':<12}" + "".join(f"{'상태'+str(k):>12}" for k in range(K)))
    print("    " + "-" * (12 + 12 * K))
    delta_by_state = {}
    for nm in ["M2", "M4", "M5"]:
        dd = [rates[nm]["by_state"][k] - rates["M3"]["by_state"][k] for k in range(K)]
        delta_by_state[nm] = dd
        print(f"    {LABEL[nm]:<12}" + "".join(f"{x:>+12.4f}" for x in dd))

    # ---------- 3. 로짓 ----------
    absr = d.r.abs().to_numpy()
    logv = np.log(d.V.to_numpy())
    intens = (d.flow_cvd.abs() / d.V).to_numpy()
    z = lambda v: (v - v.mean()) / v.std()
    cov = pd.DataFrame({"absr": z(absr), "logv": z(logv), "intens": z(intens)})

    print(f"\n  [3] 로짓 — 프록시 우위가 봉의 성격에 따라 달라지는가")
    print(f"       종속변수: M1 과 일치(1/0).  두 모형을 쌓고 is_proxy 및 교차항.")
    print(f"       봉 단위 군집 표준오차.  계수는 로그오즈.\n")
    logit_out = {}
    for nm in ["M2", "M4", "M5"]:
        y = np.concatenate([(st[nm] == ref).astype(int), (st["M3"] == ref).astype(int)])
        X = pd.concat([cov, cov], ignore_index=True)
        X["is_proxy"] = np.concatenate([np.ones(n), np.zeros(n)])
        for c in ("absr", "logv", "intens"):
            X[f"{c}:proxy"] = X[c] * X["is_proxy"]
        X = sm.add_constant(X)
        groups = np.concatenate([np.arange(n), np.arange(n)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                   cov_kwds={"groups": groups})
        keep = ["is_proxy", "absr:proxy", "logv:proxy", "intens:proxy"]
        logit_out[nm] = {k: dict(coef=float(m.params[k]), se=float(m.bse[k]),
                                 p=float(m.pvalues[k])) for k in keep}
        print(f"    {LABEL[nm]}")
        print(f"      {'항':<16}{'계수':>10}{'표준오차':>10}{'p':>9}")
        for k in keep:
            star = " *" if m.pvalues[k] < 0.05 else ""
            print(f"      {k:<16}{m.params[k]:>+10.4f}{m.bse[k]:>10.4f}"
                  f"{m.pvalues[k]:>9.4f}{star}")
        print()

    return dict(interval=interval, K=K, n_bars=n, state_counts=counts,
                agree_by_state=rates, delta_by_state=delta_by_state, logit=logit_out)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("E043 — 조건부 분해")
    rep = {f"{iv}_K{K}": run(iv, K) for iv, K in CONFIGS}
    p = TABLES / "E043_conditional.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트: {p}")


if __name__ == "__main__":
    sys.exit(main())
