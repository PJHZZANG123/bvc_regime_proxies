"""
E042b — 순환 시차 이동 영가설 (E042 블록 셔플을 대체)

E042 의 문제
  블록 셔플은 프록시 자신의 자기상관까지 파괴한다. 그러면 영분포가 낮아진 이유가
  「시간 정렬이 깨져서」인지 「입력의 지속성이 줄어서」인지 구분되지 않는다.
  BVC 는 V 를 통해 거래량 군집을 물려받으므로 섞으면 덜 지속적인 입력이 된다.

이 검정
  프록시 계열을 무작위 시차만큼 **순환 이동**한다. 주변분포와 자기상관이
  **정확히 보존**되고 M1 과의 시점 정렬만 깨진다. 따라서 관측 Δ 가 이 영분포를
  넘으면 그 우위는 지속성이 아니라 **정렬**에서 온 것이다.

시차는 표본의 5~95% 구간에서 뽑아 원래 위치 근처를 피한다.
"""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "bootstrap"

INTERVAL, K = "4h", 3
N_REP = 500
WORKERS = 7
SEEDS_OBS, SEEDS_REP = list(range(10)), [0, 1, 2]
PROXY = {"M2": ("flow_bvc", "BVC"), "M4": ("flow_clv", "CLV"), "M5": ("flow_res", "직교 잔차")}


def fit_best(X, seeds, n_iter):
    best = None
    for s in seeds:
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=n_iter, tol=1e-4, random_state=s)
        try:
            m.fit(X); ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll)
    return best


def scores(ref, other):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    al = np.array([mp[int(s)] for s in other])
    return float((al == ref).mean()), float(cohen_kappa_score(ref, al))


def _one(args):
    r_col, flow, ref, base_a, base_k, seed = args
    n = len(flow)
    rng = np.random.default_rng(seed)
    lag = int(rng.integers(int(0.05 * n), int(0.95 * n)))
    shifted = np.roll(flow, lag)                    # 순환 이동 — 자기상관 완전 보존
    X = transform.apply(np.column_stack([r_col, shifted]), transform.METHOD)
    b = fit_best(X, SEEDS_REP, 300)
    if b is None:
        return None
    a, k = scores(ref, b[0].predict(X))
    return a - base_a, k - base_k, lag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pq.read_table(FEAT / f"features_{INTERVAL}.parquet").to_pandas()
    n = len(d)
    r_col = d.r.to_numpy(float)

    X1 = transform.apply(d[["r", "flow_cvd"]].to_numpy(float), transform.METHOD)
    ref = fit_best(X1, SEEDS_OBS, 500)[0].predict(X1)
    X3 = transform.apply(d[["r"]].to_numpy(float), transform.METHOD)
    s3 = fit_best(X3, SEEDS_OBS, 500)[0].predict(X3)
    base_a, base_k = scores(ref, s3)

    print(f"E042b — 순환 시차 이동 영가설  ({INTERVAL}, K={K}, {n:,}봉, {N_REP}회)")
    print(f"통제군 기준선: 일치율 {base_a:.4f}, 카파 {base_k:.4f}\n", flush=True)

    rep = {}
    for nm, (col, lbl) in PROXY.items():
        flow = d[col].to_numpy(float)
        Xo = transform.apply(np.column_stack([r_col, flow]), transform.METHOD)
        ao, ko = scores(ref, fit_best(Xo, SEEDS_OBS, 500)[0].predict(Xo))
        da, dk = ao - base_a, ko - base_k

        seeds = np.random.SeedSequence(20260916).spawn(N_REP)
        jobs = [(r_col, flow, ref, base_a, base_k, s) for s in seeds]
        na, nk = [], []
        t0 = time.time(); done = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(_one, j) for j in jobs]):
                res = fut.result(); done += 1
                if res:
                    na.append(res[0]); nk.append(res[1])
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f"    {nm} {done:>4}/{N_REP}  경과 {el/60:4.1f}분  "
                          f"남은예상 {el/done*(N_REP-done)/60:4.1f}분", flush=True)

        na, nk = np.array(na), np.array(nk)
        rep[nm] = dict(
            label=lbl, n_null=len(na),
            delta_obs_agree=da, delta_obs_kappa=dk,
            null_agree=dict(mean=float(na.mean()), sd=float(na.std()),
                            q025=float(np.percentile(na, 2.5)),
                            q975=float(np.percentile(na, 97.5))),
            null_kappa=dict(mean=float(nk.mean()), sd=float(nk.std()),
                            q025=float(np.percentile(nk, 2.5)),
                            q975=float(np.percentile(nk, 97.5))),
            p_agree=float((na >= da).mean()), p_kappa=float((nk >= dk).mean()),
        )
        print(f"\n  {nm} · {lbl}")
        print(f"    Δ 일치율  관측 {da:+.4f}   영분포 {na.mean():+.4f} (sd {na.std():.4f}) "
              f"95% [{np.percentile(na,2.5):+.4f}, {np.percentile(na,97.5):+.4f}]  p {(na>=da).mean():.4f}")
        print(f"    Δ 카파    관측 {dk:+.4f}   영분포 {nk.mean():+.4f} (sd {nk.std():.4f}) "
              f"95% [{np.percentile(nk,2.5):+.4f}, {np.percentile(nk,97.5):+.4f}]  p {(nk>=dk).mean():.4f}\n",
              flush=True)

    p = OUT / "E042b_circular_shift.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {p}")


if __name__ == "__main__":
    sys.exit(main())
