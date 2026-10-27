"""
E071 — 강건성 변형에서의 전체 카파와 시기 조정 카파 (4시간봉, K=3).

robustness_core.py(E05x)의 변형을 그대로 쓰되, 시간 대조군(T)과 시기 조정 카파(연도·분기)를 더한다.
변형마다 M1–M5 와 T 를 다시 추정하고, 통제군 대비 증분의 95% 구간을 같은 재추출 위치로 구한다
(순환 블록 200봉, 2,000회, 시드 20260915). 시기 조정 카파의 정의는 time_structure.py(E070)와 같다.

    python src/robustness_period.py
"""

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform  # noqa: E402
from robustness_core import BARS, BASE, INTERVAL, K, VARIANTS, align, make_features  # noqa: E402
from time_structure import circ_blocks, kappa_fast, kappa_strat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
SEEDS = list(range(10))
N_BOOT, BLOCK, RNG_SEED = 2000, 200, 20260915
WORKERS = 7
SPEC = {"M1": ["r", "flow_cvd"], "M2": ["r", "flow_bvc"], "M3": ["r"],
        "M4": ["r", "flow_clv"], "M5": ["r", "flow_res"], "T": ["r", "t"]}


def fit_job(args):
    key, raw, cov = args
    X = transform.apply(np.asarray(raw, float), transform.METHOD)
    best = None
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type=cov, n_iter=500, tol=1e-4, random_state=s)
        try:
            m.fit(X); ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll)
    return key, best[0].predict(X).astype(int)


def main():
    b = pq.read_table(BARS / f"bars_{INTERVAL}.parquet").to_pandas().sort_values("bar_start").reset_index(drop=True)
    data, jobs = {}, []
    for name, over, _ in VARIANTS:
        p = {**BASE, **over}
        d, start = make_features(p)
        d["t"] = np.arange(len(d), dtype=float)
        ts = pd.to_datetime(b.bar_start.iloc[start:start + len(d)].to_numpy(), unit="ms", utc=True)
        data[name] = (d, ts, p)
        jobs += [((name, nm), d[cols].to_numpy(float), p["cov"]) for nm, cols in SPEC.items()]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        states = dict(pool.map(fit_job, jobs, chunksize=1))

    rep = {}
    print(f"E071 — 강건성 변형의 전체·시기 조정 카파  ({INTERVAL}, K={K})")
    for name, (d, ts, p) in data.items():
        ref = states[(name, "M1")]
        n = len(ref)
        al = {nm: align(ref, states[(name, nm)]) for nm in ["M2", "M3", "M4", "M5", "T"]}
        codes = {"year": np.unique(ts.year, return_inverse=True)[1].astype(np.int64),
                 "quarter": np.unique(ts.year * 10 + ts.quarter, return_inverse=True)[1].astype(np.int64)}
        nS = {k: int(v.max()) + 1 for k, v in codes.items()}
        point = {nm: {"kappa": kappa_fast(ref, a, K),
                      **{f"kappa_{st}": kappa_strat(ref, a, codes[st], K, nS[st]) for st in codes}}
                 for nm, a in al.items()}
        rng = np.random.default_rng(RNG_SEED)
        labs = ["kappa"] + [f"kappa_{st}" for st in codes]
        B = {lab: {nm: np.empty(N_BOOT) for nm in al} for lab in labs}
        for i in range(N_BOOT):
            idx = circ_blocks(n, BLOCK, rng)
            rb = ref[idx]
            cb = {st: c[idx] for st, c in codes.items()}
            for nm, a in al.items():
                ab = a[idx]
                B["kappa"][nm][i] = kappa_fast(rb, ab, K)
                for st in codes:
                    B[f"kappa_{st}"][nm][i] = kappa_strat(rb, ab, cb[st], K, nS[st])
        deltas = {}
        for nm in ["M2", "M4", "M5", "T"]:
            deltas[nm] = {}
            for lab in labs:
                dist = B[lab][nm] - B[lab]["M3"]
                lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
                deltas[nm][lab] = dict(point=point[nm][lab] - point["M3"][lab], ci=[lo, hi], zero_in=bool(lo <= 0 <= hi))
        rep[name] = dict(params=p, n_bars=n, point=point, deltas=deltas)
        cells = "  ".join(f"{nm} {deltas[nm]['kappa']['point']:+.3f}/{deltas[nm]['kappa_quarter']['point']:+.3f}"
                          f"[{deltas[nm]['kappa_quarter']['ci'][0]:+.3f},{deltas[nm]['kappa_quarter']['ci'][1]:+.3f}]"
                          for nm in ["M2", "M4", "M5", "T"])
        print(f"  {name:<16}{n:>7,}  {cells}", flush=True)
    out = TABLES / "E071_robustness_period.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
