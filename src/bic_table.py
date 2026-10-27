"""
모형별·K별 BIC 정리표.

주의 — BIC 는 **같은 모형 안에서 K 끼리만** 비교 가능하다.
모형마다 관측 벡터(차원·데이터)가 달라 우도의 척도가 다르므로,
M1 의 BIC 와 M2 의 BIC 를 나란히 놓고 크기를 비교하면 안 된다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
LOGS = ROOT / "results" / "logs"
TABLES = ROOT / "results" / "tables"

SEEDS = list(range(10))
K_RANGE = range(2, 7)
MIN_SHARE, MIN_PERSIST = 0.05, 0.90

SPEC = {
    "M1": (["r", "flow_cvd"], "CVD (기준)"),
    "M2": (["r", "flow_bvc"], "BVC"),
    "M3": (["r"], "통제군"),
    "M4": (["r", "flow_clv"], "CLV"),
    "M5": (["r", "flow_res"], "직교 잔차"),
}


def fit_best(X, K):
    best = None
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=s)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll, s)
    return best


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    ksel = json.loads((LOGS / "E020_k_selection.json").read_text(encoding="utf-8"))
    out = {}

    for iv in ("4h", "1d"):
        Kstar = ksel[iv]["k_star"]
        d = pq.read_table(FEAT / f"features_{iv}.parquet").to_pandas()
        print(f"\n{'='*104}")
        print(f"{iv}   {len(d):,}봉   K* = {Kstar} (M1 기준)   변환 {transform.METHOD}")
        print(f"필터: 최소비중 ≥ {MIN_SHARE}, 지속확률 ≥ {MIN_PERSIST}")
        print(f"{'='*104}")
        out[iv] = {}

        for nm, (cols, label) in SPEC.items():
            X = transform.apply(d[cols].to_numpy(float), transform.METHOD)
            rows = []
            for K in K_RANGE:
                b = fit_best(X, K)
                if b is None:
                    continue
                m, ll, seed = b
                st = m.predict(X)
                share = np.bincount(st, minlength=K) / len(st)
                pers = np.diag(m.transmat_)
                rows.append(dict(
                    K=K, bic=float(m.bic(X)), aic=float(m.aic(X)), loglik=float(ll),
                    seed=int(seed), converged=bool(m.monitor_.converged),
                    min_share=float(share.min()), min_persist=float(pers.min()),
                    passes=bool(share.min() >= MIN_SHARE and pers.min() >= MIN_PERSIST),
                ))
            out[iv][nm] = dict(cols=cols, label=label, rows=rows)

            bic_min_K = min(rows, key=lambda r: r["bic"])["K"]
            surv = [r for r in rows if r["passes"]]
            own_K = min(surv, key=lambda r: r["bic"])["K"] if surv else None

            print(f"\n  {nm} · {label}   관측 [{', '.join(cols)}]")
            print(f"    {'K':>3}{'BIC':>13}{'ΔBIC':>10}{'logL':>13}"
                  f"{'최소비중':>10}{'지속확률':>10}{'seed':>6}   필터")
            print(f"    {'-'*76}")
            b0 = min(r["bic"] for r in rows)
            for r in rows:
                flag = "통과" if r["passes"] else "탈락"
                star = " *" if r["K"] == bic_min_K else "  "
                own = " <- 자기 K*" if own_K == r["K"] else ""
                print(f"    {r['K']:>3}{r['bic']:>13,.1f}{r['bic']-b0:>10,.1f}"
                      f"{r['loglik']:>13,.1f}{r['min_share']:>10.3f}"
                      f"{r['min_persist']:>10.3f}{r['seed']:>6}   {flag}{star}{own}")
            print(f"    BIC 최소 K = {bic_min_K}   필터 통과 K = {[r['K'] for r in surv] or '없음'}"
                  f"   자기 K* = {own_K}")

    p = TABLES / "E020_bic_table.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*104}")
    print("주의: BIC 는 같은 모형 안에서 K 끼리만 비교 가능. 모형 간 BIC 크기 비교는 무의미.")
    print(f"저장: {p}")


if __name__ == "__main__":
    sys.exit(main())
