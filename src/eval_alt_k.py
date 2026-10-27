"""
E041c — 필터를 통과한 다른 K* 후보에서 Δ를 재평가.

규약(CLAUDE.md): 필터를 통과한 K* 후보가 둘 이상이면 전부에서 Δ를 보고한다.
4시간봉 M1은 K=2와 K=3이 모두 통과했으므로 둘 다 싣는다.
같은 추정 대상을 규약이 허용한 다른 K*에서 평가하는 것이므로 설계 안쪽 작업이다.

학습·정렬·지표 산출은 본 파이프라인(E030/E040/E041)과 동일한 설정을 쓴다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

SEEDS = list(range(10))
N_ITER, TOL, COV = 500, 1e-4, "full"

SPEC = {
    "M1": (["r", "flow_cvd"], "CVD (기준)"),
    "M2": (["r", "flow_bvc"], "BVC"),
    "M3": (["r"], "통제군"),
    "M4": (["r", "flow_clv"], "CLV"),
    "M5": (["r", "flow_res"], "직교 잔차"),
}

# (바, 평가할 K) — 규약상 필터를 통과한 후보
TARGETS = [("4h", 2)]


def fit_best(X, K):
    best = None
    for seed in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type=COV,
                        n_iter=N_ITER, tol=TOL, random_state=seed)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best["ll"]):
            best = {"model": m, "ll": float(ll), "seed": seed,
                    "converged": bool(m.monitor_.converged)}
    return best


def align(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def run(interval, K):
    d = pq.read_table(FEAT / f"features_{interval}.parquet").to_pandas()
    n = len(d)
    print(f"\n{'='*100}\n{interval}   K = {K}   {n:,}봉   변환 {transform.METHOD}\n{'='*100}")

    states, fits = {}, {}
    for nm, (cols, label) in SPEC.items():
        X = transform.apply(d[cols].to_numpy(float), transform.METHOD)
        b = fit_best(X, K)
        m = b["model"]
        st = m.predict(X)
        states[nm] = st
        share = np.bincount(st, minlength=K) / n
        pers = np.diag(m.transmat_)
        fits[nm] = dict(seed=b["seed"], loglik=b["ll"], bic=float(m.bic(X)),
                        converged=b["converged"],
                        shares=[round(float(s), 4) for s in share],
                        min_share=float(share.min()), min_persist=float(pers.min()))
        fn = MODELS / f"{nm}_{interval}_K{K}_seed{b['seed']}.npz"
        np.savez(fn, startprob=m.startprob_, transmat=m.transmat_, means=m.means_,
                 covars=m.covars_, states=st, bar_start=d.bar_start.to_numpy(),
                 cols=np.array(cols), transform=transform.METHOD)
        print(f"  {nm}  [{', '.join(cols):<18}]  logL {b['ll']:>11,.1f}  seed {b['seed']}  "
              f"비중 {np.round(np.sort(share),3)}  최소지속 {pers.min():.3f}")

    ref = states["M1"]
    res = {}
    print(f"\n  [M1 대비 일치도]")
    print(f"    {'모형':<16}{'일치율':>10}{'카파':>10}")
    for nm in ["M2", "M3", "M4", "M5"]:
        al = align(ref, states[nm], K)
        a = float((al == ref).mean())
        kp = float(cohen_kappa_score(ref, al))
        res[nm] = dict(agree=a, kappa=kp)
        print(f"    {SPEC[nm][1]:<16}{a:>10.4f}{kp:>10.4f}")

    base_a, base_k = res["M3"]["agree"], res["M3"]["kappa"]
    print(f"\n  [Δ]  통제군 기준선: 일치율 {base_a:.4f}, 카파 {base_k:.4f}")
    deltas = {}
    for nm in ["M2", "M4", "M5"]:
        da = res[nm]["agree"] - base_a
        dk = res[nm]["kappa"] - base_k
        deltas[nm] = dict(delta_agree=da, delta_kappa=dk)
        print(f"    Δ({nm}) {SPEC[nm][1]:<12}{da:>+11.4f}{dk:>+11.4f}")

    return dict(interval=interval, K=K, n_bars=n, fits=fits,
                agreement=res, deltas=deltas,
                baseline_agree=base_a, baseline_kappa=base_k)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    print("E041c — 규약상 통과한 다른 K* 후보에서 Δ 재평가")
    out = {f"{iv}_K{K}": run(iv, K) for iv, K in TARGETS}

    # 기존 K* 결과와 나란히
    main_res = json.loads((TABLES / "E041_delta.json").read_text(encoding="utf-8"))
    print(f"\n{'='*100}\n  K* 후보별 Δ 비교 (4시간봉)\n{'='*100}")
    print(f"    {'':<16}{'K=3 (BIC 최소)':>20}{'K=2':>20}")
    print(f"    {'':<16}{'일치율':>10}{'카파':>10}{'일치율':>10}{'카파':>10}")
    for nm, lbl in [("M2","Δ₂ BVC"), ("M4","Δ₄ CLV"), ("M5","Δ₅ 잔차")]:
        a3 = main_res["4h"]["deltas"][nm]
        a2 = out["4h_K2"]["deltas"][nm]
        print(f"    {lbl:<16}{a3['delta_agree']:>+10.4f}{a3['delta_kappa']:>+10.4f}"
              f"{a2['delta_agree']:>+10.4f}{a2['delta_kappa']:>+10.4f}")

    p = TABLES / "E041c_alt_k.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {p}")


if __name__ == "__main__":
    sys.exit(main())
