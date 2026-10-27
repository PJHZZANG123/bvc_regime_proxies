"""
E030 — 다섯 모형을 K* 로 고정해 학습.

| 모형 | 관측 벡터        | 역할        |
|------|------------------|-------------|
| M1   | [r, CVD 흐름]    | 검증 기준   |
| M2   | [r, BVC 흐름]    | 기존 프록시 |
| M3   | [r]              | 통제군      |
| M4   | [r, CLV 흐름]    | 신규 프록시 |
| M5   | [r, 직교 잔차]   | 신규 프록시 |

K* 는 E020 에서 M1 만으로 결정한 값(4h=3, 1d=2)이며 다섯 모형에 동일하게 적용한다.
모형마다 K 가 다르면 라벨 매칭이 성립하지 않는다.

관측 변환(부호보존 로그)은 각 모형의 관측 벡터에 동일하게 적용한다.
시드 10개를 돌려 최고 우도를 채택하고 파일명에 기록한다.
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
MODELS = ROOT / "results" / "models"
LOGS = ROOT / "results" / "logs"

SEEDS = list(range(10))
N_ITER = 500
TOL = 1e-4
COV = "full"

SPEC = {
    "M1": ["r", "flow_cvd"],
    "M2": ["r", "flow_bvc"],
    "M3": ["r"],
    "M4": ["r", "flow_clv"],
    "M5": ["r", "flow_res"],
}


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
                    "converged": bool(m.monitor_.converged),
                    "n_iter_used": int(m.monitor_.iter)}
    return best


def run(interval, K):
    d = pq.read_table(FEAT / f"features_{interval}.parquet").to_pandas()
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*96}\n{interval}   K* = {K}   {len(d):,}봉   변환 {transform.METHOD}\n{'='*96}", flush=True)

    out = {}
    for name, cols in SPEC.items():
        X = transform.apply(d[cols].to_numpy(float), transform.METHOD)
        best = fit_best(X, K)
        if best is None:
            print(f"  {name}: 전 시드 실패"); continue
        m = best["model"]
        states = m.predict(X)
        share = np.bincount(states, minlength=K) / len(states)
        persist = np.diag(m.transmat_)

        fn = MODELS / f"{name}_{interval}_K{K}_seed{best['seed']}.npz"
        np.savez(fn, startprob=m.startprob_, transmat=m.transmat_,
                 means=m.means_, covars=m.covars_, states=states,
                 bar_start=d.bar_start.to_numpy(), cols=np.array(cols),
                 transform=transform.METHOD)

        out[name] = dict(
            cols=cols, K=K, seed=best["seed"], loglik=best["ll"],
            bic=float(m.bic(X)), aic=float(m.aic(X)),
            converged=best["converged"], n_iter_used=best["n_iter_used"],
            shares=[round(float(s), 4) for s in share],
            persist=[round(float(p), 4) for p in persist],
            min_share=float(share.min()), min_persist=float(persist.min()),
            file=fn.name,
        )
        print(f"  {name}  [{', '.join(cols):<20}]  logL {best['ll']:>11,.1f}  "
              f"BIC {float(m.bic(X)):>11,.1f}  seed {best['seed']}  "
              f"수렴 {'O' if best['converged'] else 'X'}({best['n_iter_used']:>3}회)  "
              f"비중 {np.round(np.sort(share), 3)}  최소지속 {persist.min():.3f}", flush=True)

    # 다섯 모형의 시점 축이 동일한지 확인 — 검증 단계 행렬 연산의 전제
    n = {k: len(np.load(MODELS / v["file"])["states"]) for k, v in out.items()}
    assert len(set(n.values())) == 1, f"모형별 길이 불일치: {n}"
    print(f"  시점 축 일치 확인: 다섯 모형 모두 {list(n.values())[0]:,}봉")
    return out


def main():
    ksel = json.loads((LOGS / "E020_k_selection.json").read_text(encoding="utf-8"))
    print("E030 — 다섯 모형 학습 (K* 고정)")
    rep = {}
    for iv in ("4h", "1d"):
        K = ksel[iv]["k_star"]
        if K is None:
            print(f"  {iv}: K* 미확정 — 건너뜀"); continue
        rep[iv] = {"k_star": K, "models": run(iv, K)}
    p = LOGS / "E030_models.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트: {p}")


if __name__ == "__main__":
    sys.exit(main())
