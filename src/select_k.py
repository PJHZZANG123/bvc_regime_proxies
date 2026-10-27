"""
E020 — K 결정. **M1(벤치마크)만** 가지고 K 를 정한다.

절차 (CLAUDE.md)
  1. M1 = [r_t, CVD 흐름] 로 K = 2~6 학습
  2. 최소 비중 5% 미만 또는 지속확률 0.90 미만인 K 는 탈락
  3. 남은 것 중 BIC 최소를 K* 로 확정
  4. M2~M5 는 전부 K* 로 고정 (여기서는 확정만 하고 학습은 E030)

바 길이마다 K* 가 달라지는 것은 무방하다(바 안에서만 통일).

수렴 실패 대비로 여러 시드를 돌려 최고 우도를 채택하고, 시드를 결과에 기록한다.
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
OUT = ROOT / "results" / "k_selection"
LOGS = ROOT / "results" / "logs"

K_RANGE = range(2, 7)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
N_ITER = 500
TOL = 1e-4
COV = "full"

MIN_SHARE = 0.05      # 최소 비중 기준 (임의 설정 — 로버스트니스 E056 에서 3%/8% 확인)
MIN_PERSIST = 0.90    # 지속확률 기준 (임의 설정 — 로버스트니스 E057 에서 0.85/0.95 확인)


TRANSFORM = transform.METHOD    # 부호보존 로그 (E020b 확정)


def fit_best(X, K):
    """여러 시드로 돌려 최고 우도 모형을 고른다."""
    best = None
    for seed in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type=COV,
                        n_iter=N_ITER, tol=TOL, random_state=seed)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception as e:
            print(f"      seed {seed}: 실패 ({type(e).__name__})", flush=True)
            continue
        if not np.isfinite(ll):
            continue
        if best is None or ll > best["ll"]:
            best = {"model": m, "ll": float(ll), "seed": seed,
                    "converged": bool(m.monitor_.converged),
                    "n_iter_used": int(m.monitor_.iter)}
    return best


def evaluate(interval):
    d = pq.read_table(FEAT / f"features_{interval}.parquet").to_pandas()
    X_raw = d[["r", "flow_cvd"]].to_numpy(float)
    X = transform.apply(X_raw, TRANSFORM)
    from scipy.stats import kurtosis
    print(f"\n{'='*92}\n{interval}  M1 = [r, CVD 흐름]   {len(X):,}봉   변환: {TRANSFORM}\n"
          f"  초과첨도  변환전 {kurtosis(X_raw, axis=0).round(2)}  ->  변환후 {kurtosis(X, axis=0).round(2)}\n"
          f"{'='*92}", flush=True)

    rows = []
    for K in K_RANGE:
        print(f"  K={K} 학습 중 (시드 {len(SEEDS)}개)...", flush=True)
        best = fit_best(X, K)
        if best is None:
            print(f"    K={K}: 전 시드 실패"); continue
        m = best["model"]
        states = m.predict(X)
        share = np.bincount(states, minlength=K) / len(states)
        persist = np.diag(m.transmat_)
        bic = float(m.bic(X))
        aic = float(m.aic(X))

        pass_share = share.min() >= MIN_SHARE
        pass_persist = persist.min() >= MIN_PERSIST
        survives = pass_share and pass_persist

        rows.append(dict(
            K=K, bic=bic, aic=aic, loglik=best["ll"], seed=best["seed"],
            converged=best["converged"], n_iter_used=best["n_iter_used"],
            min_share=float(share.min()), min_persist=float(persist.min()),
            shares=[round(float(s), 4) for s in sorted(share)],
            persist=[round(float(p), 4) for p in sorted(persist)],
            pass_share=bool(pass_share), pass_persist=bool(pass_persist),
            survives=bool(survives),
        ))

        tag = "통과" if survives else "탈락"
        why = "" if survives else "  (" + ", ".join(
            ([] if pass_share else [f"최소비중 {share.min():.3f}<{MIN_SHARE}"]) +
            ([] if pass_persist else [f"지속확률 {persist.min():.3f}<{MIN_PERSIST}"])) + ")"
        print(f"    K={K}  BIC {bic:>12,.1f}  logL {best['ll']:>12,.1f}  "
              f"seed {best['seed']}  수렴 {'O' if best['converged'] else 'X'}({best['n_iter_used']}회)  "
              f"최소비중 {share.min():.3f}  최소지속 {persist.min():.3f}  [{tag}]{why}", flush=True)

    survivors = [r for r in rows if r["survives"]]
    if survivors:
        kstar = min(survivors, key=lambda r: r["bic"])["K"]
        print(f"\n  통과한 K: {[r['K'] for r in survivors]}  ->  BIC 최소인 "
              f"K* = {kstar}")
    else:
        kstar = None
        print(f"\n  [!] 통과한 K 없음 — 기준 재검토 필요")

    # K* 모형을 다시 적합해 저장 (시드 파일명에 기록)
    saved = None
    if kstar is not None:
        best = fit_best(X, kstar)
        m = best["model"]
        OUT.mkdir(parents=True, exist_ok=True)
        fn = OUT / f"M1_{interval}_K{kstar}_seed{best['seed']}.npz"
        np.savez(fn,
                 startprob=m.startprob_, transmat=m.transmat_,
                 means=m.means_, covars=m.covars_,
                 states=m.predict(X), transform=TRANSFORM,
                 bar_start=d.bar_start.to_numpy())
        saved = str(fn.name)
        print(f"  저장: {fn.name}")

    return dict(interval=interval, n_bars=int(len(X)), seeds=SEEDS,
                transform=TRANSFORM,
                min_share_rule=MIN_SHARE, min_persist_rule=MIN_PERSIST,
                covariance_type=COV, candidates=rows, k_star=kstar, model_file=saved)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("E020 — K 결정 (M1 기준)")
    print(f"규칙: 최소비중 ≥ {MIN_SHARE}, 지속확률 ≥ {MIN_PERSIST}, 남은 것 중 BIC 최소")
    rep = {iv: evaluate(iv) for iv in ("4h", "1d")}
    p = LOGS / "E020_k_selection.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*92}")
    for iv, r in rep.items():
        print(f"  {iv}:  K* = {r['k_star']}")
    print(f"리포트: {p}")


if __name__ == "__main__":
    sys.exit(main())
