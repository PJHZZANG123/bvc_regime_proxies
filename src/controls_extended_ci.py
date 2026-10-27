"""
E041e-CI — 확장 통제군의 Δκ 와 95% 구간 (4시간봉, K=3).

E041e(controls_extended.py)와 같은 설정으로 다섯 모형을 다시 추정하고,
E044(bootstrap_ci.py)와 같은 방식으로 구간을 낸다.

  M1   [r, CVD]          기준
  M2   [r, BVC]
  M3   [r]               통제군
  M3v  [r, V]            거래대금을 더한 통제군
  M3z  [r, V·(r/σ)]      거래대금에 표준화 수익률을 곱한 값 (CDF 압축 없음)

  - 상태 대응은 전체 표본에서 한 번만 (재추출마다 다시 맞추지 않음)
  - 순환 블록 재추출 2,000회, 블록 200봉, 난수 시드 20260915 (E044 와 같음)
  - 다섯 모형에 같은 재추출 위치를 쓴다

검산: M2 − M3 의 점추정치와 구간이 E044 의 주 결과(+0.2175, [+0.1068, +0.3240])와 같아야 한다.

    python src/controls_extended_ci.py
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
TABLES = ROOT / "results" / "tables"

INTERVAL, K = "4h", 3
SEEDS = list(range(10))
N_BOOT, BLOCK, RNG_SEED = 2000, 200, 20260915


def fit_states(X):
    best = None
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=500, tol=1e-4, random_state=s)
        try:
            m.fit(X); ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll)
    return best[0].predict(X), np.diag(best[0].transmat_)


def align_once(ref, other):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def circ_blocks(n, L, rng):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nb)
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:n] % n


def main():
    d = pq.read_table(FEAT / f"features_{INTERVAL}.parquet").to_pandas()
    r = d.r.to_numpy(float)
    V = d.V.to_numpy(float)
    z = r / d.sigma.to_numpy(float)
    specs = {
        "M1": ("[r, CVD]", np.column_stack([r, d.flow_cvd.to_numpy()])),
        "M2": ("[r, BVC]", np.column_stack([r, d.flow_bvc.to_numpy()])),
        "M3": ("[r]", r[:, None]),
        "M3v": ("[r, V]", np.column_stack([r, V])),
        "M3z": ("[r, V·(r/σ)]", np.column_stack([r, V * z])),
    }
    states, pers = {}, {}
    for nm, (_, raw) in specs.items():
        states[nm], pers[nm] = fit_states(transform.apply(raw, transform.METHOD))
    ref = states["M1"]
    n = len(ref)
    al = {nm: align_once(ref, states[nm]) for nm in specs if nm != "M1"}
    kap = {nm: float(cohen_kappa_score(ref, a)) for nm, a in al.items()}

    pairs = [("M2", "M3"), ("M2", "M3v"), ("M3v", "M3"), ("M3z", "M3"), ("M2", "M3z")]
    rng = np.random.default_rng(RNG_SEED)
    boot = {nm: [] for nm in al}
    for _ in range(N_BOOT):
        idx = circ_blocks(n, BLOCK, rng)
        rb = ref[idx]
        for nm, a in al.items():
            boot[nm].append(float(cohen_kappa_score(rb, a[idx])))
    boot = {nm: np.array(v) for nm, v in boot.items()}

    out = dict(interval=INTERVAL, K=K, n_bars=n, n_boot=N_BOOT, block_len=BLOCK, rng_seed=RNG_SEED,
               hmm_seeds=SEEDS, transform=transform.METHOD,
               kappa={nm: dict(label=specs[nm][0], kappa=kap[nm], min_persist=float(pers[nm].min()))
                      for nm in al},
               deltas={})
    print(f"E041e-CI  ({INTERVAL}, K={K}, {n:,}봉, 블록 {BLOCK}봉, {N_BOOT:,}회)")
    for nm in al:
        print(f"  {nm:<4} {specs[nm][0]:<16} κ {kap[nm]:.4f}   최소 지속확률 {pers[nm].min():.3f}")
    for a, b in pairs:
        pt = kap[a] - kap[b]
        dk = boot[a] - boot[b]
        ci = (float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5)))
        out["deltas"][f"{a}-{b}"] = dict(point=pt, ci=ci, zero_in=ci[0] <= 0 <= ci[1])
        print(f"  Δκ {a:>3} − {b:<3} {pt:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]"
              f"{'  0 포함' if ci[0] <= 0 <= ci[1] else ''}")

    p = TABLES / "E041e_ci.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
