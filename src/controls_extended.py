"""
E041e — 확장 통제군. Δ₂ 가 '정보'인가 '표현'인가.

BVC 는 (r, V, σ) 의 결정론적 함수이므로 그 너머의 정보를 담을 수 없다.
그런데 [r, V] 통제군은 실패하고 BVC 는 성공했다. 가능한 해석은 두 가지다.
  (a) 정보  — BVC 의 특정 형태에 고유한 무언가가 있다
  (b) 표현  — 가우시안 HMM 이 상태 안에서 선형 공분산만 쓰므로
             V·g(r/σ) 같은 곱 구조를 스스로 만들지 못하고, 미리 만들어 주면 쓴다

가르는 방법 — 곱 구조를 다른 형태로 미리 만들어 준 통제군을 세운다.
  M3v  [r, V]              거래량만 (기존)
  M3x  [r, V, r·V]         곱을 성분으로 직접 제공
  M3z  [r, V·(r/σ)]        부호 있는 거래량, CDF 없이 선형
  M2   [r, V·(2F_ν(r/σ)−1)] BVC — CDF 압축 있음

M3x 나 M3z 가 M2 만큼 하면 (b), 못 하면 (a) 쪽이다.

우연 수준은 **헝가리안 정렬 후의 대응**으로 계산한다.
(이전 진단에서 두 비중을 각각 내림차순 정렬해 짝지어 잘못 보고했다.)
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


def score(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    al = np.array([mp[int(s)] for s in other])
    n = len(ref)
    p = np.bincount(ref, minlength=K) / n
    q = np.bincount(al, minlength=K) / n          # 정렬 후 비중으로 우연 계산
    chance = float((p * q).sum())
    return (float((al == ref).mean()), float(cohen_kappa_score(ref, al)), chance)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    d = pq.read_table(FEAT / f"features_{INTERVAL}.parquet").to_pandas()
    n = len(d)
    r = d.r.to_numpy(float)
    V = d.V.to_numpy(float)
    sig = d.sigma.to_numpy(float)
    z = r / sig

    specs = {
        "M1":  ("[r, CVD]",            np.column_stack([r, d.flow_cvd.to_numpy()])),
        "M2":  ("[r, BVC]",            np.column_stack([r, d.flow_bvc.to_numpy()])),
        "M3":  ("[r]",                 r[:, None]),
        "M3v": ("[r, V]",              np.column_stack([r, V])),
        "M3x": ("[r, V, r·V]",         np.column_stack([r, V, r * V])),
        "M3z": ("[r, V·(r/σ)]",        np.column_stack([r, V * z])),
        "M4":  ("[r, CLV]",            np.column_stack([r, d.flow_clv.to_numpy()])),
        "M5":  ("[r, 잔차]",            np.column_stack([r, d.flow_res.to_numpy()])),
    }

    print(f"E041e — 확장 통제군  ({INTERVAL}, K={K}, {n:,}봉)")
    print("BVC 의 우위가 '정보'인지 '표현'인지 가른다\n")

    states, pers = {}, {}
    for nm, (lbl, raw) in specs.items():
        X = transform.apply(raw, transform.METHOD)
        states[nm], pers[nm] = fit_states(X)

    ref = states["M1"]
    base_a, base_k, _ = score(ref, states["M3"], K)

    print(f"{'모형':<6}{'관측 벡터':<18}{'일치율':>9}{'카파':>9}{'우연':>9}"
          f"{'Δ 일치율':>11}{'Δ 카파':>10}{'최소지속':>10}")
    print("-" * 92)
    out = {}
    for nm, (lbl, _) in specs.items():
        if nm == "M1":
            print(f"{nm:<6}{lbl:<18}{'기준':>9}{'':>9}{'':>9}{'':>11}{'':>10}"
                  f"{pers[nm].min():>10.3f}")
            continue
        a, k, ch = score(ref, states[nm], K)
        out[nm] = dict(label=lbl, agree=a, kappa=k, chance=ch,
                       delta_agree=a - base_a, delta_kappa=k - base_k,
                       min_persist=float(pers[nm].min()))
        mark = "  <-" if nm in ("M3x", "M3z") else ""
        print(f"{nm:<6}{lbl:<18}{a:>9.4f}{k:>9.4f}{ch:>9.4f}"
              f"{a-base_a:>+11.4f}{k-base_k:>+10.4f}{pers[nm].min():>10.3f}{mark}")

    print(f"\n해석")
    m2, m3x, m3z = out["M2"], out["M3x"], out["M3z"]
    print(f"  BVC        Δ 카파 {m2['delta_kappa']:+.4f}")
    print(f"  [r,V,r·V]  Δ 카파 {m3x['delta_kappa']:+.4f}   "
          f"BVC 대비 {m3x['delta_kappa']/m2['delta_kappa']*100 if m2['delta_kappa'] else 0:.0f}%")
    print(f"  [r,V·(r/σ)] Δ 카파 {m3z['delta_kappa']:+.4f}   "
          f"BVC 대비 {m3z['delta_kappa']/m2['delta_kappa']*100 if m2['delta_kappa'] else 0:.0f}%")
    print("  -> 통제군이 BVC 에 육박하면 '표현' 문제, 크게 못 미치면 'BVC 형태 고유' 쪽")

    p = TABLES / "E041e_extended_controls.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {p}")


if __name__ == "__main__":
    sys.exit(main())
