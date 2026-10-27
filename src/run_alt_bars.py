"""
E058/E060 하류 — 1시간봉·달러바에 대해 변수 생성 → K 선택 → 다섯 모형 → Δ.

규약상 표집 변형은 전체 그리드를 돌리지 않고 **Δ 부호만 확인**한다.
여기에 하나를 더 본다 — **기준 분할이 시기별로 쏠리는가.**
4시간봉에서 M1 의 상태가 2021~22 / 2023 / 2024~25 로 갈렸고 시간 3등분 분할이
κ 0.638 을 기록했다. 달러바는 봉마다 거래대금이 같아 V 를 통한 시기 정보가
사라지므로, 그 쏠림이 줄어드는지가 핵심 확인 사항이다.

윈도우는 4시간봉 기준을 봉 수에 맞춰 환산한다.
  1시간봉  4배 촘촘 → σ 200, W 400
  달러바   4시간봉과 봉 수가 비슷 → σ 50, W 100
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as tdist
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "bars"
FEAT = ROOT / "data" / "features"
TABLES = ROOT / "results" / "tables"

NU = 0.25
SEEDS = list(range(10))
MIN_SHARE, MIN_PERSIST = 0.05, 0.90
EPS = 1e-12

CONFIGS = {"1h": dict(sig_w=200, reg_w=400), "dollar": dict(sig_w=50, reg_w=100)}
SPEC = {"M1": ["r", "flow_cvd"], "M2": ["r", "flow_bvc"], "M3": ["r"],
        "M4": ["r", "flow_clv"], "M5": ["r", "flow_res"]}


def rolling_resid(y, x, W):
    r = lambda s: s.rolling(W, min_periods=W).sum().shift(1)
    sx, sy, sxx, sxy = r(x), r(y), r(x * x), r(x * y)
    den = W * sxx - sx * sx
    beta = np.where(np.abs(den) < EPS, np.nan, (W * sxy - sx * sy) / den)
    return y - ((sy - beta * sx) / W + beta * x)


def make_features(name, p):
    b = pq.read_table(BARS / f"bars_{name}.parquet").to_pandas().sort_values("bar_start")
    d = pd.DataFrame({"bar_start": b.bar_start.to_numpy(),
                      "timestamp": b.timestamp.to_numpy(),
                      "V": b.V_usdt.to_numpy(), "flow_cvd": b.cvd_usdt.to_numpy()})
    d["r"] = np.log(b.close / b.close.shift(1)).to_numpy()
    sig = d.r.rolling(p["sig_w"], min_periods=p["sig_w"]).std().shift(1)
    z = (d.r / sig).replace([np.inf, -np.inf], np.nan)
    mult = 2.0 * tdist.cdf(z.to_numpy(), df=NU) - 1.0
    d["flow_bvc"] = d.V * np.where(np.isnan(z.to_numpy()), np.nan, mult)
    rng = (b.high - b.low).to_numpy()
    clv = np.where(rng <= 0, 0.0,
                   (2 * b.close - b.high - b.low).to_numpy() / np.where(rng <= 0, 1, rng))
    d["flow_clv"] = d.V * clv
    d["flow_res"] = rolling_resid(d.flow_clv, d.r, p["reg_w"])
    need = ["r", "flow_cvd", "flow_bvc", "flow_clv", "flow_res"]
    start = max(int(d[c].first_valid_index()) for c in need)
    cut = d.iloc[start:].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(cut, preserve_index=False),
                   FEAT / f"features_{name}.parquet", compression="zstd")
    return cut, start, int((rng <= 0).sum())


def fit(X, K):
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
    return best


def sc(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    al = np.array([mp[int(s)] for s in other])
    return float((al == ref).mean()), float(cohen_kappa_score(ref, al)), al


def run(name, p):
    d, start, n_flat = make_features(name, p)
    n = len(d)
    print(f"\n{'='*100}\n{name}   {n:,}봉   σ={p['sig_w']}  W={p['reg_w']}  "
          f"(웜업 {start}, H==L {n_flat}봉)\n{'='*100}", flush=True)

    X1 = transform.apply(d[["r", "flow_cvd"]].to_numpy(float), transform.METHOD)
    rows = []
    for K in range(2, 7):
        b = fit(X1, K)
        if b is None: continue
        m = b[0]; st = m.predict(X1)
        share = np.bincount(st, minlength=K) / n
        pers = np.diag(m.transmat_)
        ok = share.min() >= MIN_SHARE and pers.min() >= MIN_PERSIST
        rows.append((K, float(m.bic(X1)), ok, float(pers.min()), float(share.min())))
        print(f"  K={K}  BIC {m.bic(X1):>12,.1f}  최소비중 {share.min():.3f}  "
              f"최소지속 {pers.min():.3f}  {'통과' if ok else '탈락'}", flush=True)
    surv = [r for r in rows if r[2]]
    if not surv:
        print("  통과한 K 없음"); return dict(interval=name, k_star=None, n_bars=n)
    K = min(surv, key=lambda r: r[1])[0]
    print(f"  통과 {[r[0] for r in surv]} -> K* = {K}")

    states = {}
    for nm, cols in SPEC.items():
        Xm = transform.apply(d[cols].to_numpy(float), transform.METHOD)
        states[nm] = fit(Xm, K)[0].predict(Xm)
    ref = states["M1"]
    res = {}
    for nm in ["M2", "M3", "M4", "M5"]:
        a, k, _ = sc(ref, states[nm], K)
        res[nm] = dict(agree=a, kappa=k)
    ba, bk = res["M3"]["agree"], res["M3"]["kappa"]
    print(f"\n  {'모형':<8}{'일치율':>10}{'카파':>10}{'Δ일치율':>11}{'Δ카파':>10}")
    for nm in ["M2", "M3", "M4", "M5"]:
        v = res[nm]
        print(f"  {nm:<8}{v['agree']:>10.4f}{v['kappa']:>10.4f}"
              f"{v['agree']-ba:>+11.4f}{v['kappa']-bk:>+10.4f}")

    # --- 시기 쏠림 확인 ---
    yr = pd.to_datetime(d.timestamp).dt.year.to_numpy()
    sh_all = [float((ref == k).mean()) for k in range(K)]
    print(f"\n  [M1 국면의 연도별 비중]")
    print(f"    {'연도':<8}{'봉수':>9}" + "".join(f"{'상태'+str(k):>10}" for k in range(K)))
    conc = []
    for y in sorted(set(yr)):
        m_ = yr == y
        sh = [float((ref[m_] == k).mean()) for k in range(K)]
        print(f"    {y:<8}{m_.sum():>9,}" + "".join(f"{s:>10.3f}" for s in sh))
    for k in range(K):
        pk = np.array([float((ref[yr == y] == k).mean()) for y in sorted(set(yr))])
        w = np.array([float((yr == y).mean()) for y in sorted(set(yr))])
        conc.append(float(np.abs(pk - sh_all[k]).dot(w) / (2*sh_all[k]*(1-sh_all[k]))))
    print(f"    쏠림지수 {np.round(conc,3)}   (4시간봉: [0.635 0.781 0.528])")

    terc = np.minimum((np.arange(n) * K) // n, K - 1)
    a_t, k_t, _ = sc(ref, terc, K)
    print(f"\n  [시간 {K}등분 분할 vs M1]  일치율 {a_t:.4f}  카파 {k_t:.4f}"
          f"   (4시간봉: 0.7587 / 0.6381)")
    print(f"    참고 — BVC 카파 {res['M2']['kappa']:.4f}")

    return dict(interval=name, k_star=K, n_bars=n, deltas={
        nm: dict(delta_agree=res[nm]["agree"]-ba, delta_kappa=res[nm]["kappa"]-bk)
        for nm in ["M2", "M4", "M5"]},
        agreement=res, concentration=conc, tercile_kappa=k_t, tercile_agree=a_t)


def main():
    FEAT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    print("E058/E060 — 1시간봉·달러바")
    rep = {}
    for name, p in CONFIGS.items():
        if not (BARS / f"bars_{name}.parquet").exists():
            print(f"  {name}: 바 파일 없음 — 건너뜀"); continue
        rep[name] = run(name, p)
    (TABLES / "E058_E060_alt_bars.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {TABLES / 'E058_E060_alt_bars.json'}")


if __name__ == "__main__":
    sys.exit(main())
