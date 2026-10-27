"""
ETH 검증 2단계 — 변수 → K 선택 → 다섯 모형 → Δ → 시기 쏠림.

BTC 기준 설정을 그대로 쓴다.
  ν=0.25, σ 윈도우 50, 회귀 W = 100(4h) / 50(1d), V=USDT, 부호보존 로그 변환,
  완전 공분산, 최소비중 5%, 지속확률 0.90, 시드 0~9 중 최고 우도

확인 사항 둘
  (1) K 선택이 BTC 와 같은 양상인가 — 통과 K, K*
  (2) 기준 국면의 연도별 뭉침이 ETH 에서도 나타나는가 — 쏠림지수
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as tdist
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "bars"
TABLES = ROOT / "results" / "tables"

NU, EPS = 0.25, 1e-12
SEEDS = list(range(10))
MIN_SHARE, MIN_PERSIST = 0.05, 0.90
N_BOOT = 2000
CONFIGS = {"4h": dict(sig_w=50, reg_w=100, block=200),
           "1d": dict(sig_w=50, reg_w=50, block=60)}
SPEC = {"M1": ["r", "flow_cvd"], "M2": ["r", "flow_bvc"], "M3": ["r"],
        "M4": ["r", "flow_clv"], "M5": ["r", "flow_res"]}
BTC_REF = {"4h": dict(kstar=3, surv=[2, 3], conc=[0.635, 0.781, 0.528],
                      dk={"M2": 0.2175, "M4": 0.0175, "M5": 0.0219}),
           "1d": dict(kstar=2, surv=[2], conc=None,
                      dk={"M2": 0.2408, "M4": 0.0148, "M5": -0.0208})}


def rolling_resid(y, x, W):
    r = lambda s: s.rolling(W, min_periods=W).sum().shift(1)
    sx, sy, sxx, sxy = r(x), r(y), r(x * x), r(x * y)
    den = W * sxx - sx * sx
    beta = np.where(np.abs(den) < EPS, np.nan, (W * sxy - sx * sy) / den)
    return y - ((sy - beta * sx) / W + beta * x)


def make_features(name, p):
    b = pq.read_table(BARS / f"bars_{name}_ETHUSDT.parquet").to_pandas().sort_values("bar_start")
    d = pd.DataFrame({"timestamp": b.timestamp.to_numpy(), "V": b.V_usdt.to_numpy(),
                      "flow_cvd": b.cvd_usdt.to_numpy()})
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
    return d.iloc[start:].reset_index(drop=True), start, int((rng <= 0).sum())


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


def align(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def run(name, p):
    d, start, n_flat = make_features(name, p)
    n = len(d)
    ref_btc = BTC_REF[name]
    print(f"\n{'='*100}\nETH {name}   {n:,}봉   σ={p['sig_w']} W={p['reg_w']}"
          f"   (웜업 {start}, H==L {n_flat})\n{'='*100}", flush=True)

    X1 = transform.apply(d[["r", "flow_cvd"]].to_numpy(float), transform.METHOD)
    rows = []
    for K in range(2, 7):
        b = fit(X1, K)
        if b is None: continue
        m = b[0]; st = m.predict(X1)
        sh = np.bincount(st, minlength=K) / n
        pe = np.diag(m.transmat_)
        ok = sh.min() >= MIN_SHARE and pe.min() >= MIN_PERSIST
        rows.append((K, float(m.bic(X1)), ok))
        print(f"  K={K}  BIC {m.bic(X1):>12,.1f}  최소비중 {sh.min():.3f}  "
              f"최소지속 {pe.min():.3f}  {'통과' if ok else '탈락'}", flush=True)
    surv = [r[0] for r in rows if r[2]]
    if not surv:
        print("  통과한 K 없음"); return dict(interval=name, k_star=None, n_bars=n)
    K = min([r for r in rows if r[2]], key=lambda r: r[1])[0]
    print(f"  통과 {surv} -> K* = {K}    [BTC: 통과 {ref_btc['surv']}, K*={ref_btc['kstar']}]")

    states = {}
    for nm, cols in SPEC.items():
        Xm = transform.apply(d[cols].to_numpy(float), transform.METHOD)
        states[nm] = fit(Xm, K)[0].predict(Xm)
    ref = states["M1"]
    al = {nm: align(ref, states[nm], K) for nm in ["M2", "M3", "M4", "M5"]}
    kap = {nm: float(cohen_kappa_score(ref, al[nm])) for nm in al}
    agr = {nm: float((al[nm] == ref).mean()) for nm in al}
    bk = kap["M3"]

    # 신뢰구간
    rng = np.random.default_rng(20260916)
    nb = int(np.ceil(n / p["block"]))
    acc = {nm: [] for nm in al}
    for _ in range(N_BOOT):
        s0 = rng.integers(0, n, size=nb)
        idx = (s0[:, None] + np.arange(p["block"])[None, :]).ravel()[:n] % n
        rb = ref[idx]
        for nm, a in al.items():
            acc[nm].append(float(cohen_kappa_score(rb, a[idx])))
    base = np.array(acc["M3"])

    print(f"\n  {'모형':<6}{'일치율':>10}{'카파':>10}{'Δ카파':>10}"
          f"{'95% CI':>22}{'0포함':>7}{'BTC Δκ':>10}")
    print("  " + "-" * 76)
    out = {}
    for nm in ["M2", "M3", "M4", "M5"]:
        dk = kap[nm] - bk
        if nm == "M3":
            print(f"  {nm:<6}{agr[nm]:>10.4f}{kap[nm]:>10.4f}{'기준':>10}{'':>22}{'':>7}{'':>10}")
            continue
        dist = np.array(acc[nm]) - base
        lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
        z = "포함" if lo <= 0 <= hi else "-"
        out[nm] = dict(agree=agr[nm], kappa=kap[nm], delta_kappa=dk, ci=(lo, hi),
                       zero_in=(z == "포함"))
        print(f"  {nm:<6}{agr[nm]:>10.4f}{kap[nm]:>10.4f}{dk:>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{z:>7}{ref_btc['dk'][nm]:>+10.4f}")

    # 시기 쏠림
    yr = pd.to_datetime(d.timestamp).dt.year.to_numpy()
    sh_all = [float((ref == k).mean()) for k in range(K)]
    print(f"\n  [M1 국면의 연도별 비중]")
    print(f"    {'연도':<7}{'봉수':>9}" + "".join(f"{'상태'+str(k):>10}" for k in range(K)))
    for y in sorted(set(yr)):
        m_ = yr == y
        print(f"    {y:<7}{m_.sum():>9,}"
              + "".join(f"{float((ref[m_]==k).mean()):>10.3f}" for k in range(K)))
    conc = []
    for k in range(K):
        pk = np.array([float((ref[yr == y] == k).mean()) for y in sorted(set(yr))])
        w = np.array([float((yr == y).mean()) for y in sorted(set(yr))])
        conc.append(float(np.abs(pk - sh_all[k]).dot(w) / (2*sh_all[k]*(1-sh_all[k]))))
    print(f"    쏠림지수 {np.round(conc,3)}"
          + (f"   [BTC: {ref_btc['conc']}]" if ref_btc['conc'] else ""))

    return dict(interval=name, n_bars=n, k_star=K, survivors=surv,
                deltas=out, concentration=conc, kappa=kap, agree=agr)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("ETH 검증 2단계 — BTC 기준 설정 그대로 적용")
    rep = {}
    for name, p in CONFIGS.items():
        if not (BARS / f"bars_{name}_ETHUSDT.parquet").exists():
            print(f"  {name}: 바 파일 없음"); continue
        rep[name] = run(name, p)
    (TABLES / "ETH_validation.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {TABLES / 'ETH_validation.json'}")


if __name__ == "__main__":
    sys.exit(main())
