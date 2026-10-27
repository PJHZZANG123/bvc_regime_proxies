"""
E050~E055 — 핵심 로버스트니스 (구성 모수 + 모형 설정).

기준 설정: 4시간봉, K*=3, ν=0.25, σ윈도우 50, W=100, 잔차 그대로·봉별, 완전 공분산.
**한 번에 하나씩만** 이탈한다. 조합하지 않는다.

재적합 범위
  ν, σ윈도우   : 웜업이 안 바뀌므로 공통 표본 동일 → M2 만 재적합
  W            : 웜업이 바뀌어 공통 표본이 달라짐 → 전 모형 재적합
  CDF압축, 누적 : 공통 표본 동일 → M5 만 재적합
  공분산       : 전 모형 재적합

볼 것은 하나 — Δ 의 부호가 뒤집히는가. 신뢰구간도 함께 구한다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from scipy.stats import t as tdist
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data" / "bars"
TABLES = ROOT / "results" / "tables"

INTERVAL, K = "4h", 3
SEEDS = list(range(10))
N_BOOT, BLOCK = 2000, 200
EPS = 1e-12

BASE = dict(nu=0.25, sig_w=50, reg_w=100, cdf=False, cumul=False, cov="full")


# ---------- 변수 생성 ----------
def rolling_ols_residual(y, x, W):
    r = lambda s: s.rolling(W, min_periods=W).sum().shift(1)
    sx, sy, sxx, sxy = r(x), r(y), r(x * x), r(x * y)
    den = W * sxx - sx * sx
    beta = np.where(np.abs(den) < EPS, np.nan, (W * sxy - sx * sy) / den)
    alpha = (sy - beta * sx) / W
    return y - (alpha + beta * x)


def make_features(p):
    b = pq.read_table(BARS / f"bars_{INTERVAL}.parquet").to_pandas().sort_values("bar_start")
    d = pd.DataFrame({"V": b.V_usdt.to_numpy(), "flow_cvd": b.cvd_usdt.to_numpy()})
    d["r"] = np.log(b.close / b.close.shift(1)).to_numpy()

    sig = d.r.rolling(p["sig_w"], min_periods=p["sig_w"]).std().shift(1)
    z = (d.r / sig).replace([np.inf, -np.inf], np.nan)
    mult = 2.0 * tdist.cdf(z.to_numpy(), df=p["nu"]) - 1.0
    d["flow_bvc"] = d.V * np.where(np.isnan(z.to_numpy()), np.nan, mult)

    rng = (b.high - b.low).to_numpy()
    clv = np.where(rng <= 0, 0.0, (2 * b.close - b.high - b.low).to_numpy() / np.where(rng <= 0, 1, rng))
    d["flow_clv"] = d.V * clv

    res = rolling_ols_residual(d.flow_clv, d.r, p["reg_w"])
    if p["cdf"]:                       # Φ 통과 — 표준화 후 2Φ(z)−1 로 꼬리 압축
        s = res.std()
        res = 2.0 * pd.Series(norm.cdf((res / s).to_numpy()), index=res.index) - 1.0
        res = res.where(~res.isna() & ~pd.isna(rolling_ols_residual(d.flow_clv, d.r, p["reg_w"])))
    if p["cumul"]:                     # 봉별 → 누적
        res = res.cumsum()
    d["flow_res"] = res

    need = ["r", "flow_cvd", "flow_bvc", "flow_clv", "flow_res"]
    start = max(int(d[c].first_valid_index()) for c in need)
    return d.iloc[start:].reset_index(drop=True), start


# ---------- 적합·평가 ----------
SPEC = {"M1": ["r", "flow_cvd"], "M2": ["r", "flow_bvc"], "M3": ["r"],
        "M4": ["r", "flow_clv"], "M5": ["r", "flow_res"]}


def fit_states(d, cols, cov):
    X = transform.apply(d[cols].to_numpy(float), transform.METHOD)
    best = None
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type=cov, n_iter=500, tol=1e-4, random_state=s)
        try:
            m.fit(X); ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll)
    return best[0].predict(X)


def align(ref, other):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def deltas_with_ci(ref, aligned):
    n = len(ref)
    rng = np.random.default_rng(20260915)
    acc = {nm: {"a": [], "k": []} for nm in aligned}
    nb = int(np.ceil(n / BLOCK))
    for _ in range(N_BOOT):
        st = rng.integers(0, n, size=nb)
        idx = (st[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
        rb = ref[idx]
        for nm, al in aligned.items():
            ab = al[idx]
            acc[nm]["a"].append(float((ab == rb).mean()))
            acc[nm]["k"].append(float(cohen_kappa_score(rb, ab)))
    ba, bk = np.array(acc["M3"]["a"]), np.array(acc["M3"]["k"])
    out = {}
    for nm in ["M2", "M4", "M5"]:
        da = np.array(acc[nm]["a"]) - ba
        dk = np.array(acc[nm]["k"]) - bk
        out[nm] = dict(
            # 점추정치 — 표본에서 직접 계산한다. 아래 kappa_boot_mean 은 재추출
            # 분포의 중심이라 블록 길이에 따라 움직이므로 점추정치가 아니다.
            agree=float((aligned[nm] == ref).mean() - (aligned["M3"] == ref).mean()),
            kappa=float(cohen_kappa_score(ref, aligned[nm])
                        - cohen_kappa_score(ref, aligned["M3"])),
            kappa_boot_mean=float(dk.mean()),
            ci_agree=(float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))),
            ci_kappa=(float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5))),
        )
    return out


def evaluate(p, refit):
    d, start = make_features(p)
    states = {nm: fit_states(d, cols, p["cov"]) for nm, cols in SPEC.items()
              if nm in refit or nm == "M1" or nm == "M3"}
    # 재적합 범위 밖의 모형도 같은 표본에서 필요하므로 전부 적합한다
    for nm, cols in SPEC.items():
        if nm not in states:
            states[nm] = fit_states(d, cols, p["cov"])
    ref = states["M1"]
    aligned = {nm: align(ref, states[nm]) for nm in ["M2", "M3", "M4", "M5"]}
    return deltas_with_ci(ref, aligned), len(d), start


VARIANTS = [
    ("기준",              dict(),                    {"-"}),
    ("ν = 0.1",          dict(nu=0.1),              {"M2"}),
    ("ν = 1",            dict(nu=1.0),              {"M2"}),
    ("ν = 30 (큰 값)",    dict(nu=30.0),             {"M2"}),
    ("σ 윈도우 = 20",     dict(sig_w=20),            {"M2"}),
    ("σ 윈도우 = 100",    dict(sig_w=100),           {"M2"}),
    ("W = 50",           dict(reg_w=50),            {"전 모형"}),
    ("W = 200",          dict(reg_w=200),           {"전 모형"}),
    ("CDF 압축 (Φ)",      dict(cdf=True),            {"M5"}),
    ("잔차 누적",          dict(cumul=True),          {"M5"}),
    ("공분산 대각",        dict(cov="diag"),          {"전 모형"}),
]


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print(f"E050~E055 — 핵심 로버스트니스  ({INTERVAL}, K={K})")
    print(f"기준: ν={BASE['nu']}, σ윈도우={BASE['sig_w']}, W={BASE['reg_w']}, "
          f"잔차 그대로·봉별, 완전 공분산\n")
    print(f"{'변형':<18}{'봉수':>8}{'Δ₂ 카파':>26}{'Δ₄ 카파':>26}{'Δ₅ 카파':>26}")
    print("-" * 104)

    rep = {}
    for name, over, scope in VARIANTS:
        p = {**BASE, **over}
        res, n, start = evaluate(p, scope)
        rep[name] = dict(params=p, n_bars=n, common_start=start, deltas=res)
        cells = []
        for nm in ["M2", "M4", "M5"]:
            v = res[nm]
            lo, hi = v["ci_kappa"]
            zero = "0포함" if lo <= 0 <= hi else "     "
            cells.append(f"{v['kappa']:+.4f} [{lo:+.4f},{hi:+.4f}] {zero}")
        print(f"{name:<18}{n:>8,}" + "".join(f"{c:>26}" for c in cells), flush=True)

    p_out = TABLES / "E05x_robustness_core.json"
    p_out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장: {p_out}")


if __name__ == "__main__":
    sys.exit(main())
