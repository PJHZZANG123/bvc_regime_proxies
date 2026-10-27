"""
E070 — 기준 국면의 시간 구조와 시기 조정 일치도.

외부 검토(2026-10-10)의 핵심 지적을 정면으로 다룬다. 기준 모형(M1)의 국면이 시기(연도)별로 뭉쳐 있으면,
국면 일치도는 주문흐름의 재현과 시기 구분의 재현을 섞어 잰다. 이를 가르기 위해 다음을 계산한다.

  1) 비정보 시간 대조군
       T     [r, t]        t = 0, 1, 2, … (봉 순번)
       RW_i  [r, w_i]      w_i = 자료와 무관한 표준정규 누적합(무작위 보행), i = 1..N_RW
       N_i   [r, e_i]      e_i = 자료와 무관한 표준정규 잡음(빠르게 변함), i = 1..N_IID
  2) BVC 분해 사다리
       M3v   [r, V]                 거래대금 크기만
       M2s   [r, sign(r)·V]         방향은 수익률, 크기는 거래대금
       M3z   [r, V·(r/σ)]           표준화 수익률에 거래대금을 곱한 값(CDF 압축 없음)
       M2m   [r, 2F(r/σ)−1]         BVC 의 승수만(거래대금을 곱하지 않음)
       M3s   [r, σ̂]                이동창(직전 50봉) 수익률 표준편차만 — 승수에 들어 있는 느린 변동성 성분
  3) 시기 조정 카파 — 기대 일치를 시기(연도·분기)마다의 국면 비중으로 계산한다.
       κ_s = (p_o − p_e,s) / (1 − p_e,s),   p_e,s = Σ_s w_s Σ_j π^s_{M1,j} π^s_{Mk,j}
     각 시기의 국면 구성만 맞히는 분할(예: 기간을 셋으로 자른 분할)은 κ_s ≈ 0 이 된다.
     시기는 연도·분기·월 세 가지로 나눈다. 시간 대조군이 0 근처가 되는 가장 거친 구분을 주 분석으로 삼는다.

설정은 기존 분석과 같다: 부호보존 로그 변환, 완전 공분산, 시드 0~9 중 최고 우도, 반복 500, tol 1e-4,
상태 대응은 전체 표본에서 한 번, 순환 블록 재추출 2,000회(4시간봉 200봉, 일봉 60봉, 시드 20260915),
모든 모형에 같은 재추출 위치. 검산: BTC 4시간봉 K=3 의 BVC−통제군 Δκ 가 +0.2175 [+0.1068, +0.3240].

    python src/time_structure.py
"""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as tdist
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "results" / "models"

SEEDS = list(range(10))
N_ITER, TOL = 500, 1e-4
N_BOOT, RNG_SEED = 2000, 20260915
NU = 0.25
WORKERS = 7

SETTINGS = [  # (자산, 봉, K, 블록, 무작위 보행 수, 잡음 수, 시간 대조군 이동 검정 횟수)
    ("BTC", "4h", 3, 200, 20, 5, 200),
    ("BTC", "4h", 2, 200, 10, 0, 0),
    ("BTC", "1d", 2, 60, 10, 0, 0),
    ("ETH", "4h", 3, 200, 10, 0, 0),
    ("ETH", "1d", 4, 60, 10, 0, 0),
]
CORE = ["M2", "M3", "M4", "M5"]
LADDER = ["M3v", "M2s", "M3z", "M2m"]
LADDER = LADDER + ["M3s"]          # M3s = [r, σ̂]
PAIRS = [("M2", "M3"), ("M4", "M3"), ("M5", "M3"), ("T", "M3"), ("M2", "T"),
         ("M3v", "M3"), ("M2s", "M3"), ("M3z", "M3"), ("M2m", "M3"), ("M3s", "M3"),
         ("M2", "M3z"), ("M2", "M2s"), ("M2", "M3s"), ("M2m", "M3s")]
STRATA = ["year", "quarter", "month"]


# ------------------------------------------------------------------ 자료
def load_btc(interval):
    d = pq.read_table(FEAT / f"features_{interval}.parquet").to_pandas()
    out = pd.DataFrame({"timestamp": pd.to_datetime(d.timestamp), "r": d.r, "flow_cvd": d.flow_cvd,
                        "flow_bvc": d.flow_bvc, "flow_clv": d.flow_clv, "flow_res": d.flow_res,
                        "V": d.V, "z": d.z, "mult": d.bvc_mult, "sigma": d.sigma})
    return out.reset_index(drop=True)


def load_eth(interval):
    from eth_stage2 import CONFIGS, make_features
    p = CONFIGS[interval]
    d, _, _ = make_features(interval, p)
    sig = d.r.rolling(p["sig_w"], min_periods=p["sig_w"]).std().shift(1)
    # make_features 는 표본 시작을 웜업 뒤로 잘라 둔다. σ 는 잘리기 전 계열로 다시 계산해야 하므로 원 봉에서 구한다.
    from eth_stage2 import BARS
    b = pq.read_table(BARS / f"bars_{interval}_ETHUSDT.parquet").to_pandas().sort_values("bar_start")
    r_full = np.log(b.close / b.close.shift(1)).reset_index(drop=True)
    sig_full = r_full.rolling(p["sig_w"], min_periods=p["sig_w"]).std().shift(1)
    off = len(b) - len(d)
    z = (r_full / sig_full).to_numpy()[off:]
    assert np.allclose(r_full.to_numpy()[off:], d.r.to_numpy(), equal_nan=True)
    d = d.assign(timestamp=pd.to_datetime(d.timestamp), z=z, mult=2.0 * tdist.cdf(z, df=NU) - 1.0,
                 sigma=sig_full.to_numpy()[off:])
    return d.reset_index(drop=True)


def specs_of(d, n_rw, n_iid):
    r, V, z = d.r.to_numpy(float), d.V.to_numpy(float), d.z.to_numpy(float)
    n = len(d)
    S = {
        "M1": np.column_stack([r, d.flow_cvd]), "M2": np.column_stack([r, d.flow_bvc]),
        "M3": r[:, None], "M4": np.column_stack([r, d.flow_clv]), "M5": np.column_stack([r, d.flow_res]),
        "M3v": np.column_stack([r, V]), "M2s": np.column_stack([r, np.sign(r) * V]),
        "M3z": np.column_stack([r, V * z]), "M2m": np.column_stack([r, d.mult]),
        "M3s": np.column_stack([r, d.sigma]),
        "T": np.column_stack([r, np.arange(n, dtype=float)]),
    }
    for i in range(1, n_rw + 1):
        w = np.cumsum(np.random.default_rng(1000 + i).standard_normal(n))
        S[f"RW{i}"] = np.column_stack([r, w])
    for i in range(1, n_iid + 1):
        S[f"N{i}"] = np.column_stack([r, np.random.default_rng(2000 + i).standard_normal(n)])
    return S


# ------------------------------------------------------------------ 추정
def fit_job(args):
    name, raw, K, seeds, n_iter = args
    X = transform.apply(np.asarray(raw, float), transform.METHOD)
    best = None
    for s in seeds:
        m = GaussianHMM(n_components=K, covariance_type="full", n_iter=n_iter, tol=TOL, random_state=s)
        try:
            m.fit(X); ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll, s)
    if best is None:
        return name, None
    m, ll, s = best
    return name, dict(states=m.predict(X).astype(int), persist=np.diag(m.transmat_).tolist(),
                      loglik=float(ll), seed=int(s))


def shift_job(args):
    r, t, K, ref, seed = args
    rng = np.random.default_rng(seed)
    n = len(t)
    lag = int(rng.integers(int(0.05 * n), int(0.95 * n)))
    _, res = fit_job(("shift", np.column_stack([r, np.roll(t, lag)]), K, [0, 1, 2], 300))
    if res is None:
        return None
    return float(cohen_kappa_score(ref, align(ref, res["states"], K)))


# ------------------------------------------------------------------ 평가
def align(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def kappa_fast(ref, a, K):
    """cohen_kappa_score 와 같은 값(무가중)."""
    n = len(ref)
    po = float((ref == a).mean())
    pe = float((np.bincount(ref, minlength=K) * np.bincount(a, minlength=K)).sum()) / (n * n)
    return (po - pe) / (1 - pe)


def kappa_strat(ref, a, code, K, S):
    """시기 조정 카파. code 는 0..S-1 의 시기 번호."""
    n = len(ref)
    po = float((ref == a).mean())
    cr = np.bincount(code * K + ref, minlength=S * K).reshape(S, K)
    ca = np.bincount(code * K + a, minlength=S * K).reshape(S, K)
    ns = cr.sum(1)
    ok = ns > 0
    pe = float(((cr[ok] * ca[ok]).sum(1) / ns[ok]).sum()) / n
    return (po - pe) / (1 - pe)


def n_runs(s):
    return int(1 + (s[1:] != s[:-1]).sum())


def circ_blocks(n, L, rng):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nb)
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:n] % n


def run_setting(asset, interval, K, block, n_rw, n_iid, n_shift, pool):
    t0 = time.time()
    d = load_btc(interval) if asset == "BTC" else load_eth(interval)
    n = len(d)
    yr = d.timestamp.dt.year.to_numpy()
    raw_codes = {"year": yr, "quarter": (d.timestamp.dt.year * 10 + d.timestamp.dt.quarter).to_numpy(),
                 "month": (d.timestamp.dt.year * 100 + d.timestamp.dt.month).to_numpy()}
    codes = {k: np.unique(v, return_inverse=True)[1].astype(np.int64) for k, v in raw_codes.items()}
    nS = {k: int(v.max()) + 1 for k, v in codes.items()}
    S = specs_of(d, n_rw, n_iid)
    jobs = [(nm, raw, K, SEEDS, N_ITER) for nm, raw in S.items()]
    fits = dict(pool.map(fit_job, jobs, chunksize=1))
    assert all(v is not None for v in fits.values()), [k for k, v in fits.items() if v is None]
    ref = fits["M1"]["states"]

    # BTC 는 저장된 주 모형과 같은 상태열이어야 한다(재현 검산)
    if asset == "BTC":
        for nm in ["M1", "M2", "M3", "M4", "M5"]:
            saved = sorted(MODELS.glob(f"{nm}_{interval}_K{K}_seed*.npz"))
            if saved:
                s0 = np.load(saved[0])["states"].astype(int)
                same = float((align(s0, fits[nm]["states"], K) == s0).mean())
                assert same > 0.999, (nm, same)

    al = {nm: align(ref, f["states"], K) for nm, f in fits.items() if nm != "M1"}
    al["T3"] = align(ref, np.minimum((np.arange(n) * K) // n, K - 1), K)       # 모형 없이 기간을 K 등분
    names = [nm for nm in al]

    point = {}
    for nm in names:
        a = al[nm]
        point[nm] = dict(agree=float((a == ref).mean()), kappa=float(cohen_kappa_score(ref, a)),
                         **{f"kappa_{st}": kappa_strat(ref, a, codes[st], K, nS[st]) for st in STRATA},
                         runs=n_runs(fits[nm]["states"]) if nm in fits else n_runs(a),
                         min_persist=float(min(fits[nm]["persist"])) if nm in fits else None,
                         seed=fits[nm]["seed"] if nm in fits else None)
    point["M1"] = dict(runs=n_runs(ref), min_persist=float(min(fits["M1"]["persist"])), seed=fits["M1"]["seed"])

    # 연도별
    years = sorted(set(yr))
    by_year = {nm: {int(y): float(cohen_kappa_score(ref[yr == y], al[nm][yr == y])) for y in years}
               for nm in CORE + ["T", "T3"] + LADDER}
    m1_year = {int(y): [float((ref[yr == y] == k).mean()) for k in range(K)] for y in years}
    model_year = {nm: {int(y): [float((al[nm][yr == y] == k).mean()) for k in range(K)] for y in years}
                  for nm in CORE + ["T"] + LADDER}

    # 재추출 구간 (같은 위치를 모든 모형에)
    boot_models = CORE + LADDER + ["T"]
    rng = np.random.default_rng(RNG_SEED)
    B = {lab: {nm: np.empty(N_BOOT) for nm in boot_models} for lab in ["kappa"] + [f"kappa_{st}" for st in STRATA]}
    for b in range(N_BOOT):
        idx = circ_blocks(n, block, rng)
        rb = ref[idx]
        cb = {st: codes[st][idx] for st in STRATA}
        for nm in boot_models:
            ab = al[nm][idx]
            B["kappa"][nm][b] = kappa_fast(rb, ab, K)
            for st in STRATA:
                B[f"kappa_{st}"][nm][b] = kappa_strat(rb, ab, cb[st], K, nS[st])
    deltas = {}
    for a_, b_ in PAIRS:
        rec = {}
        for lab in B:
            P, Bl = lab, B[lab]
            pt = point[a_][P] - point[b_][P]
            dist = Bl[a_] - Bl[b_]
            lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
            rec[lab] = dict(point=pt, ci=[lo, hi], zero_in=bool(lo <= 0 <= hi))
        deltas[f"{a_}-{b_}"] = rec

    placebo = {}
    for pre, lab in [("RW", "random_walk"), ("N", "iid_noise")]:
        ks = [point[nm]["kappa"] for nm in names if nm.startswith(pre) and nm[len(pre):].isdigit()]
        if ks:
            rec = dict(n=len(ks), kappa=dict(mean=float(np.mean(ks)), min=float(np.min(ks)),
                                             max=float(np.max(ks)), all=ks),
                       share_above_control=float(np.mean(np.array(ks) > point["M3"]["kappa"])),
                       share_above_bvc=float(np.mean(np.array(ks) > point["M2"]["kappa"])))
            for st in STRATA:
                v = [point[nm][f"kappa_{st}"] for nm in names if nm.startswith(pre) and nm[len(pre):].isdigit()]
                rec[f"kappa_{st}"] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
            placebo[lab] = rec

    shift = None
    if n_shift:
        seeds = np.random.SeedSequence(20261010).spawn(n_shift)
        jobs = [(d.r.to_numpy(float), np.arange(n, dtype=float), K, ref, s) for s in seeds]
        null = np.array([v for v in pool.map(shift_job, jobs, chunksize=4) if v is not None])
        dk_obs = point["T"]["kappa"] - point["M3"]["kappa"]
        dnull = null - point["M3"]["kappa"]
        shift = dict(n=len(null), delta_obs=dk_obs, null_mean=float(dnull.mean()),
                     null_q025=float(np.percentile(dnull, 2.5)), null_q975=float(np.percentile(dnull, 97.5)),
                     p=float((dnull >= dk_obs).mean()))

    # 시기를 따라가는 원천: 연도별 규모
    raw = pd.DataFrame({"year": yr, "V": d.V, "abs_cvd": d.flow_cvd.abs(), "abs_bvc": d.flow_bvc.abs(),
                        "abs_r": d.r.abs(), "cvd_over_v": d.flow_cvd / d.V})
    Xc = transform.apply(np.column_stack([d.r, d.flow_cvd]), transform.METHOD)
    Xb = transform.apply(np.column_stack([d.r, d.flow_bvc]), transform.METHOD)
    raw["cvd_t"], raw["bvc_t"], raw["r_t"] = Xc[:, 1], Xb[:, 1], Xc[:, 0]
    yearly = {}
    for y, g in raw.groupby("year"):
        yearly[int(y)] = dict(bars=int(len(g)), median_V=float(g.V.median()), median_abs_cvd=float(g.abs_cvd.median()),
                              median_abs_bvc=float(g.abs_bvc.median()), median_abs_r=float(g.abs_r.median()),
                              mean_cvd_over_v=float(g.cvd_over_v.mean()),
                              mean_cvd_t=float(g.cvd_t.mean()), mean_abs_cvd_t=float(g.cvd_t.abs().mean()),
                              mean_bvc_t=float(g.bvc_t.mean()), mean_abs_bvc_t=float(g.bvc_t.abs().mean()),
                              sd_r_t=float(g.r_t.std()))

    keep = {nm: al[nm] for nm in CORE + LADDER + ["T"]}
    np.savez(MODELS / f"E070_states_{asset}_{interval}_K{K}.npz", M1=ref, **keep,
             timestamp=d.timestamp.astype("int64").to_numpy())
    if asset == "BTC" and interval == "4h" and K == 3:
        np.savez(MODELS / "E070_states_4h_K3.npz", M1=ref, **keep, timestamp=d.timestamp.astype("int64").to_numpy())

    rep = dict(asset=asset, interval=interval, K=K, n_bars=n, block=block, n_boot=N_BOOT,
               point=point, deltas=deltas, placebo=placebo, shift_test_T=shift,
               by_year_kappa=by_year, m1_year_shares=m1_year, model_year_shares=model_year, yearly=yearly)
    el = (time.time() - t0) / 60
    pr = lambda nm: point[nm]
    print(f"\n{asset} {interval} K={K}  {n:,}봉  ({el:.1f}분)")
    print(f"  {'모형':<5}{'κ':>8}{'κ_연도':>9}{'κ_분기':>9}{'κ_월':>9}{'런':>7}{'최소지속':>9}")
    for nm in ["M2", "M3", "M4", "M5", "M3v", "M2s", "M3z", "M2m", "M3s", "T", "T3"]:
        mp = pr(nm)["min_persist"]
        print(f"  {nm:<5}{pr(nm)['kappa']:>8.3f}{pr(nm)['kappa_year']:>9.3f}{pr(nm)['kappa_quarter']:>9.3f}"
              f"{pr(nm)['kappa_month']:>9.3f}{pr(nm)['runs']:>7}{(f'{mp:.3f}' if mp is not None else '—'):>9}", flush=True)
    print(f"  M1 런 {point['M1']['runs']}  최소지속 {point['M1']['min_persist']:.3f}")
    for k, v in placebo.items():
        print(f"  위약 {k}: κ 평균 {v['kappa']['mean']:.3f} [{v['kappa']['min']:.3f}, {v['kappa']['max']:.3f}]"
              f"  κ_분기 평균 {v['kappa_quarter']['mean']:.3f}  통제군보다 높은 비율 {v['share_above_control']:.2f}"
              f"  BVC 보다 높은 비율 {v['share_above_bvc']:.2f}", flush=True)
    for k, v in deltas.items():
        a, b, c = v["kappa"], v["kappa_year"], v["kappa_quarter"]
        print(f"  Δ {k:<9} κ {a['point']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]  "
              f"연도 {b['point']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]  "
              f"분기 {c['point']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]", flush=True)
    if shift:
        print(f"  시간 대조군 이동 검정: 관측 {shift['delta_obs']:+.4f}  영분포 평균 {shift['null_mean']:+.4f} "
              f"[{shift['null_q025']:+.4f}, {shift['null_q975']:+.4f}]  p {shift['p']:.4f}  ({shift['n']}회)")
    return rep


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for asset, iv, K, block, n_rw, n_iid, n_shift in SETTINGS:
            out[f"{asset}_{iv}_K{K}"] = run_setting(asset, iv, K, block, n_rw, n_iid, n_shift, pool)
            (TABLES / "E070_time_structure.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n저장: {TABLES / 'E070_time_structure.json'}")


if __name__ == "__main__":
    main()
