"""
E042 — 블록 부트스트랩으로 Δ의 영가설 분포를 만든다.

영가설
  프록시 흐름은 **시간 정렬된** 국면 정보를 담고 있지 않다.
  (주변분포와 자기상관은 그대로 두되, r·CVD 와의 시점 대응만 깬다)

절차 — 갈래마다 따로 (CLAUDE.md)
  1. 프록시 흐름 열만 순환 블록 셔플. 수익률 r 은 원래 자리에 둔다.
  2. 그 관측 벡터로 M_k 를 재적합 → 상태열
  3. Δ_k^null = Agree(M1, M_k^shuf) − Agree(M1, M3)
     M1·M3 는 관측 데이터로 한 번만 적합해 고정한다.
  4. 1,000회 반복 → 영분포. 관측 Δ_k 가 그 안에서 어디인지 본다.

반복수 1,000 은 사전 등록에 따른 값(Δ₅>0 인 경우 200 → 1,000 상향).
블록 길이는 n^(1/3) 올림 — 블록 부트스트랩의 표준 규칙.
부트스트랩 적합은 시드 3개(관측 적합은 10개)로 줄인다. 영분포의 각 표본은
최고 우도 하나를 대표값으로만 쓰므로 관측 적합만큼의 정밀도가 필요하지 않다.
"""

import json
import os
import sys
import time
from pathlib import Path

# 워커마다 BLAS 스레드를 1로 묶는다. 묶지 않으면 프로세스 8개가 각자 스레드를
# 띄워 서로 경합해 오히려 느려진다. numpy import 전에 설정해야 한다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transform

WORKERS = 7     # 논리 코어 8개 중 1개는 남겨둔다

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
OUT = ROOT / "results" / "bootstrap"
LOGS = ROOT / "results" / "logs"

N_BOOT = 1000
SEEDS_OBS = list(range(10))
SEEDS_BOOT = [0, 1, 2]
N_ITER_OBS, N_ITER_BOOT = 500, 300
TOL, COV = 1e-4, "full"

PROXY = {"M2": ("flow_bvc", "BVC"), "M4": ("flow_clv", "CLV"), "M5": ("flow_res", "직교 잔차")}
CONFIGS = [("4h", 3), ("1d", 2), ("4h", 2)]     # 앞의 둘이 주 결과


def fit_best(X, K, seeds, n_iter):
    best = None
    for s in seeds:
        m = GaussianHMM(n_components=K, covariance_type=COV,
                        n_iter=n_iter, tol=TOL, random_state=s)
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[1]):
            best = (m, ll)
    return best


def align_scores(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    al = np.array([mp[int(s)] for s in other])
    return float((al == ref).mean()), float(cohen_kappa_score(ref, al))


def circular_block_shuffle(x: np.ndarray, L: int, rng) -> np.ndarray:
    """순환 블록 부트스트랩: 길이 L 블록을 무작위 시작점에서 떼어 이어붙인다."""
    n = len(x)
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(L)[None, :]).ravel() % n
    return x[idx[:n]]


def _one_rep(args):
    """워커 하나가 도는 부트스트랩 1회. 피클 가능한 인자만 받는다."""
    r_col, flow, L, K, ref, base_a, base_k, seed = args
    rng = np.random.default_rng(seed)
    shuf = circular_block_shuffle(flow, L, rng)
    Xb = transform.apply(np.column_stack([r_col, shuf]), transform.METHOD)
    fb = fit_best(Xb, K, SEEDS_BOOT, N_ITER_BOOT)
    if fb is None:
        return None
    a, kp = align_scores(ref, fb[0].predict(Xb), K)
    return a - base_a, kp - base_k


def run(interval, K):
    d = pq.read_table(FEAT / f"features_{interval}.parquet").to_pandas()
    n = len(d)
    L = int(np.ceil(n ** (1 / 3)))
    print(f"\n{'='*100}\n{interval}  K={K}  {n:,}봉  블록 길이 {L}  반복 {N_BOOT:,}회\n{'='*100}", flush=True)

    # 관측 적합 (고정)
    Xs = {nm: transform.apply(d[cols].to_numpy(float), transform.METHOD)
          for nm, cols in [("M1", ["r", "flow_cvd"]), ("M3", ["r"])]}
    ref = fit_best(Xs["M1"], K, SEEDS_OBS, N_ITER_OBS)[0].predict(Xs["M1"])
    s3 = fit_best(Xs["M3"], K, SEEDS_OBS, N_ITER_OBS)[0].predict(Xs["M3"])
    base_a, base_k = align_scores(ref, s3, K)
    print(f"  통제군 기준선: 일치율 {base_a:.4f}, 카파 {base_k:.4f}", flush=True)

    r_col = d["r"].to_numpy(float)
    out = {}
    for nm, (col, label) in PROXY.items():
        flow = d[col].to_numpy(float)
        X_obs = transform.apply(np.column_stack([r_col, flow]), transform.METHOD)
        st_obs = fit_best(X_obs, K, SEEDS_OBS, N_ITER_OBS)[0].predict(X_obs)
        a_obs, k_obs = align_scores(ref, st_obs, K)
        d_obs_a, d_obs_k = a_obs - base_a, k_obs - base_k

        # 반복마다 독립 시드를 주어 재현 가능하게 한다
        seeds = np.random.SeedSequence(20260915).spawn(N_BOOT)
        jobs = [(r_col, flow, L, K, ref, base_a, base_k, s) for s in seeds]
        null_a, null_k = [], []
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(_one_rep, j) for j in jobs]):
                res = fut.result()
                done += 1
                if res is not None:
                    null_a.append(res[0])
                    null_k.append(res[1])
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f"    {nm} {done:>5,}/{N_BOOT:,}  경과 {el/60:5.1f}분  "
                          f"남은예상 {el/done*(N_BOOT-done)/60:5.1f}분", flush=True)

        na, nk = np.array(null_a), np.array(null_k)
        p_a = float((na >= d_obs_a).mean())
        p_k = float((nk >= d_obs_k).mean())
        out[nm] = dict(
            label=label, n_null=len(na), block_len=L,
            delta_obs_agree=d_obs_a, delta_obs_kappa=d_obs_k,
            null_agree=dict(mean=float(na.mean()), sd=float(na.std()),
                            q025=float(np.percentile(na, 2.5)),
                            q50=float(np.percentile(na, 50)),
                            q975=float(np.percentile(na, 97.5))),
            null_kappa=dict(mean=float(nk.mean()), sd=float(nk.std()),
                            q025=float(np.percentile(nk, 2.5)),
                            q50=float(np.percentile(nk, 50)),
                            q975=float(np.percentile(nk, 97.5))),
            p_upper_agree=p_a, p_upper_kappa=p_k,
        )
        print(f"  {nm} · {label}", flush=True)
        print(f"    관측 Δ 일치율 {d_obs_a:+.4f}   영분포 평균 {na.mean():+.4f} "
              f"sd {na.std():.4f}  95% [{np.percentile(na,2.5):+.4f}, {np.percentile(na,97.5):+.4f}]"
              f"   p(상단) {p_a:.4f}", flush=True)
        print(f"    관측 Δ 카파   {d_obs_k:+.4f}   영분포 평균 {nk.mean():+.4f} "
              f"sd {nk.std():.4f}  95% [{np.percentile(nk,2.5):+.4f}, {np.percentile(nk,97.5):+.4f}]"
              f"   p(상단) {p_k:.4f}", flush=True)

    return dict(interval=interval, K=K, n_bars=n, block_len=L, n_boot=N_BOOT,
                baseline_agree=base_a, baseline_kappa=base_k, proxies=out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E042 — 블록 부트스트랩  ({N_BOOT:,}회, 갈래마다 따로)")
    rep = {}
    for iv, K in CONFIGS:
        key = f"{iv}_K{K}"
        rep[key] = run(iv, K)
        # 구성 하나 끝날 때마다 저장 (중단돼도 앞 결과는 남는다)
        (OUT / "E042_bootstrap.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [저장] {key} 까지", flush=True)
    print(f"\n완료: {OUT / 'E042_bootstrap.json'}")


if __name__ == "__main__":
    sys.exit(main())
