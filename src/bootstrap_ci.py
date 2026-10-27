"""
E044 — Δ 의 신뢰구간 (규약에 없던 항목, 2026-09-15 추가)

추가 이유
  Δ 의 부호를 주장하려면 불확실성이 필요하다. 이는 결과와 무관하게 성립하는
  요건이므로 사후 추가여도 명세 탐색이 아니다. 규약에 빠져 있던 것을 메운다.

방법 — 순환 블록 부트스트랩 (재적합 없음)
  이미 적합된 상태열을 전 표본에서 한 번 정렬한 뒤, 봉 인덱스를 블록 단위로
  재추출해 일치율·카파의 분포를 만든다.
  * 다섯 모형에 **같은 인덱스**를 쓴다. Δ 는 같은 봉에서의 차이여야 한다.
  * 정렬(헝가리안)은 전 표본에서 한 번만 한다. 재추출마다 다시 정렬하면
    표본마다 일치를 최대화해 낙관 편향이 생긴다.

블록 길이
  일치 지표는 국면이 지속되는 동안 강하게 자기상관된다. 블록이 국면 지속보다
  짧으면 불확실성을 과소평가한다. 따라서 국면 지속 규모(수십~200봉)에 맞춘
  길이를 주로 쓰고, 짧은 길이도 함께 제시해 민감도를 드러낸다.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"

N_BOOT = 2000
CONFIGS = [("4h", 3, [23, 100, 200]), ("1d", 2, [13, 30, 60]), ("4h", 2, [23, 100, 200])]
PRIMARY_IDX = -1          # 블록 길이 목록의 마지막(가장 긴 것)을 주 결과로
LABEL = {"M2": "BVC", "M3": "통제군", "M4": "CLV", "M5": "직교 잔차"}


def load_states(interval, K):
    out = {}
    for nm in ["M1", "M2", "M3", "M4", "M5"]:
        hits = sorted(MODELS.glob(f"{nm}_{interval}_K{K}_seed*.npz"))
        out[nm] = np.load(hits[0], allow_pickle=True)["states"].astype(int)
    return out


def align_once(ref, other, K):
    C = confusion_matrix(ref, other, labels=list(range(K)))
    r, c = linear_sum_assignment(-C)
    mp = {int(cc): int(rr) for rr, cc in zip(r, c)}
    return np.array([mp[int(s)] for s in other])


def circ_blocks(n, L, rng):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nb)
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:n] % n


def run(interval, K, block_lens):
    st = load_states(interval, K)
    ref = st["M1"]
    n = len(ref)
    aligned = {nm: align_once(ref, st[nm], K) for nm in ["M2", "M3", "M4", "M5"]}

    print(f"\n{'='*104}\n{interval}  K={K}  {n:,}봉  재추출 {N_BOOT:,}회\n{'='*104}")
    # 점추정치는 표본에서 직접 계산한다. 블록 길이와 무관하며 재추출은 구간만 제공한다.
    pa = {nm: float((aligned[nm] == ref).mean() - (aligned["M3"] == ref).mean())
          for nm in ["M2", "M4", "M5"]}
    pk = {nm: float(cohen_kappa_score(ref, aligned[nm]) - cohen_kappa_score(ref, aligned["M3"]))
          for nm in ["M2", "M4", "M5"]}
    print("  점추정  " + "   ".join(
        f"{LABEL[nm]} Δ일치율 {pa[nm]:+.4f} Δ카파 {pk[nm]:+.4f}" for nm in ["M2", "M4", "M5"]))

    out = {}
    for L in block_lens:
        rng = np.random.default_rng(20260915)
        acc = {nm: {"a": [], "k": []} for nm in aligned}
        for _ in range(N_BOOT):
            idx = circ_blocks(n, L, rng)
            r_b = ref[idx]
            for nm, al in aligned.items():
                a_b = al[idx]
                acc[nm]["a"].append(float((a_b == r_b).mean()))
                acc[nm]["k"].append(float(cohen_kappa_score(r_b, a_b)))

        base_a = np.array(acc["M3"]["a"])
        base_k = np.array(acc["M3"]["k"])
        print(f"\n  [블록 {L}봉]")
        print(f"    {'갈래':<12}{'Δ 일치율':>11}{'95% CI':>24}{'0 포함':>8}"
              f"{'Δ 카파':>11}{'95% CI':>24}{'0 포함':>8}")
        print("    " + "-" * 98)
        res = {}
        for nm in ["M2", "M4", "M5"]:
            da = np.array(acc[nm]["a"]) - base_a
            dk = np.array(acc[nm]["k"]) - base_k
            ca = (float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5)))
            ck = (float(np.percentile(dk, 2.5)), float(np.percentile(dk, 97.5)))
            za = "포함" if ca[0] <= 0 <= ca[1] else "-"
            zk = "포함" if ck[0] <= 0 <= ck[1] else "-"
            res[nm] = dict(
                point_agree=pa[nm], ci_agree=ca, zero_in_agree=(za == "포함"),
                boot_mean_agree=float(da.mean()),
                point_kappa=pk[nm], ci_kappa=ck, zero_in_kappa=(zk == "포함"),
                boot_mean_kappa=float(dk.mean()),
            )
            print(f"    {LABEL[nm]:<12}{pa[nm]:>+11.4f}"
                  f"{f'[{ca[0]:+.4f}, {ca[1]:+.4f}]':>24}{za:>8}"
                  f"{pk[nm]:>+11.4f}{f'[{ck[0]:+.4f}, {ck[1]:+.4f}]':>24}{zk:>8}")
        out[f"L{L}"] = res
    return dict(interval=interval, K=K, n_bars=n, n_boot=N_BOOT,
                block_lens=block_lens, primary_block=block_lens[PRIMARY_IDX], by_block=out)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print(f"E044 — Δ 신뢰구간 (순환 블록 부트스트랩 {N_BOOT:,}회, 재적합 없음)")
    rep = {f"{iv}_K{K}": run(iv, K, Ls) for iv, K, Ls in CONFIGS}
    p = TABLES / "E044_delta_ci.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*104}\n  요약 — 주 블록 길이 기준, 95% CI 가 0 을 포함하는가\n{'='*104}")
    print(f"    {'설정':<14}{'블록':>6}{'Δ₂ BVC':>22}{'Δ₄ CLV':>22}{'Δ₅ 잔차':>22}")
    for key, r in rep.items():
        L = r["primary_block"]
        row = r["by_block"][f"L{L}"]
        cells = []
        for nm in ["M2", "M4", "M5"]:
            v = row[nm]
            cells.append(f"{v['point_kappa']:+.4f} {'(0포함)' if v['zero_in_kappa'] else '       '}")
        print(f"    {key:<14}{L:>6}" + "".join(f"{c:>22}" for c in cells))
    print("\n  (카파 기준)")
    print(f"리포트: {p}")


if __name__ == "__main__":
    sys.exit(main())
