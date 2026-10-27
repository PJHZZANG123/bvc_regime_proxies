"""
E040~E041 — 라벨 정렬, 일치율·카파, 핵심 지표 Δ.

1. 라벨 정렬 : M1 과 대상 모형의 교차표에서 총 불일치를 최소화하는 1:1 매칭(헝가리안)
2. 일치율·카파 : Cohen's kappa. 국면 비중표를 함께 제시
   (비중이 치우치면 일치율은 높은데 카파가 낮아지는 현상 대비)
3. 핵심 지표 : Δ_k = Agree(M1, M_k) − Agree(M1, M3),  k ∈ {2, 4, 5}
   Δ_k > 0 이면 해당 프록시가 수익률만으로는 얻을 수 없는 국면 정보를 준 것.

사전 기대(E041, 결과를 보기 전에 고정): Δ₂ < 0, Δ₄ < 0, Δ₅ < 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"
LOGS = ROOT / "results" / "logs"
TABLES = ROOT / "results" / "tables"

MODEL_ORDER = ["M1", "M2", "M3", "M4", "M5"]
LABEL = {"M1": "CVD (기준)", "M2": "BVC", "M3": "통제군 (수익률만)",
         "M4": "CLV", "M5": "직교 잔차"}


def load_states(interval, K):
    out = {}
    for name in MODEL_ORDER:
        hits = sorted(MODELS.glob(f"{name}_{interval}_K{K}_seed*.npz"))
        if not hits:
            raise FileNotFoundError(f"{name}_{interval}_K{K} 모형 파일 없음")
        z = np.load(hits[0], allow_pickle=True)
        out[name] = (z["states"].astype(int), hits[0].name)
    return out


def align(ref: np.ndarray, other: np.ndarray, K: int):
    """other 의 라벨을 ref 에 맞춰 1:1 재배열 (총 일치 최대화)."""
    C = confusion_matrix(ref, other, labels=list(range(K)))
    row, col = linear_sum_assignment(-C)          # 일치 최대화 = 음수 비용 최소화
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    return np.array([mapping[int(s)] for s in other]), mapping, C


def evaluate(interval, K):
    st = load_states(interval, K)
    ref, ref_file = st["M1"]
    n = len(ref)
    print(f"\n{'='*100}\n{interval}   K* = {K}   {n:,}봉\n{'='*100}")

    # 국면 비중표 — 일치율/카파 괴리를 해석하려면 반드시 함께 봐야 한다
    print(f"\n  [국면 비중]")
    print(f"    {'모형':<20}" + "".join(f"{'상태'+str(k):>10}" for k in range(K)))
    shares = {}
    for name in MODEL_ORDER:
        s, _ = st[name]
        sh = np.bincount(s, minlength=K) / n
        shares[name] = [round(float(x), 4) for x in sh]
        print(f"    {LABEL[name]:<20}" + "".join(f"{x:>10.3f}" for x in sh))

    res = {}
    print(f"\n  [M1 대비 일치도]")
    print(f"    {'모형':<20}{'일치율':>10}{'카파':>10}   매칭")
    for name in ["M2", "M3", "M4", "M5"]:
        other, _ = st[name]
        aligned, mapping, C = align(ref, other, K)
        agree = float((aligned == ref).mean())
        kappa = float(cohen_kappa_score(ref, aligned))
        res[name] = dict(agree=agree, kappa=kappa,
                         mapping={str(k): v for k, v in mapping.items()},
                         confusion=C.tolist(), file=st[name][1])
        mp = " ".join(f"{k}->{v}" for k, v in sorted(mapping.items()))
        print(f"    {LABEL[name]:<20}{agree:>10.4f}{kappa:>10.4f}   {mp}")

    # --- 핵심 지표 ---
    base = res["M3"]["agree"]
    base_k = res["M3"]["kappa"]
    print(f"\n  [핵심 지표]  통제군 M3 기준선: 일치율 {base:.4f}, 카파 {base_k:.4f}")
    print(f"    {'':<20}{'Δ 일치율':>12}{'Δ 카파':>12}   판정")
    deltas = {}
    for name in ["M2", "M4", "M5"]:
        d = res[name]["agree"] - base
        dk = res[name]["kappa"] - base_k
        deltas[name] = dict(delta_agree=d, delta_kappa=dk)
        verdict = "프록시가 추가 정보 제공" if d > 0 else "통제군을 이기지 못함"
        print(f"    Δ({name}) {LABEL[name]:<12}{d:>+12.4f}{dk:>+12.4f}   {verdict}")

    return dict(interval=interval, K=K, n_bars=n, shares=shares,
                agreement=res, baseline_agree=base, baseline_kappa=base_k,
                deltas=deltas, ref_file=ref_file)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    ksel = json.loads((LOGS / "E020_k_selection.json").read_text(encoding="utf-8"))
    print("E040~E041 — 라벨 정렬 · 일치율/카파 · Δ")
    print("사전 기대(E041): Δ₂ < 0, Δ₄ < 0, Δ₅ < 0")
    rep = {}
    for iv in ("4h", "1d"):
        K = ksel[iv]["k_star"]
        rep[iv] = evaluate(iv, K)

    p = TABLES / "E041_delta.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*100}\n  요약 — Δ (일치율 기준)\n{'='*100}")
    print(f"    {'바':<8}{'K*':>4}{'Δ₂ (BVC)':>14}{'Δ₄ (CLV)':>14}{'Δ₅ (잔차)':>14}")
    for iv, r in rep.items():
        print(f"    {iv:<8}{r['K']:>4}"
              + "".join(f"{r['deltas'][m]['delta_agree']:>+14.4f}" for m in ["M2", "M4", "M5"]))
    print(f"\n리포트: {p}")


if __name__ == "__main__":
    sys.exit(main())
