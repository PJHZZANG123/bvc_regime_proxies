"""
적합된 M1 을 실시간 추론용 JSON 하나로 동결한다.

봇은 이 파일만 있으면 된다 — parquet 도 hmmlearn 도 필요 없다.
담는 것
  · HMM 모수 (startprob, transmat, means, covars)
  · 관측 변환 상수 (열별 MAD, 변환 후 mean/std)
  · 상태 라벨 (평균의 흐름축 값으로 자동 판정)

**변환 상수를 고정하는 이유.** `transform.py` 는 주어진 배열에서 MAD 와 mean/std 를
그때그때 계산한다. 실시간에서 그대로 쓰면 새 봉이 들어올 때마다 척도가 움직여
같은 관측이 다른 값으로 변환된다. 학습 표본에서 한 번 구한 값을 박아 둬야
모형이 학습 때와 같은 공간을 본다.

    python src/freeze_model.py
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "results" / "models" / "M1_4h_K3_seed9.npz"
BARS = ROOT / "data" / "bars" / "bars_4h.parquet"
OUT = ROOT / "results" / "models" / "live_M1_4h_K3.json"
WARMUP = 101          # 공통 표본 시작점 (학습과 동일)


def main():
    z = np.load(MODEL, allow_pickle=True)
    b = pq.read_table(BARS).to_pandas().sort_values("bar_start")
    r = np.log(b.close / b.close.shift(1)).to_numpy()
    cvd = b.cvd_usdt.to_numpy()
    X = np.column_stack([r, cvd])[WARMUP:]

    # 열별 변환 상수 — 학습 표본에서 한 번만
    consts = []
    for j in range(X.shape[1]):
        x = X[:, j]
        mad = float(np.median(np.abs(x - np.median(x))))
        y = np.sign(x) * np.log1p(np.abs(x) / mad)
        consts.append({"mad": mad, "mean": float(y.mean()), "std": float(y.std())})

    means = z["means"]
    # 흐름축(2번째 성분) 평균으로 상태 성격을 정한다.
    # 가장 낮으면 순매도, 가장 높으면 순매수, 나머지가 저변동.
    flow = means[:, 1]
    lo, hi = int(flow.argmin()), int(flow.argmax())
    labels = {}
    for k in range(len(flow)):
        labels[str(k)] = "순매도 우세" if k == lo else "순매수 우세" if k == hi else "저변동"

    doc = {
        "model": MODEL.name,
        "symbol": "BTCUSDT",
        "interval": "4h",
        "K": int(len(z["startprob"])),
        "cols": [str(c) for c in z["cols"]],
        "transform": str(z["transform"]),
        "startprob": z["startprob"].tolist(),
        "transmat": z["transmat"].tolist(),
        "means": means.tolist(),
        "covars": z["covars"].tolist(),
        "consts": consts,
        "labels": labels,
        "train_bars": int(len(X)),
        "train_from_ms": int(b.bar_start.to_numpy()[WARMUP]),
        "train_to_ms": int(b.bar_start.to_numpy()[-1]),
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"  학습 표본 {doc['train_bars']:,}봉   K={doc['K']}   관측 {doc['cols']}")
    print("  변환 상수")
    for nm, c in zip(doc["cols"], consts):
        print(f"    {nm:<10} MAD {c['mad']:.6g}   mean {c['mean']:+.6f}   std {c['std']:.6f}")
    print("  상태 라벨 (흐름축 평균 기준)")
    for k in range(doc["K"]):
        print(f"    상태 {k}  흐름축 평균 {flow[k]:+.4f}  →  {labels[str(k)]}")


if __name__ == "__main__":
    raise SystemExit(main())
