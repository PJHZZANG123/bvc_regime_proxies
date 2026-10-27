"""
E001d — 전 기간(2021-01~2025-12) agg_trade_id 결손 구간 전수 조사.

기존 감사(E001b)는 월별로 갭 예시를 최대 5개까지만 저장했고, 월 경계를 넘는
갭은 조립 단계에서 따로 봤다. 여기서는 두 가지를 한 번에 확정한다.
  - 월 내부 갭 (개수 제한 없이 전부)
  - 월 경계 갭 (직전 월 마지막 id -> 다음 월 첫 id)

parquet 은 agg_trade_id 오름차순으로 조립돼 있으므로 배치 스트리밍으로 훑는다
(월 전체를 메모리에 올리지 않는다). 스트리밍 중 정렬이 깨지면 즉시 세운다.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "data" / "trades"
LOGS = ROOT / "results" / "logs"

BATCH = 4_000_000


def u(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S.%f")[:-3]


def scan_month(path: Path, prev_id, prev_t, prev_month):
    """한 월을 배치로 훑으며 모든 갭을 찾는다. 직전 월의 마지막 id/시각을 받아
    월 경계 갭도 같은 로직으로 처리한다."""
    gaps = []
    f = pq.ParquetFile(path)
    last_id, last_t = prev_id, prev_t
    boundary_from = prev_month
    rows = 0

    for b in f.iter_batches(batch_size=BATCH, columns=["agg_trade_id", "transact_time"]):
        ids = b.column("agg_trade_id").to_numpy()
        ts = b.column("transact_time").to_numpy()
        rows += len(ids)

        if not (np.diff(ids) >= 0).all():
            raise SystemExit(f"{path.name}: id 정렬 깨짐 — 조립 단계 재확인 필요")

        # 직전 배치(또는 직전 월)의 마지막 id 와 이 배치 첫 id 사이
        if last_id is not None and ids[0] - last_id > 1:
            gaps.append(dict(
                missing=int(ids[0] - last_id - 1),
                id_before=int(last_id), id_after=int(ids[0]),
                t_before=int(last_t), t_after=int(ts[0]),
                cross_month=(boundary_from is not None),
                from_month=boundary_from,
            ))
        boundary_from = None

        d = np.diff(ids)
        for k in np.where(d > 1)[0]:
            gaps.append(dict(
                missing=int(d[k] - 1),
                id_before=int(ids[k]), id_after=int(ids[k + 1]),
                t_before=int(ts[k]), t_after=int(ts[k + 1]),
                cross_month=False, from_month=None,
            ))

        last_id, last_t = int(ids[-1]), int(ts[-1])

    return gaps, last_id, last_t, rows


def main():
    files = sorted(TRADES.glob("aggTrades-*.parquet"))
    print(f"E001d 결손 전수 조사  |  {len(files)}개월\n", flush=True)

    all_gaps = []
    prev_id = prev_t = None
    prev_month = None
    total_rows = 0

    for k, path in enumerate(files, 1):
        month = path.stem.replace("aggTrades-", "")
        gaps, prev_id, prev_t, rows = scan_month(path, prev_id, prev_t, prev_month)
        prev_month = month
        total_rows += rows
        for g in gaps:
            g["month"] = month
        all_gaps += gaps
        tag = f"  갭 {len(gaps)}건 / {sum(g['missing'] for g in gaps):,}개" if gaps else ""
        print(f"[{k:2d}/{len(files)}] {month}  {rows:>11,}행{tag}", flush=True)

    print(f"\n{'='*100}")
    print(f"전체 {total_rows:,}행,  결손 구간 {len(all_gaps)}건,  "
          f"결손 id 총 {sum(g['missing'] for g in all_gaps):,}개")
    print(f"{'='*100}\n")

    hdr = f"{'#':>3}  {'결손 시작 (UTC)':<25}{'결손 끝 (UTC)':<25}{'결손 id':>12}  {'실경과':>10}  비고"
    print(hdr)
    print("-" * len(hdr))
    for i, g in enumerate(sorted(all_gaps, key=lambda x: x["t_before"]), 1):
        dur = (g["t_after"] - g["t_before"]) / 1000
        dur_s = f"{dur:,.1f}초" if dur < 60 else f"{dur/60:,.1f}분"
        note = f"월경계({g['from_month']}→{g['month']})" if g["cross_month"] else g["month"]
        print(f"{i:>3}  {u(g['t_before']):<25}{u(g['t_after']):<25}"
              f"{g['missing']:>12,}  {dur_s:>10}  {note}")

    out = LOGS / "E001d_all_gaps.json"
    out.write_text(json.dumps(
        dict(total_rows=total_rows,
             n_gaps=len(all_gaps),
             total_missing=sum(g["missing"] for g in all_gaps),
             gaps=[{**g, "utc_before": u(g["t_before"]), "utc_after": u(g["t_after"])}
                   for g in sorted(all_gaps, key=lambda x: x["t_before"])]),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    sys.exit(main())
