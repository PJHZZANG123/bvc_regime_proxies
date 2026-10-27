"""
E001b — 수집된 60개월 aggTrades parquet 전수 감사.

초기 manifest 검사(월 경계 id 불연속 + 시간 단조)로는 2022-09 블록 중복이
잡히지 않았다 (중복 블록이 시간순으로 삽입돼 두 검사를 모두 통과함).
그래서 이 감사는 월 *내부* 기준으로 본다:
  - 고유 agg_trade_id 수 대 행 수 (블록 중복 탐지)
  - id 결손 구간 (id 정렬 후 diff > 1)
  - 일자별 행 수 분포 (누락일 / 이상 희박일)
  - 행 순서가 시간순인지, id 순인지, 둘 다 아닌지
  - id 순 정렬 시 시간이 단조인지 (정렬 키로 id 를 써도 되는지 확인)
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "data" / "trades"
LOGS = ROOT / "results" / "logs"
DAY_MS = 86_400_000


def u(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def audit_month(path: Path) -> dict:
    f = pq.ParquetFile(path)
    ts_parts, id_parts = [], []
    for b in f.iter_batches(batch_size=8_000_000, columns=["transact_time", "agg_trade_id"]):
        ts_parts.append(b.column("transact_time").to_numpy())
        id_parts.append(b.column("agg_trade_id").to_numpy())
    t = np.concatenate(ts_parts)
    i = np.concatenate(id_parts)
    del ts_parts, id_parts

    n = len(t)
    n_unique_id = len(np.unique(i))
    n_dup_rows = n - n_unique_id

    order = np.argsort(i, kind="stable")
    isr, tsr = i[order], t[order]
    d = np.diff(isr)
    gap_mask = d > 1
    n_gap_segments = int(gap_mask.sum())
    n_missing_ids = int((d[gap_mask] - 1).sum()) if n_gap_segments else 0
    gap_examples = [
        {"id_before": int(isr[k]), "id_after": int(isr[k + 1]),
         "missing": int(d[k] - 1), "t_before": u(tsr[k]), "t_after": u(tsr[k + 1])}
        for k in np.where(gap_mask)[0][:5]
    ]

    time_sorted_as_is = bool((np.diff(t) >= 0).all())
    id_sorted_as_is = bool((np.diff(i) >= 0).all())
    time_monotonic_when_id_sorted = bool((np.diff(tsr) >= 0).all())

    days = np.floor_divide(t, DAY_MS)
    uniq_days, cnt = np.unique(days, return_counts=True)
    full_range = np.arange(int(days.min()), int(days.max()) + 1)
    missing_days = np.setdiff1d(full_range, uniq_days)
    mean_cnt = cnt.mean()
    thin_days = [
        {"date": u(int(x) * DAY_MS)[:10], "rows": int(c), "ratio": round(c / mean_cnt, 3)}
        for x, c in zip(uniq_days, cnt) if c < mean_cnt * 0.35
    ]

    return dict(
        rows=n,
        unique_ids=int(n_unique_id),
        dup_rows=int(n_dup_rows),
        id_gap_segments=n_gap_segments,
        missing_ids=n_missing_ids,
        gap_examples=gap_examples,
        time_sorted_as_is=time_sorted_as_is,
        id_sorted_as_is=id_sorted_as_is,
        time_monotonic_when_id_sorted=time_monotonic_when_id_sorted,
        days_present=len(uniq_days),
        days_expected=len(full_range),
        missing_days=[u(int(x) * DAY_MS)[:10] for x in missing_days],
        thin_days=thin_days,
        t_min=u(int(t.min())),
        t_max=u(int(t.max())),
    )


def main():
    files = sorted(TRADES.glob("aggTrades-*.parquet"))
    print(f"E001b 전수 감사  |  {len(files)}개 파일\n", flush=True)

    report = {}
    flagged = []
    for k, path in enumerate(files, 1):
        month = path.stem.replace("aggTrades-", "")
        r = audit_month(path)
        report[month] = r

        issues = []
        if r["dup_rows"] > 0:
            issues.append(f"중복 {r['dup_rows']:,}행")
        if r["missing_ids"] > 0:
            issues.append(f"id결손 {r['missing_ids']:,}개/{r['id_gap_segments']}구간")
        if r["missing_days"]:
            issues.append(f"누락일 {r['missing_days']}")
        if r["thin_days"]:
            issues.append(f"희박일 {[d['date'] for d in r['thin_days']]}")
        if not r["time_sorted_as_is"]:
            issues.append("행순서 시간비단조")
        if not r["time_monotonic_when_id_sorted"]:
            issues.append("[!] id정렬해도 시간비단조 -- 정렬키 재검토 필요")

        tag = "OK" if not issues else "ISSUE"
        print(f"[{k:2d}/{len(files)}] {month}  {r['rows']:>11,}행  {tag}"
              + (("  " + " | ".join(issues)) if issues else ""), flush=True)
        if issues:
            flagged.append(month)

    out = LOGS / "E001b_audit_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n감사 완료  {len(files)}개월 중 {len(flagged)}개월에 이슈: {flagged}")
    print(f"상세 리포트: {out}")


if __name__ == "__main__":
    main()
