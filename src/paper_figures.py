"""
논문 그림 후보 네 개 (matplotlib 기본 스타일).

  fig_design.png      연구 설계 — 자료에서 Δκ 까지 (fig_design_en.png 은 학술지용 영문판)
  fig_multiplier.png  BVC 승수 2F_df(z) - 1 을 자유도별로
  fig_forest.png      설정별 Δκ 와 95% 구간 (BTC·ETH, 4시간봉·일봉, K=3·K=2)
  fig_regime.png      국면별 일치율 차이 (프록시 - 통제군), 4시간봉 K=3

한글 글꼴만 지정하고 나머지는 matplotlib 기본값을 쓴다.

    python src/paper_figures.py

출력: docs/figures/
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import norm, t as tdist

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "figures"
PROXY = {"M2": "BVC", "M4": "CLV 흐름", "M5": "직교 잔차"}


FONT_DIR = ROOT / "src" / "fonts"


def noto_static():
    """Noto Sans KR 가변 글꼴의 기본값이 가장 가는 굵기(100)라 matplotlib 에서 그대로 쓰면 글씨가 흐리다.
    PDF 변환(export_memo_pdf.py)과 같은 고정 굵기 파일(src/fonts)을 쓴다. 없으면 같은 방법으로 만든다."""
    from export_memo_pdf import _make_kr
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for w, tag in [(400, "Regular"), (700, "Bold")]:
        p = FONT_DIR / f"NotoSansKR-{tag}.ttf"
        if not p.exists():
            _make_kr(w, tag, p)
        out[tag] = p
    return out


def setup():
    for p in noto_static().values():
        font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "Noto Sans KR",
        "axes.unicode_minus": False,
        # 수식(Δ, σ, τ 등)도 같은 글꼴로
        "mathtext.fontset": "custom",
        "mathtext.rm": "Noto Sans KR",
        "mathtext.it": "Noto Sans KR",
        "mathtext.bf": "Noto Sans KR:bold",
        "mathtext.sf": "Noto Sans KR",
    })
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- 1. 연구 설계
# 학술지 작성요령: 국문논문도 그림 안의 글자는 모두 영문. 국문판(fig_design.png)은 원고 검토용으로 남긴다.
DESIGN_TEXT = {
    "ko": dict(trades="체결 기록\n(aggTrades\n28.1억 건)", bars="봉\n(4시간봉·일봉)", ret="수익률 r",
               flows={0: "CVD", 1: "BVC", 3: "CLV 흐름", 4: "직교 잔차"},
               models=["M1 [r, CVD]", "M2 [r, BVC]", "M3 [r]", "M4 [r, CLV]", "M5 [r, 잔차]"],
               match="헝가리안\n대응", delta="Δκ\n(M3 대비)",
               heads=["자료", "주문흐름 측정치", "HMM", "평가"], out="fig_design.png"),
    "en": dict(trades="Trade records\n(aggTrades,\n2.81 billion)", bars="Bars\n(4-hour, daily)", ret="Return r",
               flows={0: "CVD", 1: "BVC", 3: "CLV flow", 4: "Orthogonal residual"},
               models=["M1 [r, CVD]", "M2 [r, BVC]", "M3 [r]", "M4 [r, CLV]", "M5 [r, residual]"],
               match="Hungarian\nmatching", delta="Δκ\n(vs. M3)",
               heads=["Data", "Order-flow measures", "HMM", "Evaluation"], out="fig_design_en.png"),
}


def fig_design(lang="ko"):
    tx = DESIGN_TEXT[lang]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(7, 60); ax.axis("off")

    def box(x, y, w, h, text):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="black"))

    rows = [46, 37, 28, 19, 10]
    H = 7
    mid = [y + H / 2 for y in rows]

    box(1, 40, 18, 11, tx["trades"])
    box(1, 22, 18, 11, tx["bars"])
    arrow(10, 40, 10, 33)

    for i, nm in tx["flows"].items():
        box(26, rows[i], 20, H, nm)
        arrow(19, mid[2], 26, mid[i])
        arrow(46, mid[i], 53, mid[i])
    arrow(19, mid[2], 53, mid[2])
    ax.text(36, mid[2] + 0.8, tx["ret"], ha="center", va="bottom", fontsize=8)

    for y, nm in zip(rows, tx["models"]):
        box(53, y, 19, H, nm)
        ax.plot([72, 76], [y + H / 2] * 2, color="black", lw=0.8)
    ax.plot([76, 76], [mid[-1], mid[0]], color="black", lw=0.8)

    box(81, 34, 18, 11, tx["match"])
    box(81, 13, 18, 11, tx["delta"])
    arrow(76, 39.5, 81, 39.5)
    arrow(90, 34, 90, 24)

    for x, s in zip([10, 36, 62.5, 90], tx["heads"]):
        ax.text(x, 56, s, ha="center", fontsize=9)
    save(fig, tx["out"])


# ---------------------------------------------------------------- 2. BVC 승수
def fig_multiplier():
    z = np.linspace(-4, 4, 801)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z, 2 * tdist.cdf(z, 0.25) - 1, "-", label="df = 0.25 (본 연구)")
    ax.plot(z, 2 * tdist.cdf(z, 1) - 1, "--", label="df = 1")
    ax.plot(z, 2 * tdist.cdf(z, 30) - 1, "-.", label="df = 30")
    ax.plot(z, 2 * norm.cdf(z) - 1, ":", label="표준정규")
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(r"$\Delta P_{\tau} / \sigma_{\Delta P}$")
    ax.set_ylabel(r"$2F_{\mathrm{df}}(\Delta P_{\tau} / \sigma_{\Delta P}) - 1$")
    ax.legend()
    save(fig, "fig_multiplier.png")


# ---------------------------------------------------------------- 3. Δκ 숲그림
def fig_forest(lang="ko"):
    """lang="en" 은 학술지용 영문판(fig_forest_en.png)."""
    en = lang == "en"
    proxy = {"M2": "BVC", "M4": "CLV flow", "M5": "Orthogonal residual"} if en else PROXY
    ci = json.loads((TABLES / "E044_delta_ci.json").read_text(encoding="utf-8"))
    eth = json.loads((TABLES / "ETH_validation.json").read_text(encoding="utf-8"))

    def btc(key):
        v = ci[key]; b = v["by_block"][f"L{v['primary_block']}"]
        return {m: (b[m]["point_kappa"], *b[m]["ci_kappa"]) for m in PROXY}

    def e(iv):
        d = eth[iv]["deltas"]
        return {m: (d[m]["delta_kappa"], *d[m]["ci"]) for m in PROXY}

    h4, d1 = ("4-hour", "daily") if en else ("4시간봉", "일봉")
    rows = [(f"BTC {h4} K=3", btc("4h_K3")),
            (f"BTC {h4} K=2", btc("4h_K2")),
            (f"BTC {d1} K=2", btc("1d_K2")),
            (f"ETH {h4} K=3", e("4h")),
            (f"ETH {d1} K=4", e("1d"))]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ys = np.arange(len(rows))[::-1].astype(float)
    for j, (m, mk) in enumerate(zip(PROXY, ["o", "s", "^"])):
        pts = np.array([r[1][m] for r in rows])
        yy = ys + (1 - j) * 0.2
        ax.errorbar(pts[:, 0], yy, xerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]],
                    fmt=mk, capsize=3, label=proxy[m])
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Δκ (95% CI)" if en else "Δκ (95% 구간)")
    # 그림 안에는 구간과 겹치지 않는 빈자리가 없어 범례를 위쪽 바깥에 한 줄로 둔다
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, "fig_forest_en.png" if en else "fig_forest.png")


# ---------------------------------------------------------------- 4. 국면별 차이
def fig_regime():
    d = json.loads((TABLES / "E043_conditional.json").read_text(encoding="utf-8"))["4h_K3"]
    regimes = ["순매도", "고변동", "저변동"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3); w = 0.25
    for i, m in enumerate(PROXY):
        ax.bar(x + (i - 1) * w, d["delta_by_state"][m], w, label=PROXY[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(regimes)
    ax.set_xlabel("기준 모형의 국면")
    ax.set_ylabel("일치율 차이 (프록시 - 통제군)")
    ax.legend()
    save(fig, "fig_regime.png")


# ---------------------------------------------------------------- 논문_수정안: 연구 설계(시간 대조군 포함)
DESIGN_V2_TEXT = {
    "en": dict(trades="Trade records\n(aggTrades,\n2.81 billion)", bars="Bars\n(4-hour, daily)",
               measures=["CVD", "BVC", "CLV flow", "CLV residual", "Return r only", "Time index t"],
               models=["M1 [r, CVD]\nbenchmark", "M2 [r, BVC]", "M4 [r, CLV]", "M5 [r, residual]",
                       "M3 [r]\nreturns control", "T [r, t]\ntemporal control"],
               match="Hungarian\nmatching", kappa="Overall κ\nPeriod-adjusted κ", delta="Δκ vs. M3\nand vs. T",
               heads=["Data", "Measures", "HMM (K from M1)", "Evaluation"], out="fig_design_v2_en.png"),
    "ko": dict(trades="집계 체결\n(aggTrades,\n28.1억 행)", bars="봉\n(4시간봉·일봉)",
               measures=["CVD", "BVC", "CLV 흐름", "CLV 잔차", "수익률 r 만", "봉 순번 t"],
               models=["M1 [r, CVD]\n기준 모형", "M2 [r, BVC]", "M4 [r, CLV 흐름]", "M5 [r, CLV 잔차]",
                       "M3 [r]\n수익률 모형", "T [r, t]\n순번 모형"],
               match="헝가리안\n대응", kappa="전체 κ\n시기 조정 κ", delta="M3·T 대비\nΔκ",
               heads=["자료", "측정치", "은닉마르코프모형 (K: M1 기준)", "평가"], out="fig_design_v2.png"),
}


def fig_design_v2(lang="en", noclv=False):
    """수정안의 연구 설계. 수익률 통제군 옆에 비정보 시간 대조군을 두고, 일치도를 전체·시기 조정 두 가지로 잰다.
    lang="ko" 는 투고본(지능정보연구)용 국문판(fig_design_v2.png). noclv=True 는 CLV 흐름·잔차를 뺀 판(…_noclv.png)."""
    tx = dict(DESIGN_V2_TEXT[lang])
    if noclv:
        keep = [0, 1, 4, 5]
        tx["measures"] = [tx["measures"][i] for i in keep]
        tx["models"] = [tx["models"][i] for i in keep]
        tx["out"] = tx["out"].replace(".png", "_noclv.png")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(2, 62); ax.axis("off")

    def box(x, y, w, h, text, fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="black"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="black"))

    rows = [47, 37, 27, 17] if noclv else [50, 42, 34, 26, 18, 10]
    H = 6
    mid = [y + H / 2 for y in rows]
    box(1, 40, 17, 11, tx["trades"])
    box(1, 21, 17, 11, tx["bars"])
    arrow(9.5, 40, 9.5, 32)
    for i, (ms, md) in enumerate(zip(tx["measures"], tx["models"])):
        box(25, rows[i], 19, H, ms)
        arrow(18, 26.5, 25, mid[i])
        box(50, rows[i], 21, H, md, fs=8.5)
        arrow(44, mid[i], 50, mid[i])
        ax.plot([71, 75], [mid[i]] * 2, color="black", lw=0.8)
    ax.plot([75, 75], [mid[-1], mid[0]], color="black", lw=0.8)
    box(80, 38, 19, 9, tx["match"])
    box(80, 22, 19, 11, tx["kappa"])
    box(80, 7, 19, 10, tx["delta"])
    arrow(75, 42.5, 80, 42.5)
    arrow(89.5, 38, 89.5, 33)
    arrow(89.5, 22, 89.5, 17)
    for x, s in zip([9.5, 34.5, 60.5, 89.5], tx["heads"]):
        ax.text(x, 59, s, ha="center", fontsize=9)
    save(fig, tx["out"])


# ---------------------------------------------------------------- 투고본 그림 1(2026-11-03): 비교 모형 전부를 역할별로 묶는다
def fig_design_v3():
    """투고본(지능정보연구) 그림 1 → fig_design_v3.png. fig_design_v2(noclv) 의 네 모형에 BVC 분해 모형(표 8)과
    랜덤워크·백색잡음을 더해 주 비교 / BVC 분해 / 시기 구분 점검(주문흐름 정보 없음)으로 묶고, 난수는 따로 된 자료
    상자에서 나오게 그린다. 좌표는 가로 2000·세로 1616 단위(위가 0)."""
    W, H = 2000, 1616
    fig = plt.figure(figsize=(8, 8 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    def box(x0, y0, x1, y1, text, fs=9):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="black"))

    def line(xs, ys):
        ax.plot(xs, ys, color="black", lw=0.8)

    sg = r"$\hat{\sigma}$"
    groups = [("주 비교", [("CVD", "M1 [r, CVD]\n기준 모형"), ("BVC", "M2 [r, BVC]"),
                         ("수익률 r 만", "M3 [r]\n수익률 모형")]),
              ("BVC 분해 (표 8)", [("거래대금 V", "M3v [r, V]"), ("부호 × 거래대금", "M2s [r, sign(r)·V]"),
                                ("표준화 수익률 × 거래대금", f"M3z [r, V·r/{sg}]"),
                                ("BVC 승수", f"M2m [r, 2F(r/{sg}) − 1]"),
                                (f"롤링 표준편차 {sg}", f"M3σ [r, {sg}]\n표준편차 모형")]),
              ("시기 구분 점검 (주문흐름 정보 없음)", [("봉 순번 t", "T [r, t]\n순번 모형"), ("랜덤워크 20개", "RW [r, 랜덤워크]"),
                                             ("백색잡음 5개", "N [r, 백색잡음]")])]
    BH, STEP = 90, 117                 # 상자 높이, 같은 묶음 안의 줄 간격
    MX0, MX1, HX0, HX1 = 468, 902, 1000, 1452   # 측정치·모형 상자의 좌우
    mids, label_y = [], 165
    for label, rows in groups:
        ax.text(MX0, label_y, label, ha="left", va="center", fontsize=8, color="0.35")
        y = label_y + 45
        for ms, md in rows:
            box(MX0, y, MX1, y + BH, ms)
            box(HX0, y, HX1, y + BH, md, fs=8.5)
            m = y + BH / 2
            arrow(MX1 + 6, m, HX0 - 10, m)
            line([HX1, 1522], [m, m])
            mids.append(m)
            y += STEP
        label_y = y - STEP + BH + 37

    # 자료: 체결 기록 → 봉 → (CVD … 봉 순번), 난수 → (랜덤워크, 백색잡음)
    box(35, 355, 330, 550, "집계 체결\n(aggTrades,\n28.1억 행)")
    box(35, 689, 330, 867, "봉\n(4시간봉·일봉)")
    box(35, 1388, 330, 1566, "표준정규 난수\n(자료와 무관)")
    arrow(182, 557, 182, 680)
    for (src_y, i0, i1) in [(778, 0, 8), (1477, 9, 10)]:
        line([330, 410], [src_y, src_y])
        line([410, 410], [mids[i0], mids[i1]])
        for m in mids[i0:i1 + 1]:
            arrow(410, m, MX0 - 8, m)

    # 평가
    line([1522, 1522], [mids[0], mids[-1]])
    box(1590, 521, 1965, 668, "헝가리안\n대응")
    box(1590, 815, 1965, 993, "전체 κ\n시기 조정 κ")
    box(1590, 1139, 1965, 1301, "M3·T 대비\nΔκ")
    arrow(1522, 594.5, 1582, 594.5)
    arrow(1778, 675, 1778, 806)
    arrow(1778, 1000, 1778, 1130)

    for x, s in zip([182, 685, 1226, 1778], ["자료", "측정치", "은닉마르코프모형 (K: M1 기준)", "평가"]):
        ax.text(x, 48, s, ha="center", va="center", fontsize=9)
    fig.savefig(OUT / "fig_design_v3.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- 논문_수정안: 국면 상태열의 시간 배치
def fig_timeline(lang="en"):
    """기준 모형, BVC 모형, 수익률 통제군, 시간 대조군의 상태열을 날짜 축에 나란히 (BTC 4시간봉, K=3).
    lang="ko" 는 투고본용 국문판(fig_timeline.png).
    상태 이름은 기준 모형 상태의 특성으로 붙인다: 수익률 절댓값 중앙값이 가장 큰 상태 = 고변동, 가장 작은 상태 = 저변동,
    나머지 = 순매도."""
    import matplotlib.dates as mdates
    import pandas as pd
    import pyarrow.parquet as pq
    z = np.load(ROOT / "results" / "models" / "E070_states_4h_K3.npz")
    v = int(z["timestamp"][0])                       # pandas 3 는 parquet 시각을 ms 단위로 읽는다
    unit = "ms" if v < 1e14 else ("us" if v < 1e17 else "ns")
    ts = pd.to_datetime(z["timestamp"], unit=unit, utc=True).tz_convert(None)
    d = pq.read_table(ROOT / "data" / "features" / "features_4h.parquet").to_pandas()
    absr = np.abs(d.r.to_numpy())
    med = [np.median(absr[z["M1"] == k]) for k in range(3)]
    hi, lo = int(np.argmax(med)), int(np.argmin(med))
    ns = 3 - hi - lo
    ko = lang == "ko"
    label = ({hi: "고변동", lo: "저변동", ns: "순매도"} if ko
             else {hi: "High volatility", lo: "Low volatility", ns: "Net selling"})
    color = {hi: "tab:orange", lo: "tab:blue", ns: "tab:red"}
    rows = ([("기준 모형 (M1)", "M1"), ("BVC (M2)", "M2"), ("수익률 모형 (M3)", "M3"), ("순번 모형 (T)", "T")] if ko
            else [("Benchmark (M1)", "M1"), ("BVC (M2)", "M2"), ("Returns control (M3)", "M3"), ("Time index (T)", "T")])
    fig, ax = plt.subplots(figsize=(8, 3.0))
    x = mdates.date2num(ts.to_pydatetime())
    dx = np.median(np.diff(x))
    for i, (_, key) in enumerate(rows):
        s = z[key]
        y = len(rows) - 1 - i
        for k in range(3):
            m = s == k
            # 같은 상태가 이어지는 구간을 한 막대로 그린다
            starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
            ends = np.flatnonzero(m & ~np.r_[m[1:], False])
            ax.broken_barh([(x[a], x[b] - x[a] + dx) for a, b in zip(starts, ends)], (y - 0.38, 0.76),
                           facecolors=color[k], edgecolor="none")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    import datetime as _dt
    ax.set_xlim(mdates.date2num(_dt.datetime(2021, 1, 1)), mdates.date2num(_dt.datetime(2026, 1, 1)))
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[k]) for k in (hi, lo, ns)]
    ax.legend(handles, [label[k] for k in (hi, lo, ns)], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False)
    save(fig, "fig_timeline.png" if ko else "fig_timeline_en.png")


# ---------------------------------------------------------------- 논문_수정안: 전체 대 분기 조정 Δκ
def fig_adjusted(lang="en", noclv=False):
    """설정별로 통제군 대비 Δκ 를 전체(왼쪽)와 분기 조정(오른쪽)으로 나란히. 결과는 E070.
    lang="ko" 는 투고본용 국문판(fig_adjusted.png). noclv=True 는 BVC 와 시간 대조군만 그린 판(…_noclv.png)."""
    ko = lang == "ko"
    d = json.loads((TABLES / "E070_time_structure.json").read_text(encoding="utf-8"))
    bar = ("4시간봉", "일봉") if ko else ("4-hour", "daily")
    settings = [("BTC_4h_K3", f"BTC {bar[0]} K=3"), ("BTC_4h_K2", f"BTC {bar[0]} K=2"), ("BTC_1d_K2", f"BTC {bar[1]} K=2"),
                ("ETH_4h_K3", f"ETH {bar[0]} K=3"), ("ETH_1d_K4", f"ETH {bar[1]} K=4")]
    settings = [s for s in settings if s[0] in d]
    models = ([("M2", "BVC", "o"), ("M4", "CLV 흐름", "s"), ("M5", "CLV 잔차", "^"), ("T", "순번 모형", "x")] if ko
              else [("M2", "BVC", "o"), ("M4", "CLV flow", "s"), ("M5", "CLV residual", "^"), ("T", "Time index", "x")])
    if noclv:
        models = [m for m in models if m[0] in ("M2", "T")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6), sharey=True)
    ys = np.arange(len(settings))[::-1].astype(float)
    step = 0.22 if noclv else 0.17
    titles = ("전체", "분기 조정") if ko else ("Overall", "Quarter-adjusted")
    for ax, lab, title in [(axes[0], "kappa", titles[0]), (axes[1], "kappa_quarter", titles[1])]:
        for j, (m, name, mk) in enumerate(models):
            pts = np.array([[d[s]["deltas"][f"{m}-M3"][lab]["point"], *d[s]["deltas"][f"{m}-M3"][lab]["ci"]]
                            for s, _ in settings])
            yy = ys + ((len(models) - 1) / 2 - j) * step
            ax.errorbar(pts[:, 0], yy, xerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]],
                        fmt=mk, capsize=2.5, label=name, ms=5)
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("수익률 모형 대비 Δκ (95% 구간)" if ko else "Δκ vs. returns control (95% CI)")
    axes[0].set_yticks(ys); axes[0].set_yticklabels([s[1] for s in settings])
    handles, labels = axes[0].get_legend_handles_labels()
    # 범례는 그림 전체 위에 둔다(tight_layout 이 축 범례를 기준으로 간격을 벌리지 않게, 저장도 직접 한다)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    name = "fig_adjusted.png" if ko else "fig_adjusted_en.png"
    fig.savefig(OUT / (name.replace(".png", "_noclv.png") if noclv else name), dpi=300)
    plt.close(fig)


def main():
    setup()
    fig_design("ko"); fig_design("en"); fig_multiplier(); fig_forest("ko"); fig_forest("en"); fig_regime()
    fig_design_v2(); fig_timeline(); fig_adjusted()
    fig_design_v2("ko"); fig_timeline("ko"); fig_adjusted("ko")
    fig_design_v3()
    print("저장:", ", ".join(p.name for p in sorted(OUT.glob("fig_*.png"))))


if __name__ == "__main__":
    main()
