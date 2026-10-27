"""
논문 3.1.4절 그림 1 — BTCUSDT 4시간봉 종가와 로그수익률 (분석 표본).

위아래 두 칸으로 나눈다(한 칸에 축 두 개를 겹치지 않는다).
  (a) 종가 — 5년 동안 가격 수준이 크게 달라 로그 눈금
  (b) 로그수익률 (%)
계열이 하나라 범례는 두지 않는다. 색은 파랑 한 가지.

    python src/fig_returns.py

출력: docs/figures/fig1_btc_4h.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "features_4h.parquet"
OUT = ROOT / "docs" / "figures" / "fig1_btc_4h.png"

# 논문 PDF 와 같은 잉크·선 색 (export_memo_pdf.py 의 CSS 변수와 맞춤)
INK, INK2, GRID = "#171b21", "#4c5563", "#e3e7ee"
SERIES = "#2a78d6"


def contrast(hex_fg: str, hex_bg: str = "#ffffff") -> float:
    """WCAG 대비비."""
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted([lum(hex_fg), lum(hex_bg)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def main():
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "font.size": 8.5,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.unicode_minus": False,
    })

    f = pd.read_parquet(FEAT)
    t = f.timestamp.dt.tz_convert(None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.7, 4.3), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28})
    for ax in (ax1, ax2):
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    ax1.plot(t, f.close, color=SERIES, linewidth=0.9, solid_joinstyle="round")
    ax1.set_yscale("log")
    ax1.yaxis.set_minor_locator(NullLocator())      # 로그 눈금의 보조 눈금은 지운다
    ax1.set_yticks([20000, 40000, 60000, 80000, 100000])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 10000:,.0f}만"))
    ax1.set_title("(a) 종가 (USDT, 로그 눈금)", loc="left", fontsize=9, color=INK, pad=6)

    ax2.plot(t, f.r * 100, color=SERIES, linewidth=0.45)
    ax2.axhline(0, color=INK2, linewidth=0.6)
    ax2.set_title("(b) 4시간봉 로그수익률 (%)", loc="left", fontsize=9, color=INK, pad=6)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"대비비(흰 바탕) {contrast(SERIES):.2f}:1   저장: {OUT}")


if __name__ == "__main__":
    main()
