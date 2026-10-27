"""E070/E071 결과에서 논문_수정안의 결과 표(마크다운)를 만든다. 손으로 옮겨 적다 생기는 오류를 막기 위함.

    python src/revision_tables.py      → results/tables/rev_tables.json (표 이름별 마크다운)
"""
import json
import re
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "results" / "tables"
OUT = Path(__file__).resolve().parent.parent / "results" / "tables"
d = json.loads((T / "E070_time_structure.json").read_text(encoding="utf-8"))
m = d["BTC_4h_K3"]


def ci(v):
    return f"{v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"


def neg(x):
    return f"{x:.3f}".replace("-", "−")


def fmt(s):
    """숫자 앞의 빼기만 수학 기호 − 로 바꾼다. 단어의 하이픈(4-hour)과 수식($…$) 안은 그대로 둔다."""
    parts = re.split(r"(\$[^$]*\$)", s)
    return "".join(x if x.startswith("$") else re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "−", x) for x in parts)


out = {}

# Table 9: 통제군 대비 증분 (주 설정)
rows = [("M2 BVC − M3", "M2-M3"), ("M4 CLV flow − M3", "M4-M3"), ("M5 CLV residual − M3", "M5-M3"),
        ("T Time index − M3", "T-M3"), ("M2 BVC − T", "M2-T")]
t9 = ["&lt;Table 9&gt; Increments in Agreement: Overall and Period-Adjusted (BTCUSDT 4-Hour, $K=3$)", "",
      "| Comparison | Overall $\\Delta\\kappa$ | Year-adjusted | Quarter-adjusted | Month-adjusted |", "|---|---|---|---|---|"]
for lab, key in rows:
    v = m["deltas"][key]
    t9.append(fmt(f"| {lab} | {ci(v['kappa'])} | {ci(v['kappa_year'])} | {ci(v['kappa_quarter'])} | {ci(v['kappa_month'])} |"))
t9 += ["", "Note: Each cell gives the point estimate and the 95% confidence interval from the circular block bootstrap "
       "(blocks of 200 bars, 2,000 replications, the same indices for all models). Period-adjusted increments use Eq. (10)."]
out["TABLE9"] = "\n".join(t9)

# Table 10: BVC 분해 (주 설정)
lad = [("M3v", "$V_{\\tau}$"), ("M2s", "$\\mathrm{sign}(r_{\\tau})\\,V_{\\tau}$"), ("M3z", "$V_{\\tau}\\,r_{\\tau}/\\hat{\\sigma}_{\\tau}$"),
       ("M2m", "$2F_{\\mathrm{df}}(r_{\\tau}/\\hat{\\sigma}_{\\tau})-1$"), ("M3s", "$\\hat{\\sigma}_{\\tau}$"), ("M2", "BVC")]
name = {"M3s": "M3σ"}
t10 = ["&lt;Table 10&gt; Decomposition of the BVC Model (BTCUSDT 4-Hour, $K=3$)", "",
       "| Model | Second observation | $\\kappa$ | $\\Delta\\kappa$ vs. M3 | Quarter-adjusted $\\kappa_s$ | Quarter-adjusted $\\Delta\\kappa_s$ vs. M3 |",
       "|---|---|---|---|---|---|"]
for nm, obs in lad:
    p = m["point"][nm]
    v = m["deltas"][f"{nm}-M3"]
    t10.append(fmt(f"| {name.get(nm, nm)} | {obs} | {p['kappa']:.3f} | {ci(v['kappa'])} | {p['kappa_quarter']:.3f} | {ci(v['kappa_quarter'])} |"))
t10 += ["", fmt(f"Note: $\\hat{{\\sigma}}_{{\\tau}}$ is the standard deviation of the previous 50 returns. M3 has $\\kappa$ = "
                f"{m['point']['M3']['kappa']:.3f} and quarter-adjusted $\\kappa_s$ = {m['point']['M3']['kappa_quarter']:.3f}. "
                f"Confidence intervals as in &lt;Table 9&gt;.")]
out["TABLE10"] = "\n".join(t10)

# Table 11: 다른 설정
sets = [("BTC_4h_K3", "BTC 4-hour, $K=3$ (main)"), ("BTC_4h_K2", "BTC 4-hour, $K=2$"), ("BTC_1d_K2", "BTC daily, $K=2$"),
        ("ETH_4h_K3", "ETH 4-hour, $K=3$"), ("ETH_1d_K4", "ETH daily, $K=4$")]
t11 = ["&lt;Table 11&gt; Results across Bar Lengths, Numbers of Regimes and Assets", "",
       "| Setting | Benchmark runs | Control $\\kappa$ | Time index $\\kappa$ | Random walks: mean $\\kappa$ (share above BVC) | BVC $\\Delta\\kappa$: overall | BVC $\\Delta\\kappa$: quarter-adjusted |",
       "|---|---|---|---|---|---|---|"]
for key, lab in sets:
    r = d[key]; P = r["point"]; rw = r["placebo"]["random_walk"]
    t11.append(fmt(f"| {lab} | {P['M1']['runs']:,} | {P['M3']['kappa']:.3f} | {P['T']['kappa']:.3f} | "
                   f"{rw['kappa']['mean']:.3f} ({rw['share_above_bvc']:.2f}) | {ci(r['deltas']['M2-M3']['kappa'])} | "
                   f"{ci(r['deltas']['M2-M3']['kappa_quarter'])} |"))
t11 += ["", "Note: Each setting fixes the number of regimes chosen by the benchmark model. Random walks are 20 draws in the main setting "
        "and 10 draws elsewhere. Blocks are 200 bars (4-hour) and 60 bars (daily)."]
out["TABLE11"] = "\n".join(t11)

# 부록 E1: 연도별 카파
yrs = sorted(m["by_year_kappa"]["M2"].keys())
te = ["&lt;Table E1&gt; Kappa within Each Year (BTCUSDT 4-Hour, $K=3$)", "",
      "| Model | " + " | ".join(yrs) + " |", "|---|" + "---|" * len(yrs)]
for nm, lab in [("M3", "M3 Returns control"), ("M2", "M2 BVC"), ("M4", "M4 CLV flow"), ("M5", "M5 CLV residual"), ("T", "T Time index")]:
    te.append(fmt("| " + lab + " | " + " | ".join(f"{m['by_year_kappa'][nm][y]:.3f}" for y in yrs) + " |"))
te += ["", "Note: Kappa computed within each calendar year with the state matching fixed from the full sample. Within-year kappa tends "
       "to be low because the regime shares within a year are uneven."]
out["TABLEE1"] = "\n".join(te)

# 부록 B1: 강건성(E071) — 있을 때만
p71 = T / "E071_robustness_period.json"
if p71.exists():
    r71 = json.loads(p71.read_text(encoding="utf-8"))
    lab = {"기준": "Baseline", "ν = 0.1": "Degrees of freedom df = 0.1", "ν = 1": "Degrees of freedom df = 1",
           "ν = 30 (큰 값)": "Degrees of freedom df = 30", "σ 윈도우 = 20": "Volatility window 20 bars",
           "σ 윈도우 = 100": "Volatility window 100 bars", "W = 50": "Regression window $W=50$", "W = 200": "Regression window $W=200$",
           "CDF 압축 (Φ)": "Residual passed through the standard normal CDF", "잔차 누적": "Cumulative residual",
           "공분산 대각": "Diagonal covariance"}
    tb = ["&lt;Table 12&gt; Variations: Overall and Quarter-Adjusted Increments over the Returns Control (BTCUSDT 4-Hour, $K=3$)", "",
          "| Variation | Bars | BVC: overall | BVC: quarter-adjusted | CLV flow: quarter-adjusted | CLV residual: quarter-adjusted | Time index: quarter-adjusted |",
          "|---|---|---|---|---|---|---|"]
    for k, v in r71.items():
        dd = v["deltas"]
        tb.append(fmt(f"| {lab.get(k, k)} | {v['n_bars']:,} | {ci(dd['M2']['kappa'])} | {ci(dd['M2']['kappa_quarter'])} | "
                      f"{ci(dd['M4']['kappa_quarter'])} | {ci(dd['M5']['kappa_quarter'])} | {ci(dd['T']['kappa_quarter'])} |"))
    tb += ["", "Note: Each variation changes one setting from the baseline and re-estimates all models, including the time index model. "
           "Confidence intervals as in &lt;Table 9&gt;."]
    out["TABLE12"] = "\n".join(tb)
    # 전체 카파 증분(부록용)
    tb2 = ["&lt;Table B1&gt; Variations: Overall Increments in Kappa (BTCUSDT 4-Hour, $K=3$)", "",
           "| Variation | Bars | BVC | CLV flow | CLV residual | Time index |", "|---|---|---|---|---|---|"]
    for k, v in r71.items():
        dd = v["deltas"]
        tb2.append(fmt(f"| {lab.get(k, k)} | {v['n_bars']:,} | {ci(dd['M2']['kappa'])} | {ci(dd['M4']['kappa'])} | "
                       f"{ci(dd['M5']['kappa'])} | {ci(dd['T']['kappa'])} |"))
    out["TABLEB1"] = "\n".join(tb2)

(OUT / "rev_tables.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in out.items():
    print(f"===== {k} =====\n{v}\n")
