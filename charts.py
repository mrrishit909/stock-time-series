"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_tails.png, 02_acf.png, 03_leverage.png, 04_volforecast.png, 05_randomwalk.png
The charts are our own aggregates (histogram bins, autocorrelations, scores); the price file itself is never published."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import SERIES, STOCKS, load

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
AC = pd.read_csv(R / "acf.csv"); LB = pd.read_csv(R / "ljungbox.csv").set_index("series"); TL = pd.read_csv(R / "tails.csv"); LV = pd.read_csv(R / "leverage.csv").set_index("series")
VS = pd.read_csv(R / "volscores.csv"); MV = pd.read_csv(R / "spy_monthly_vol.csv"); VR = pd.read_csv(R / "vr.csv"); TR = pd.read_csv(R / "trend.csv").set_index("series"); SU = pd.read_csv(R / "summary.csv").set_index("key")["value"]
px = load(); ret = 100 * np.log(px).diff().dropna(); N = len(ret)


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def acf_panel(ax, fs=9):
    g = AC.groupby("lag")[["acf_r", "acf_abs"]].mean(); b = 1.96 / np.sqrt(N); lag = g.index.values
    ax.axhspan(-b, b, color=GRAY, alpha=0.25, linewidth=0, label="chance band, one series (95%)")
    ax.bar(lag - 0.2, g["acf_r"], width=0.4, color=BLUE, label="return"); ax.bar(lag + 0.2, g["acf_abs"], width=0.4, color=ORANGE, label="absolute return")
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xlabel("lag (trading days)"); ax.set_ylabel("autocorrelation, mean of 11 series"); ax.set_ylim(-0.12, 0.3); ax.legend(frameon=False, labelcolor=INK, fontsize=fs, loc="upper right"); ax.set_title("Yesterday's move says little about today's direction, much about its size", fontsize=12, loc="left")


def tail_panel(ax):
    z = ((ret - ret.mean()) / ret.std()).to_numpy().ravel(); edges = np.linspace(-12, 12, 97); h, _ = np.histogram(z, edges, density=True); c = (edges[1:] + edges[:-1]) / 2
    ax.semilogy(c[h > 0], h[h > 0], "o", color=BLUE, markersize=4.5, label="11 series, pooled"); xs = np.linspace(-12, 12, 400); ax.semilogy(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, linewidth=2, label="normal curve")
    ax.set_ylim(1e-6, 1); ax.set_xlim(-12, 12); ax.set_xlabel("daily return, in standard deviations of its own series"); ax.set_ylabel("density (log scale)"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right"); ax.set_title("Far more big days than a normal curve allows", fontsize=12)


def hill_panel(ax):
    for tail, c, lab in (("loss", ORANGE, "losses"), ("gain", BLUE, "gains")):
        g = TL[TL["tail"] == tail].groupby("k_pct").alpha; ax.fill_between(g.min().index, g.min(), g.max(), color=c, alpha=0.18, linewidth=0); ax.plot(g.median().index, g.median(), color=c, linewidth=2.2, marker="o", markersize=4, label=f"{lab}: median and range of 11")
    ax.axvline(5, color=GRAY, linestyle=":", linewidth=1.2); ax.axhline(4, color=DIM, linestyle="--", linewidth=1); ax.text(10, 4.07, "4: below it the kurtosis is not finite", ha="right", va="bottom", fontsize=8.5, color=DIM)
    ax.set_ylim(0, 5); ax.set_xlim(0.5, 10.5); ax.set_xlabel("k, the share of days used as the tail (%); 5 is the headline"); ax.set_ylabel("Hill tail index"); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right"); md = TL.groupby(["tail", "k_pct"]).alpha.median(); ax.set_title(f"Median tail index {md.min():.1f} to {md.max():.1f}, for any k", fontsize=12)


def lb_panel(ax):
    x = np.arange(len(SERIES)); ax.bar(x - 0.2, LB.loc[SERIES, "q10_r"], width=0.4, color=BLUE, label="return"); ax.bar(x + 0.2, LB.loc[SERIES, "q10_abs"], width=0.4, color=ORANGE, label="absolute return")
    ax.set_yscale("log"); ax.axhline(18.307, color=GRAY, linestyle=":", linewidth=1.4, label="5% critical value, 18.3")
    ax.set_xticks(x); ax.set_xticklabels(SERIES, fontsize=7.5, rotation=60); ax.set_ylim(10, 6000); ax.set_ylabel("Ljung-Box Q(10), log scale"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left"); rt = LB.loc[SERIES, "q10_abs"] / LB.loc[SERIES, "q10_r"]; ax.set_title(f"The test statistic is {rt.min():.0f} to {rt.max():.0f} times larger for size", fontsize=12)


def lev_panel(ax):
    x = np.arange(len(SERIES)); v = (LV.loc[SERIES, "next_abs_after_down"] / LV.loc[SERIES, "next_abs_after_up"]).values
    ax.bar(x, v, width=0.6, color=[ORANGE if s == "SPY" else BLUE for s in SERIES]); ax.axhline(1, color=DIM, linewidth=1)
    for i, t in enumerate(v): ax.text(i, t + 0.015, f"{t:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(SERIES, fontsize=7.5, rotation=60); ax.set_ylim(0, 1.5); ax.set_ylabel("next-day size: after a down day / after an up day", fontsize=9.5); ax.grid(axis="x", visible=False); ax.set_title("A down day is followed by a bigger move", fontsize=12)


def vix_panel(ax):
    vx = pd.read_csv(HERE / "data" / "VIXCLS.csv", index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0].reindex(px.index); ok = vx.notna() & vx.shift(1).notna(); dv = np.log(vx).diff()[ok]; rs = ret["SPY"].reindex(dv.index)
    ax.scatter(rs, 100 * dv, s=6, color=BLUE, alpha=0.35, linewidths=0); xs = np.array([-11, 10]); b = np.polyfit(rs, 100 * dv, 1); ax.plot(xs, np.polyval(b, xs), color=ORANGE, linewidth=2)
    ax.axhline(0, color=DIM, linewidth=0.8); ax.axvline(0, color=DIM, linewidth=0.8); ax.set_xlabel("SPY's return on the day (%)"); ax.set_ylabel("change of the VIX on the day (log points x 100)")
    ax.set_title(f"Falling prices, rising fear (correlation {float(SU['corr_spy_return_dlogvix']):.2f})", fontsize=12)


def volscore_panel(ax):
    x = np.arange(len(SERIES)); w = 0.27
    for k, (m, c) in enumerate((("EWMA", GRAY), ("GARCH", BLUE), ("GJR-GARCH", ORANGE))):
        d = VS[VS.model == m].set_index("series").loc[SERIES]; ax.bar(x + (k - 1) * w, d["gain_vs_constant"], width=w * 0.92, color=c, label=m)
    ax.set_xticks(x); ax.set_xticklabels(SERIES, fontsize=7.5, rotation=60); ax.set_ylabel("QLIKE better than a constant by"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left"); ax.set_ylim(0, 0.62); ax.set_title("All three beat the constant, on every series", fontsize=12)


def spy_vol_panel(ax):
    t = pd.PeriodIndex(MV["month"], freq="M").to_timestamp(); ax.plot(t, MV["realised_vol"], color=INK, linewidth=1.6, label="what happened (realised)"); ax.plot(t, MV["constant"], color=GRAY, linewidth=2, linestyle="--", label="constant")
    ax.plot(t, MV["GARCH"], color=BLUE, linewidth=2.2, label="GARCH(1,1)"); ax.set_ylim(0, 90); ax.set_ylabel("SPY volatility, % a year, by month"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right"); ax.set_title("SPY's monthly volatility and its day-ahead forecasts", fontsize=12)


def vr_panel(ax):
    for s in SERIES:
        d = VR[VR.series == s]; ax.plot(d["q"], d["vr"], color=GRAY, linewidth=1, alpha=0.7)
    g = VR.groupby("q").vr.median(); ax.plot(g.index, g.values, color=BLUE, linewidth=2.6, marker="o", label="median of 11 series")
    ax.axhline(1, color=ORANGE, linewidth=1.6, linestyle="--", label="random walk: 1"); ax.set_xscale("log"); ax.set_xticks(list(g.index)); ax.set_xticklabels([str(q) for q in g.index]); ax.minorticks_off(); ax.set_ylim(0, 1.2)
    ax.set_xlabel("q, days in the return (lines: one per series)"); ax.set_ylabel("variance ratio VR(q)"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left"); ax.set_title("Multi-day variance is a little below the random walk's", fontsize=12)


def trend_panel(ax):
    y = np.arange(len(SERIES))[::-1]
    for N_, c, m in ((21, ORANGE, "o"), (63, BLUE, "s"), (252, GREEN, "D")): ax.scatter(TR.loc[SERIES, f"rmse_trend{N_}"] / TR.loc[SERIES, "rmse_random_walk"], y, color=c, s=34, marker=m, label=f"trend of last {N_} days", zorder=3)
    ax.axvline(1, color=DIM, linewidth=1.2, linestyle="--"); ax.set_yticks(y); ax.set_yticklabels(SERIES, fontsize=8.5); ax.set_xlim(0.9, 1.9); ax.set_xlabel("forecast error of the trend / error of 'no change' (above 1 = worse)"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right"); ax.set_title("Extending a trend makes the 21-day forecast worse", fontsize=12)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    g = AC.groupby("lag")[["acf_r", "acf_abs"]].mean(); sd1 = float(np.mean([LB.loc[s, "q10_abs"] / LB.loc[s, "q10_r"] for s in SERIES]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8)); acf_panel(axes[0], 10.5); axes[0].set_title("Direction: almost no memory. Size: long memory", fontsize=14); tail_panel(axes[1]); axes[1].set_title("Fat tails: big days are far more common than a normal curve says", fontsize=14)
    fig.suptitle(f"Stocks move unpredictably, but not randomly in size: yesterday's swing forecasts today's (autocorrelation {g.loc[1, 'acf_abs']:.2f} against {g.loc[1, 'acf_r']:.2f})", x=0.01, ha="left", fontsize=17, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); tail_panel(axes[0]); hill_panel(axes[1]); save(fig, "01_tails.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1]}); acf_panel(axes[0], 8.5); axes[0].set_title("Autocorrelation by lag", fontsize=12); lb_panel(axes[1]); save(fig, "02_acf.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1)); lev_panel(axes[0]); vix_panel(axes[1]); save(fig, "03_leverage.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.1, 1]}); volscore_panel(axes[0]); spy_vol_panel(axes[1]); save(fig, "04_volforecast.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); vr_panel(axes[0]); trend_panel(axes[1]); save(fig, "05_randomwalk.png")


if __name__ == "__main__":
    main()
