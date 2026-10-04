"""Step 5: recompute every number in results/ from data/ in plain Python (csv, math, loops; no pandas, numpy, scipy).   ./venv/bin/python check.py

Everything is recomputed with different code than analyze.py, to a relative tolerance of 1e-8 (counts exactly), EXCEPT the quasi-maximum-likelihood fits of GARCH and GJR-GARCH, which are not refitted here (that is an optimiser's job).
Those are verified by properties instead: for each of the 682 stored fits, check.py recomputes the log-likelihood at the stored parameters from the window's returns and compares it to the stored one, checks the parameters are
feasible (omega >= 1e-6, alpha, beta, gamma >= 0, persistence <= 0.999), and checks that no feasible parameter vector formed by moving one parameter by +-3% (or adding +0.001 to a parameter at zero) has a higher log-likelihood
(by more than 0.5 log-likelihood units on a window of 750+ returns). Given the stored parameters, every forecast, score and test below is recomputed exactly.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "results"
STOCKS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "XOM", "JNJ", "WMT", "KO"]
SERIES = STOCKS + ["SPY"]
BURN, REFIT, EW, QS, NW = 750, 63, 0.94, (2, 5, 10, 20), 5
n_ok = [0]


def eq(name, got, want, rel=1e-8, absol=1e-10):
    want = float(want)
    if not (abs(got - want) <= rel * abs(want) + absol): raise SystemExit(f"MISMATCH {name}: recomputed {got!r} vs stored {want!r}")
    n_ok[0] += 1


def exact(name, got, want):
    if str(got) != str(want): raise SystemExit(f"MISMATCH {name}: recomputed {got!r} vs stored {want!r}")
    n_ok[0] += 1


def table(name): return list(csv.DictReader(open(R / name)))


def mean(x): return sum(x) / len(x)


def nsf2(z): return math.erfc(abs(z) / math.sqrt(2))                    # two-sided normal p-value


def chi2_even_sf(x, df): return math.exp(-x / 2) * sum((x / 2) ** i / math.factorial(i) for i in range(df // 2))


def acf(x, nl):
    m = mean(x); d = [v - m for v in x]; den = sum(v * v for v in d)
    return [sum(d[t] * d[t - l] for t in range(l, len(d))) / den for l in range(1, nl + 1)]


def lbq(x, m=10):
    n = len(x); a = acf(x, m); q = n * (n + 2) * sum(a[l - 1] ** 2 / (n - l) for l in range(1, m + 1)); return q, chi2_even_sf(q, m)


def hill(x, k):
    s = sorted(x, reverse=True); return k / sum(math.log(s[i] / s[k]) for i in range(k)), s[k]


def recur(r, om, al, ga, be, h0):
    h = [h0]
    for t in range(1, len(r)): h.append(om + (al + (ga if r[t - 1] < 0 else 0.0)) * r[t - 1] ** 2 + be * h[-1])
    return h


def ll(r, p, h0):
    h = recur(r, *p, h0); return -0.5 * sum(math.log(h[t]) + r[t] * r[t] / h[t] for t in range(len(r)))


def nw(d, L=NW):
    n = len(d); m = mean(d); e = [v - m for v in d]; v = sum(x * x for x in e) / n
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * sum(e[t] * e[t - l] for t in range(l, n)) / n
    z = m / math.sqrt(v / n); return z, nsf2(z)


def load():
    px = defaultdict(dict); hl = []
    for row in csv.DictReader(open(HERE / "data" / "bars.csv")):
        if row["symbol"] in SERIES: px[row["symbol"]][row["date"]] = float(row["close"]); hl.append((float(row["close"]), float(row["high"]), float(row["low"])))
    dates = sorted(px["SPY"]); assert all(sorted(px[s]) == dates for s in SERIES), "series have different dates"
    return px, dates, hl


def fred(name):
    out = {}
    for row in csv.DictReader(open(HERE / "data" / f"{name}.csv")):
        v = list(row.values())
        if v[1] not in (".", ""): out[v[0]] = float(v[1])
    return out


def main():
    px, dates, hl = load(); N = len(dates) - 1; rd = dates[1:]
    close = {s: [px[s][d] for d in dates] for s in SERIES}; lp = {s: [math.log(c) for c in close[s]] for s in SERIES}
    ret = {s: [100 * (lp[s][t] - lp[s][t - 1]) for t in range(1, N + 1)] for s in SERIES}
    from datetime import date as D
    # audit
    au = {r["check"]: r["value"] for r in table("audit.csv")}
    exact("closes", len(dates), au["closes per series"]); exact("returns", N, au["returns per series"]); exact("first", dates[0], au["first close"]); exact("last", dates[-1], au["last close"])
    exact("missing closes", 0, au["missing closes (any series)"])
    exact("max gap", max((D.fromisoformat(dates[t]) - D.fromisoformat(dates[t - 1])).days for t in range(1, len(dates))), au["longest gap between closes (days)"])
    exact("outside range", sum(1 for c, h, l in hl if c > h + 1e-9 or c < l - 1e-9), au["closes outside the day's high-low range"])
    exact("|r|>20", sum(1 for s in SERIES for v in ret[s] if abs(v) > 20), au["returns with |r| > 20%"])
    eq("max |r|", max(abs(v) for s in SERIES for v in ret[s]), au["largest |return| %"], absol=6e-4)
    exact("zero returns", sum(1 for s in SERIES for v in ret[s] if v == 0), au["returns of exactly 0"])
    sp = fred("SP500"); both = [d for d in dates if d in sp]; a = [math.log(px["SPY"][d]) for d in both]; b = [math.log(sp[d]) for d in both]
    x = [a[t] - a[t - 1] for t in range(1, len(a))]; y = [b[t] - b[t - 1] for t in range(1, len(b))]; mx, my = mean(x), mean(y)
    exact("spy days", len(x), au["SPY vs FRED SP500: days compared"])
    eq("spy corr", sum((u - mx) * (v - my) for u, v in zip(x, y)) / math.sqrt(sum((u - mx) ** 2 for u in x) * sum((v - my) ** 2 for v in y)), au["SPY vs FRED SP500: correlation of daily log returns"], absol=6e-6)
    # biggest days
    for row in table("top_moves.csv"):
        s = row["series"]; order = sorted(range(N), key=lambda k: (-abs(ret[s][k]), k)); k = order[int(row["rank"]) - 1]
        exact(f"top date {s}{row['rank']}", rd[k], row["date"]); eq(f"top ret {s}{row['rank']}", ret[s][k], row["return_pct"], absol=6e-5)
    # moments
    for row in table("moments.csv"):
        s = row["series"]; r = ret[s]; m = mean(r); sd = math.sqrt(sum((v - m) ** 2 for v in r) / N); z = [(v - m) / sd for v in r]
        exact("n", N, row["n"]); eq(f"mean {s}", m, row["mean_pct"]); eq(f"sd {s}", sd, row["sd_pct"]); eq(f"annvol {s}", sd * math.sqrt(252), row["ann_vol_pct"]); eq(f"skew {s}", mean([v ** 3 for v in z]), row["skew"])
        eq(f"kurt {s}", mean([v ** 4 for v in z]) - 3, row["excess_kurtosis"]); eq(f"min {s}", min(r), row["min_pct"]); eq(f"max {s}", max(r), row["max_pct"]); eq(f"3sd {s}", 100 * sum(1 for v in z if abs(v) > 3) / N, row["share_beyond_3sd_pct"])
    # tails
    for row in table("tails.csv"):
        s = row["series"]; k = int(round(int(row["k_pct"]) / 100 * N)); x = [-v for v in ret[s] if v < 0] if row["tail"] == "loss" else [v for v in ret[s] if v > 0]
        al, th = hill(x, k); exact("k", k, row["k"]); eq(f"hill {s}{row['tail']}{row['k_pct']}", al, row["alpha"]); eq(f"thr {s}", th, row["threshold_pct"])
    # acf, Ljung-Box
    A = {s: (acf(ret[s], 40), acf([abs(v) for v in ret[s]], 40)) for s in SERIES}
    for row in table("acf.csv"):
        s, l = row["series"], int(row["lag"]) - 1; eq(f"acf {s}{l}", A[s][0][l], row["acf_r"], absol=1e-12); eq(f"acfabs {s}{l}", A[s][1][l], row["acf_abs"])
    for row in table("ljungbox.csv"):
        s = row["series"]; q1, p1 = lbq(ret[s]); q2, p2 = lbq([abs(v) for v in ret[s]])
        eq(f"q r {s}", q1, row["q10_r"]); eq(f"p r {s}", p1, row["p_r"], absol=1e-12); eq(f"q abs {s}", q2, row["q10_abs"]); eq(f"p abs {s}", p2, row["p_abs"], absol=1e-12)
    # GARCH parameters: properties, then forecasts
    P = defaultdict(list)
    for row in table("garch_params.csv"): P[(row["series"], row["model"])].append(row)
    worst = 0.0; nprop = 0
    for (s, m), rows in P.items():
        r = ret[s]
        for row in rows:
            T = int(row["first_day_index"]); om, al, ga, be = (float(row[k]) for k in ("omega", "alpha", "gamma", "beta")); gj = m == "GJR-GARCH"
            assert T == int(row["window_days"]) and om >= 1e-6 and al >= 0 and be >= 0 and ga >= 0 and (gj or ga == 0) and al + (ga / 2 if gj else 0) + be <= 0.999 + 1e-9, ("infeasible", s, m, T)
            w = r[:T]; h0 = mean([v * v for v in w]); base = ll(w, (om, al, ga, be), h0); eq(f"loglik {s}{m}{T}", base, row["loglik"], rel=1e-9)
            names = (0, 1, 2, 3) if gj else (0, 1, 3)
            for i in names:
                for mult in (0.97, 1.03, None):
                    q = [om, al, ga, be]; q[i] = q[i] * mult if mult else q[i] + 1e-3
                    if mult is None and q[i] != q[i] or (q[i] < 0) or (q[0] < 1e-6) or q[1] + (q[2] / 2 if gj else 0) + q[3] > 0.999: continue
                    if mult and q[i] == 0.0: continue
                    worst = max(worst, ll(w, tuple(q), h0) - base); nprop += 1
    assert worst <= 0.5, ("a perturbation beats a stored optimum by", worst)
    print(f"  {len(P) and sum(len(v) for v in P.values())} stored fits verified (log-likelihood recomputed, feasible, {nprop} perturbations; best perturbation gains {worst:.2g})")
    # forecasts and scores
    H = {}
    for s in SERIES:
        r = ret[s]; r2 = [v * v for v in r]; h = {m: [None] * N for m in ("constant", "EWMA", "GARCH", "GJR-GARCH")}; run = 0.0
        for t in range(N):
            if t >= BURN: h["constant"][t] = run / t
            run += r2[t]
        e = mean(r2[:60]); h["EWMA"][60] = e
        for t in range(61, N): e = EW * e + (1 - EW) * r2[t - 1]; h["EWMA"][t] = e
        for m in ("GARCH", "GJR-GARCH"):
            for row in P[(s, m)]:
                T = int(row["first_day_index"]); end = min(T + REFIT, N); h0 = mean(r2[:T]); hh = recur(r[:end], float(row["omega"]), float(row["alpha"]), float(row["gamma"]), float(row["beta"]), h0)
                for t in range(T, end): h["GJR-GARCH" if m == "GJR-GARCH" else "GARCH"][t] = hh[t]
        assert all(v is not None for m in h for v in h[m][BURN:]); H[s] = h
    for row in table("volscores.csv"):
        s, m = row["series"], row["model"]; r = ret[s]
        lo = {k: [r[t] ** 2 / H[s][k][t] + math.log(H[s][k][t]) for t in range(BURN, N)] for k in ("constant", m)}
        d = [a - b for a, b in zip(lo["constant"], lo[m])]; exact("n oos", N - BURN, row["n"]); eq(f"qlike {s}{m}", mean(lo[m]), row["qlike"]); eq(f"gain {s}{m}", mean(d), row["gain_vs_constant"], absol=1e-12)
        if m != "constant": z, p = nw(d); eq(f"dm z {s}{m}", z, row["dm_z"]); eq(f"dm p {s}{m}", p, row["dm_p"], absol=1e-12)
    # monthly SPY volatility
    mon = defaultdict(list)
    for t in range(BURN, N): mon[rd[t][:7]].append(t)
    for row in table("spy_monthly_vol.csv"):
        ts = mon[row["month"]]; eq(f"realised {row['month']}", math.sqrt(mean([ret["SPY"][t] ** 2 for t in ts]) * 252), row["realised_vol"])
        for m in ("constant", "EWMA", "GARCH", "GJR-GARCH"): eq(f"mvol {row['month']}{m}", math.sqrt(mean([H["SPY"][m][t] for t in ts]) * 252), row[m])
    exact("months", len(mon), len(table("spy_monthly_vol.csv")))
    # leverage
    last = {s: P[(s, "GJR-GARCH")][-1] for s in SERIES}
    for row in table("leverage.csv"):
        s = row["series"]; r = ret[s]; dn = [r[t] < 0 for t in range(N - 1)]; nxt = [abs(r[t + 1]) for t in range(N - 1)]
        eq(f"after down {s}", mean([nxt[t] for t in range(N - 1) if dn[t]]), row["next_abs_after_down"]); eq(f"after up {s}", mean([nxt[t] for t in range(N - 1) if not dn[t]]), row["next_abs_after_up"])
        x = r[:-1]; mx, my = mean(x), mean(nxt); eq(f"corr lev {s}", sum((a - mx) * (b - my) for a, b in zip(x, nxt)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in nxt)), row["corr_r_next_abs"])
        exact("down days", sum(dn), row["n_down_days"]); eq(f"gamma {s}", float(last[s]["gamma"]), row["gjr_gamma_last"]); eq(f"alpha {s}", float(last[s]["alpha"]), row["gjr_alpha_last"])
    # variance ratios
    for row in table("vr.csv"):
        s, q = row["series"], int(row["q"]); r = ret[s]; mu = mean(r); d = [v - mu for v in r]; s_a = sum(v * v for v in d) / (N - 1)
        cs = [0.0]
        for v in r: cs.append(cs[-1] + v)
        m = q * (N - q + 1) * (1 - q / N); s_c = sum((cs[t + q] - cs[t] - q * mu) ** 2 for t in range(N - q + 1)) / m
        d2 = [v * v for v in d]; den = sum(d2) ** 2; th = sum((2 * (q - j) / q) ** 2 * sum(d2[t] * d2[t - j] for t in range(j, N)) / den for j in range(1, q))
        z = (s_c / s_a - 1) / math.sqrt(th); eq(f"vr {s}{q}", s_c / s_a, row["vr"]); eq(f"z {s}{q}", z, row["z_star"]); eq(f"p {s}{q}", nsf2(z), row["p"], absol=1e-12)
    # trend forecasts
    for row in table("trend.csv"):
        s = row["series"]; p = lp[s]; cl = close[s]; idx = list(range(252, len(p) - 21, 21)); fut = [p[i + 21] - p[i] for i in idx]; exact("blocks", len(idx), row["blocks"])
        eq(f"rmse rw {s}", math.sqrt(mean([f * f for f in fut])), row["rmse_random_walk"])
        for Nn in (21, 63, 252):
            e = [f - 21 * (p[i] - p[i - Nn]) / Nn for f, i in zip(fut, idx)]; d = [a * a - f * f for a, f in zip(e, fut)]; md = mean(d); sdv = math.sqrt(sum((v - md) ** 2 for v in d) / (len(d) - 1))
            eq(f"rmse {s}{Nn}", math.sqrt(mean([a * a for a in e])), row[f"rmse_trend{Nn}"]); eq(f"t {s}{Nn}", md / (sdv / math.sqrt(len(d))), row[f"t_trend{Nn}"], absol=1e-9)
        above = [cl[i] > mean(cl[i - 199:i + 1]) for i in idx]; up = [f > 0 for f in fut]
        eq(f"above {s}", mean(above), row["share_above_200d"]); eq(f"hit ma {s}", mean([a == u for a, u in zip(above, up)]), row["hit_rate_ma_rule"]); eq(f"hit up {s}", mean(up), row["hit_rate_always_up"])
        eq(f"above ret {s}", 100 * mean([f for f, a in zip(fut, above) if a]), row["mean_next21_above_pct"]); eq(f"below ret {s}", 100 * mean([f for f, a in zip(fut, above) if not a]), row["mean_next21_below_pct"]); exact("n above", sum(above), row["n_above"])
    # SPY and the VIX
    S = {r["key"]: r["value"] for r in table("summary.csv")}; vx = fred("VIXCLS"); rs, dv = [], []
    for t in range(1, len(dates)):
        if dates[t] in vx and dates[t - 1] in vx: rs.append(ret["SPY"][t - 1]); dv.append(math.log(vx[dates[t]]) - math.log(vx[dates[t - 1]]))
    mx, my = mean(rs), mean(dv); sxy = sum((a - mx) * (b - my) for a, b in zip(rs, dv)); sxx = sum((a - mx) ** 2 for a in rs); syy = sum((b - my) ** 2 for b in dv)
    exact("vix days", len(rs), S["vix_days"]); exact("vix missing", len(dates) - len(rs), S["vix_missing_trading_days"]); eq("vix corr", sxy / math.sqrt(sxx * syy), S["corr_spy_return_dlogvix"]); eq("vix slope", sxy / sxx, S["slope_dlogvix_per_return_pct"])
    exact("oos", N - BURN, S["oos_days"]); exact("oos first", rd[BURN], S["oos_first_day"])
    print(f"check.py: {n_ok[0]} recomputed numbers agree with results/")


if __name__ == "__main__":
    main()
