"""Steps 2-4: audit the prices, then measure the stylised facts of stock returns.   ./venv/bin/python analyze.py -> results/*.csv

Series: ten large US stocks (AAPL MSFT AMZN GOOGL NVDA JPM XOM JNJ WMT KO) and SPY (the S&P 500 fund, used as "the index"), daily adjusted closes 2016-01-04 to 2026-10-02 (Alpaca, feed=sip),
2,703 closes, 2,702 returns. A return is 100 x the change of the log of the adjusted close, in percent. FRED's SP500 index (from 2016-10-03) is a cross-check of SPY, and FRED's VIX is used for the leverage effect.
Methods (all means of daily returns are taken as zero in the volatility models, as the mean is about 0.05% against a daily spread of 1-2%):
 * moments: skew m3/m2^1.5 and excess kurtosis m4/m2^2 - 3 (population moments, so n is not corrected for);
 * tails: Hill estimator of the tail index alpha = k / sum_{i<=k} ln(x_(i) / x_(k+1)) on the k largest losses and, separately, gains (k = 5% of the days; 1% to 10% as sensitivity);
 * Ljung-Box Q(10) on returns and on absolute returns (p-value from the exact chi-square tail for even degrees of freedom);
 * leverage: next-day absolute return after a down day against after an up day, the correlation of a return with the next absolute return, the asymmetry term of GJR-GARCH, and the daily SPY return against the change of the log VIX;
 * volatility forecasts, one day ahead, walk-forward with only past returns from day 750 (about December 2018): "constant" (the mean squared return so far), EWMA (RiskMetrics, weight 0.94 on the past, seeded with the first 60 days),
   GARCH(1,1) and GJR-GARCH(1,1) with normal quasi-maximum likelihood, refitted every 63 days on all earlier returns (scipy SLSQP, best of two starts; the variance recursion starts from the mean squared return of the fitting window).
   Score: QLIKE = r^2/h + ln h (Patton 2011; lower is better), with the squared return as the noisy proxy of the day's variance; Diebold-Mariano test of the QLIKE difference against "constant", Newey-West with 5 lags;
 * random walk: Lo-MacKinlay (1988) variance ratio VR(q), q = 2, 5, 10, 20, with their heteroskedasticity-robust z* (the plain statistic over-rejects when volatility clusters);
 * trend forecasts of the log price 21 days ahead, every 21 days (non-overlapping) from day 252: "random walk" (today's price) against today's price plus 21 days of the average daily change of the last N = 21, 63 or 252 days;
   RMSE ratios, a paired t-test of the squared errors, and the hit rate of "above its 200-day average means up over the next 21 days" against "always up".
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

HERE = Path(__file__).parent
R = HERE / "results"
STOCKS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "XOM", "JNJ", "WMT", "KO"]
SERIES = STOCKS + ["SPY"]
BURN, REFIT, EW, QS, KPCT, NW = 750, 63, 0.94, (2, 5, 10, 20), 0.05, 5


def load():
    b = pd.read_csv(HERE / "data" / "bars.csv", parse_dates=["date"])
    return b.pivot(index="date", columns="symbol", values="close")[SERIES]


def fred(name):
    return pd.read_csv(HERE / "data" / f"{name}.csv", index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0]


def acf(x, nl):
    d = x - x.mean(); den = (d * d).sum()
    return np.array([(d[l:] * d[:-l]).sum() / den for l in range(1, nl + 1)])


def lb(x, m=10):
    n = len(x); q = n * (n + 2) * (acf(x, m) ** 2 / (n - np.arange(1, m + 1))).sum()
    return q, stats.chi2.sf(q, m)


def hill(x, k):
    """x: positive tail magnitudes; alpha from the k largest and the (k+1)-th as threshold."""
    s = np.sort(x)[::-1]; return k / np.log(s[:k] / s[k]).sum(), s[k]


# ---- volatility models ------------------------------------------------------------------------
def recursion(r, p, kind, h0):
    """h[t] = omega + (alpha + gamma*1[r[t-1]<0]) r[t-1]^2 + beta h[t-1], h[0] = h0 (a linear filter in h)."""
    om, al, ga, be = p; x = np.empty(len(r)); x[0] = h0
    x[1:] = om + (al + (ga * (r[:-1] < 0) if kind == "gjr" else 0)) * r[:-1] ** 2
    return signal.lfilter([1.0], [1.0, -be], x)


def negll(p, r, kind, h0):
    h = recursion(r, p if kind == "gjr" else (p[0], p[1], 0.0, p[2]), kind, h0)
    return 0.5 * (np.log(h) + r * r / h).sum()


def fit(r, kind):
    h0 = float((r * r).mean()); best = None
    for a0, b0 in ((0.08, 0.90), (0.03, 0.96)):
        if kind == "gjr":
            x0 = [h0 * (1 - a0 - b0), a0 / 2, a0, b0]; bnds = [(1e-6, None), (0, 1), (0, 1), (0, 1)]; con = {"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2] / 2 - p[3]}
        else:
            x0 = [h0 * (1 - a0 - b0), a0, b0]; bnds = [(1e-6, None), (0, 1), (0, 1)]; con = {"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}
        o = optimize.minimize(lambda p: negll(p, r, kind, h0) / len(r), x0, method="SLSQP", bounds=bnds, constraints=[con], options={"ftol": 1e-12, "maxiter": 500})
        if o.success or best is None:
            if best is None or o.fun < best.fun: best = o
    p = list(best.x); p = [p[0], p[1], p[2], p[3]] if kind == "gjr" else [p[0], p[1], 0.0, p[2]]
    return p, -negll(p if kind == "gjr" else (p[0], p[1], p[3]), r, kind, h0), h0


def nw_p(d, L=NW):
    n = len(d); m = d.mean(); e = d - m; v = (e * e).sum() / n
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * (e[l:] * e[:-l]).sum() / n
    z = m / np.sqrt(v / n); return z, 2 * stats.norm.sf(abs(z))


def vol_models(r, name, params_out):
    n = len(r); H = {"constant": np.full(n, np.nan), "EWMA": np.full(n, np.nan), "GARCH": np.full(n, np.nan), "GJR-GARCH": np.full(n, np.nan)}
    r2 = r * r; cs = np.cumsum(r2); H["constant"][BURN:] = cs[BURN - 1:n - 1] / np.arange(BURN, n)
    e = np.empty(n); e[60] = r2[:60].mean()
    for t in range(61, n): e[t] = EW * e[t - 1] + (1 - EW) * r2[t - 1]
    H["EWMA"][60:] = e[60:]
    for T in range(BURN, n, REFIT):
        for kind, nm in (("garch", "GARCH"), ("gjr", "GJR-GARCH")):
            p, ll, h0 = fit(r[:T], kind); end = min(T + REFIT, n)
            H[nm][T:end] = recursion(r[:end], p, kind, h0)[T:end]
            params_out.append({"series": name, "model": nm, "first_day_index": T, "omega": p[0], "alpha": p[1], "gamma": p[3 - 1] if kind == "gjr" else 0.0, "beta": p[3], "loglik": ll, "window_days": T})
    return H


def vr_test(r, q):
    N = len(r); mu = r.mean(); d = r - mu; s_a = (d * d).sum() / (N - 1)
    cs = np.concatenate([[0], np.cumsum(r)]); rq = cs[q:] - cs[:-q]; m = q * (N - q + 1) * (1 - q / N); s_c = ((rq - q * mu) ** 2).sum() / m
    d2 = d * d; den = d2.sum() ** 2; th = sum((2 * (q - j) / q) ** 2 * (d2[j:] * d2[:-j]).sum() / den for j in range(1, q))
    z = (s_c / s_a - 1) / np.sqrt(th); return s_c / s_a, z, 2 * stats.norm.sf(abs(z))


def main():
    R.mkdir(exist_ok=True); px = load(); dates = px.index; lp = np.log(px.to_numpy()); ret = 100 * np.diff(lp, axis=0); rd = dates[1:]; N = len(ret)
    # audit
    gaps = pd.Series(dates).diff().dt.days; spy_f = fred("SP500").dropna(); vix = fred("VIXCLS")
    j = pd.concat([np.log(px["SPY"]), np.log(spy_f)], axis=1, sort=True).dropna(); jr = j.diff().dropna()
    absr = np.abs(ret); au = {"closes per series": len(px), "returns per series": N, "first close": dates[0].date(), "last close": dates[-1].date(), "missing closes (any series)": int(px.isna().sum().sum()),
          "longest gap between closes (days)": int(gaps.max()), "closes outside the day's high-low range": 0,
          "returns with |r| > 20%": int((absr > 20).sum()), "largest |return| %": round(float(absr.max()), 3), "returns of exactly 0": int((ret == 0).sum()),
          "SPY vs FRED SP500: days compared": len(jr), "SPY vs FRED SP500: correlation of daily log returns": round(float(jr.corr().iloc[0, 1]), 5)}
    b = pd.read_csv(HERE / "data" / "bars.csv"); au["closes outside the day's high-low range"] = int(((b.close > b.high + 1e-9) | (b.close < b.low - 1e-9)).sum())
    pd.DataFrame({"check": list(au), "value": list(au.values())}).to_csv(R / "audit.csv", index=False)
    # biggest days
    rows = []
    for i, s in enumerate(SERIES):
        for rk, k in enumerate(np.argsort(-absr[:, i], kind="stable")[:3], 1): rows.append({"series": s, "rank": rk, "date": rd[k].date(), "return_pct": round(float(ret[k, i]), 4)})
    pd.DataFrame(rows).to_csv(R / "top_moves.csv", index=False)
    # moments, tails, acf, Ljung-Box, leverage
    mom, tails, ac, lbr, lev = [], [], [], [], []
    for i, s in enumerate(SERIES):
        r = ret[:, i]; m, sd = r.mean(), r.std(); z = (r - m) / sd; k = int(round(KPCT * N))
        mom.append({"series": s, "n": N, "mean_pct": m, "sd_pct": sd, "ann_vol_pct": sd * np.sqrt(252), "skew": (z ** 3).mean(), "excess_kurtosis": (z ** 4).mean() - 3, "min_pct": r.min(), "max_pct": r.max(), "share_beyond_3sd_pct": 100 * (np.abs(z) > 3).mean()})
        for kp in range(1, 11):
            kk = int(round(kp / 100 * N))
            for tail, x in (("loss", -r[r < 0]), ("gain", r[r > 0])):
                a, th = hill(x, kk); tails.append({"series": s, "tail": tail, "k_pct": kp, "k": kk, "threshold_pct": th, "alpha": a})
        a1, a2 = acf(r, 40), acf(np.abs(r), 40)
        for l in range(40): ac.append({"series": s, "lag": l + 1, "acf_r": a1[l], "acf_abs": a2[l]})
        q1, p1 = lb(r); q2, p2 = lb(np.abs(r)); lbr.append({"series": s, "q10_r": q1, "p_r": p1, "q10_abs": q2, "p_abs": p2})
        dn = r[:-1] < 0
        lev.append({"series": s, "next_abs_after_down": np.abs(r[1:])[dn].mean(), "next_abs_after_up": np.abs(r[1:])[~dn].mean(), "corr_r_next_abs": np.corrcoef(r[:-1], np.abs(r[1:]))[0, 1], "n_down_days": int(dn.sum())})
    pd.DataFrame(mom).to_csv(R / "moments.csv", index=False, float_format="%.10g"); pd.DataFrame(tails).to_csv(R / "tails.csv", index=False, float_format="%.10g")
    pd.DataFrame(ac).to_csv(R / "acf.csv", index=False, float_format="%.10g"); pd.DataFrame(lbr).to_csv(R / "ljungbox.csv", index=False, float_format="%.10g")
    # volatility forecasts
    prm, sc, monthly = [], [], None
    for i, s in enumerate(SERIES):
        r = ret[:, i]; H = vol_models(r, s, prm); t0 = BURN; lo = {m: r[t0:] ** 2 / H[m][t0:] + np.log(H[m][t0:]) for m in H}
        for m in H:
            d = lo["constant"] - lo[m]; z, p = nw_p(d) if m != "constant" else (np.nan, np.nan)
            sc.append({"series": s, "model": m, "n": N - t0, "qlike": lo[m].mean(), "gain_vs_constant": d.mean(), "dm_z": z, "dm_p": p})
        if s == "SPY":
            mm = pd.DataFrame({"month": rd[t0:].strftime("%Y-%m"), "realised_vol": r[t0:] ** 2, **{f"{m}": H[m][t0:] for m in H}}).groupby("month").mean()
            monthly = np.sqrt(mm * 252).reset_index()
    pr = pd.DataFrame(prm); pr["first_day"] = [rd[k].date() for k in pr["first_day_index"]]; pr.to_csv(R / "garch_params.csv", index=False, float_format="%.12g")
    pd.DataFrame(sc).to_csv(R / "volscores.csv", index=False, float_format="%.12g"); monthly.to_csv(R / "spy_monthly_vol.csv", index=False, float_format="%.10g")
    # leverage extras: GJR asymmetry in the last refit; SPY against VIX
    last = pr[(pr.model == "GJR-GARCH")].groupby("series").tail(1).set_index("series"); L = pd.DataFrame(lev)
    L["gjr_gamma_last"] = [last.loc[s, "gamma"] for s in L.series]; L["gjr_alpha_last"] = [last.loc[s, "alpha"] for s in L.series]; L.to_csv(R / "leverage.csv", index=False, float_format="%.10g")
    # random walk and trends
    vrs, trs = [], []
    for i, s in enumerate(SERIES):
        for q in QS: v, z, p = vr_test(ret[:, i], q); vrs.append({"series": s, "q": q, "vr": v, "z_star": z, "p": p})
        p_ = lp[:, i]; cl = px[s].to_numpy(); T0 = 252; idx = np.arange(T0, len(p_) - 21, 21); fut = p_[idx + 21] - p_[idx]
        row = {"series": s, "blocks": len(idx), "rmse_random_walk": np.sqrt((fut ** 2).mean())}
        for Nn in (21, 63, 252):
            e = fut - 21 * (p_[idx] - p_[idx - Nn]) / Nn; d = e ** 2 - fut ** 2; row[f"rmse_trend{Nn}"] = np.sqrt((e ** 2).mean()); row[f"t_trend{Nn}"] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        sma = np.array([cl[k - 199:k + 1].mean() for k in idx]); up = fut > 0; above = cl[idx] > sma
        row.update({"share_above_200d": above.mean(), "hit_rate_ma_rule": ((above & up) | (~above & ~up)).mean(), "hit_rate_always_up": up.mean(),
                    "mean_next21_above_pct": 100 * fut[above].mean(), "mean_next21_below_pct": 100 * fut[~above].mean() if (~above).any() else np.nan, "n_above": int(above.sum())})
        trs.append(row)
    pd.DataFrame(vrs).to_csv(R / "vr.csv", index=False, float_format="%.10g"); pd.DataFrame(trs).to_csv(R / "trend.csv", index=False, float_format="%.10g")
    # SPY and the VIX
    v = vix.reindex(dates); ok = v.notna() & v.shift(1).notna(); dv = np.log(v).diff()[ok].to_numpy(); rs = pd.Series(ret[:, SERIES.index("SPY")], index=rd).reindex(dv_idx := np.log(v).diff()[ok].index).to_numpy()
    sl = np.polyfit(rs, dv, 1)[0]
    summ = {"vix_days": len(rs), "vix_missing_trading_days": int((~ok).sum()), "corr_spy_return_dlogvix": float(np.corrcoef(rs, dv)[0, 1]), "slope_dlogvix_per_return_pct": float(sl), "burn_in_days": BURN, "refit_days": REFIT, "oos_days": N - BURN,
            "oos_first_day": str(rd[BURN].date()), "hill_k_pct": KPCT * 100}
    pd.DataFrame({"key": list(summ), "value": list(summ.values())}).to_csv(R / "summary.csv", index=False)
    print("done", len(prm), "fits")


if __name__ == "__main__":
    main()
