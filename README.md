# What part of a stock's move is predictable?

Ten large US stocks and the S&P 500 fund (SPY), 2,703 daily closes from 4 January 2016 to 2 October 2026, put through the "stylised facts" of stock returns, each as a test that can fail. Five questions:
- how fat are the tails of daily returns, and how sure can we be of their shape;
- is there any memory in the direction of a return, and in its size (volatility clustering);
- do falling prices make the next day more volatile (the leverage effect);
- how much better does an exponentially weighted or GARCH-type forecast of volatility do than a constant, walking forward with only the past;
- do price-level trends forecast anything, or is "no change" the better forecast (random-walk and variance-ratio tests).

The questions follow GeeksforGeeks' "Stock Price Time Series Analysis". They differ from the crypto-forecasting and gold-price-drivers projects in asking what is stable across stocks, and in testing volatility forecasts and trend forecasts out of sample. The page with the results and the charts is the build log:

Build log: https://mrrishit909.github.io/projects/stock-time-series/

## Data

Daily bars from the Alpaca Markets market-data API (https://data.alpaca.markets/v2/stocks/bars: 1-day bars, split- and dividend-adjusted, retrieved 2026-10-04 with a free paper-trading account; the keys are read from a local `.env` inside `download.py` and are never printed or stored). The feed is the consolidated tape (`feed=sip`; the free plan serves it for bars older than 15 minutes). The free `iex` feed was tried first, but it starts only on 27 July 2020 and would miss the 2020 crash. Alpaca's market-data terms forbid redistributing the raw data, so **the price file is not in this repository** (`data/` is ignored) and everything published here is our own statistics. Two FRED series (Federal Reserve Bank of St. Louis) supply a cross-check and the fear index: SP500 (S&P Dow Jones Indices; FRED marks it "copyrighted: pre-approval required", from October 2016) and VIXCLS (CBOE; "copyrighted: citation required"); only a correlation and a slope computed from them are published. The download is 2.8 MB (the same price file also holds eight restaurant stocks used by the fast-food-stocks project; they are not used here); checksums are in `results/pull.json`.

```
./venv/bin/pip install pandas numpy scipy matplotlib certifi      # in a venv
./venv/bin/python download.py    # data/*  (needs ALPACA_API_KEY and ALPACA_SECRET_KEY in a .env; set ENV_FILE to its path)
./venv/bin/python analyze.py     # results/*  (about 10 seconds)
./venv/bin/python check.py       # about 5 seconds
./venv/bin/python charts.py
```

## Steps

1. **Download** the bars of 19 symbols (ten stocks, SPY and eight others for a sibling project) and the two FRED series, with checksums.
2. **Audit.** Dates, gaps, missing closes, closes outside the day's high-low range, the largest moves. All 11 series have the same 2,703 trading days; the longest gap is 4 days; no close lies outside its day's range; the largest daily move is 26.1% (NVDA), and 4 moves exceed 20%. SPY's daily returns correlate 0.994 with the FRED S&P 500 index (2,513 days).
3. **Describe the returns.** A return is 100 x the change of the log of the adjusted close, in percent. Mean, spread, skew and excess kurtosis (population moments), the share of days beyond 3 standard deviations, the Hill tail index of the 5.0% largest losses and of the largest gains (k = 5% of the days, the sensitivity of k from 1% to 10%), autocorrelations of returns and absolute returns to lag 40, Ljung-Box Q(10).
4. **Leverage.** The next day's absolute return after a down day against after an up day, the correlation of a return with the next absolute return, the asymmetry term of GJR-GARCH, and SPY's daily return against the change of the log of the VIX.
5. **Forecast volatility, walking forward.** One day ahead, on 1,952 days from 27 December 2018: a constant (the mean squared return so far), EWMA (weight 0.94 on the past), GARCH(1,1) and GJR-GARCH(1,1), the last two by normal quasi-maximum likelihood (hand-coded around scipy's SLSQP, because the `arch` package is not installed), refitted every 63 days on all earlier returns. Scored by QLIKE, r^2/h + ln h, with a Diebold-Mariano test against the constant (Newey-West, 5 lags).
6. **Test the random walk and the trend.** Lo-MacKinlay variance ratios (q = 2, 5, 10, 20) with the heteroskedasticity-robust z*; forecasts of the log price 21 days ahead (every 21 days, 116 non-overlapping forecasts per series) by "no change" and by extending the last 21, 63 or 252 days' average change, by RMSE and a paired t-test; and the hit rate of "above its 200-day average means up".
7. **Check.** `check.py` recomputes everything in plain Python (csv, math and loops: no pandas, numpy or scipy): 3,475 numbers, to a relative tolerance of 1e-8, including all the walk-forward variance forecasts, the scores and tests, the Ljung-Box p-values (the exact chi-square tail), and the variance ratios. The GARCH fits are not refitted there; each of the 682 stored fits is verified by property instead (its log-likelihood is recomputed, its parameters are feasible, and none of 6,819 small perturbations has a higher likelihood).
8. **Charts.**

## Results

- **Fat tails.** Daily returns have excess kurtosis 4.4 to 13.2 (median 8.7; a normal distribution has 0), and 1.1% to 1.7% of days lie beyond 3 standard deviations (a normal distribution: 0.27%). The Hill tail index with k = 5% of the days is 2.3 to 3.3, for losses and for gains alike (medians 2.60 for losses, 2.67 for gains); for every k from 1% to 10% the median index stays between 2.1 and 3.2. An index below 4 means the kurtosis would be infinite in theory, so the sample kurtosis values above are dominated by a few days: 16 of the 33 largest moves (three per series) fall between February and April 2020.
- **Direction has almost no memory.** The lag-1 autocorrelation of returns averages -0.06 (-0.12 to -0.01), and no lag to 40 exceeds 0.12 in size. A Ljung-Box test on 10 lags still rejects zero autocorrelation at 5% for 11 of 11 series (10 at 1%), and 20% of the 440 series-lags lie outside the 95% band (5% by chance): small, but not nothing. The test assumes independent, evenly spread returns, which fat tails and clustering violate, so its rejections are probably overstated.
- **Size has long memory.** The lag-1 autocorrelation of absolute returns averages 0.24 (0.14 to 0.35); it is outside the 95% band at each of the first 10 lags in every series, and at 92% of the 40 lags. It decays slowly (mean 0.17 at lag 10, 0.12 at 20, 0.07 at 40). Q(10) is 10 to 68 times larger for absolute returns than for returns and rejects at p < 1e-10 for 11 of 11 series.
- **Down days are followed by bigger moves.** The next day's absolute return is higher after a down day than after an up day in 11 of 11 series, by a factor of 1.02 to 1.27 (median 1.11; SPY 1.27). The GJR-GARCH asymmetry term is positive in all 11 (larger than the symmetric term in 9); for SPY (fit of 8 July 2026) a 1% fall raises the next day's variance 6.7 times as much as a 1% rise (alpha 0.038, gamma 0.215). SPY's daily return and the change of the log VIX have correlation -0.73: the VIX rises 5.3% for each 1% fall in SPY, on average (2,701 days).
- **Volatility can be forecast.** On 1,952 walk-forward days, the constant has an average QLIKE of 2.16; EWMA, GARCH and GJR-GARCH are better by 0.21, 0.23 and 0.23 on average, and better than the constant on 11, 11 and 11 of 11 series. The gains are statistically significant at 5% on 4, 7 and 6 of 11 (at 1%: 2, 4, 4); SPY gains most (0.51 for GARCH). GARCH beats EWMA on 9 of 11 series and GJR-GARCH beats GARCH on 5: the asymmetry matters little for the forecast. In March 2020, SPY's realised volatility was 83% a year; the GARCH forecasts averaged 71%, EWMA 60% and the constant 16%.
- **Trends fail as forecasts.** Variance ratios are below 1 in 43 of 44 cases (medians 0.95 at q = 2, 0.91 at 5, 0.88 at 10, 0.85 at 20), a little mean reversion, but the robust test rejects a random walk in only 6 of 44 cases (GOOGL, MSFT, NVDA, SPY). For the log price 21 days ahead, extending the last 21 days' trend has an RMSE 1.44 times that of "no change" (median; worse in 11 of 11 series, significantly in 11), the last 63 days' trend 1.15 (worse in 11, significantly in 9), and the last 252 days' 1.01 (worse in 11, significantly in 0). "Above its 200-day average means up" is right 57% of the time against 62% for "always up", and loses to it in 9 of 11 series (wins in 0); the average next-21-day return is 1.63% after being above the average and 1.81% after being below it.

## Not done, and caveats

- The stocks are today's large companies, chosen after the fact (NVDA's average day is +0.21%), which inflates drift and makes "always up" hard to beat (62% of 21-day periods were up); the trend results should be read in that light. The volatility and tail results do not depend on the drift.
- One decade and one market: a high-inflation or low-volatility era was not tested, and the 11 series are strongly correlated, so statements such as "11 of 11" are not 11 independent confirmations. No correction for multiple testing was made.
- The Hill estimator is noisy, depends on k (shown from 1% to 10%, not optimised) and is biased by volatility clustering; the tail index is a description, not an estimate with an interval. Skew and kurtosis are population moments and have no standard errors here.
- Volatility is scored against squared daily returns, a noisy stand-in for the true daily variance; intraday data would give a sharper score. The gains may be concentrated in turbulent periods (March 2020, April 2025); that was not separated. Means are taken as zero, and only one-day-ahead forecasts with normal quasi-likelihood were tried (no Student-t, no longer horizons, no value-at-risk test).
- The GARCH models were hand-coded and fitted by scipy SLSQP and are not refitted in `check.py` (verified by properties instead); they were not compared with the `arch` package. Refits every 63 days and a 750-day burn-in were chosen once, not tuned.
- The prices are Alpaca's adjusted consolidated closes; their adjustment was not independently verified, beyond finding no close outside its day's range and no jump of split size. The feed is `sip`, not `iex` as first planned (see Data).
- No ADF or other unit-root test was run (their p-values are not reproducible in plain Python); the variance ratio and the forecast comparison are the evidence on the random walk.
- A historical description only, not investment advice. The raw prices are not published (Alpaca's terms); only our own statistics are.
