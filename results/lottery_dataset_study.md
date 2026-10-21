# Powerball dataset comparison

This study asks whether any series Neural Forge can train on moves with Powerball draws. The draw is at 10:59 pm Eastern. Each series contributes only the last value that would already have been known at that moment, and the models are scored on a later block of draws they were not fit on.

Series actually compared: weather, stocks, fx, crypto, commodities, electricity, air_quality, earthquakes, sunspots, sports, ucdp, space_f107, space_flares, space_kp_ap, space_solar_wind, space_dst, space_neos, space_launches, space_exoplanets, horse_racing, ai_progress, quantum_qubits, quantum_gate_error, quantum_neutrons, arxiv_quant_ph. Left out of the numbers: economy (FRED download was not a data table), traffic (no fetcher yet), qrng (no calendar).

The association family had 16160 tests. If none of the series had anything to do with the lottery, and the tests were independent, about 808.0 of them would fall under p = 0.05 by luck (dependence makes the usual count smaller). 846 did. Benjamini–Hochberg kept 0; Bonferroni kept 0. The ticket-scoring family had 496 tests and would expect about 24.8 raw hits; it produced 45 raw, 29 after Benjamini–Hochberg, and 29 after Bonferroni. Pooling every test (16656) raises the luck expectation to about 832.8 raw hits. Permutation p-values cannot go below 0.0020 with this shuffle count.

Horse racing stops on 2018-06-27. Draws after that keep the last race day as the last known value. That carry-forward is stale, and it is not a look-ahead. Annual series are the same kind of step: only a year that has already ended can inform the draw.

The strongest Pearson correlation was +0.464 between crypto.night and white_04 on 91 aligned draws (raw p 3.57e-06, pooled Benjamini–Hochberg p 0.00198). Inside the association family the Benjamini–Hochberg p is 0.0577.

No association survived Benjamini–Hochberg or Bonferroni inside the association family. That includes the white-ball sum, the Powerball, the odd count, each individual ball, near-repeats (a draw sharing 3 or 4 white balls with the draw before it), and neighbor numbers (a ball one away from the previous draw, or a consecutive pair inside the draw).

Pooling association tests with the ticket-scoring tests lets 6 association rows cross the pooled Benjamini–Hochberg line: crypto.ada vs powerball_10, r=-0.125, n=1155, family BH p 0.116; crypto.night vs white_04, r=+0.464, n=91, family BH p 0.0577; crypto.trx vs white_01, r=+0.117, n=1139, family BH p 0.226; crypto.vet vs powerball_10, r=-0.117, n=1126, family BH p 0.226; crypto.vvv vs white_58, r=+0.260, n=266, family BH p 0.116; space_flares.b_class vs white_11, r=+0.105, n=1422, family BH p 0.226. The ticket-scoring rejects are mostly white-ball sum errors beating a random ticket, and those tiny p-values loosen the pooled threshold. The same association rows stay above the association-family Benjamini–Hochberg line and above Bonferroni. They are not a surviving signal.

On the holdout, a random ticket is expected to match 2 or more whites about 3.89% of the time, 3 or more 0.18%, and 4 or more 0.00%. The plain transformer matched 3 or more on 0.00% of draws and its played white-sum MAE was 32.04. The historical-mean ticket was at 32.00. Chance white-sum MAE on these draws is 47.37. Powerball exact-match chance is 3.85%; the plain model hit 3.60% and the historical mean hit 6.12%. No dataset-trained model beat both the plain model and the historical-mean ticket on a pre-specified close-call metric after the multiple-comparison correction.

The only ticket-scoring rows that stay under the Benjamini–Hochberg line are white-ball sum errors beating a random ticket: Plain transformer, Weather, Stocks, FX, Crypto, Commodities, Electricity, Air quality, Earthquakes, Sunspots, Sports, UCDP, Space F10.7, Solar flares, Space Kp/Ap, Solar wind, Space Dst, Near-Earth objects, Launches, Exoplanets, Horse racing, AI progress, Quantum qubits, Quantum gate error, Oulu neutrons, arXiv quant-ph, All datasets, Historical mean, Cold numbers. A forecast that sits near the average sum will do that, and the historical mean does it with no outside data at all. It is not evidence that a dataset predicts the draw. No association survived the association-family correction, and no dataset model beat both the plain transformer and the historical-mean ticket. Close calls — matching 2, 3, or 4 whites, or landing within 1 or 2 of a drawn ball — sit on the chance rates. The draws still look independent.

## Ranked tickets

| Rank | Predictor | PB exact | PB MAE | Sum MAE | Mean whites | 2+ | 3+ | 4+ | Within ±1 | Within ±2 | Signal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Stocks | 18/278 (6.47%) | 6.796 | 31.61 | 0.345 | 2.88% | 0.36% | 0.00% | 0.95 | 1.49 | no |
| 2 | Horse racing | 13/278 (4.68%) | 6.828 | 31.73 | 0.356 | 5.40% | 0.00% | 0.00% | 0.98 | 1.54 | no |
| 3 | Electricity | 16/278 (5.76%) | 6.830 | 31.74 | 0.353 | 3.96% | 0.36% | 0.00% | 1.00 | 1.60 | no |
| 4 | Quantum gate error | 15/278 (5.40%) | 6.815 | 31.74 | 0.317 | 3.60% | 0.00% | 0.00% | 1.00 | 1.52 | no |
| 5 | arXiv quant-ph | 18/278 (6.47%) | 6.794 | 31.76 | 0.317 | 3.60% | 0.00% | 0.00% | 1.00 | 1.53 | no |
| 6 | Oulu neutrons | 13/278 (4.68%) | 6.810 | 31.78 | 0.338 | 2.88% | 0.36% | 0.00% | 0.97 | 1.53 | no |
| 7 | Exoplanets | 17/278 (6.12%) | 6.821 | 31.78 | 0.345 | 4.32% | 0.00% | 0.00% | 0.97 | 1.55 | no |
| 8 | Solar flares | 13/278 (4.68%) | 6.819 | 31.82 | 0.371 | 5.04% | 0.00% | 0.00% | 0.99 | 1.53 | no |
| 9 | Launches | 18/278 (6.47%) | 6.806 | 31.82 | 0.295 | 2.52% | 0.00% | 0.00% | 0.97 | 1.52 | no |
| 10 | Sports | 19/278 (6.83%) | 6.818 | 31.84 | 0.349 | 4.68% | 0.00% | 0.00% | 1.01 | 1.55 | no |
| 11 | Quantum qubits | 10/278 (3.60%) | 6.846 | 31.85 | 0.324 | 3.60% | 0.00% | 0.00% | 0.96 | 1.51 | no |
| 12 | Weather | 11/278 (3.96%) | 6.780 | 31.86 | 0.367 | 5.40% | 0.36% | 0.00% | 0.94 | 1.53 | no |
| 13 | AI progress | 8/278 (2.88%) | 6.875 | 31.87 | 0.327 | 3.24% | 0.00% | 0.00% | 0.98 | 1.55 | no |
| 14 | FX | 12/278 (4.32%) | 6.881 | 31.90 | 0.324 | 3.24% | 0.00% | 0.00% | 0.96 | 1.52 | no |
| 15 | Cold numbers | 12/278 (4.32%) | 6.982 | 31.90 | 0.335 | 3.60% | 0.00% | 0.00% | 1.02 | 1.65 | no |
| 16 | Air quality | 18/278 (6.47%) | 6.801 | 31.91 | 0.403 | 5.04% | 0.36% | 0.00% | 1.03 | 1.55 | no |
| 17 | Space Dst | 15/278 (5.40%) | 6.824 | 31.94 | 0.335 | 3.96% | 0.36% | 0.00% | 1.00 | 1.52 | no |
| 18 | Space F10.7 | 11/278 (3.96%) | 6.833 | 31.95 | 0.345 | 3.24% | 0.72% | 0.00% | 1.04 | 1.54 | no |
| 19 | Near-Earth objects | 12/278 (4.32%) | 6.857 | 31.95 | 0.306 | 2.16% | 0.36% | 0.00% | 1.03 | 1.55 | no |
| 20 | Commodities | 12/278 (4.32%) | 6.916 | 31.96 | 0.327 | 2.52% | 0.36% | 0.00% | 1.01 | 1.58 | no |
| 21 | Earthquakes | 11/278 (3.96%) | 6.824 | 31.96 | 0.331 | 2.88% | 0.00% | 0.00% | 0.97 | 1.48 | no |
| 22 | Historical mean | 17/278 (6.12%) | 6.870 | 32.00 | 0.371 | 3.96% | 0.72% | 0.00% | 1.04 | 1.51 | no |
| 23 | UCDP | 13/278 (4.68%) | 6.864 | 32.01 | 0.349 | 2.88% | 0.36% | 0.00% | 1.00 | 1.57 | no |
| 24 | Sunspots | 12/278 (4.32%) | 6.821 | 32.03 | 0.378 | 4.32% | 0.36% | 0.00% | 1.00 | 1.55 | no |
| 25 | Plain transformer | 10/278 (3.60%) | 6.799 | 32.04 | 0.371 | 4.32% | 0.00% | 0.00% | 1.00 | 1.53 | no |
| 26 | Solar wind | 7/278 (2.52%) | 6.850 | 32.14 | 0.345 | 3.60% | 0.00% | 0.00% | 1.03 | 1.58 | no |
| 27 | Space Kp/Ap | 9/278 (3.24%) | 6.861 | 32.20 | 0.381 | 4.32% | 0.00% | 0.00% | 0.99 | 1.56 | no |
| 28 | Crypto | 7/278 (2.52%) | 7.115 | 32.27 | 0.327 | 3.60% | 0.00% | 0.00% | 0.94 | 1.53 | no |
| 29 | All datasets | 5/278 (1.80%) | 7.153 | 32.53 | 0.345 | 3.96% | 0.00% | 0.00% | 1.10 | 1.68 | no |
| 30 | Persistence | 6/278 (2.16%) | 9.471 | 43.97 | 0.356 | 2.52% | 0.36% | 0.00% | 1.04 | 1.65 | no |
| 31 | Hot numbers | 10/278 (3.60%) | 9.860 | 43.98 | 0.353 | 3.60% | 0.36% | 0.00% | 0.92 | 1.44 | no |
| 32 | Neighbor shift | 9/278 (3.24%) | 9.504 | 44.09 | 0.367 | 4.68% | 1.08% | 0.00% | 1.05 | 1.65 | no |
| 33 | Random tickets | 5/278 (1.80%) | 8.968 | 48.29 | 0.324 | 2.88% | 0.36% | 0.00% | 0.97 | 1.62 | no |

Chance for a uniform ticket on this holdout: Powerball exact 3.85%, 2+ 3.89%, 3+ 0.18%, 4+ 0.00%, mean whites within ±1 1.02, within ±2 1.60, white-sum MAE 47.37.

## Strongest associations

| Rank | Series | Target | Test | r | MI bits | p | Pooled BH | Family BH | Bonferroni | Perm p | Family survives |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | crypto.night | white_04 | pearson | +0.464 | — | 3.57e-06 | 0.00198 | 0.0577 | 0.0595 | 0.002 | no |
| 2 | crypto.vvv | white_58 | pearson | +0.260 | — | 1.77e-05 | 0.00954 | 0.116 | 0.296 | 0.002 | no |
| 3 | crypto.ada | powerball_10 | pearson | -0.125 | — | 2.15e-05 | 0.0112 | 0.116 | 0.358 | 0.002 | no |
| 4 | space_flares.b_class | white_11 | pearson | +0.105 | — | 7.2e-05 | 0.0363 | 0.226 | 1 | 0.002 | no |
| 5 | crypto.trx | white_01 | pearson | +0.117 | — | 8e-05 | 0.0392 | 0.226 | 1 | 0.002 | no |
| 6 | crypto.vet | powerball_10 | pearson | -0.117 | — | 8.4e-05 | 0.04 | 0.226 | 1 | 0.002 | no |
| 7 | crypto.avax | powerball_10 | pearson | -0.127 | — | 0.000125 | 0.0577 | 0.263 | 1 | 0.002 | no |
| 8 | crypto.u | near_repeat_3 | pearson | -0.348 | — | 0.00013 | 0.0586 | 0.263 | 1 | 0.012 | no |
| 9 | crypto.apt | white_46 | pearson | +0.150 | — | 0.00017 | 0.0746 | 0.306 | 1 | 0.002 | no |
| 10 | crypto.atom | powerball_10 | pearson | -0.114 | — | 0.000209 | 0.0894 | 0.318 | 1 | 0.004 | no |
| 11 | crypto.pepe | powerball_10 | pearson | -0.159 | — | 0.000216 | 0.0901 | 0.318 | 1 | 0.002 | no |
| 12 | crypto.crv | powerball_10 | pearson | -0.120 | — | 0.000275 | 0.112 | 0.355 | 1 | 0.004 | no |
| 13 | sports.goal_diff | powerball_12 | pearson | +0.096 | — | 0.000289 | 0.114 | 0.355 | 1 | 0.002 | no |
| 14 | fx.myr | white_40 | pearson | -0.096 | — | 0.000307 | 0.119 | 0.355 | 1 | 0.002 | no |
| 15 | crypto.apt | powerball_22 | pearson | +0.143 | — | 0.000334 | 0.126 | 0.36 | 1 | 0.002 | no |

## Multiple comparisons

- **association**: 16160 tests, expected luck hits 808.00, raw p ≤ 0.05: 846, Benjamini–Hochberg: 0, Bonferroni: 0 (cutoff 3.09e-06).
- **predictive**: 496 tests, expected luck hits 24.80, raw p ≤ 0.05: 45, Benjamini–Hochberg: 29, Bonferroni: 29 (cutoff 0.000101).
- **combined**: 16656 tests, expected luck hits 832.80, raw p ≤ 0.05: 891, Benjamini–Hochberg: 35, Bonferroni: 29 (cutoff 3e-06).

A dataset signal in the ticket table requires a paired improvement on the same metric against both the plain transformer and the historical-mean ticket, with the Benjamini–Hochberg p-value from the combined family at or below alpha. Beating a random ticket is not enough: the historical mean already gets a closer white-ball sum, and more Powerball exact hits, by sitting near the middle.

## Datasets

- **weather** (available, environment): loaded 16726 rows from /workspace/data/weather_central_park.csv; alignment calendar:prior_calendar_day Columns: temperature_c, doy_sin, doy_cos. NOAA NCEI daily maximum temperature at Central Park (station USW00094728), with the day-of-year harmonics stored in that file. Public domain (17 U.S.C. § 105).
- **stocks** (available, markets): loaded 4216 days and 4 log-return series from /workspace/data/markets/features; alignment calendar:same_day_before_draw Columns: aapl, msft, nvda, sp500. Daily log returns for the equities and S&P 500 series in src/markets. The US cash close is 16:00 ET, before the draw. vol_10 is a function of those same returns and is not a second test.
- **fx** (available, markets): loaded 7109 days and 41 log-return series from /workspace/data/markets/features; alignment calendar:same_day_before_draw Columns: aud, bgn, brl, cad, chf, cny, cyp, czk, dkk, eek, eur, gbp, hkd, hrk, huf, idr, ils, inr, isk, jpy, krw, ltl, lvl, mtl, mxn, myr, nok, nzd, php, pln, rol, ron, rub, sek, sgd, sit, skk, thb, trl, try, zar. ECB and H.10 daily FX log returns, one column per currency. Reference rates are published before the 22:59 ET draw.
- **crypto** (available, markets): loaded 3343 days and 62 log-return series from /workspace/data/markets/features; alignment calendar:prior_calendar_day Columns: aave, ada, aero, algo, apt, arb, aster, atom, avax, bch, bnb, btc, cake, cro, crv, dash, doge, dot, ena, etc, eth, ethfi, fil, gram, hbar, hype, icp, inj, jst, jup, link, ltc, morpho, near, nexo, night, ondo, paxg, pepe, pol, pump, pyth, qnt, render, shib, sky, sol, stx, sui, tao, tia, trx, u, uni, vet, vvv, wld, wlfi, xlm, xrp, zec, zro. Daily log returns for the top crypto sample. A date-only candle has no clock, and crypto trades through the draw, so the value used is the previous calendar day.
- **commodities** (available, markets): loaded 4217 days and 9 log-return series from /workspace/data/markets/features; alignment calendar:same_day_before_draw Columns: aluminum, brent, copper, gold, natgas, palladium, platinum, silver, wti. Commodity futures daily log returns. The daily settle is treated as known by the 16:00 ET cash close.
- **electricity** (available, infrastructure): loaded 68409 rows from /workspace/data/realworld/electricity_us48_hourly_full.csv; alignment timestamp:at_or_before_draw Columns: demand. EIA US48 hourly electricity demand. The hour stamp is the start of the hour, so the value is treated as known one hour later.
- **air_quality** (available, environment): loaded 1754 rows from /workspace/data/realworld/air_berlin_daily.csv; alignment calendar:prior_calendar_day Columns: pm2_5, ozone, pollen. Berlin daily PM2.5, ozone, and pollen from Open-Meteo. A daily mean is not known until the day is over.
- **earthquakes** (available, environment): loaded 13442 rows from /workspace/data/realworld/earthquakes_m45_daily.csv; alignment calendar:prior_calendar_day Columns: count, log10_energy. USGS daily count of magnitude 4.5+ earthquakes and the log energy of those events. The day's total is used from the next day.
- **sunspots** (available, environment): loaded 3333 rows from /workspace/data/realworld/sunspots_monthly.csv; alignment calendar:prior_calendar_day Columns: sunspot_number. SILSO monthly sunspot number. The month is stamped at month end so a draw during the month cannot see that month's mean.
- **economy** (TODO, economy): loader failed: FRED CSV has no value column Columns: —. FRED unemployment, CPI inflation, and weekly regular gasoline. Monthly series are stamped at month end.
- **sports** (available, society): loaded 1269 rows from /workspace/data/realworld/sports_arsenal_goal_diff.csv; alignment calendar:prior_calendar_day Columns: goal_diff. Arsenal match goal difference from football-data.co.uk. The result is used from the next calendar day.
- **traffic** (TODO, infrastructure): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/traffic.csv Columns: —. Traffic volume. TODO: fetcher is not on main yet.
- **ucdp** (available, society): loaded 736 rows from /workspace/data/realworld/ucdp_monthly.csv, /workspace/data/realworld/ucdp_onset_gaps.csv; alignment calendar:prior_calendar_day Columns: events, fatalities, onset_gap. UCDP monthly conflict events and fatalities, plus onset gaps. Monthly totals are stamped at month end.
- **space_f107** (available, space): loaded 25222 rows from /workspace/data/space/f107_daily.csv; alignment calendar:prior_calendar_day Columns: f107. Daily solar radio flux F10.7. The day's value is used from the next day.
- **space_flares** (available, space): loaded 501 rows from /workspace/data/space/flares_monthly.csv; alignment calendar:prior_calendar_day Columns: b_class, c_class, m_class, x_class. Monthly GOES flare counts by class. A month is stamped at month end, after the partial month has been dropped.
- **space_kp_ap** (available, space): loaded 25222 rows from /workspace/data/space/kp_daily.csv; alignment calendar:prior_calendar_day Columns: kp_max, ap. Daily geomagnetic Kp and Ap. Used from the next calendar day.
- **space_solar_wind** (available, space): loaded 11663 rows from /workspace/data/space/solar_wind_daily.csv; alignment calendar:prior_calendar_day Columns: speed, density. Daily solar-wind speed and density. Used from the next calendar day.
- **space_dst** (available, space): loaded 23300 rows from /workspace/data/space/dst_daily.csv; alignment calendar:prior_calendar_day Columns: dst. Daily disturbance storm time (Dst). Used from the next calendar day.
- **space_neos** (available, space): loaded 1596 rows from /workspace/data/space/neo_approaches_monthly.csv, /workspace/data/space/neo_discoveries_monthly.csv; alignment calendar:prior_calendar_day Columns: approaches, discoveries. Monthly near-Earth close approaches and first observations. Stamped at month end.
- **space_launches** (available, space): loaded 828 rows from /workspace/data/space/launches_monthly.csv; alignment calendar:prior_calendar_day Columns: attempts. Monthly orbital launch attempts. Stamped at month end.
- **space_exoplanets** (available, space): loaded 34 rows from /workspace/data/space/exoplanets_yearly.csv; alignment calendar:prior_calendar_day Columns: discoveries. Yearly exoplanet discovery counts. Stamped at year end.
- **horse_racing** (available, society): loaded 2875 race days 1979-09-22 to 2018-06-27 from /workspace/data/raw/horse_racing/performances.csv.gz. Later draws keep that last race day.; alignment calendar:prior_calendar_day Columns: n_races, n_runners, mean_placing, median_odds. Hong Kong race-day counts, field size, average placing, and median dividend. A race day is used from the next calendar day.
- **ai_progress** (available, technology): loaded 193 rows from /workspace/datasets/ai_progress/arxiv_counts_monthly.csv, /workspace/datasets/ai_progress/frontier_compute_monthly.csv, /workspace/datasets/ai_progress/github_ml_repos_monthly.csv; alignment calendar:prior_calendar_day Columns: cs_ai, cs_lg, cs_cl, frontier_log10_compute, github_ml_repos. Monthly arXiv cs.AI, cs.LG, and cs.CL counts, Epoch AI frontier compute, and new machine-learning GitHub repositories. Months are stamped at month end. An incomplete GitHub month is dropped.
- **quantum_qubits** (available, technology): loaded 29 rows from /workspace/datasets/quantum/qubit_record_annual.csv; alignment calendar:prior_calendar_day Columns: log10_qubits, log10_qubits_excluding_unrearranged. Annual published qubit-count records, including the series that excludes unrearranged arrays. Stamped at year end.
- **quantum_gate_error** (available, technology): loaded 29 rows from /workspace/datasets/quantum/two_qubit_error_annual.csv; alignment calendar:prior_calendar_day Columns: log10_error. Annual best reported two-qubit error, as log10. Stamped at year end.
- **quantum_neutrons** (available, technology): loaded 751 rows from /workspace/datasets/quantum/oulu_neutrons_monthly.csv; alignment calendar:prior_calendar_day Columns: count_rate. Oulu monthly cosmic-ray neutron count rate. Stamped at month end.
- **arxiv_quant_ph** (available, technology): loaded 177 rows from /workspace/datasets/quantum/arxiv_quant_ph_monthly.csv; alignment calendar:prior_calendar_day Columns: count. Monthly arXiv quant-ph counts. Stamped at month end.
- **qrng** (TODO, technology): /workspace/datasets/quantum/qrng_symbols.csv is an ordered symbol stream (column 'step'), not a calendar series, so it cannot be placed before a draw without inventing a clock. Columns: —. Quantum random symbols from the shipped QRNG file. The file is an ordered stream with no calendar date.

## Definitions

White-ball sum, Powerball, and odd count are properties of the draw being predicted. Odd count is the number of odd white balls. Individual ball rows are indicators that the number was drawn. A near-repeat shares 3 or 4 white balls with the previous draw. A neighbor event is a white ball whose nearest ball in the previous draw is exactly one away (an exact repeat is not a neighbor), or two consecutive whites in the same draw. Within ±1 and ±2 count predicted whites whose nearest drawn white is at most that far away, so exact matches are included. Near-miss counts in the JSON are the same comparison at distance exactly 1 or exactly 2. Partial-match chance rates are hypergeometric. The chart shows white-sum error against the random-ticket baseline and the smallest association p-values.

Chart: `/workspace/results/lottery_dataset_study.svg`
