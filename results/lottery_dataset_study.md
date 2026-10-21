# Powerball dataset comparison

This study asks whether any series Neural Forge can train on moves with Powerball draws. The draw is at 10:59 pm Eastern. Each series contributes only the last value that would already have been known at that moment, and the models are scored on a later block of draws they were not fit on.

Series actually compared: weather. Still waiting on a fetcher, so not in the numbers: stocks, fx, crypto, commodities, electricity, air_quality, earthquakes, sunspots, economy, sports, traffic, ucdp, space_f107, space_flares, space_kp_ap, space_solar_wind, space_neos, space_launches, space_exoplanets, horse_racing, ai_progress, quantum_qubits, arxiv_quant_ph, qrng.

The association family had 321 tests. If none of the series had anything to do with the lottery, and the tests were independent, about 16.1 of them would fall under p = 0.05 by luck (dependence makes the usual count smaller). 12 did. Benjamini–Hochberg kept 0; Bonferroni kept 0. The ticket-scoring family had 71 tests and would expect about 3.6 raw hits; it produced 8 raw, 4 after Benjamini–Hochberg, and 4 after Bonferroni. Pooling every test (392) raises the luck expectation to about 19.6 raw hits. Permutation p-values cannot go below 0.0020 with this shuffle count.

The strongest Pearson correlation was -0.090 between weather.doy_sin and powerball_22 (raw p 0.000653, Benjamini–Hochberg p 0.0512).

No association survived Benjamini–Hochberg or Bonferroni. That includes the white-ball sum, the Powerball, the odd count, each individual ball, near-repeats (a draw sharing 3 or 4 white balls with the draw before it), and neighbor numbers (a ball one away from the previous draw, or a consecutive pair inside the draw).

On the holdout, a random ticket is expected to match 2 or more whites about 3.89% of the time, 3 or more 0.18%, and 4 or more 0.00%. The plain transformer matched 3 or more on 0.00% of draws and its played white-sum MAE was 32.04. The historical-mean ticket was at 32.00. Chance white-sum MAE on these draws is 47.37. Powerball exact-match chance is 3.85%; the plain model hit 3.60% and the historical mean hit 6.12%. No dataset-trained model beat both the plain model and the historical-mean ticket on a pre-specified close-call metric after the multiple-comparison correction.

The only rows that stay under the Benjamini–Hochberg line are white-ball sum errors beating a random ticket: Plain transformer, Weather, Historical mean, Cold numbers. A forecast that sits near the average sum will do that, and the historical mean does it with no outside data at all. It is not evidence that a dataset predicts the draw. No association survived, and no dataset model beat both the plain transformer and the historical-mean ticket after the correction. Close calls — matching 2, 3, or 4 whites, or landing within 1 or 2 of a drawn ball — sit on the chance rates. The draws still look independent. Re-run this study after the other fetchers merge; the registry will pick them up without a new design.

## Ranked tickets

| Rank | Predictor | PB exact | PB MAE | Sum MAE | Mean whites | 2+ | 3+ | 4+ | Within ±1 | Within ±2 | Signal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Weather | 11/278 (3.96%) | 6.780 | 31.86 | 0.367 | 5.40% | 0.36% | 0.00% | 0.94 | 1.53 | no |
| 2 | All datasets | 11/278 (3.96%) | 6.780 | 31.86 | 0.367 | 5.40% | 0.36% | 0.00% | 0.94 | 1.53 | no |
| 3 | Cold numbers | 12/278 (4.32%) | 6.982 | 31.90 | 0.335 | 3.60% | 0.00% | 0.00% | 1.02 | 1.65 | no |
| 4 | Historical mean | 17/278 (6.12%) | 6.870 | 32.00 | 0.371 | 3.96% | 0.72% | 0.00% | 1.04 | 1.51 | no |
| 5 | Plain transformer | 10/278 (3.60%) | 6.799 | 32.04 | 0.371 | 4.32% | 0.00% | 0.00% | 1.00 | 1.53 | no |
| 6 | Persistence | 6/278 (2.16%) | 9.471 | 43.97 | 0.356 | 2.52% | 0.36% | 0.00% | 1.04 | 1.65 | no |
| 7 | Hot numbers | 10/278 (3.60%) | 9.860 | 43.98 | 0.353 | 3.60% | 0.36% | 0.00% | 0.92 | 1.44 | no |
| 8 | Neighbor shift | 9/278 (3.24%) | 9.504 | 44.09 | 0.367 | 4.68% | 1.08% | 0.00% | 1.05 | 1.65 | no |
| 9 | Random tickets | 5/278 (1.80%) | 8.968 | 48.29 | 0.324 | 2.88% | 0.36% | 0.00% | 0.97 | 1.62 | no |

Chance for a uniform ticket on this holdout: Powerball exact 3.85%, 2+ 3.89%, 3+ 0.18%, 4+ 0.00%, mean whites within ±1 1.02, within ±2 1.60, white-sum MAE 47.37.

## Strongest associations

| Rank | Series | Target | Test | r | MI bits | p | BH | Bonferroni | Perm p | Survives |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | weather.doy_sin | powerball_22 | pearson | -0.090 | — | 0.000653 | 0.0512 | 0.256 | 0.002 | no |
| 2 | weather.doy_sin | white_52 | pearson | +0.077 | — | 0.00379 | 0.248 | 1 | 0.004 | no |
| 3 | weather.doy_sin | white_39 | pearson | +0.073 | — | 0.0058 | 0.325 | 1 | 0.008 | no |
| 4 | weather.doy_sin | powerball_06 | pearson | +0.071 | — | 0.00745 | 0.326 | 1 | 0.006 | no |
| 5 | weather.temperature_c | white_18 | pearson | -0.070 | — | 0.00831 | 0.326 | 1 | 0.012 | no |
| 6 | weather.temperature_c | white_10 | pearson | +0.067 | — | 0.0111 | 0.394 | 1 | 0.008 | no |
| 7 | weather.temperature_c | white_45 | pearson | +0.065 | — | 0.0139 | 0.446 | 1 | 0.012 | no |
| 8 | weather.doy_cos | white_12 | pearson | +0.062 | — | 0.0198 | 0.555 | 1 | 0.018 | no |
| 9 | weather.doy_cos | white_51 | pearson | +0.056 | — | 0.0357 | 0.881 | 1 | 0.038 | no |
| 10 | weather.doy_cos | white_36 | pearson | -0.053 | — | 0.0437 | 0.881 | 1 | 0.038 | no |
| 11 | weather.temperature_c | powerball_10 | pearson | -0.053 | — | 0.0457 | 0.881 | 1 | 0.044 | no |
| 12 | weather.doy_cos | white_22 | pearson | -0.052 | — | 0.05 | 0.881 | 1 | 0.046 | no |
| 13 | weather.doy_sin | white_36 | pearson | +0.051 | — | 0.0563 | 0.881 | 1 | — | no |
| 14 | weather.doy_cos | white_28 | pearson | +0.050 | — | 0.0571 | 0.881 | 1 | — | no |
| 15 | weather.doy_cos | white_45 | pearson | -0.050 | — | 0.0603 | 0.881 | 1 | — | no |

## Multiple comparisons

- **association**: 321 tests, expected luck hits 16.05, raw p ≤ 0.05: 12, Benjamini–Hochberg: 0, Bonferroni: 0 (cutoff 0.000156).
- **predictive**: 71 tests, expected luck hits 3.55, raw p ≤ 0.05: 8, Benjamini–Hochberg: 4, Bonferroni: 4 (cutoff 0.000704).
- **combined**: 392 tests, expected luck hits 19.60, raw p ≤ 0.05: 20, Benjamini–Hochberg: 4, Bonferroni: 4 (cutoff 0.000128).

A dataset signal in the ticket table requires a paired improvement on the same metric against both the plain transformer and the historical-mean ticket, with the Benjamini–Hochberg p-value from the combined family at or below alpha. Beating a random ticket is not enough: the historical mean already gets a closer white-ball sum, and more Powerball exact hits, by sitting near the middle.

## Datasets

- **weather** (available, environment): loaded 16726 rows from /workspace/data/weather_central_park.csv; alignment calendar:prior_calendar_day Columns: temperature_c, doy_sin, doy_cos. NOAA NCEI daily maximum temperature at Central Park (station USW00094728), with the day-of-year harmonics stored in that file. Public domain (17 U.S.C. § 105).
- **stocks** (TODO, markets): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/stocks.csv, data/equities.csv, data/stock_prices.csv, data/markets/stocks.csv Columns: —. Equity prices or index levels. TODO: fetcher is not on main yet.
- **fx** (TODO, markets): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/fx.csv, data/forex.csv, data/currencies.csv, data/markets/fx.csv Columns: —. Foreign-exchange rates, about 30 currencies. TODO: fetcher is not on main yet.
- **crypto** (TODO, markets): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/crypto.csv, data/cryptocurrencies.csv, data/markets/crypto.csv Columns: —. Large-cap crypto prices, about the top 50 coins. TODO: fetcher is not on main yet.
- **commodities** (TODO, markets): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/commodities.csv, data/futures.csv, data/markets/commodities.csv Columns: —. Commodity or futures settles. TODO: fetcher is not on main yet.
- **electricity** (TODO, infrastructure): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/electricity.csv, data/power.csv, data/grid.csv Columns: —. Electricity load or price. TODO: fetcher is not on main yet.
- **air_quality** (TODO, environment): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/air_quality.csv, data/aqi.csv Columns: —. Air-quality measurements. TODO: fetcher is not on main yet.
- **earthquakes** (TODO, environment): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/earthquakes.csv, data/seismic.csv Columns: —. Earthquake counts or magnitudes. TODO: fetcher is not on main yet.
- **sunspots** (TODO, environment): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/sunspots.csv Columns: —. Sunspot counts. TODO: fetcher is not on main yet.
- **economy** (TODO, economy): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/economy.csv, data/fred.csv, data/macro.csv Columns: —. Macroeconomic series. TODO: fetcher is not on main yet.
- **sports** (TODO, society): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/sports.csv Columns: —. Sports results or betting lines. TODO: fetcher is not on main yet.
- **traffic** (TODO, infrastructure): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/traffic.csv Columns: —. Traffic volume. TODO: fetcher is not on main yet.
- **ucdp** (TODO, society): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/ucdp.csv, data/conflict.csv Columns: —. UCDP conflict events. TODO: fetcher is not on main yet.
- **space_f107** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/f107.csv, data/space_f107.csv, data/solar_f107.csv Columns: —. Solar radio flux F10.7. TODO: fetcher is not on main yet.
- **space_flares** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/flares.csv, data/solar_flares.csv Columns: —. Solar flares. TODO: fetcher is not on main yet.
- **space_kp_ap** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/kp_ap.csv, data/geomagnetic.csv, data/kp.csv Columns: —. Geomagnetic Kp and Ap. TODO: fetcher is not on main yet.
- **space_solar_wind** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/solar_wind.csv Columns: —. Solar-wind measurements. TODO: fetcher is not on main yet.
- **space_neos** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/neos.csv, data/near_earth_objects.csv Columns: —. Near-Earth objects. TODO: fetcher is not on main yet.
- **space_launches** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/launches.csv Columns: —. Rocket launches. TODO: fetcher is not on main yet.
- **space_exoplanets** (TODO, space): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/exoplanets.csv Columns: —. Exoplanet discoveries. TODO: fetcher is not on main yet.
- **horse_racing** (TODO, society): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/horse_racing.csv, data/horses.csv Columns: —. Horse-racing results. TODO: fetcher is not on main yet.
- **ai_progress** (TODO, technology): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/ai_progress.csv Columns: —. AI progress benchmarks. TODO: fetcher is not on main yet.
- **quantum_qubits** (TODO, technology): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/qubits.csv, data/quantum_qubits.csv Columns: —. Published qubit counts. TODO: fetcher is not on main yet.
- **arxiv_quant_ph** (TODO, technology): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/arxiv_quant_ph.csv, data/quant_ph.csv Columns: —. arXiv quant-ph publication counts. TODO: fetcher is not on main yet.
- **qrng** (TODO, technology): not on main yet; export LOTTERY_STUDY_HOOKS or write one of: data/qrng.csv Columns: —. Quantum random-number generator output. TODO: fetcher is not on main yet.

## Definitions

White-ball sum, Powerball, and odd count are properties of the draw being predicted. Odd count is the number of odd white balls. Individual ball rows are indicators that the number was drawn. A near-repeat shares 3 or 4 white balls with the previous draw. A neighbor event is a white ball whose nearest ball in the previous draw is exactly one away (an exact repeat is not a neighbor), or two consecutive whites in the same draw. Within ±1 and ±2 count predicted whites whose nearest drawn white is at most that far away, so exact matches are included. Near-miss counts in the JSON are the same comparison at distance exactly 1 or exactly 2. Partial-match chance rates are hypergeometric. The chart shows white-sum error against the random-ticket baseline and the smallest association p-values.

Chart: `/workspace/results/lottery_dataset_study.svg`
