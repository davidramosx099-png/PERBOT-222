# PERBOT-222

PERBOT-222 is an independent research project for MT5/XAUUSD short-horizon prediction.

## Objective

Build and validate a tick-based engine that estimates whether price is likely to rise over a very short future horizon, then turns only statistically supported predictions into entries.

The project is **independent from PERBOT**. Do not mix code, data, experiments, results, or assumptions between the two projects.

## Research sequence

1. Ingest and validate tick data.
2. Build causal microstructure features from past ticks only.
3. Define forward-looking labels for multiple horizons.
4. Establish a leakage-safe baseline.
5. Evaluate probability calibration, precision, expectancy and drawdown after spread/slippage/commission.
6. Run chronological OOS and walk-forward validation.
7. Search for robustness across regimes and time periods.
8. Only then design the live MT5 execution layer.

## Initial hypothesis

At some moments the recent tick stream contains enough information to increase the probability of a short upward move. The first target is to measure that probability rather than assume it.

Candidate lookbacks: 50, 100, 250, 500, 1000 and 2000 ticks.

Candidate horizons: 1, 3, 5, 10, 30 and 60 seconds.

No model is considered valid because of a single profitable run.

## Current status

Repository initialized. Tick schema, feature pipeline, label generation and a leakage-safe baseline are being built first.
