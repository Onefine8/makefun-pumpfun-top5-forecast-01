# pumpfun-top5-forecast-01

Entry for make.fun's "Predict the top 5 new Solana coins by market cap,
24 hours out" bounty.

`main.py` (Python 3.13, standard library only) reads the house's 10-minute
market-cap snapshots of pump.fun coins from `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes one forecast to
`$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output`).

Method: honest Monte Carlo. Each candidate coin's log10 market cap is
simulated forward to each of the seven answer reads as a drift / volatility /
death process fitted on its own trajectory, shrunk toward venue priors
(pool coins hold up better than curve coins). Fresh rockets born after the
history cut are simulated as a Poisson influx with caps drawn at top-5
scale — in the practice rehearsal, 3 of the top 5 at the last read were
born after the cut, so a large share of every place is deliberately left
unnamed. Place probabilities and size-bin probabilities are the simulated
frequencies, averaged over the seven reads. One fixed seed, so two runs on
the same history write identical bytes.

Run locally:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
