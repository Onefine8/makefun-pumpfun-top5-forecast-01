#!/usr/bin/env python3
"""make.fun pumpfun-top5-forecast-01 entry.

Forecasts the 5 largest pump.fun coins by market cap at the seven answer
reads, from the house's 10-minute snapshots in /input/history.ndjson.

Method: honest Monte Carlo. Every candidate coin's log10 market cap is
simulated forward to each answer read as a (drift, volatility, death)
process fitted on its own trajectory, shrunk toward venue priors; fresh
rockets born after the history cut are simulated as a Poisson influx with
caps drawn at top-5 scale (the practice shows 2-3 of the top 5 at the last
read were born after the cut, so a large share of every place is deliberately
left unnamed). Place probabilities and size-bin probabilities are the
simulated frequencies, averaged over the seven reads. Deterministic:
one fixed seed, so two runs on the same history write identical bytes.

Standard library only. Reads $MAKEFUN_INPUT_DIR (default /input), writes
$MAKEFUN_OUTPUT_DIR/forecast.json (default /output/forecast.json).
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261112

# --- tunables (calibrated on the kit's 2-hour practice rehearsal) ---
N_SIMS = 3000
MAX_CANDIDATES = 60
# drift (log10 cap per hour) from a coin's own slope, shrunk 50% toward the
# venue prior, then clamped. Pooled 10-min log changes of top-30 coins:
# pool mean +0.086 sd 0.143, curve mean -0.130 sd 0.289 (frenzy phase);
# priors below are the same numbers shrunk hard toward 0, since a launch
# frenzy does not compound for 10 hours.
DRIFT_SHRINK = 0.5
DRIFT_CLAMP = 0.15
DRIFT_PRIOR = {"pool": 0.03, "curve": -0.05, "migrating": -0.02}
# per-hour log10 volatility: pooled 10-min sd scaled by sqrt(6) is the
# honest prior (pool 0.35, curve 0.71); a coin's own residual only blends
# in once it has >=6 observations, else the prior rules.
SIGMA_CLAMP = (0.30, 0.75)
SIGMA_PRIOR = {"pool": 0.35, "curve": 0.70, "migrating": 0.50}
# death hazard per hour (coin fades off the chart for good).
HAZARD = {"pool": 0.08, "curve": 0.25, "migrating": 0.15}
# drift/volatility stop growing past this horizon (caps are redrawn, not
# random-walked to infinity): effective horizon in hours.
H_EFF_CAP = 4.0
# newcomers born after the cut: Poisson mean per read, caps lognormal around
# the log10 cap of the current #1 coin with a wide spread (fresh rockets
# arrive at top scale: in the practice a coin 9 minutes old held 2,940 SOL,
# and 3 of the top 5 at the last practice read were born after the cut).
# Sweeps on the practice backtest: lam 3 -> 4 -> 5 improves the longest
# (70-min) horizon each step; 4.0 is the compromise that still beats the
# house baseline on every cut. Env overrides exist only for local tuning.
NEWCOMER_LAM = float(os.environ.get("MF_LAM", "4.0"))
NEWCOMER_SD = float(os.environ.get("MF_NEW_SD", "0.45"))
# forecast rows below this probability are dropped (their mass stays unnamed).
P_MIN = 0.004
SIZE_P_MIN = 0.0015


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log10_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    n_bins = len(edges) + 1

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    cut = parse_ts(marks[-1]["at"])
    reads = [parse_ts(r) for r in task["reads"]]
    horizons = [max(0.0, (r - cut).total_seconds() / 3600.0) for r in reads]

    # --- per-coin series: mint -> list of (hours since cut, log10 cap, rank, venue)
    series = {}
    for m in marks:
        t = (parse_ts(m["at"]) - cut).total_seconds() / 3600.0
        for row in m["rows"]:
            mint = row["mint"]
            series.setdefault(mint, []).append(
                (t, log10_sol(row["market_cap_lamports"]), int(row["rank"]), row["venue"])
            )

    last = marks[-1]["rows"]
    if len(last) < places:
        raise SystemExit("the last mark ranks fewer coins than the places")
    last_by_mint = {r["mint"]: r for r in last}

    # --- candidate set: last mark's top 25, any coin top-10 in the last
    # 6 marks, and the top growers of the last hour among the last top 100.
    cand = set(r["mint"] for r in last[:25])
    for m in marks[-6:]:
        for r in m["rows"][:10]:
            cand.add(r["mint"])
    growth = []
    for mint, pts in series.items():
        if mint in last_by_mint and len(pts) >= 2:
            (t0, x0, _, _), (t1, x1, _, _) = pts[0], pts[-1]
            if t1 - t0 >= 0.5:
                growth.append((mint, (x1 - x0) / (t1 - t0)))
    growth.sort(key=lambda kv: -kv[1])
    for mint, _ in growth[:10]:
        cand.add(mint)
    cand = sorted(cand)[:MAX_CANDIDATES]

    # --- fit each candidate: (x0, drift/hr, sigma/hr, hazard/hr)
    # pooled per-venue 10-min log-change stats over the whole history, as
    # the volatility/drift prior (a coin's own few points cannot estimate
    # its volatility: a 2-point linear fit has zero residuals by construction).
    pooled = {}
    for i in range(len(marks) - 1):
        prev = {r["mint"]: r for r in marks[i]["rows"]}
        cur = {r["mint"]: r for r in marks[i + 1]["rows"]}
        for mint, r in prev.items():
            if r["rank"] <= 30 and mint in cur:
                d = log10_sol(cur[mint]["market_cap_lamports"]) - log10_sol(
                    r["market_cap_lamports"]
                )
                pooled.setdefault(r["venue"], []).append(d)
    pool_stats = {}
    for venue, ds in pooled.items():
        if len(ds) >= 5:
            m = sum(ds) / len(ds)
            sd = math.sqrt(sum((d - m) ** 2 for d in ds) / len(ds))
            pool_stats[venue] = (m * 6.0, sd * math.sqrt(6.0))
    fitted = {}
    for mint in cand:
        pts = sorted(series[mint])
        x0 = pts[-1][1]
        venue = pts[-1][3]
        prior_mu = DRIFT_PRIOR.get(venue, 0.0)
        prior_sigma = SIGMA_PRIOR.get(venue, 0.35)
        if venue in pool_stats:
            prior_mu = max(-0.05, min(0.05, pool_stats[venue][0]))
            prior_sigma = pool_stats[venue][1]
        if len(pts) >= 2:
            n = len(pts)
            mt = sum(p[0] for p in pts) / n
            mx = sum(p[1] for p in pts) / n
            num = sum((p[0] - mt) * (p[1] - mx) for p in pts)
            den = sum((p[0] - mt) ** 2 for p in pts)
            slope = num / den if den > 0 else 0.0
            mu = DRIFT_SHRINK * slope + (1.0 - DRIFT_SHRINK) * prior_mu
            if n >= 6 and den > 0:
                resid = [p[1] - (mx + slope * (p[0] - mt)) for p in pts]
                var = sum(r * r for r in resid) / max(1, n - 2)
                own_sigma = math.sqrt(var) * math.sqrt(6.0)
                sigma = 0.5 * own_sigma + 0.5 * prior_sigma
            else:
                sigma = prior_sigma
        else:
            mu = prior_mu
            sigma = prior_sigma
        mu = max(-DRIFT_CLAMP, min(DRIFT_CLAMP, mu))
        sigma = max(SIGMA_CLAMP[0], min(SIGMA_CLAMP[1], sigma))
        fitted[mint] = (x0, mu, sigma, HAZARD.get(venue, 0.15))

    # newcomer cap centre: log10 cap of the current #1 coin
    ref_cap = log10_sol(last[0]["market_cap_lamports"])

    rng = random.Random(SEED)
    # place_counts[k][mint] and size_counts[k][bin], over sims x reads
    place_counts = [dict() for _ in range(places)]
    size_counts = [dict() for _ in range(places)]
    total_draws = 0
    for h in horizons:
        he = min(h, H_EFF_CAP)
        sq = math.sqrt(he)
        for _ in range(N_SIMS):
            total_draws += 1
            scored = []  # (log10 cap, mint or None for newcomer)
            for mint in cand:
                x0, mu, sigma, haz = fitted[mint]
                if rng.random() < 1.0 - math.exp(-haz * h):
                    continue  # died before this read
                x = x0 + mu * he + sigma * sq * rng.gauss(0.0, 1.0)
                scored.append((x, mint))
            n_new = 0
            # Poisson draw by Knuth's method (lam small)
            lam = NEWCOMER_LAM
            l = math.exp(-lam)
            p = 1.0
            while True:
                p *= rng.random()
                if p <= l:
                    break
                n_new += 1
            for _ in range(n_new):
                scored.append((ref_cap + NEWCOMER_SD * rng.gauss(0.0, 1.0), None))
            scored.sort(key=lambda kv: -kv[0])
            for k in range(places):
                if k < len(scored):
                    x, mint = scored[k]
                    if mint is not None:
                        d = place_counts[k]
                        d[mint] = d.get(mint, 0) + 1
                    b = 0
                    while b < len(edges) and x >= edges[b]:
                        b += 1
                    d = size_counts[k]
                    d[b] = d.get(b, 0) + 1
                # else: no coin at this place (cannot happen: newcomers fill)

    # --- build the forecast
    probabilities = []
    for k in range(places):
        rows = sorted(place_counts[k].items(), key=lambda kv: -kv[1])
        s = 0.0
        for mint, c in rows:
            p = c / total_draws
            if p < P_MIN:
                continue
            p = math.floor(p * 1e6) / 1e6
            if p <= 0:
                continue
            s += p
            if s > 1.0:
                break
            probabilities.append({"mint": mint, "place": k + 1, "p": p})
    sizes = []
    for k in range(places):
        rows = sorted(size_counts[k].items())
        s = 0.0
        for b, c in rows:
            p = c / total_draws
            if p < SIZE_P_MIN:
                continue
            p = math.floor(p * 1e6) / 1e6
            if p <= 0:
                continue
            s += p
            if s > 1.0:
                break
            sizes.append({"place": k + 1, "bin": int(b), "p": p})
    # top5: most total place mass, weighted toward the top places
    mass = {}
    for k in range(places):
        for mint, c in place_counts[k].items():
            mass[mint] = mass.get(mint, 0.0) + (places + 1 - (k + 1)) * c
    top5 = [m for m, _ in sorted(mass.items(), key=lambda kv: -kv[1])[:places]]
    # top5 must be exactly `places` distinct mints seen in the history
    if len(top5) < places:
        for r in last:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) == places:
                break

    doc = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
