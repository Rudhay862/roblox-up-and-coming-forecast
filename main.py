#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Model: Monte Carlo over latent "growth scores" (score = -rank at the cut).
Each game keeps a score that drifts (momentum + lifecycle ageing) and jitters
each hour; new games enter at the bottom of the chart and a few climb fast.
At every simulated hour the games are ranked by score and the top 10 recorded.
The one forecast is the average over the 72 scored hours, because the same
forecast is scored at every hourly read of the window.

Calibration (from the kit's dev history, 34 reads, 2026-10-30..11-07):
- The chart churns hard: #1 changed 13 times in 45 reads; the top 10 changes
  a median of 6 places between reads; top-10 survival is 87% after ~6h,
  79% after ~13h, 65% after ~26h, 47% after ~52h. Persistence is weak.
- Rank velocity has weak momentum (corr 0.22 past -> next velocity).
- Ageing is the strongest signal: games drift down faster the longer they have
  been on the chart (mean rank velocity +0.10/h at age 0-24h, +0.35/h at
  24-72h, +0.48/h at 72-150h, +0.79/h past 150h).
- New games enter at median rank 88, ~0.8/hour; games that reach the top 10
  enter at median rank 55 and take ~33h to climb there.
- Player counts do NOT predict rank moves (corr -0.015), so they are ignored.

Stdlib only, deterministic (fixed seed), no network.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261113
N_PATHS = 1000

# --- calibration constants (see header) ---
# AGE_DRIFT_SLOPE was tuned on two backtest splits (history 24->10 reads and
# 20->14 reads): 0.011 beat 0.0037/0.005/0.007/0.009/0.013 on both. The slope
# is steeper than the raw linear fit (0.0037) because first-seen age
# underestimates true age, so the steeper slope corrects for it.
AGE_DRIFT0 = 0.06        # places/hour worsening at age 0
AGE_DRIFT_SLOPE = 0.011  # extra places/hour worsening per hour of age
AGE_DRIFT_CAP = 300.0     # hours of age after which the drift stops growing
MOMENTUM = 0.15           # weight on a game's recent rank velocity
VEL_CLIP_LO = -2.5        # fastest climb, places/hour
VEL_CLIP_HI = 2.0         # fastest decline, places/hour
SIGMA_TOP = 1.3           # score noise per hour for a top-20 game
SIGMA_BOTTOM = 3.0        # score noise per hour for a bottom game
ENTRY_PER_HOUR = 1.2      # new games entering the chart (measured 0.8; 1.2 won both splits)
ENTRY_RANK_MEAN = 88.0    # rank an ordinary new game enters at
ENTRY_RANK_SD = 8.0
CONTENDER_RANK_MEAN = 55.0  # rank a future top-10 game enters at
CONTENDER_RANK_SD = 12.0
ENTRY_RANK_LO = 35.0
ENTRY_RANK_HI = 97.0
CLIMBER_P = 0.15          # share of entrants that climb fast (future top 10)
CLIMBER_BOOST = 2.0       # extra score drift (places/hour) for a climber
CLIMBER_HALF_LIFE_H = 30.0
# A climber's boost fades as it nears the top: full at rank 60, ~0 at rank 5.
CLIMBER_FADE_TOP = 5.0
CLIMBER_FADE_BOTTOM = 60.0
EXIT_SCORE = -160.0       # score below which a game is dropped from the pool
CHART_SIZE_FALLBACK = 97


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json")))
    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    return task, snaps


def build_games(snaps):
    """Per-game trajectories from the snapshots.

    Returns a list of dicts: uid, obs [(t_hours, rank)], first_seen_t,
    created_t (or None), last_rank, last_t.
    Times are hours since the first snapshot.
    """
    t0 = parse_time(snaps[0]["at"])
    games = {}
    for s in snaps:
        t = (parse_time(s["at"]) - t0).total_seconds() / 3600.0
        details = s.get("details") or {}
        for r in s.get("up_and_coming") or []:
            uid = r["universe_id"]
            g = games.get(uid)
            if g is None:
                g = games[uid] = {
                    "uid": uid,
                    "obs": [],
                    "first_seen_t": t,
                    "created_t": None,
                    "last_rank": r["rank"],
                }
            g["obs"].append((t, r["rank"]))
            g["last_rank"] = r["rank"]
            # created date, when the snapshot carries details
            det = None
            if isinstance(details, dict):
                det = details.get(str(uid)) or details.get(uid)
            if isinstance(det, dict) and det.get("created"):
                try:
                    g["created_t"] = (parse_time(det["created"]) - t0).total_seconds() / 3600.0
                except Exception:
                    pass
    last_t = (parse_time(snaps[-1]["at"]) - t0).total_seconds() / 3600.0
    return games, last_t


def recent_velocity(obs, last_t):
    """Shrunk slope of rank vs time (places/hour, negative = climbing).

    Weighted least squares over the last 96h, recency-weighted; shrunk toward
    0 by the observation count because single-snapshot velocities are noisy.
    """
    pts = [(t, r) for (t, r) in obs if last_t - t <= 96.0]
    if len(pts) < 2:
        return 0.0
    wsum = 0.0
    mt = 0.0
    mr = 0.0
    for (t, r) in pts:
        w = math.exp(-(last_t - t) / 24.0)
        wsum += w
        mt += w * t
        mr += w * r
    mt /= wsum
    mr /= wsum
    num = 0.0
    den = 0.0
    for (t, r) in pts:
        w = math.exp(-(last_t - t) / 24.0)
        num += w * (t - mt) * (r - mr)
        den += w * (t - mt) * (t - mt)
    if den <= 0:
        return 0.0
    v = num / den
    n = len(pts)
    return v * n / (n + 3.0)


def main():
    task, snaps = load_history()
    places = int(task.get("places", 10))
    window = task.get("window", {})
    hours = int(window.get("slots", 72))
    out_name = task.get("output", "forecast.json")

    if not snaps:
        # Nothing to forecast from; emit an empty but well-formed file.
        with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
            f.write(json.dumps({"probabilities": [], "top10": []}) + "\n")
        return

    games, last_t = build_games(snaps)
    last_chart = snaps[-1].get("up_and_coming") or []
    chart_size = len(last_chart) if last_chart else CHART_SIZE_FALLBACK

    # --- per-game drift and noise, in score units (score = -rank) ---
    init = []
    for g in games.values():
        age_h = last_t - (g["created_t"] if g["created_t"] is not None
                          else g["first_seen_t"])
        age_h = max(0.0, age_h)
        age_v = AGE_DRIFT0 + AGE_DRIFT_SLOPE * min(age_h, AGE_DRIFT_CAP)
        v = age_v + MOMENTUM * recent_velocity(g["obs"], last_t)
        # Games within a day of the 28-day eligibility end fall off fast.
        if g["created_t"] is not None and age_h > 26.0 * 24.0:
            v += 2.0
        v = max(VEL_CLIP_LO, min(VEL_CLIP_HI, v))
        rank = g["last_rank"]
        sigma = SIGMA_TOP + (SIGMA_BOTTOM - SIGMA_TOP) * (rank / 100.0)
        init.append({
            "uid": g["uid"],
            "score": -float(rank),
            "drift": -v,          # score drift: positive = climbing
            "sigma": sigma,
            "age_h": age_h,
        })

    rng = random.Random(SEED)
    gauss = rng.gauss
    rand = rng.random

    # counts[place][uid] and entrant_counts[place]
    counts = [dict() for _ in range(places)]
    entrant_counts = [0] * places
    total_slots = N_PATHS * hours

    for _ in range(N_PATHS):
        scores = [g["score"] for g in init]
        drifts = [g["drift"] for g in init]
        sigmas = [g["sigma"] for g in init]
        uids = [g["uid"] for g in init]
        extra = [0.0] * len(init)   # climber boost, decays with half-life
        decay = 0.5 ** (1.0 / CLIMBER_HALF_LIFE_H)
        for _h in range(hours):
            # one hourly step
            for i in range(len(scores)):
                b = extra[i]
                if b:
                    # climber boost fades near the top (score ~ -rank)
                    f = (-scores[i] - CLIMBER_FADE_TOP) / (CLIMBER_FADE_BOTTOM - CLIMBER_FADE_TOP)
                    if f <= 0.0:
                        b = 0.0
                    elif f < 1.0:
                        b *= f
                    extra[i] *= decay
                scores[i] += drifts[i] + b + sigmas[i] * gauss(0.0, 1.0)
            # new games enter the chart
            n_new = 0
            x = rand()
            if x < ENTRY_PER_HOUR:
                n_new = 1
                if rand() < 0.05:
                    n_new = 2
            for _ in range(n_new):
                if rand() < CLIMBER_P:
                    er = gauss(CONTENDER_RANK_MEAN, CONTENDER_RANK_SD)
                    boost = CLIMBER_BOOST
                else:
                    er = gauss(ENTRY_RANK_MEAN, ENTRY_RANK_SD)
                    boost = 0.0
                er = max(ENTRY_RANK_LO, min(ENTRY_RANK_HI, er))
                scores.append(-er)
                drifts.append(-AGE_DRIFT0)
                sigmas.append(2.5)
                uids.append(-1)  # entrant: cannot be named, stays unnamed
                extra.append(boost)
            # rank and record the top 10
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for p in range(min(places, len(order))):
                uid = uids[order[p]]
                if uid == -1:
                    entrant_counts[p] += 1
                else:
                    d = counts[p]
                    d[uid] = d.get(uid, 0) + 1
            # drop games far below the chart to bound the pool
            if len(scores) > 220:
                keep = [i for i in range(len(scores)) if scores[i] > EXIT_SCORE]
                if len(keep) < len(scores):
                    scores = [scores[i] for i in keep]
                    drifts = [drifts[i] for i in keep]
                    sigmas = [sigmas[i] for i in keep]
                    uids = [uids[i] for i in keep]
                    extra = [extra[i] for i in keep]

    # --- build the forecast ---
    # unnamed share per place = probability an entrant (a game not in the
    # history) holds the place. It stays unnamed by construction.
    probabilities = []
    total_mass = {}   # uid -> sum of p over places (for top10 ordering)
    expected_place = {}  # uid -> sum of p * place
    for p in range(places):
        unnamed = entrant_counts[p] / total_slots
        cand = sorted(counts[p].items(), key=lambda kv: -kv[1])
        named = []  # (uid, p)
        s_named = 0.0
        k_named = 0
        for uid, c in cand:
            prob = c / total_slots
            # Name it only if it beats what the game would get unnamed:
            # the remaining unnamed share spread over the chart's unnamed games.
            per_game_unnamed = (1.0 - s_named) / max(1, chart_size - k_named)
            if prob <= per_game_unnamed:
                break
            named.append((uid, prob))
            s_named += prob
            k_named += 1
        for uid, prob in named:
            p6 = math.floor(prob * 1_000_000) / 1_000_000
            if p6 <= 0:
                continue
            probabilities.append({"universe_id": uid, "place": p + 1, "p": p6})
            total_mass[uid] = total_mass.get(uid, 0.0) + p6
            expected_place[uid] = expected_place.get(uid, 0.0) + p6 * (p + 1)
        # (the unnamed share, entrant probability included, is left unnamed)

    # top10: most total top-10 mass first, ties to the better expected place
    ranked = sorted(total_mass.items(),
                    key=lambda kv: (-kv[1], expected_place.get(kv[0], 99.0), kv[0]))
    top10 = [uid for uid, _ in ranked[:places]]
    # Fill from the last snapshot's chart if fewer than `places` games qualified.
    if len(top10) < places:
        seen = set(top10)
        for r in sorted(last_chart, key=lambda r: r["rank"]):
            if r["universe_id"] not in seen:
                top10.append(r["universe_id"])
                seen.add(r["universe_id"])
            if len(top10) >= places:
                break
    top10 = top10[:places]

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    out = {"probabilities": probabilities, "top10": top10}
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
