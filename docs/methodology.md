# T20X Rating Methodology

## Overview

The primary rating signal in T20X is **Win Probability Added (WPA)** — the cricket analog of baseball's WPA. For every delivery in every match, we compute the change in win probability and attribute it to the batter (positive) and bowler (negative). Aggregated across a career and normalized against a replacement-level baseline, this becomes **Wins Above Replacement (WAR)**.

Per-delivery **Elo** and **Bradley–Terry** ratings remain in the codebase as secondary signals — useful as sanity references and for visualizing temporal trajectories, but **demoted** from primary rankings after they failed face-validity tests (see *History of the rating algorithm* at the bottom of this document).

---

## Why traditional batting stats are misleading

Average and strike rate, in isolation, distort player value in T20:

- A batter averaging **45 at SR 115** is *actively hurting* their team — eating balls at a sub-par scoring rate.
- A batter averaging **22 at SR 165** is highly valuable — every ball they face is a positive expected-runs contribution.

Three failure modes drive the distortion:

1. **No pitch / venue / situation adjustment.** SR 140 on a Bengaluru flat track is below par. SR 130 on a Chennai turner is match-winning.
2. **No phase isolation.** A middle-order batter's powerplay numbers (often as nightwatchman or pinch-hitter) shouldn't pollute their middle-overs evaluation. The three phases are functionally three different games.
3. **No outcome decomposition.** 30 off 20 with 5 sixes and 15 dots vs. 30 off 20 via constant strike rotation are completely different profiles, both for the batter and for the partnership they enable. Boundary % and dot % matter more than runs/balls alone.

T20X addresses all three by working at the **per-delivery level**, with context-aware expected baselines (xR), independent phase models, and outcome-class probabilities that natively expose dot and boundary rates.

### Runs Above Average (RAA) — the match-level analog

A simple but powerful decomposition of batter contribution:

$$\text{RAA} = \text{Runs Scored} - \left( \text{Balls Faced} \times \frac{\text{Total Match Runs}}{\text{Total Match Balls}} \right)$$

This subtracts off the **match-baseline runs per ball** — a batter is only credited for runs above what an average batter would have scored on the same surface, against the same attack, in the same match.

RAA generalizes naturally to **per-ball xR-residual**:

$$\text{RAA}_{\text{ball}} = \text{Actual Runs}_i - \widehat{xR}_i(\text{context})$$

The match-level RAA uses the *match RPB* as a single baseline. xR uses a context-aware baseline per delivery — phase, over, wickets fallen, required rate, and (eventually) bowler quality. Same concept; xR is the contextual generalization.

T20X surfaces both:

- **RAA per match / season** as a quick-read summary stat
- **xR-residual cumulative** as the rating-driver

---

## Why traditional bowling stats are also misleading

Wickets and economy alone are similarly distorted:

- Three wickets cleaning up tailenders in the 20th over of a chase already won looks identical on a scorecard to three top-order wickets in the powerplay — but the latter is **several wins worth** of value, the former is essentially zero.
- Conceding 8 in the 19th over is a phenomenal achievement (death-overs economy is typically 10+); conceding 8 in the 8th over is below the middle-overs baseline (~7.5).

Three metrics fix this:

### Contextual Economy Rate (CERA)

For each over a bowler bowls, compare actual runs conceded against the **expected runs for that over number, phase, and innings**, computed from the league baseline:

$$\text{CERA}_{\text{over}} = \text{Expected RPO}_{\text{phase, over, innings}} - \text{Actual RPO}$$

Positive = better than baseline. Aggregated across a season, this is the bowler's mirror of batter RAA — and it falls out of the same xR model: `Σ_over (xR_runs − actual_runs)`.

### Dot Ball Frequency

Dot % is the ultimate pressure metric in T20 — it correlates strongly with wickets at the *other* end (forcing the next batter to take risks). In T20X this is **natively exposed** by the outcome-class model: the predicted distribution over `{0, 1, 2, 3, 4, 6, W}` directly gives `P(dot)`, both actual and expected.

We track:

- **Actual dot %** — straightforward
- **xR-baseline dot %** — what a league-average bowler would have produced in the same situations
- **Dot % above expected** — the bowler's true dot-creation ability

### Impact Wickets

A wicket is not a wicket. Removing the opposition's #1 batter in the powerplay is fundamentally different from removing #10 in the 18th over. Weighting each dismissal by **the batter's current rating** transforms the wickets column into something meaningful:

$$\text{Impact Wickets} = \sum_{\text{wickets}} \left( \frac{R_{\text{batter dismissed}}}{R_{\text{avg}}} \right)$$

This creates a positive feedback loop with the rating system: the more accurately we rate batters, the more accurately we credit bowlers who dismiss them — and vice versa. It's the same circular dependency that the Bradley–Terry pass already resolves, just exposed as a summary stat.

---

## Fielding — the inefficiency frontier

If on-base percentage was the undervalued metric in 2002 baseball, **fielding is the undervalued asset in T20 cricket today**. Because run-saving and catch difficulty don't show up in batting or bowling columns, franchise auctions persistently underprice elite fielders.

The full Statcast-style fielding model — **Runs Saved** from a 5m fielder radius using ball-tracking, **Catch Probability** by trajectory and reaction time — is the right destination. But:

> ⚠️ **Cricsheet does not ship fielder positions, ball-tracking, or trajectory data.** Only dismissal records (catcher, run-out fielder, stumping) and wicketkeeper attribution.

So T20X tackles fielding in two tiers:

### Tier 1 — what we can do with Cricsheet today

From dismissal records and aggregate boundary data:

- **Catch conversion rate** — caught dismissals per delivery fielded (very rough; relies on identifying which non-bowler fielder was involved)
- **Run-out involvement rate**
- **Boundary suppression proxy** — for grounds where the same fielders play many matches, infer which fielders are at the boundary in low-4% innings

These are crude but reveal extreme outliers (Pollard, Pant, Jadeja).

### Tier 2 — when richer data arrives

Integrating ball-tracking from Hawk-Eye or IPL's official feeds (likely paid / scraping) would unlock:

- **Runs Saved per chance** — actual fielder GPS position vs expected runs conceded if ball were unfielded
- **Catch Probability** — classifier on `{distance run, reaction time, trajectory, ball speed}` → catch likelihood. A fielder's value is then their conversion rate on **hard** chances, not raw catches.
- **Diving range / arm strength** — separable from catching ability

This is a Phase 6+ data-acquisition problem, not a modeling problem. Flagged here so the design doesn't accidentally bake batter-only or bowler-only assumptions into the WAR composite.

### Where fielding lands in WAR

Even with crude Tier 1 metrics:

```
Total_WAR = Batting_WAR + Bowling_WAR + Fielding_WAR
```

Fielding_WAR will be small in early versions (low signal-to-noise from Cricsheet alone) but is explicitly kept as a first-class term so the framework doesn't need re-architecting when better data lands.

---

## Win Probability Added (WPA) — primary rating

### The model

A classifier predicts `P(batting_team_wins | match state)`. Features are situation-only — no player identity, no team identity:

| Feature | Notes |
|---------|-------|
| `innings` | 1 or 2 |
| `balls_remaining` | 0–120, before the delivery |
| `wickets_in_hand` | 10 − wickets fallen |
| `runs_so_far` | cumulative runs |
| `target_remaining` | NaN for innings 1 |
| `required_rate` | NaN for innings 1 |

Trained on every delivery from completed matches in the dataset (target = did the batting team in this innings win?). The classifier is `HistGradientBoostingClassifier` with NaN-aware splits, so innings-1 and innings-2 features are handled jointly.

> Calibration check on IPL: `WP_pre.mean() = 0.493`, batting team wins 48.9% of all balls — exactly aligned. Δ_WP mean = −0.0041 per ball (essentially zero), std = 0.032.

### Per-delivery attribution

```
Δ_WP = WP(post_state) − WP(pre_state)
batter += Δ_WP        (positive = batter helped their team)
bowler -= Δ_WP        (positive = bowler helped their team)
```

This naturally captures everything outcome scores and xR-residual missed:

- **Clutch context** — a single in the 19th of a tight chase swings WP ~5%; a single in the 6th of a routine chase swings ~0.2%.
- **Wicket value** — a wicket falling at 30/0 in the powerplay costs ~3% WP; the same wicket at 80/4 chasing 180 costs ~15%.
- **Wicket avoidance** — surviving a ball in pressure is *positive* Δ_WP, even at zero runs. This is what rescues anchor batters from xR-residual's blind spot.

### From WPA to WAR

WPA is the career sum of Δ_WP. WAR normalizes against a replacement-level baseline:

```
replacement_per_ball = 25th percentile of (WPA / balls) among players with ≥ 500 balls
WAR = WPA − replacement_per_ball × balls
```

A player who just barely clears replacement-level on a per-ball basis but plays a huge volume can accumulate substantial WAR. **This is the insight that fixes anchor batters.**

### Worked example: Kohli vs Gayle in IPL

| Player | Balls | WPA (raw) | WPA/ball | WAR |
|--------|------:|----------:|---------:|----:|
| CH Gayle  | 3,420 | **+4.82** | +0.00141 | +23.68 |
| V Kohli   | 6,653 | **−19.49** | −0.00293 | +17.19 |

Gayle has positive raw WPA — his per-ball impact is above the league average. Kohli's raw WPA is *negative* (the average batter would have done slightly better per ball). But replacement-level is −0.00551 WPA/ball — much worse than Kohli. Over 6,653 balls, Kohli clears replacement by 17.2 wins.

**The volume of "staying in" is itself value.** A replacement batter wouldn't have lasted 6,653 balls.

### Known limitation: dot-ball specialists

WPA undervalues bowlers whose primary contribution is "preventing scoring without taking wickets." Rashid Khan, on IPL: WPA +8.61 over 3,250 balls = +0.00265/ball — *below* the +0.00329/ball replacement, so his WAR is mildly negative.

Why: in middle overs, a dot ball barely moves WP because the state hasn't changed (one fewer ball, same wickets, same runs). A bowler who specializes in middle-overs economic stranglehold doesn't generate big Δ_WP swings. The "induced pressure → wicket at the other end" effect — real and well-known in cricket — accrues to the *other* bowler in our framework.

This is the same class of issue baseball's WPA has with defensive specialists. Documented; not patched. Future work: introduce a "pressure index" feature into WP that lets us credit sustained low-scoring periods.

### Implementation

[`src/t20x/ratings/win_probability.py`](../src/t20x/ratings/win_probability.py) — `WinProbabilityModel`, `compute_wpa`, `compute_war`.

---

## Legacy: Elo + Bradley–Terry (secondary signals)

What follows below remains in the codebase but is **not** how T20X ranks players. Treat these as visualization signals (Elo for time-series trajectories, BT for pairwise matchup probabilities) and as historical context for the rating-engine evolution.

---

## Step 1 — Outcome score mapping

Each delivery's result is mapped to a score in `[0, 1]` from the batter's perspective:

| Outcome | Score |
|---------|------:|
| Wicket  | 0.0 |
| Dot     | 0.1 |
| 1 run   | 0.3 |
| 2 runs  | 0.5 |
| 3 runs  | 0.6 |
| 4 runs  | 0.8 |
| 6 runs  | 1.0 |
| Wide / No-ball | 0.2 |

The bowler's score is the complement (`1 − S_bat`), making each ball strictly zero-sum.

> ⚠️ **Known calibration issue:** the empirical mean of this mapping on real T20 data is ≈ 0.25, not 0.5. Batters drift below 1500 systematically and dot-tolerant accumulators (Kohli, Rohit) are buried behind strike-rate-heavy power hitters. The xR upgrade below fixes this.

---

## Step 2 — Per-delivery Elo

For each ball, in strict chronological order:

```
E_bat   = 1 / (1 + 10^((R_bowl − R_bat) / 400))
R_bat  ← R_bat  + K · (S_bat   − E_bat)
R_bowl ← R_bowl + K · ((1−S_bat) − (1−E_bat))
```

- Starting rating: **1500** for every player
- Scale factor: **400** (standard Elo)
- K-factor: **0.5** per ball (small, because the cumulative number of balls is in the millions)
- Optional **opponent-weighted K**: from epoch 2 onwards, K is scaled by the opponent's strength relative to the league mean — facing a top bowler amplifies the update

Implementation: [`src/t20x/ratings/elo.py`](../src/t20x/ratings/elo.py)

---

## Step 3 — Bradley–Terry global fit

Aggregate every batter–bowler pair into win and contest counts, then fit strengths `α_i` (batters) and `β_j` (bowlers) via iterative MLE:

```
P(batter dominates | i, j) = α_i / (α_i + β_j)
α_i ← Σⱼ wᵢⱼ / Σⱼ nᵢⱼ / (αᵢ + βⱼ)
```

- Players with fewer than **30 deliveries** in the slice are excluded
- After each iteration: normalize so the geometric mean of all strengths is 1
- Convergence: max change < 1e-6 or 100 iterations
- Convert to an Elo-comparable scale for blending: `rating = 1500 + 400 · log₁₀(strength)`

Why both Elo and BT? Elo is sequential and biased by who-you-faced-when. BT solves the whole pairwise graph simultaneously, so it neutralizes path dependence — at the cost of losing time order.

Implementation: [`src/t20x/ratings/bradley_terry.py`](../src/t20x/ratings/bradley_terry.py)

---

## Step 4 — Blend

```
rating = 0.6 · Elo + 0.4 · Bradley–Terry
```

- **Elo** contributes temporal sensitivity (form, recent peaks)
- **BT** contributes global consistency (calibration across the whole pool)

The weight (currently 60/40) is a hyperparameter and will be tuned via the validation framework once it exists.

---

## Step 5 — Iterate to convergence

Each epoch:

1. Elo sweep over all deliveries chronologically (initialized from the previous epoch's ratings)
2. Bradley–Terry refit on aggregate matchups
3. Blend
4. Check `max |new_rating − old_rating|` across all players with ≥ 100 deliveries

**Stopping criterion:** max change < **1.0** rating point, or 10 epochs.

Observed on IPL smoke run (282K deliveries, 1,188 matches): max change went 0.0 → 33.4 → 18.2 → 10.0 → 5.5 across 5 epochs — converging at roughly half per step.

Implementation: [`src/t20x/ratings/convergence.py`](../src/t20x/ratings/convergence.py)

---

## Phase-specific ratings

The same pipeline runs independently for each T20 phase:

- **Powerplay** — overs 1–6
- **Middle**    — overs 7–15
- **Death**     — overs 16–20

This produces separate `(player, phase)` ratings so we can ask questions like *"top death-overs bowler vs left-handed batters."*

---

## Micro-matchup ratings

Phase splits are the *first* axis of conditioning, not the last. A "Moneyball" T20 squad isn't built from generalists — it's built from specialists who dominate a specific micro-situation. You don't draft "a middle-order batter." You draft **a left-handed batter who averages 40+ at SR 150+ against right-arm leg-spin in overs 7–15**, and then you deploy them at the exact ball the opposition turns to their leg-spinner.

T20X enables this by producing **conditional ratings** along multiple axes:

### Conditioning axes

| Axis | Levels |
|------|--------|
| **Phase** | powerplay / middle / death |
| **Bowling type** (for batters) | right-arm pace / left-arm pace / right-arm off-spin / right-arm leg-spin / left-arm orthodox / left-arm wrist-spin |
| **Batting hand** (for bowlers) | vs right-hand bat / vs left-hand bat |
| **Innings** | batting first / chasing |
| **Match state** *(future)* | pressure index buckets — `runs needed`, `wickets in hand`, `balls remaining` |
| **Venue / pitch type** *(future)* | flat / spinning / seaming, inferred from match-aggregate baselines |

Each `(player, axis-combination)` cell gets its own rating. A batter's profile becomes a small matrix:

```
                  PP    Mid   Death
vs right-pace    1572  1605  1488
vs left-pace     1550  1622  1502
vs off-spin       --   1550   --
vs leg-spin       --   1485   --
vs left-orthodox  --   1531   --
```

(Cells marked `--` are too sparse to rate at the cut-off threshold and need shrinkage from broader cells.)

### Sparsity is the real problem

Slicing by phase × bowling type × batting hand creates 36+ cells per player. Most are sparse. Mishandled, this collapses into noise — a batter with 12 balls against right-arm leg-spin in the death will get a wild rating either way.

Two mitigations:

1. **Hierarchical shrinkage.** Each fine-grained cell is partially pooled toward broader cells: `R(batter, middle, leg-spin) = w · R̂_cell + (1−w) · R(batter, middle, all-spin)`, with `w` scaling with sample size. This is the standard Bayesian hierarchical approach (think baseball's Empirical Bayes for batting splits).
2. **Minimum-deliveries gating.** Cells under a threshold (e.g., 60 balls) display as "insufficient data" rather than a spurious number. The recommender treats them as the parent-cell rating with extra uncertainty.

### Data prerequisite: bowler type taxonomy

Micro-matchups require **classifying every bowler's type**, which is *not* in Cricsheet ball-by-ball data. This is what Phase 2 (player metadata enrichment via ESPNCricinfo) unblocks. Until that runs:

- We have **batting hand** (in Cricsheet)
- We do **not** have **bowling type** at the resolution needed (off-spin vs leg-spin, etc.)

So fine-grained micro-matchups are gated on Phase 2. Phase-only splits work today.

### How the recommender uses this (Phase 5)

The downstream comparison engine asks queries like:

```
t20x recommend --situation "middle overs, need wicket, opposition has 2 lefties set"
```

…and ranks players by their **conditional** rating in the matching cell, not their overall rating. This is the operational point of the whole exercise: a player whose overall rating is mediocre can still be the **best in the world** for a specific micro-moment, and that's the inefficiency a smart franchise exploits.

---

---

## History of the rating algorithm

This document went through three rating designs before settling on WPA/WAR as the primary signal. The history is documented because each design exposed a real and instructive failure mode.

### v1 — Outcome-score Elo

Map each delivery to a `[0, 1]` score (dot = 0.1, six = 1.0, wicket = 0.0), run zero-sum Elo updates. Smoke test on IPL: top-15 bowlers excellent (Bumrah, Narine, Rashid, Archer); top-15 batters wrong (Salt #1 at 1494.8 — *below* the 1500 starting line, Kohli #36, Rohit #73).

**Failure mode:** the outcome map has empirical mean ≈ 0.25 per ball in T20, not 0.5. The "zero-sum" game is structurally biased toward bowlers, so batter ratings systematically drift down and the system rewards strike-rate over durability.

### v2 — xR-residual Elo

Train a context-only xR classifier (`HistGradientBoostingClassifier`) over outcome classes `{0, 1, 2, 3, 4, 6, W}`, replace the rating-diff expected score with `xR_score = Σ P(outcome) · score(outcome)`. Elo update becomes `R += K · (actual − xR)`.

Smoke test: xR calibration excellent (expected mean 0.330 vs actual 0.326). Bowlers stayed clean. Batters: power hitters surged correctly (ABdV #1, SKY #2, Russell #3) but **anchor batters collapsed** — Kohli fell to #135, Dhoni #133, Rohit #114.

**Failure mode:** xR conditions on `innings_runs` and `wickets_fallen`, so an anchor batter who *creates* a favorable state then has to beat the elevated baseline he himself caused. xR-residual can't see the value of balls-survived. This is a real and known tension — residual-based methods reward rate exceedance, not occupancy.

### v3 — Win Probability Added (current)

Train a WP classifier on match outcomes, compute per-delivery Δ_WP, attribute to batter/bowler, aggregate to career WPA, normalize against replacement-level → WAR.

Smoke test: Kohli #4, Rohit #29, Dhoni #95, Gayle #1, ABdV #24, Bumrah #6 bowler. Face validity passes for the first time across the full distribution, including anchors.

The Elo + Bradley–Terry machinery from v1 remains in the codebase because it's still useful as:
- A **temporal-trajectory visualizer** (Elo gives you rating-over-time charts; WPA is a cumulative scalar)
- A **pairwise matchup probability estimator** (BT gives you `P(batter A vs bowler B)` which WPA does not)

Both are computed alongside WPA but neither drives the primary rankings.

---

## Future — Cricket WAR composite

Player WAR currently comes from career WPA against replacement-level batters/bowlers. The full composite metric will extend this:

```
Total_WAR = Batting_WAR + Bowling_WAR + Fielding_WAR
```

- **Batting_WAR** = WPA-derived (current implementation)
- **Bowling_WAR** = WPA-derived (current implementation)
- **Fielding_WAR** = Tier 1 from dismissal-record attribution; Tier 2 once ball-tracking data is integrated (see Fielding section above)

The WPA-based formulation is a direct improvement on the original `(Actual_Runs − xR_Runs) / Runs_Per_Win` framing — WPA already accounts for context, clutch, and innings state, so we don't need a separate runs-per-win conversion. WAR is already denominated in wins.

---

## Validation (not yet built — planned)

Before any rating ships as "real," it has to pass:

| Check | What it tests | Target |
|-------|---------------|--------|
| **Holdout prediction** | Train on 2008–2023, predict 2024–2025 match outcomes | Beat naive baseline on log-loss & Brier |
| **Calibration plot** | Predicted vs actual outcome frequency by decile | Diagonal |
| **Rank correlation** | Spearman vs ICC T20I rankings | > 0.6 |
| **Convergence trace** | Rating changes per epoch | Monotone-decreasing |
| **Sensitivity sweep** | K-factor, blend weights, min-deliveries | Top-10 stable |
| **Cross-league**     | Train IPL, predict BBL (and reverse) | Generalizes |
| **Face validity**    | Top-10 lists vs expert consensus | No obvious howlers |

---

## References

- Bradley–Terry in cricket — [Cardiff University paper](https://orca.cardiff.ac.uk/id/eprint/107993/1/BT.pdf)
- Bayesian batting survival — [arXiv:1609.04078](https://arxiv.org/pdf/1609.04078)
- Expected Runs — [Cricket Savant](https://cricketsavant.wordpress.com/2016/12/26/introducing-expected-runs/)
- WASP — [Wikipedia](https://en.wikipedia.org/wiki/WASP_(cricket_calculation_tool))
- Baseball WAR — [Wikipedia](https://en.wikipedia.org/wiki/Wins_above_replacement)
- Football xG + Elo — [xGELO](https://medium.com/@worville/xgelo-combining-expected-goals-and-elo-ratings-6aa987481479)
