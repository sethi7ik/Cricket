# T20X: T20 Cricket Analytics Package — Implementation Plan

## Vision

A Python package for comparing and ranking T20 cricket players across leagues (IPL, BBL, PSL, CPL, T20I) using opponent-quality-adjusted, situation-specific metrics. Goes beyond batting averages and strike rates to provide context-aware player evaluation.

**Core insight**: A batsman's true ability depends on the quality of bowlers faced, and vice versa. This circular dependency requires an iterative rating algorithm that alternately refines batsman and bowler ratings until convergence.

**Inspiration**: Baseball WAR/wOBA, football xG/Dixon-Coles/Elo, cricket Bradley-Terry models, Bayesian hierarchical methods.

---

## Package Structure

```
src/t20x/
  __init__.py, __main__.py, config.py, constants.py
  db/           — DuckDB schema, engine, migrations, queries
  ingest/       — Cricsheet parser, Kaggle adapter, ESPN scraper, player registry, enrichment
  models/       — Pydantic domain objects (Match, Delivery, Player), enums, context
  ratings/      — Elo engine, Bradley-Terry, Bayesian, convergence orchestrator, xR model
  metrics/      — WAR, phase splits, matchups, situation-weighted, opposition-adjusted
  compare/      — Ranker, recommender, player similarity
  cli/          — Typer CLI app with ingest/ratings/compare/rank/query commands
  api/          — FastAPI REST endpoints (Phase 6)
tests/          — Unit + integration tests
notebooks/      — Exploratory Jupyter notebooks
data/           — .gitignored local data storage
```

---

## Data Schema (DuckDB)

### matches
| Column | Type | Description |
|--------|------|-------------|
| match_id | VARCHAR PK | Cricsheet numeric ID |
| league | VARCHAR | IPL, BBL, PSL, CPL, T20I, etc. |
| season | VARCHAR | e.g., "2023/24" |
| date | DATE | Match date |
| venue | VARCHAR | Ground name |
| city | VARCHAR | City |
| team_1, team_2 | VARCHAR | Team names |
| toss_winner | VARCHAR | |
| toss_decision | VARCHAR | bat / field |
| winner | VARCHAR | |
| win_margin | INTEGER | |
| win_type | VARCHAR | runs / wickets |
| player_of_match | VARCHAR | |
| gender | VARCHAR | male / female |

### players
| Column | Type | Description |
|--------|------|-------------|
| player_id | VARCHAR PK | Cricsheet registry UUID |
| name | VARCHAR | Display name |
| full_name | VARCHAR | Full name |
| dob | DATE | Date of birth |
| batting_style | VARCHAR | right-hand bat / left-hand bat |
| bowling_type | VARCHAR | Composite type |
| bowling_arm | VARCHAR | right / left |
| bowling_pace | VARCHAR | fast / medium / spin |
| bowling_style | VARCHAR | off-break, leg-break, orthodox, wrist, seam, swing |
| primary_role | VARCHAR | batsman / bowler / allrounder / wicketkeeper |
| country | VARCHAR | |
| espn_id | INTEGER | ESPNCricinfo player ID |

### deliveries (central fact table, ~2-3M rows)
| Column | Type | Description |
|--------|------|-------------|
| match_id | VARCHAR | FK → matches |
| innings | SMALLINT | 1 or 2 |
| over_number | SMALLINT | 0-19 |
| ball_number | SMALLINT | Within over |
| delivery_seq | INTEGER | Global sequence within innings |
| batter_id | VARCHAR | FK → players |
| bowler_id | VARCHAR | FK → players |
| non_striker_id | VARCHAR | FK → players |
| batter_runs | SMALLINT | |
| extra_runs | SMALLINT | |
| total_runs | SMALLINT | |
| is_wide, is_noball, is_bye, is_legbye | BOOLEAN | |
| is_boundary_four, is_boundary_six | BOOLEAN | |
| is_dot | BOOLEAN | |
| is_wicket | BOOLEAN | |
| dismissal_kind | VARCHAR | bowled, caught, lbw, etc. |
| player_out_id | VARCHAR | |
| phase | VARCHAR | powerplay / middle / death |
| innings_runs | INTEGER | Cumulative runs at this point |
| innings_wickets | SMALLINT | Cumulative wickets |
| run_rate | FLOAT | Current run rate |
| required_rate | FLOAT | Required rate (innings 2 only) |
| balls_remaining | SMALLINT | |

### player_ratings
| Column | Type | Description |
|--------|------|-------------|
| player_id | VARCHAR | FK → players |
| rating_type | VARCHAR | elo_bat, elo_bowl, bt_bat, bt_bowl, bayesian_bat, bayesian_bowl |
| phase | VARCHAR | NULL for overall, or powerplay/middle/death |
| league | VARCHAR | NULL for cross-league |
| rating_value | FLOAT | |
| rating_variance | FLOAT | Uncertainty (Bayesian) |
| matches_count | INTEGER | |
| deliveries_count | INTEGER | |
| epoch | INTEGER | Iteration number |

### expected_runs
| Column | Type | Description |
|--------|------|-------------|
| match_id, innings, over_number, ball_number | | FK → deliveries |
| xr_value | FLOAT | Expected runs for this delivery |
| actual_runs | SMALLINT | |
| runs_above_expected | FLOAT | actual - expected |
| model_version | VARCHAR | |

---

## Rating Algorithm

### Iterative Elo + Bradley-Terry Convergence

**Epoch 0**: All players start at Elo 1500.

**Each epoch**:
1. **Elo sweep**: Replay every delivery chronologically. Each ball is a zero-sum game.
   - Outcome mapping: 6→1.0, 4→0.8, 3→0.6, 2→0.5, 1→0.3, dot→0.1, wicket→0.0
   - `E_bat = 1 / (1 + 10^((R_bowl - R_bat) / 400))`
   - `R_bat_new = R_bat + K * (S_bat - E_bat)`, K ~0.5 adjusted by opponent strength
2. **Bradley-Terry fit**: Aggregate pairwise outcomes, fit via MLE.
   - `P(batter scores | batter i, bowler j) = alpha_i / (alpha_i + beta_j)`
   - Solves circular dependencies globally in one optimization
3. **Blend**: 60% Elo (temporal sensitivity) + 40% BT (global consistency)
4. **Convergence**: Stop when max rating change < 1.0 for all players with >100 deliveries

**Phase-specific**: Run independently for powerplay (overs 1-6), middle (7-15), death (16-20).

### Expected Runs (xR)
- Gradient-boosted model (LightGBM/scikit-learn)
- Features: phase, over, innings, cumulative runs/wickets, required rate, bowler type, batter/bowler Elo, venue
- Predicts expected runs per delivery
- Players measured by actual vs expected

### Cricket WAR
```
Batting_WAR  = (Actual_Runs - xR_Runs) / Runs_Per_Win
Bowling_WAR  = (xR_Conceded - Actual_Conceded) / Runs_Per_Win
Total_WAR    = Batting_WAR + Bowling_WAR + Fielding_WAR
```

---

## Bowler Type Taxonomy

| Category | Subcategories |
|----------|--------------|
| Pace | Right-arm fast, right-arm medium-fast, right-arm medium |
| Pace | Left-arm fast, left-arm medium-fast, left-arm medium |
| Seam/Swing | Subset of pace with movement classification |
| Finger Spin | Right-arm off-break, left-arm orthodox |
| Wrist Spin | Right-arm leg-break, left-arm wrist spin |

Source: ESPNCricinfo player profiles + inference heuristics for missing data.

---

## Implementation Phases

### Phase 1: Foundation + Data Ingestion ✅ DONE
- [x] Package scaffolding (pyproject.toml, directory structure)
- [x] Config, constants, enums
- [x] DuckDB schema + engine
- [x] Cricsheet JSON parser with auto-download
- [x] Player registry (Cricsheet UUIDs)
- [x] CLI: `t20x ingest --league ipl`
- [x] Tests (17 passing) with real Cricsheet fixture data

### Phase 2: Player Metadata Enrichment
- [ ] ESPNCricinfo scraper for bowling style, batting hand
- [ ] Bowler type classification (with inference fallback)
- [ ] CLI: `t20x enrich --source espn`
- [ ] Target: >90% bowler classification coverage

### Phase 3: Core Rating Engine + Validation
- [ ] Per-delivery Elo engine (`ratings/elo.py`)
- [ ] Bradley-Terry pairwise MLE (`ratings/bradley_terry.py`)
- [ ] Iterative convergence orchestrator (`ratings/convergence.py`)
- [ ] CLI: `t20x ratings compute`, `t20x ratings show <player>`
- [ ] Validation framework (`validation/` module):
  - [ ] `holdout.py` — train on 2008-2023, predict 2024-2025 match outcomes (accuracy, log-loss, Brier score vs ICC/naive baselines)
  - [ ] `calibration.py` — calibration plots: predicted vs actual outcome frequency by decile
  - [ ] `rank_compare.py` — Spearman correlation with ICC T20I rankings (target > 0.6)
  - [ ] `convergence.py` — plot rating changes per epoch, verify stabilization and zero-sum
  - [ ] `sensitivity.py` — parameter sweep: K-factor, Elo/BT blend weights, stability of top-10
  - [ ] Cross-league validation: train on IPL, predict BBL (and vice versa)
  - [ ] Face validity: sanity-check top-10 lists against expert consensus

### Phase 4: Situation-Aware Metrics + xR
- [ ] Expected Runs model (`ratings/expected_runs.py`)
- [ ] Phase splits with opponent-quality weighting
- [ ] Batsman vs bowler-type matchup matrices
- [ ] Opposition-quality-adjusted averages
- [ ] Pressure index weighting
- [ ] Cricket WAR composite metric

### Phase 5: Comparison + Recommendation Engine
- [ ] Multi-criteria ranking engine
- [ ] Situation-specific recommender
- [ ] Player similarity via embeddings
- [ ] CLI: `t20x compare`, `t20x rank`

### Phase 6: Bayesian + GP Enhancement + API
- [ ] PyMC/NumPyro hierarchical model (Elo as priors)
- [ ] Gaussian Process player ratings (`ratings/gaussian_process.py`)
  - GP with temporal kernel for time-varying ability
  - GP classification for delivery outcomes
  - A/B comparison: linear (Bradley-Terry) vs GP on held-out season
  - Blog post: "Linear Models vs Gaussian Processes: Which Predicts Cricket Better?"
- [ ] Posterior distributions with credible intervals
- [ ] FastAPI REST endpoints
- [ ] PyPI packaging

---

## Data Sources

| Source | Type | Coverage | Access |
|--------|------|----------|--------|
| **Cricsheet.org** (primary) | Ball-by-ball JSON | 21,500+ matches: IPL, BBL, PSL, CPL, T20I | Free download |
| **Kaggle** | Ball-by-ball CSV | IPL 2008-2025, T20I 2003-2023 | Free |
| **ESPNCricinfo Statsguru** | Player metadata | All international + domestic | Scraping |
| **CricAPI** | Live data, player info | Current matches | Free tier: 100/day |

---

## Key Dependencies

```
duckdb>=1.0, pydantic>=2.0, typer>=0.12, rich>=13.0,
httpx>=0.27, tqdm>=4.66, numpy>=1.26, scipy>=1.12,
scikit-learn>=1.4, pandas>=2.2
Optional: pymc>=5.10, numpyro>=0.14, fastapi>=0.110
Dev: pytest>=8.0, ruff, mypy, hypothesis
```

---

## CLI Commands (Target)

```bash
t20x ingest --source cricsheet              # Auto-download + ingest
t20x ingest --source cricsheet --path ./data # Ingest from local file
t20x enrich --source espn                    # Enrich player metadata
t20x ratings compute --epochs 5             # Compute iterative ratings
t20x ratings show "Virat Kohli"             # Show player ratings
t20x rank --role bowler --phase death --league IPL
t20x compare "Rashid Khan" "Wanindu Hasaranga" --phase middle
t20x query "top 5 death bowlers vs left-handers"
```

---

## References

- Bradley-Terry in cricket: [Cardiff University paper](https://orca.cardiff.ac.uk/id/eprint/107993/1/BT.pdf)
- Bayesian batting survival: [arXiv:1609.04078](https://arxiv.org/pdf/1609.04078)
- Expected Runs: [Cricket Savant](https://cricketsavant.wordpress.com/2016/12/26/introducing-expected-runs/)
- WASP: [Wikipedia](https://en.wikipedia.org/wiki/WASP_(cricket_calculation_tool))
- Baseball WAR: [Wikipedia](https://en.wikipedia.org/wiki/Wins_above_replacement)
- Football xG + Elo: [xGELO](https://medium.com/@worville/xgelo-combining-expected-goals-and-elo-ratings-6aa987481479)
- Dixon-Coles model: Journal of Applied Statistics, 1997

---

*Last updated: 2026-04-12 — Phase 1 in progress*
