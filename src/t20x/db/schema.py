"""DuckDB schema definitions for t20x."""

SCHEMA_SQL = """
-- Match-level information
CREATE TABLE IF NOT EXISTS matches (
    match_id        VARCHAR PRIMARY KEY,
    league          VARCHAR NOT NULL,
    season          VARCHAR,
    date            DATE NOT NULL,
    venue           VARCHAR,
    city            VARCHAR,
    team_1          VARCHAR NOT NULL,
    team_2          VARCHAR NOT NULL,
    toss_winner     VARCHAR,
    toss_decision   VARCHAR,
    winner          VARCHAR,
    win_margin      INTEGER,
    win_type        VARCHAR,
    player_of_match VARCHAR,
    gender          VARCHAR DEFAULT 'male'
);

-- Player registry
CREATE TABLE IF NOT EXISTS players (
    player_id       VARCHAR PRIMARY KEY,
    name            VARCHAR NOT NULL,
    full_name       VARCHAR,
    dob             DATE,
    batting_style   VARCHAR,
    bowling_type    VARCHAR,
    bowling_arm     VARCHAR,
    bowling_pace    VARCHAR,
    bowling_style   VARCHAR,
    primary_role    VARCHAR,
    country         VARCHAR,
    espn_id         INTEGER,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ball-by-ball delivery data (central fact table)
CREATE TABLE IF NOT EXISTS deliveries (
    match_id        VARCHAR NOT NULL,
    innings         SMALLINT NOT NULL,
    over_number     SMALLINT NOT NULL,
    ball_number     SMALLINT NOT NULL,
    delivery_seq    INTEGER NOT NULL,
    batter_id       VARCHAR NOT NULL,
    bowler_id       VARCHAR NOT NULL,
    non_striker_id  VARCHAR NOT NULL,
    batter_runs     SMALLINT NOT NULL DEFAULT 0,
    extra_runs      SMALLINT NOT NULL DEFAULT 0,
    total_runs      SMALLINT NOT NULL DEFAULT 0,
    is_wide         BOOLEAN DEFAULT FALSE,
    is_noball       BOOLEAN DEFAULT FALSE,
    is_bye          BOOLEAN DEFAULT FALSE,
    is_legbye       BOOLEAN DEFAULT FALSE,
    is_boundary_four BOOLEAN DEFAULT FALSE,
    is_boundary_six  BOOLEAN DEFAULT FALSE,
    is_dot          BOOLEAN DEFAULT FALSE,
    is_wicket       BOOLEAN DEFAULT FALSE,
    dismissal_kind  VARCHAR,
    player_out_id   VARCHAR,
    phase           VARCHAR NOT NULL,
    innings_runs    INTEGER DEFAULT 0,
    innings_wickets SMALLINT DEFAULT 0,
    run_rate        FLOAT DEFAULT 0.0,
    required_rate   FLOAT,
    balls_remaining SMALLINT DEFAULT 120,
    PRIMARY KEY (match_id, innings, over_number, ball_number, delivery_seq)
);

-- Player ratings (snapshot per epoch)
CREATE TABLE IF NOT EXISTS player_ratings (
    player_id        VARCHAR NOT NULL,
    rating_type      VARCHAR NOT NULL,
    phase            VARCHAR,
    league           VARCHAR,
    rating_value     FLOAT NOT NULL,
    rating_variance  FLOAT,
    matches_count    INTEGER,
    deliveries_count INTEGER,
    epoch            INTEGER NOT NULL,
    computed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Expected runs per delivery
CREATE TABLE IF NOT EXISTS expected_runs (
    match_id        VARCHAR NOT NULL,
    innings         SMALLINT NOT NULL,
    over_number     SMALLINT NOT NULL,
    ball_number     SMALLINT NOT NULL,
    delivery_seq    INTEGER NOT NULL,
    xr_value        FLOAT NOT NULL,
    actual_runs     SMALLINT NOT NULL,
    runs_above_expected FLOAT,
    model_version   VARCHAR
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_del_batter ON deliveries(batter_id, phase);
CREATE INDEX IF NOT EXISTS idx_del_bowler ON deliveries(bowler_id, phase);
CREATE INDEX IF NOT EXISTS idx_del_match ON deliveries(match_id, innings);
CREATE INDEX IF NOT EXISTS idx_ratings_player ON player_ratings(player_id, rating_type, epoch);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league, date);
"""
