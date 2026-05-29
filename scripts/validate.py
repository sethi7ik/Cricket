"""Run the validation suite against the current DB."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb

from t20x.validation.cross_league import evaluate_cross_league
from t20x.validation.match_prediction import (
    MatchPredictionResult,
    evaluate_match_prediction,
)
from t20x.validation.wp_holdout import evaluate_wp_holdout

DB_PATH = Path.home() / ".t20x" / "data" / "t20x.duckdb"
CUTOFF = dt.date(2024, 1, 1)


def fmt(d: dict) -> str:
    return ", ".join(
        f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()
    )


def main() -> None:
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    print("=" * 70)
    print("WP HOLDOUT — train on deliveries before 2024-01-01, test on later")
    print("=" * 70)
    res = evaluate_wp_holdout(conn, cutoff=CUTOFF)
    print(f"\nTrain deliveries: {res.n_train:,}")
    print(f"Test  deliveries: {res.n_test:,}")
    print(f"\nWP model on test set:  {fmt(res.test_metrics)}")
    print(f"Constant-baseline on test:  {fmt(res.baseline_metrics)}")

    cal = res.calibration
    print("\nPer-decile calibration (test set):")
    print(f"  {'bin':>3}  {'range':>14}  {'count':>9}  {'pred_mean':>10}  {'actual':>8}  {'gap':>8}")
    for _, r in cal.iterrows():
        rng = f"{r.lo:.2f}-{r.hi:.2f}"
        count = f"{int(r['count']):,}"
        if r["count"] == 0:
            print(f"  {int(r.bin):>3}  {rng:>14}  {count:>9}  {'-':>10}  {'-':>8}  {'-':>8}")
            continue
        print(
            f"  {int(r.bin):>3}  {rng:>14}  {count:>9}  "
            f"{r.pred_mean:>10.3f}  {r.actual_mean:>8.3f}  {r.gap:>+8.3f}"
        )

    # A quick interpretation hint
    print()
    delta_brier = res.baseline_metrics["brier"] - res.test_metrics["brier"]
    delta_ll = res.baseline_metrics["log_loss"] - res.test_metrics["log_loss"]
    print(f"Improvement vs constant baseline: dbrier={delta_brier:+.5f}  dlog_loss={delta_ll:+.5f}")
    print(f"ECE = {res.test_metrics['ece']:.4f}  (0 = perfect calibration; <0.02 is excellent)")

    print()
    print("=" * 70)
    print("MATCH PREDICTION — winner of post-2024 matches from pre-cutoff WPA")
    print("=" * 70)

    # Window variants: career total + hard cutoffs
    window_variants = [
        ("career_total", None, None),
        ("2_year_window", 2, None),
        ("1_year_window", 1, None),
    ]
    # Exponential-decay variants: continuous recency.
    # lambda = ln(2) / half_life_years.
    import math
    decay_variants = [
        (f"decay_hl_{hl}y", None, math.log(2) / hl)
        for hl in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
    ]
    variants = window_variants + decay_variants
    print(f"\n{'Variant':18s}  {'Accuracy':>10s}  {'LogLoss':>9s}  {'Brier':>8s}  {'sigmoid_coef':>12s}")
    print("-" * 70)
    mp_career: MatchPredictionResult | None = None
    for label, lb, dl in variants:
        mp = evaluate_match_prediction(
            conn, cutoff=CUTOFF, fit_split=0.5, lookback_years=lb, decay_lambda=dl,
        )
        if label == "career_total":
            mp_career = mp
        m = mp.metrics["WPA_model"]
        print(f"  {label:16s}  {m['accuracy']*100:>9.1f}%  {m['log_loss']:>9.4f}  {m['brier']:>8.4f}  {m['coef']:>12.4f}")
    assert mp_career is not None
    print(f"\nBaselines (from career variant, same test split):")
    print(f"  {'coin_flip':14s}  {'50.0%':>10s}  "
          f"{mp_career.metrics['coin_flip']['log_loss']:>9.4f}  {mp_career.metrics['coin_flip']['brier']:>8.4f}")
    print(f"  {'toss_winner':14s}  "
          f"{mp_career.metrics['toss_winner_wins']['accuracy']*100:>9.1f}%  "
          f"{mp_career.metrics['toss_winner_wins']['log_loss']:>9.4f}  "
          f"{mp_career.metrics['toss_winner_wins']['brier']:>8.4f}")
    print(f"  {'bat_first':14s}  "
          f"{mp_career.metrics['bat_first_wins']['accuracy']*100:>9.1f}%  "
          f"{mp_career.metrics['bat_first_wins']['log_loss']:>9.4f}  "
          f"{mp_career.metrics['bat_first_wins']['brier']:>8.4f}")

    print()
    print("=" * 70)
    print("CROSS-LEAGUE TRANSFER — does WP generalize across leagues?")
    print("=" * 70)
    pairs = [
        ("Indian Premier League", "Big Bash League"),
        ("Big Bash League", "Indian Premier League"),
        ("Indian Premier League", "Pakistan Super League"),
        ("Indian Premier League", "Caribbean Premier League"),
    ]
    print(f"\n{'Source -> Target':50s}  {'transfer Brier':>14s}  {'within Brier':>13s}  {'baseline':>10s}")
    print("-" * 100)
    for src, tgt in pairs:
        try:
            r = evaluate_cross_league(conn, source_league=src, target_league=tgt)
            label = f"{src} -> {tgt}"
            print(
                f"  {label:48s}  {r.transfer_metrics['brier']:>14.4f}  "
                f"{r.within_metrics['brier']:>13.4f}  {r.baseline_metrics['brier']:>10.4f}  "
                f"[n_src={r.n_source:,}  n_tgt={r.n_target:,}]"
            )
            print(
                f"    {'ECE:':>10s} transfer={r.transfer_metrics['ece']:.4f}  "
                f"within={r.within_metrics['ece']:.4f}  "
                f"log-loss transfer={r.transfer_metrics['log_loss']:.4f}  "
                f"within={r.within_metrics['log_loss']:.4f}"
            )
        except Exception as e:
            print(f"  {src} -> {tgt}: skipped ({e})")


if __name__ == "__main__":
    main()
