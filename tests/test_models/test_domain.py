"""Tests for domain models."""

from t20x.models.enums import Phase, over_to_phase


def test_over_to_phase_powerplay():
    for over in range(6):
        assert over_to_phase(over) == Phase.POWERPLAY


def test_over_to_phase_middle():
    for over in range(6, 15):
        assert over_to_phase(over) == Phase.MIDDLE


def test_over_to_phase_death():
    for over in range(15, 20):
        assert over_to_phase(over) == Phase.DEATH


def test_parsed_match_has_deliveries(parsed_match):
    assert parsed_match is not None
    assert len(parsed_match.deliveries) > 0


def test_parsed_match_info(parsed_match):
    assert parsed_match.info.match_id == "test_001"
    assert parsed_match.info.league == "Indian Premier League"
    assert parsed_match.info.venue == "Wankhede Stadium"
    assert parsed_match.info.team_1 == "Team A"
    assert parsed_match.info.team_2 == "Team B"
    assert parsed_match.info.winner == "Team A"
    assert parsed_match.info.win_type == "runs"
    assert parsed_match.info.win_margin == 15


def test_parsed_match_players(parsed_match):
    assert len(parsed_match.players) == 22  # 11 per team
    assert "bat001" in parsed_match.players
    assert "bowl003" in parsed_match.players
    assert parsed_match.players["bat001"].name == "Batter One"


def test_delivery_parsing(parsed_match):
    deliveries = parsed_match.deliveries

    # First delivery: Batter One hits 4 off Bowler Three
    d0 = deliveries[0]
    assert d0.batter_id == "bat001"
    assert d0.bowler_id == "bowl003"
    assert d0.batter_runs == 4
    assert d0.is_boundary_four is True
    assert d0.is_boundary_six is False
    assert d0.phase == Phase.POWERPLAY


def test_wide_delivery(parsed_match):
    # The 5th delivery in over 0 (index 4) is a wide
    deliveries = [d for d in parsed_match.deliveries if d.innings == 1]
    wide = deliveries[4]
    assert wide.is_wide is True
    assert wide.extra_runs == 1


def test_wicket_delivery(parsed_match):
    # Last delivery of over 0, innings 1: Batter One bowled
    deliveries = [d for d in parsed_match.deliveries if d.innings == 1]
    wicket_del = deliveries[6]  # 7th delivery (including the wide)
    assert wicket_del.is_wicket is True
    assert wicket_del.dismissal_kind == "bowled"
    assert wicket_del.player_out_id == "bat001"


def test_six_delivery(parsed_match):
    deliveries = [d for d in parsed_match.deliveries if d.innings == 1]
    six = deliveries[3]  # Batter Two hits 6
    assert six.batter_runs == 6
    assert six.is_boundary_six is True


def test_second_innings_has_required_rate(parsed_match):
    inn2 = [d for d in parsed_match.deliveries if d.innings == 2]
    assert len(inn2) > 0
    # Second innings should have required_rate set
    # First ball: target is first innings total + 1
    assert inn2[0].required_rate is not None
