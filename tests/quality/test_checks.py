from statflow.quality.checks import (
    check_games_final_has_scores,
    check_games_no_null_ids,
    check_games_pk_unique,
    check_games_total_runs_consistent,
    check_pitcher_stats_pk_unique,
    check_pitcher_stats_two_starters_per_final_game,
    check_team_stats_pk_unique,
    check_team_stats_two_rows_per_final_game,
)


def test_all_checks_pass_on_valid_silver(silver_dir, valid_silver):
    checks = [
        check_games_pk_unique,
        check_games_no_null_ids,
        check_games_final_has_scores,
        check_games_total_runs_consistent,
        check_team_stats_pk_unique,
        check_team_stats_two_rows_per_final_game,
        check_pitcher_stats_pk_unique,
        check_pitcher_stats_two_starters_per_final_game,
    ]
    for check in checks:
        result = check(silver_dir)
        assert result.passed, f"{result.name} failed: {result.details}"


def test_games_pk_unique_fails_on_duplicate(
    silver_dir, write_games, write_team_stats, write_pitcher_stats, make_final_game
):
    write_games([make_final_game(111), make_final_game(111)])  # dup
    result = check_games_pk_unique(silver_dir)
    assert not result.passed
    assert "1 duplicate" in result.details


def test_games_no_null_ids_fails_on_null_team(silver_dir, write_games, make_final_game):
    game = make_final_game(111)
    game["home_team_id"] = None
    write_games([game])
    result = check_games_no_null_ids(silver_dir)
    assert not result.passed
    assert "home_team_id" in result.details


def test_games_final_has_scores_fails_when_score_missing(silver_dir, write_games, make_final_game):
    game = make_final_game(111)
    game["home_score"] = None
    write_games([game])
    result = check_games_final_has_scores(silver_dir)
    assert not result.passed
    assert "1 missing scores" in result.details


def test_games_total_runs_consistent_fails_on_mismatch(silver_dir, write_games, make_final_game):
    game = make_final_game(111, home_score=5, away_score=3)
    game["total_runs"] = 999  # wrong
    write_games([game])
    result = check_games_total_runs_consistent(silver_dir)
    assert not result.passed
    assert "1 rows" in result.details


def test_team_stats_pk_unique_fails_on_duplicate(silver_dir, write_team_stats, make_team_stats_row):
    write_team_stats(
        [
            make_team_stats_row(111, 147, True),
            make_team_stats_row(111, 147, True),  # dup
        ]
    )
    result = check_team_stats_pk_unique(silver_dir)
    assert not result.passed


def test_team_stats_two_rows_per_final_game_fails_on_missing_side(
    silver_dir,
    write_games,
    write_team_stats,
    make_final_game,
    make_team_stats_row,
):
    write_games([make_final_game(111)])
    write_team_stats([make_team_stats_row(111, 147, True)])  # only home
    result = check_team_stats_two_rows_per_final_game(silver_dir)
    assert not result.passed
    assert "row-count != 2" in result.details or "no team_game_stats rows" in result.details


def test_pitcher_stats_pk_unique_fails_on_duplicate(
    silver_dir, write_pitcher_stats, make_pitcher_row
):
    write_pitcher_stats(
        [
            make_pitcher_row(111, 1001, 147, True),
            make_pitcher_row(111, 1001, 147, True),  # dup
        ]
    )
    result = check_pitcher_stats_pk_unique(silver_dir)
    assert not result.passed


def test_pitcher_stats_two_starters_per_final_game_fails_with_one_starter(
    silver_dir,
    write_games,
    write_pitcher_stats,
    make_final_game,
    make_pitcher_row,
):
    write_games([make_final_game(111)])
    write_pitcher_stats([make_pitcher_row(111, 1001, 147, True)])  # only 1 starter
    result = check_pitcher_stats_two_starters_per_final_game(silver_dir)
    assert not result.passed
