# Copyright 2026 Nuri ML contributors
import numpy as np

from child_profile_experiment import band_recall, build_profiles, ndcg_at_k


def make_pools() -> dict[str, dict[int, list[str]]]:
    return {
        f"g{group}": {
            level: [f"g{group}L{level}{index}" for index in range(3)] for level in (1, 2, 3, 4)
        }
        for group in range(4)
    }


def test_ndcg_rewards_the_correct_direction() -> None:
    truth = [4, 3, 2, 1]
    ordered = [4.0, 3.0, 2.0, 1.0]
    reversed_scores = [1.0, 2.0, 3.0, 4.0]

    assert ndcg_at_k(ordered, truth, weak=False) == 1.0
    assert ndcg_at_k(reversed_scores, truth, weak=False) < 1.0
    assert ndcg_at_k(ordered, truth, weak=True) == 1.0
    assert ndcg_at_k(reversed_scores, truth, weak=True) < 1.0


def test_band_recall_uses_the_second_best_level_as_threshold() -> None:
    truth = [4, 3, 2, 1]
    ordered = [4.0, 3.0, 2.0, 1.0]
    reversed_scores = [1.0, 2.0, 3.0, 4.0]

    assert band_recall(ordered, truth, weak=False) == 1.0
    assert band_recall(reversed_scores, truth, weak=False) == 0.0
    assert band_recall(ordered, truth, weak=True) == 1.0
    assert band_recall(reversed_scores, truth, weak=True) == 0.0


def test_profiles_are_seeded_and_keep_levels_consistent() -> None:
    pools = make_pools()
    first = build_profiles(pools, children=6, items_per_child=3, observations=2, seed=11)
    second = build_profiles(pools, children=6, items_per_child=3, observations=2, seed=11)

    assert [(p.groups, p.truth, p.observations) for p in first] == [
        (p.groups, p.truth, p.observations) for p in second
    ]
    for profile in first:
        assert len(profile.groups) == 3
        assert len(set(profile.groups)) == 3
        for group, level in zip(profile.groups, profile.truth, strict=True):
            assert 1 <= level <= 4
            texts = profile.observations[group]
            assert len(texts) == 2
            assert all(text.startswith(f"{group}L{level}") for text in texts)


def test_observations_are_capped_by_the_available_rows() -> None:
    profiles = build_profiles(make_pools(), children=3, items_per_child=4, observations=9, seed=5)
    assert all(len(texts) == 3 for p in profiles for texts in p.observations.values())


def test_ndcg_is_bounded() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        truth = [int(value) for value in rng.integers(1, 5, size=8)]
        scores = [float(value) for value in rng.random(8)]
        for weak in (False, True):
            value = ndcg_at_k(scores, truth, weak=weak)
            assert 0.0 <= value <= 1.0 + 1e-12
