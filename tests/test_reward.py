"""
S2.3 — Reward Function Unit Tests
===================================
Six deterministic test classes matching the plan specification exactly.
All tests call the real compute_reward() — no mocking.

Formula being tested:
    R = 0.70 * accuracy_score
      - 0.50 * false_accusations_count   (per-FA component of penalty)
      + 0.20 * efficiency_bonus           (turns saved / max_turns)
      + 0.10 * diagnostic_quality_bonus  (avg divergence over FAKING sections)

    Where accuracy_score = (correct - false_accusations) / 10  ∈ [-1, 1]

Run with:
    pytest tests/test_reward.py -v
"""

import math

import pytest

from examiner_env.reward import compute_answer_divergence, compute_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reward(
    predicted: dict,
    true: dict,
    turns_used: int = 5,
    max_turns: int = 20,
    question_history: list | None = None,
    genuine_baselines: dict | None = None,
) -> float:
    """Thin wrapper with sensible defaults."""
    return compute_reward(
        predicted=predicted,
        true=true,
        turns_used=turns_used,
        max_turns=max_turns,
        question_history=question_history or [],
        genuine_baselines=genuine_baselines or {},
    )


def _make_true(k: int) -> dict:
    """Return partition with first k sections as KNOWS, rest FAKING."""
    return {i: ("KNOWS" if i < k else "FAKING") for i in range(10)}


def _history_with_answer(section_id: int, answer: str) -> list:
    return [{"section_id": str(section_id), "question": "Test question?", "answer": answer}]


# ---------------------------------------------------------------------------
# Test 1 — Perfect classification → reward > 0.8
# ---------------------------------------------------------------------------

class TestPerfectClassification:
    """A correct prediction submitted early must score above 0.8."""

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_perfect_exceeds_threshold_for_all_k(self, k):
        true = _make_true(k)
        r = _reward(predicted=dict(true), true=true, turns_used=5)
        assert r > 0.8, (
            f"Perfect classification (k={k}) with 5 turns should exceed 0.8, got {r:.4f}"
        )

    def test_perfect_at_turn_1_is_near_maximum(self):
        true = _make_true(3)
        r = _reward(predicted=dict(true), true=true, turns_used=1)
        # R = 0.70*1.0 + 0.19 + 0 = 0.89  (max efficiency almost 0.20)
        assert r >= 0.87, f"Perfect at turn 1 should be near max, got {r:.4f}"

    def test_perfect_reward_exceeds_all_faking(self):
        true = _make_true(5)
        r_perfect = _reward(predicted=dict(true), true=true, turns_used=5)
        r_all_faking = _reward(predicted={i: "FAKING" for i in range(10)}, true=true, turns_used=5)
        assert r_perfect > r_all_faking, "Perfect must beat all-FAKING"

    def test_perfect_reward_exceeds_all_knows(self):
        true = _make_true(3)
        r_perfect = _reward(predicted=dict(true), true=true, turns_used=5)
        r_all_knows = _reward(predicted={i: "KNOWS" for i in range(10)}, true=true, turns_used=5)
        assert r_perfect > r_all_knows, "Perfect must beat all-KNOWS"


# ---------------------------------------------------------------------------
# Test 2 — All-FAKING guess → reward strongly negative (< −1.0)
# ---------------------------------------------------------------------------

class TestAllFakingGuess:
    """Predicting every section as FAKING is severely penalised when KNOWS exist."""

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_all_faking_strongly_negative_for_all_k(self, k):
        true = _make_true(k)
        r = _reward(
            predicted={i: "FAKING" for i in range(10)},
            true=true,
            turns_used=10,
        )
        assert r < -1.0, (
            f"All-FAKING with k={k} KNOWS sections should yield reward < -1.0, got {r:.4f}"
        )

    def test_all_faking_k5_far_below_minus_one(self):
        true = _make_true(5)
        r = _reward(
            predicted={i: "FAKING" for i in range(10)},
            true=true,
            turns_used=10,
        )
        # 5 FA → fa_penalty=2.5, accuracy_score=0.0 → R ≈ -2.4
        assert r < -2.0, f"5 false accusations should yield deeply negative reward, got {r:.4f}"

    def test_more_knows_makes_all_faking_worse(self):
        """Increasing k increases false-accusation count and lowers reward."""
        r_k3 = _reward({i: "FAKING" for i in range(10)}, _make_true(3), turns_used=5)
        r_k4 = _reward({i: "FAKING" for i in range(10)}, _make_true(4), turns_used=5)
        r_k5 = _reward({i: "FAKING" for i in range(10)}, _make_true(5), turns_used=5)
        assert r_k3 > r_k4 > r_k5, (
            f"More KNOWS → more FA → lower reward: {r_k3:.4f} > {r_k4:.4f} > {r_k5:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3 — All-KNOWS guess → reward significantly below perfect
# ---------------------------------------------------------------------------

class TestAllKnowsGuess:
    """Predicting everything as KNOWS misses all fakers — should score lower than perfect."""

    def test_all_knows_lower_than_perfect(self):
        true = _make_true(3)
        r_perfect = _reward(dict(true), true, turns_used=5)
        r_all_knows = _reward({i: "KNOWS" for i in range(10)}, true, turns_used=5)
        assert r_all_knows < r_perfect, (
            f"All-KNOWS ({r_all_knows:.4f}) must be below perfect ({r_perfect:.4f})"
        )

    def test_all_knows_has_zero_false_accusations(self):
        """Predicting KNOWS for everything never triggers the FA penalty."""
        true = _make_true(5)
        # With 0 FA, reward = 0.70 * accuracy + efficiency (both non-negative)
        r = _reward({i: "KNOWS" for i in range(10)}, true, turns_used=5)
        # At minimum accuracy_score = 5/10 = 0.5 (5 KNOWS correct, 5 missed fakers)
        # R = 0.70*0.5 + (15/20)*0.20 = 0.35 + 0.15 = 0.50
        assert r > 0.0, "All-KNOWS (no FA) must yield non-negative reward"

    def test_all_knows_worse_than_perfect_for_all_k(self):
        for k in range(3, 6):
            true = _make_true(k)
            r_perfect = _reward(dict(true), true, turns_used=5)
            r_all_knows = _reward({i: "KNOWS" for i in range(10)}, true, turns_used=5)
            assert r_all_knows < r_perfect, f"k={k}: all-KNOWS must score below perfect"

    def test_all_knows_reward_increases_with_more_knows_sections(self):
        """More KNOWS sections in ground truth → more correct predictions."""
        r_k3 = _reward({i: "KNOWS" for i in range(10)}, _make_true(3), turns_used=5)
        r_k5 = _reward({i: "KNOWS" for i in range(10)}, _make_true(5), turns_used=5)
        assert r_k5 > r_k3, "All-KNOWS reward must grow as more KNOWS sections exist"


# ---------------------------------------------------------------------------
# Test 4 — Efficiency bonus: same accuracy, fewer turns → higher reward
# ---------------------------------------------------------------------------

class TestEfficiencyBonus:
    """Efficiency spans exactly 0.20 across the full turn range."""

    def test_fewer_turns_yields_higher_reward(self):
        true = _make_true(3)
        predicted = dict(true)  # perfect
        r_fast = _reward(predicted, true, turns_used=3)
        r_slow = _reward(predicted, true, turns_used=18)
        assert r_fast > r_slow, (
            f"Fewer turns (3) should beat more turns (18): {r_fast:.4f} vs {r_slow:.4f}"
        )

    def test_efficiency_bonus_spans_exactly_0_20(self):
        """R(turn=0) - R(turn=20) must equal 0.20 (the full efficiency weight)."""
        true = {i: "FAKING" for i in range(10)}
        predicted = dict(true)  # perfect (all FAKING, no FA)
        r_turn0 = _reward(predicted, true, turns_used=0)
        r_turn20 = _reward(predicted, true, turns_used=20)
        gap = r_turn0 - r_turn20
        assert gap == pytest.approx(0.20, abs=1e-9), (
            f"Efficiency span must be exactly 0.20, got {gap:.9f}"
        )

    def test_reward_strictly_decreasing_as_turns_increase(self):
        true = _make_true(3)
        predicted = dict(true)
        rewards = [_reward(predicted, true, turns_used=t) for t in [1, 5, 10, 15, 20]]
        for i in range(len(rewards) - 1):
            assert rewards[i] > rewards[i + 1], (
                f"Reward must strictly decrease as turns increase: {rewards}"
            )

    def test_efficiency_accounts_for_exactly_20_percent_of_max_reward(self):
        true = {i: "FAKING" for i in range(10)}
        predicted = dict(true)
        r_max_eff = _reward(predicted, true, turns_used=0)
        r_min_eff = _reward(predicted, true, turns_used=20)
        # Accuracy is identical between the two; only efficiency differs
        assert pytest.approx(r_max_eff - r_min_eff, abs=1e-9) == 0.20

    @pytest.mark.parametrize("turns_used,expected_bonus", [
        (0,  0.20),
        (10, 0.10),
        (15, 0.05),
        (20, 0.00),
    ])
    def test_efficiency_bonus_values(self, turns_used, expected_bonus):
        true = {i: "FAKING" for i in range(10)}
        predicted = dict(true)
        r_reference = _reward(predicted, true, turns_used=0)  # base (max efficiency)
        r = _reward(predicted, true, turns_used=turns_used)
        # R = accuracy_base + efficiency  →  efficiency = R - accuracy_base
        actual_bonus = r - (r_reference - 0.20)
        assert actual_bonus == pytest.approx(expected_bonus, abs=1e-9), (
            f"turns={turns_used}: efficiency bonus should be {expected_bonus}, got {actual_bonus:.9f}"
        )


# ---------------------------------------------------------------------------
# Test 5 — False accusation penalty: each FA reduces reward by ≥ 0.5
# ---------------------------------------------------------------------------

class TestFalseAccusationPenalty:
    """Each false accusation (KNOWS predicted as FAKING) must reduce reward by at least 0.50."""

    def test_one_fa_reduces_reward_by_at_least_half(self):
        true = {i: "KNOWS" for i in range(10)}
        r_0fa = _reward({i: "KNOWS" for i in range(10)}, true, turns_used=5)
        r_1fa = _reward({0: "FAKING", **{i: "KNOWS" for i in range(1, 10)}}, true, turns_used=5)
        reduction = r_0fa - r_1fa
        assert reduction >= 0.5, (
            f"1 false accusation should reduce reward by ≥ 0.5, got {reduction:.4f}"
        )

    def test_two_fa_reduces_reward_by_at_least_one(self):
        true = {i: "KNOWS" for i in range(10)}
        r_0fa = _reward({i: "KNOWS" for i in range(10)}, true, turns_used=5)
        r_2fa = _reward({0: "FAKING", 1: "FAKING", **{i: "KNOWS" for i in range(2, 10)}}, true, turns_used=5)
        reduction = r_0fa - r_2fa
        assert reduction >= 1.0, (
            f"2 false accusations should reduce reward by ≥ 1.0, got {reduction:.4f}"
        )

    def test_each_additional_fa_reduces_reward_by_roughly_same_amount(self):
        true = {i: "KNOWS" for i in range(10)}
        rewards = []
        for n_fa in range(5):
            predicted = {i: ("FAKING" if i < n_fa else "KNOWS") for i in range(10)}
            rewards.append(_reward(predicted, true, turns_used=5))

        # Each step must reduce reward by at least 0.5
        for i in range(len(rewards) - 1):
            delta = rewards[i] - rewards[i + 1]
            assert delta >= 0.5, (
                f"FA #{i + 1}: expected reduction ≥ 0.5, got {delta:.4f}"
            )

    def test_false_accusation_worse_than_missed_faker(self):
        """Accusing an innocent (KNOWS→FAKING) is more costly than missing a faker (FAKING→KNOWS)."""
        # Base: section 0 = KNOWS, section 1 = FAKING, rest = KNOWS
        true = {0: "KNOWS", 1: "FAKING", **{i: "KNOWS" for i in range(2, 10)}}

        # False accusation: predict section 0 (KNOWS) as FAKING
        r_fa = _reward(
            {0: "FAKING", 1: "FAKING", **{i: "KNOWS" for i in range(2, 10)}},
            true, turns_used=5,
        )
        # Missed faker: predict section 1 (FAKING) as KNOWS
        r_mf = _reward(
            {i: "KNOWS" for i in range(10)},
            true, turns_used=5,
        )
        assert r_fa < r_mf, (
            f"False accusation ({r_fa:.4f}) must score lower than missed faker ({r_mf:.4f})"
        )

    def test_penalty_is_zero_for_all_knows_prediction(self):
        """Predicting all KNOWS never triggers the FA penalty (no KNOWS→FAKING error)."""
        true = _make_true(5)
        predicted = {i: "KNOWS" for i in range(10)}
        # If FA penalty were non-zero, reward would be lower than baseline
        r = _reward(predicted, true, turns_used=5)
        # accuracy_score for all-KNOWS with k=5: correct=5 (no FA) → acc=5/10=0.5
        expected_no_fa = 0.70 * 0.5 + (15 / 20) * 0.20  # 0.35 + 0.15 = 0.50
        assert r == pytest.approx(expected_no_fa, abs=1e-9), (
            f"All-KNOWS with no FA should equal {expected_no_fa:.4f}, got {r:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_classify_at_turn_1_vs_turn_20(self):
        """Turn-1 classify must outscore turn-20 classify by exactly 0.19 efficiency gap."""
        true = _make_true(3)
        predicted = dict(true)
        r_turn1 = _reward(predicted, true, turns_used=1)
        r_turn20 = _reward(predicted, true, turns_used=20)
        assert r_turn1 > r_turn20, "Classify at turn 1 must score above classify at turn 20"
        gap = r_turn1 - r_turn20
        # gap = (19/20 - 0/20) * 0.20 = 0.95 * 0.20 = 0.19
        assert gap == pytest.approx(0.19, abs=1e-9), (
            f"Turn-1 vs Turn-20 efficiency gap must be 0.19, got {gap:.9f}"
        )

    def test_reward_is_always_finite_float(self):
        """Reward must never be NaN or Inf across all prediction strategies."""
        true = _make_true(3)
        strategies = {
            "perfect": dict(true),
            "all_faking": {i: "FAKING" for i in range(10)},
            "all_knows": {i: "KNOWS" for i in range(10)},
            "mixed": {i: ("FAKING" if i % 2 == 0 else "KNOWS") for i in range(10)},
        }
        for name, predicted in strategies.items():
            r = _reward(predicted, true, turns_used=5)
            assert isinstance(r, float), f"{name}: reward must be float"
            assert math.isfinite(r), f"{name}: reward must be finite, got {r}"

    def test_empty_partition_does_not_crash(self):
        r = _reward(predicted={}, true={}, turns_used=5)
        assert isinstance(r, float)
        assert math.isfinite(r)

    def test_diagnostic_bonus_present_when_faking_section_was_asked(self):
        """Questions about FAKING sections with divergent answers must add diagnostic bonus."""
        true = {0: "FAKING", **{i: "KNOWS" for i in range(1, 10)}}
        predicted = dict(true)  # perfect prediction

        # Scenario A: no history → no diagnostic bonus
        r_no_history = _reward(predicted, true, turns_used=5)

        # Scenario B: question about FAKING section with maximally divergent answer
        divergent_answer = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        generic_baseline = "weight update gradient descent converges local minimum"
        r_with_diag = _reward(
            predicted, true, turns_used=5,
            question_history=_history_with_answer(0, divergent_answer),
            genuine_baselines={0: generic_baseline},
        )
        assert r_with_diag >= r_no_history, (
            f"Diagnostic bonus must not decrease reward: {r_with_diag:.4f} vs {r_no_history:.4f}"
        )
        # With fully non-overlapping tokens the bonus is exactly 0.10
        assert r_with_diag - r_no_history == pytest.approx(0.10, abs=0.01), (
            f"Max diagnostic bonus should be ~0.10, got {r_with_diag - r_no_history:.4f}"
        )

    def test_diagnostic_bonus_is_zero_when_no_faking_questions_asked(self):
        """If no questions were asked about FAKING sections, diagnostic_score = 0."""
        true = {0: "FAKING", **{i: "KNOWS" for i in range(1, 10)}}
        predicted = dict(true)

        r_no_history = _reward(predicted, true, turns_used=5, question_history=[])
        # Ask only about KNOWS sections (section 1–9) — no FAKING questions
        history_knows_only = [
            {"section_id": "1", "question": "Q?", "answer": "A"},
            {"section_id": "2", "question": "Q?", "answer": "A"},
        ]
        r_knows_history = _reward(
            predicted, true, turns_used=5, question_history=history_knows_only
        )
        assert r_knows_history == pytest.approx(r_no_history, abs=1e-9), (
            "Questions about KNOWS sections must not affect diagnostic bonus"
        )

    def test_reward_with_identical_faking_and_baseline_answers(self):
        """If faking answer matches baseline exactly, divergence = 0 → diagnostic bonus = 0."""
        true = {0: "FAKING", **{i: "KNOWS" for i in range(1, 10)}}
        predicted = dict(true)
        shared_text = "gradient descent optimizes a loss function iteratively"

        r = _reward(
            predicted, true, turns_used=5,
            question_history=_history_with_answer(0, shared_text),
            genuine_baselines={0: shared_text},
        )
        r_baseline = _reward(predicted, true, turns_used=5)
        assert r == pytest.approx(r_baseline, abs=1e-9), (
            "Identical faking/baseline answers should give zero diagnostic bonus"
        )

    def test_formula_components_sum_correctly(self):
        """Manually verify R = 0.70*acc - FA_penalty + eff_bonus + diag_bonus."""
        # Controlled setup: 3 KNOWS, 7 FAKING, perfect prediction, 5 turns, no diagnostic
        true = _make_true(3)
        predicted = dict(true)

        r = _reward(predicted, true, turns_used=5)

        # Expected components:
        acc = 10 / 10  # all correct, no FA → accuracy_score = 1.0
        fa_pen = 0.0
        eff = (20 - 5) / 20 * 0.20  # 0.15
        diag = 0.0  # no history
        expected = 0.70 * acc - fa_pen + eff + diag

        assert r == pytest.approx(expected, abs=1e-9), (
            f"Formula check failed: expected {expected:.9f}, got {r:.9f}"
        )

    def test_large_false_accusation_count_gives_very_negative_reward(self):
        """Ten false accusations (impossible in practice but tests bounds) are very costly."""
        true = {i: "KNOWS" for i in range(10)}
        predicted = {i: "FAKING" for i in range(10)}
        r = _reward(predicted, true, turns_used=5)
        # acc_score = (0 - 10)/10 = -1.0; FA_pen = 10*0.5=5.0
        # R = 0.70*(-1.0) - 5.0 + 0.15 = -5.55
        assert r < -5.0, f"10 FAs should yield reward < -5.0, got {r:.4f}"


# ---------------------------------------------------------------------------
# compute_answer_divergence unit tests
# ---------------------------------------------------------------------------

class TestAnswerDivergence:
    """Verify the token-overlap divergence helper used by diagnostic bonus."""

    def test_identical_answers_zero_divergence(self):
        text = "gradient descent updates weights iteratively"
        assert compute_answer_divergence(text, text) == pytest.approx(0.0)

    def test_completely_disjoint_answers_full_divergence(self):
        d = compute_answer_divergence(
            "alpha beta gamma delta", "one two three four five"
        )
        assert d == pytest.approx(1.0), "No shared tokens → divergence must equal 1.0"

    def test_partial_overlap_is_correct(self):
        # shared: {gradient, optimizer} = 2; unique union = {gradient, descent, optimizer, momentum} = 4
        d = compute_answer_divergence(
            "gradient descent optimizer", "gradient momentum optimizer"
        )
        assert d == pytest.approx(0.5, abs=1e-9), f"Expected 0.5, got {d}"

    def test_both_empty_returns_zero(self):
        assert compute_answer_divergence("", "") == pytest.approx(0.0)

    def test_one_empty_returns_one(self):
        assert compute_answer_divergence("word", "") == pytest.approx(1.0)
        assert compute_answer_divergence("", "word") == pytest.approx(1.0)

    def test_case_insensitive(self):
        d = compute_answer_divergence("Gradient Descent", "gradient descent")
        assert d == pytest.approx(0.0), "Divergence must be case-insensitive"

    def test_divergence_symmetric(self):
        a = "neural network weights"
        b = "backpropagation chain rule"
        assert compute_answer_divergence(a, b) == pytest.approx(
            compute_answer_divergence(b, a), abs=1e-12
        )

    def test_divergence_bounded_between_0_and_1(self):
        pairs = [
            ("the cat sat on the mat", "a dog ran across a field"),
            ("loss function", "loss function gradient"),
            ("", "non empty"),
        ]
        for a, b in pairs:
            d = compute_answer_divergence(a, b)
            assert 0.0 <= d <= 1.0, f"Divergence out of [0,1]: {d} for ({a!r}, {b!r})"
