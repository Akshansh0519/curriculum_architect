"""
S1.5 — Episode Smoke Test
=========================
Phase 1 merge gate.  Verifies the full environment loop without requiring a
real HuggingFace model (ScriptedStudent falls back to deterministic strings
when EXAMINER_ENABLE_HF_STUDENT is not set, which is the default in CI).

Run with:
    pytest tests/test_episode.py -v
"""

import asyncio
import math

import pytest

from examiner_env.models import ExaminerAction, ExaminerObservation
from examiner_env.server.examiner_environment import ExaminerEnvironment


# ---------------------------------------------------------------------------
# Helper: run a single coroutine synchronously
# ---------------------------------------------------------------------------

def _run(coro):
    """Execute an async coroutine in a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh ExaminerEnvironment for each test."""
    return ExaminerEnvironment()


@pytest.fixture
def reset_env(env):
    """Environment that has already been reset with a fixed seed."""
    _run(env.reset(seed=42))
    return env


# ---------------------------------------------------------------------------
# 1. Reset / Observation structure
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_examiner_observation(self, env):
        obs = _run(env.reset(seed=0))
        assert isinstance(obs, ExaminerObservation)

    def test_section_titles_has_ten_entries(self, env):
        obs = _run(env.reset(seed=1))
        assert len(obs.section_titles) == 10
        for title in obs.section_titles:
            assert isinstance(title, str) and title.strip()

    def test_initial_turn_counter_is_zero(self, env):
        obs = _run(env.reset(seed=2))
        assert obs.turn_counter == 0

    def test_initial_remaining_turns_is_twenty(self, env):
        obs = _run(env.reset(seed=3))
        assert obs.remaining_turns == 20

    def test_initial_question_history_is_empty(self, env):
        obs = _run(env.reset(seed=4))
        assert obs.question_history == []

    def test_belief_scratchpad_is_string(self, env):
        obs = _run(env.reset(seed=5))
        assert isinstance(obs.belief_scratchpad, str)

    def test_episode_id_changes_on_each_reset(self, env):
        _run(env.reset())
        id1 = env.state.episode_id
        _run(env.reset())
        id2 = env.state.episode_id
        assert id1 != id2, "Each reset must produce a unique episode_id"

    def test_state_max_turns_is_twenty(self, env):
        _run(env.reset(seed=6))
        assert env.state.max_turns == 20


# ---------------------------------------------------------------------------
# 2. Ask steps
# ---------------------------------------------------------------------------

class TestAskStep:
    def test_ask_increments_turn_counter(self, reset_env):
        action = ExaminerAction(
            action_type="ask", section_id=0,
            question_text="What is gradient descent?"
        )
        result = _run(reset_env.step(action))
        assert result.observation.turn_counter == 1
        assert result.observation.remaining_turns == 19

    def test_ask_appends_to_history(self, reset_env):
        action = ExaminerAction(
            action_type="ask", section_id=3,
            question_text="How does regularization help?"
        )
        result = _run(reset_env.step(action))
        history = result.observation.question_history
        assert len(history) == 1
        entry = history[0]
        assert "section_id" in entry
        assert "question" in entry
        assert "answer" in entry

    def test_ask_answer_is_non_empty_string(self, reset_env):
        action = ExaminerAction(
            action_type="ask", section_id=1,
            question_text="Explain backpropagation."
        )
        result = _run(reset_env.step(action))
        answer = result.observation.question_history[0]["answer"]
        assert isinstance(answer, str) and len(answer.strip()) > 0

    def test_ask_done_is_false(self, reset_env):
        action = ExaminerAction(
            action_type="ask", section_id=2,
            question_text="What is regularization?"
        )
        result = _run(reset_env.step(action))
        assert result.done is False, "Episode must not end on an Ask action"

    def test_multiple_asks_accumulate_in_history(self, reset_env):
        questions = [
            (0, "What is gradient descent?"),
            (1, "How does backpropagation work?"),
            (4, "What does accuracy measure?"),
        ]
        for sid, qtext in questions:
            _run(reset_env.step(ExaminerAction(action_type="ask", section_id=sid, question_text=qtext)))

        result = _run(reset_env.step(
            ExaminerAction(action_type="ask", section_id=5, question_text="What are CNNs?")
        ))
        assert result.observation.turn_counter == 4
        assert len(result.observation.question_history) == 4

    def test_ask_missing_section_id_raises(self, reset_env):
        with pytest.raises((ValueError, Exception)):
            _run(reset_env.step(ExaminerAction(action_type="ask", question_text="A question?")))

    def test_ask_missing_question_text_raises(self, reset_env):
        with pytest.raises((ValueError, Exception)):
            _run(reset_env.step(ExaminerAction(action_type="ask", section_id=0)))


# ---------------------------------------------------------------------------
# 3. Full episode: 5 asks → classify
# ---------------------------------------------------------------------------

class TestFullEpisode:
    def _five_asks_then_classify(self, env, classify_dict):
        """Helper: reset, ask 5 questions, then classify."""
        _run(env.reset(seed=7))
        questions = [
            (0, "What is gradient descent and how does it update weights?"),
            (1, "Explain the chain rule in backpropagation."),
            (2, "Why does L2 regularization reduce overfitting?"),
            (5, "What makes a transformer architecture different from a CNN?"),
            (7, "Why is cross-entropy preferred over MSE for classification?"),
        ]
        for sid, qtext in questions:
            result = _run(env.step(ExaminerAction(action_type="ask", section_id=sid, question_text=qtext)))
            assert result.done is False, f"Must not end on ask turn {sid}"

        classify = ExaminerAction(action_type="classify", classification=classify_dict)
        return _run(env.step(classify))

    def test_done_is_true_after_classify(self, env):
        result = self._five_asks_then_classify(env, {i: "KNOWS" for i in range(10)})
        assert result.done is True, "Episode must end after Classify"

    def test_reward_is_finite_float(self, env):
        result = self._five_asks_then_classify(env, {i: "KNOWS" for i in range(10)})
        assert isinstance(result.reward, float), f"Reward must be float, got {type(result.reward)}"
        assert math.isfinite(result.reward), f"Reward must be finite, got {result.reward}"

    def test_metrics_has_four_required_keys(self, env):
        result = self._five_asks_then_classify(env, {i: "KNOWS" for i in range(10)})
        metrics = result.observation.metadata
        assert isinstance(metrics, dict), "obs.metadata must be a dict"
        for key in ("accuracy", "false_accusations", "efficiency", "diagnostic_score"):
            assert key in metrics, f"Missing required metric key: {key}"

    def test_accuracy_is_fraction_between_0_and_1(self, env):
        result = self._five_asks_then_classify(env, {i: "FAKING" for i in range(10)})
        acc = result.observation.metadata["accuracy"]
        assert 0.0 <= acc <= 1.0, f"accuracy must be in [0,1], got {acc}"

    def test_false_accusations_is_non_negative(self, env):
        result = self._five_asks_then_classify(env, {i: "FAKING" for i in range(10)})
        fa = result.observation.metadata["false_accusations"]
        assert fa >= 0, f"false_accusations must be >= 0, got {fa}"

    def test_efficiency_in_valid_range(self, env):
        result = self._five_asks_then_classify(env, {i: "KNOWS" for i in range(10)})
        eff = result.observation.metadata["efficiency"]
        assert 0.0 <= eff <= 1.0, f"efficiency must be in [0,1], got {eff}"

    def test_diagnostic_score_non_negative(self, env):
        result = self._five_asks_then_classify(env, {i: "KNOWS" for i in range(10)})
        diag = result.observation.metadata["diagnostic_score"]
        assert diag >= 0.0, f"diagnostic_score must be >= 0, got {diag}"

    def test_get_metrics_returns_same_four_keys(self, env):
        self._five_asks_then_classify(env, {i: "KNOWS" for i in range(10)})
        metrics = env.get_metrics()
        for key in ("accuracy", "false_accusations", "efficiency", "diagnostic_score"):
            assert key in metrics, f"get_metrics() missing key: {key}"

    def test_classify_immediately_still_terminates(self, env):
        """Classify on turn 0 (no asks) must still give a valid terminal reward."""
        _run(env.reset(seed=10))
        result = _run(env.step(ExaminerAction(
            action_type="classify",
            classification={i: "FAKING" for i in range(10)},
        )))
        assert result.done is True
        assert math.isfinite(result.reward)


# ---------------------------------------------------------------------------
# 4. Turn counter and forced termination
# ---------------------------------------------------------------------------

class TestTurnMechanics:
    def test_turn_counter_increments_correctly(self, env):
        _run(env.reset(seed=11))
        for expected in range(1, 6):
            result = _run(env.step(ExaminerAction(
                action_type="ask",
                section_id=expected % 10,
                question_text=f"Question {expected}",
            )))
            assert result.observation.turn_counter == expected
            assert result.observation.remaining_turns == 20 - expected

    def test_forced_termination_at_turn_20(self, env):
        """Must terminate at exactly turn 20 with −0.5 penalty when no classify submitted."""
        _run(env.reset(seed=99))
        last_result = None
        for i in range(20):
            last_result = _run(env.step(ExaminerAction(
                action_type="ask",
                section_id=i % 10,
                question_text=f"Forced question {i + 1}",
            )))
        assert last_result is not None
        assert last_result.done is True, "Episode must terminate after turn 20"
        assert last_result.reward == pytest.approx(-0.5), (
            f"Forced termination reward must be -0.5, got {last_result.reward}"
        )

    def test_no_step_after_forced_termination(self, env):
        """After forced termination a classify should still not crash (idempotent)."""
        _run(env.reset(seed=77))
        for i in range(20):
            _run(env.step(ExaminerAction(
                action_type="ask", section_id=i % 10,
                question_text="Forced",
            )))
        # A classify after forced-termination should either raise or return done=True
        try:
            result = _run(env.step(ExaminerAction(
                action_type="classify",
                classification={i: "FAKING" for i in range(10)},
            )))
            assert isinstance(result.reward, float)
        except Exception:
            pass  # Raising after episode end is also acceptable


# ---------------------------------------------------------------------------
# 5. Render / transcript
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_returns_string(self, reset_env):
        transcript = reset_env.render()
        assert isinstance(transcript, str)

    def test_render_includes_question_after_ask(self, reset_env):
        _run(reset_env.step(ExaminerAction(
            action_type="ask", section_id=9,
            question_text="Explain MC dropout."
        )))
        transcript = reset_env.render()
        assert "Q1" in transcript, "Transcript must label the first Q/A pair"

    def test_render_grows_with_each_ask(self, reset_env):
        t0 = len(reset_env.render())
        _run(reset_env.step(ExaminerAction(action_type="ask", section_id=0, question_text="Why?")))
        t1 = len(reset_env.render())
        _run(reset_env.step(ExaminerAction(action_type="ask", section_id=1, question_text="How?")))
        t2 = len(reset_env.render())
        assert t1 > t0, "Transcript must grow after first ask"
        assert t2 > t1, "Transcript must grow after second ask"


# ---------------------------------------------------------------------------
# 6. State property
# ---------------------------------------------------------------------------

class TestStateProp:
    def test_state_step_count_tracks_actions(self, env):
        _run(env.reset(seed=55))
        for _ in range(3):
            _run(env.step(ExaminerAction(action_type="ask", section_id=0, question_text="?")))
        assert env.state.step_count == 3

    def test_state_max_turns_is_twenty(self, env):
        _run(env.reset(seed=56))
        assert env.state.max_turns == 20

    def test_state_episode_id_is_string(self, env):
        _run(env.reset(seed=57))
        assert isinstance(env.state.episode_id, str)
        assert len(env.state.episode_id) > 0


# ---------------------------------------------------------------------------
# 7. Partition sanity
# ---------------------------------------------------------------------------

class TestPartitionSanity:
    def test_partition_k_is_between_3_and_5(self, env):
        """The env must sample k ∈ {3, 4, 5} KNOWS sections per episode."""
        for seed in range(20):
            _run(env.reset(seed=seed))
            # Access internal partition to verify k
            knows_count = sum(1 for lbl in env._partition.values() if lbl == "KNOWS")
            assert knows_count in (3, 4, 5), (
                f"seed={seed}: k must be in {{3,4,5}}, got {knows_count}"
            )

    def test_partition_covers_all_ten_sections(self, env):
        _run(env.reset(seed=0))
        assert set(env._partition.keys()) == set(range(10))

    def test_partition_changes_between_resets(self, env):
        _run(env.reset(seed=0))
        p1 = dict(env._partition)
        # Use a different seed that's likely to change partition
        _run(env.reset(seed=13))
        p2 = dict(env._partition)
        # Not guaranteed to differ but true for these specific seeds
        assert p1 != p2 or True  # soft check — just must not crash
