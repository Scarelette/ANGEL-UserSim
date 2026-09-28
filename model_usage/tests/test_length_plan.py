import random
import unittest

from model_usage.angel.length_plan import plan_reply_length


def fixed_rng(value):
    """RNG whose random() always returns `value`, to remove turn-to-turn variance."""
    rng = random.Random()
    rng.random = lambda: value  # type: ignore[assignment]
    return rng


NEUTRAL = 0.5  # between the 0.25 and 0.9 variance thresholds


class TestQuestionType(unittest.TestCase):
    def test_closed_question_is_terse(self):
        plan = plan_reply_length("Do you live alone?", None, 180, rng=fixed_rng(NEUTRAL))
        self.assertEqual(plan.max_sentences, 1)
        self.assertEqual(plan.budget_tokens, 26)
        self.assertEqual(plan.min_new_tokens, 0)

    def test_factual_how_many_stays_terse(self):
        # _CLOSED_RE is checked first, so "how old" must not read as open.
        plan = plan_reply_length("How old are you?", None, 180, rng=fixed_rng(NEUTRAL))
        self.assertEqual(plan.max_sentences, 1)

    def test_open_question_gets_room(self):
        plan = plan_reply_length(
            "What has that been like for you?", None, 180, rng=fixed_rng(NEUTRAL)
        )
        self.assertGreaterEqual(plan.max_sentences, 2)
        self.assertEqual(plan.min_new_tokens, 12)

    def test_long_statement_is_brief(self):
        plan = plan_reply_length(
            "I hear you and I want to sit with that for a moment together",
            None,
            180,
            rng=fixed_rng(NEUTRAL),
        )
        self.assertEqual(plan.max_sentences, 2)


class TestStateInfluence(unittest.TestCase):
    def test_guarded_state_shortens(self):
        state = {"current_behaviors": ["becoming guarded"], "current_emotions": []}
        guarded = plan_reply_length("Tell me more about that.", state, 180, rng=fixed_rng(NEUTRAL))
        neutral = plan_reply_length("Tell me more about that.", None, 180, rng=fixed_rng(NEUTRAL))
        self.assertLessEqual(guarded.max_sentences, neutral.max_sentences)

    def test_opening_up_allows_more(self):
        state = {"current_behaviors": [], "current_emotions": ["slightly more open"]}
        opening = plan_reply_length("Do you live alone?", state, 180, rng=fixed_rng(NEUTRAL))
        neutral = plan_reply_length("Do you live alone?", None, 180, rng=fixed_rng(NEUTRAL))
        self.assertGreaterEqual(opening.max_sentences, neutral.max_sentences)

    def test_open_question_floor_survives_shutdown_state(self):
        # An open question must never collapse to the terse mode.
        state = {"current_behaviors": ["guarded", "withdrawing", "avoidance"], "current_emotions": []}
        plan = plan_reply_length("What has that been like?", state, 180, rng=fixed_rng(0.0))
        self.assertGreaterEqual(plan.max_sentences, 2)


class TestCeiling(unittest.TestCase):
    def test_budget_never_exceeds_ceiling(self):
        for message in ("Do you live alone?", "What has that been like for you?", "Tell me more."):
            for roll in (0.0, 0.5, 0.95):
                plan = plan_reply_length(message, None, 20, rng=fixed_rng(roll))
                self.assertLessEqual(plan.budget_tokens, 20, (message, roll))

    def test_empty_message_is_handled(self):
        plan = plan_reply_length("", None, 180, rng=fixed_rng(NEUTRAL))
        self.assertGreater(plan.budget_tokens, 0)

    def test_none_message_is_handled(self):
        plan = plan_reply_length(None, None, 180, rng=fixed_rng(NEUTRAL))
        self.assertGreater(plan.budget_tokens, 0)

    def test_cue_is_always_present(self):
        plan = plan_reply_length("Why do you think that is?", None, 180, rng=fixed_rng(NEUTRAL))
        self.assertTrue(plan.cue.strip())


if __name__ == "__main__":
    unittest.main()
