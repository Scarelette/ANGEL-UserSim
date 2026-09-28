import json
import unittest
from pathlib import Path

from model_usage.angel.actor import Actor, profile_summary
from model_usage.angel.backends import StubBackend
from model_usage.angel.config import ActorConfig
from model_usage.angel.patient_profile import convert_rich_profile_to_internal

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = ROOT / "examples" / "example_custom_profile.json"


def rich_profile():
    return json.loads(EXAMPLE_PROFILE.read_text(encoding="utf-8"))


def make_actor(replies=None, prompt_style="patient_demo", **config_kwargs):
    config = ActorConfig(model_path="unused", prompt_style=prompt_style, **config_kwargs)
    return Actor(rich_profile(), config, backend=StubBackend(replies=replies))


class TestProfileBinding(unittest.TestCase):
    def test_rich_profile_is_converted(self):
        actor = make_actor()
        self.assertEqual(actor.profile["name"], "Sam")
        self.assertTrue(actor.profile["behavior_rules"])

    def test_rich_profile_is_retained_for_prompt_rendering(self):
        actor = make_actor()
        self.assertEqual(actor.rich_profile["identity"]["name"], "Sam")

    def test_internal_profile_accepted_directly(self):
        internal = convert_rich_profile_to_internal(rich_profile(), None)
        actor = Actor(internal, ActorConfig(model_path="unused"),
                      backend=StubBackend(), is_internal_profile=True)
        self.assertEqual(actor.profile["name"], "Sam")
        # _raw_profile round-trips the rich form, which patient_demo needs.
        self.assertEqual(actor.rich_profile["identity"]["name"], "Sam")

    def test_invalid_profile_rejected(self):
        with self.assertRaises(ValueError):
            Actor({"identity": {}}, ActorConfig(model_path="unused"), backend=StubBackend())

    def test_unknown_prompt_style_rejected(self):
        with self.assertRaises(ValueError):
            make_actor(prompt_style="nonsense")

    def test_set_profile_resets_conversation(self):
        actor = make_actor()
        actor.reply("Hello.")
        profile = rich_profile()
        profile["identity"]["name"] = "Other"
        actor.set_profile(profile)
        self.assertEqual(actor.num_turns, 0)
        self.assertEqual(actor.profile["name"], "Other")


class TestReply(unittest.TestCase):
    def test_reply_records_the_turn(self):
        actor = make_actor()
        self.assertTrue(actor.reply("Hi, how are you feeling?"))
        self.assertEqual(actor.num_turns, 1)

    def test_record_false_leaves_history_untouched(self):
        actor = make_actor()
        actor.reply("Hi.", record=False)
        self.assertEqual(actor.num_turns, 0)

    def test_empty_message_rejected(self):
        actor = make_actor()
        with self.assertRaises(ValueError):
            actor.reply("   ")

    def test_transcript_alternates_roles(self):
        actor = make_actor()
        actor.reply("One.")
        actor.reply("Two.")
        self.assertEqual(
            [t["role"] for t in actor.transcript()], ["user", "assistant", "user", "assistant"]
        )

    def test_reset_clears_history_and_keeps_profile(self):
        actor = make_actor()
        actor.reply("One.")
        actor.reset()
        self.assertEqual(actor.num_turns, 0)
        self.assertEqual(actor.profile["name"], "Sam")
        self.assertEqual(actor.conversation, [])


class TestPatientDemoPromptStyle(unittest.TestCase):
    """patient_demo is the default and must match the reference demo's behaviour."""

    def test_system_prompt_uses_the_demo_template(self):
        actor = make_actor()
        self.assertIn("You are role-playing as a mental health patient", actor.base_system_prompt)
        self.assertIn("Patient profile:", actor.base_system_prompt)
        # The profile arrives as the rendered short-text block.
        self.assertIn("Name: Sam", actor.base_system_prompt)
        self.assertIn("Presenting problems:", actor.base_system_prompt)

    def test_dynamic_state_starts_empty_not_profile_seeded(self):
        # the reference demo constructs PatientStateManager({}), so profile emotions do
        # NOT preload the dynamic state; they reach the model via the profile block.
        actor = make_actor()
        state = actor.state_manager.get_dynamic_state()
        self.assertEqual(state["current_emotions"], [])
        self.assertEqual(state["current_behaviors"], [])

    def test_state_block_is_appended_once_state_accumulates(self):
        actor = make_actor()
        actor.reply("Tell me about your family and your mother.")
        system = actor.backend.calls[-1][0]["content"]
        self.assertIn("Current emotions:", system)
        self.assertIn("Sensitive topics:", system)
        self.assertIn("family conflict", system)

    def test_length_cue_comes_last(self):
        actor = make_actor()
        actor.reply("Do you live alone?")
        system = actor.backend.calls[-1][0]["content"]
        self.assertIn("For THIS reply", system)
        self.assertTrue(system.rstrip().endswith("."), system[-80:])

    def test_full_history_is_sent_unwindowed(self):
        # the reference demo sends the entire conversation every turn.
        actor = make_actor(max_turns=2)
        for i in range(5):
            actor.reply(f"Question {i}?")
        self.assertEqual(actor.num_turns, 5)
        roles = [m["role"] for m in actor.backend.calls[-1]]
        self.assertEqual(roles.count("assistant"), 4)

    def test_patient_tag_block_is_preferred(self):
        actor = make_actor(replies=["<think>plan</think><patient>Just tired.</patient>"])
        self.assertEqual(actor.reply("How are you?"), "Just tired.")

    def test_duplicated_half_is_collapsed(self):
        half = "I have not been sleeping much and it is starting to really wear me down now."
        actor = make_actor(replies=[half + " " + half])
        self.assertEqual(actor.reply("How are you?"), half)

    def test_role_prefix_is_not_stripped(self):
        # Faithful to the reference demo: clean_reply there does no role-leakage strip.
        actor = make_actor(replies=["patient: I slept badly."])
        self.assertEqual(actor.reply("How did you sleep?"), "patient: I slept badly.")


class TestAngelEvalPromptStyle(unittest.TestCase):
    """The alternate path, reproducing the paper evaluation actor (experiments/profile_expansion)."""

    def test_system_prompt_uses_the_eval_template(self):
        actor = make_actor(prompt_style="angel_eval")
        self.assertIn("You are role-playing as a simulated patient", actor.base_system_prompt)
        self.assertIn("Identity:", actor.base_system_prompt)

    def test_dynamic_state_is_seeded_from_the_profile(self):
        actor = make_actor(prompt_style="angel_eval")
        self.assertTrue(actor.state_manager.get_dynamic_state()["current_emotions"])

    def test_history_is_windowed_by_max_turns(self):
        actor = make_actor(prompt_style="angel_eval", max_turns=2)
        for i in range(5):
            actor.reply(f"Question {i}?")
        roles = [m["role"] for m in actor.backend.calls[-1]]
        self.assertLessEqual(roles.count("assistant"), 2)

    def test_role_prefix_is_stripped(self):
        actor = make_actor(prompt_style="angel_eval", replies=["patient: I slept badly ."])
        self.assertEqual(actor.reply("How did you sleep?"), "I slept badly.")


class TestRegeneration(unittest.TestCase):
    def test_refusal_triggers_regeneration(self):
        actor = make_actor(replies=["I don't know.", "It started in the spring, after the move."])
        reply = actor.reply("When did it start?")
        self.assertNotIn("don't know", reply)
        self.assertGreaterEqual(len(actor.backend.calls), 2)

    def test_regeneration_is_bounded(self):
        actor = make_actor(replies=["I don't know."], max_retries=2)
        actor.reply("When did it start?")
        self.assertLessEqual(len(actor.backend.calls), 3)

    def test_no_repeat_ngram_is_passed_through(self):
        actor = make_actor()
        actor.reply("How are you?")
        self.assertEqual(actor.config.no_repeat_ngram_size, 3)


class TestProfileSummary(unittest.TestCase):
    def test_summary_exposes_public_fields_only(self):
        actor = make_actor()
        summary = profile_summary(actor.profile)
        self.assertEqual(set(summary), {"profile_id", "name", "age", "gender", "role", "source_title"})
        self.assertNotIn("_raw_profile", summary)


if __name__ == "__main__":
    unittest.main()
