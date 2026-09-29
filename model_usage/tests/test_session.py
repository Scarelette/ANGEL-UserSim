"""Session-surface parity with the HTTP API version, plus the profile-identity fix."""

import copy
import json
import unittest
from pathlib import Path

from model_usage.angel import demo_prompt
from model_usage.angel.backends import StubBackend
from model_usage.angel.config import ActorConfig, ObserverConfig, RunnerConfig
from model_usage.angel.pipeline import AngelModel, profile_fingerprint

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = ROOT / "examples" / "example_custom_profile.json"


def make_model(**overrides):
    config = RunnerConfig(
        observer=ObserverConfig(model_path="unused"),
        actor=ActorConfig(model_path="unused"),
        jsonl_path=ROOT / "examples" / "profiles.jsonl",
        backend="stub",
        seed=1234,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return AngelModel(config, actor_backend=StubBackend(), observer_backend=StubBackend())


def load_example():
    return json.loads(EXAMPLE_PROFILE.read_text(encoding="utf-8"))


class TestProfileListing(unittest.TestCase):
    def test_lists_bundled_profiles_with_numeric_ids(self):
        model = make_model()
        profiles = model.list_profiles()
        self.assertGreater(len(profiles), 0)
        self.assertEqual(profiles[0]["id"], "0")
        self.assertTrue(profiles[0]["canonical_id"])
        self.assertTrue(profiles[0]["label"])

    def test_resolves_numeric_index(self):
        model = make_model()
        expected = model.list_profiles()[0]["canonical_id"]
        self.assertEqual(model.resolve_profile_id("0"), expected)

    def test_resolves_canonical_id(self):
        model = make_model()
        canonical = model.list_profiles()[1]["canonical_id"]
        self.assertEqual(model.resolve_profile_id(canonical), canonical)

    def test_out_of_range_index_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.resolve_profile_id("9999")

    def test_unknown_id_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.resolve_profile_id("no-such-profile")


class TestSend(unittest.TestCase):
    def test_first_message_with_builtin_profile(self):
        model = make_model()
        result = model.send("alice", "Hi, how are you feeling?", profile_id="0")
        self.assertTrue(result["reply"])
        self.assertEqual(result["session"]["user"], "alice")
        self.assertEqual(result["session"]["num_turns"], 1)
        # Every profile goes through the Observer: the Actor plays the long profile.
        self.assertTrue(result["model"]["observer_used"])

    def test_later_messages_reuse_the_profile(self):
        model = make_model()
        model.send("alice", "Hi there.", profile_id="0")
        second = model.send("alice", "Do you want to talk about it?")
        self.assertEqual(second["session"]["num_turns"], 2)

    def test_first_message_without_a_profile_is_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.send("alice", "Hi there.")

    def test_custom_rich_profile_accepted(self):
        model = make_model()
        result = model.send("bob", "Hello.", profile=load_example())
        self.assertEqual(result["session"]["profile"]["name"], "Sam")

    def test_non_rich_profile_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.send("bob", "Hello.", profile={"nope": True})

    def test_empty_user_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.send("  ", "Hello.", profile_id="0")

    def test_empty_message_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.send("alice", "   ", profile_id="0")

    def test_two_profile_sources_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.send("alice", "Hi.", profile_id="0", short_profile="Mara, 34.")

    def test_sessions_are_isolated_per_user(self):
        model = make_model()
        model.send("alice", "Hi.", profile_id="0", expand=False)
        model.send("bob", "Hi.", profile_id="1", expand=False)
        self.assertNotEqual(
            model.history("alice")["profile"]["profile_id"],
            model.history("bob")["profile"]["profile_id"],
        )

    def test_sessions_are_isolated_per_session_id(self):
        model = make_model()
        model.send("alice", "Hi.", profile_id="0", session_id="a")
        model.send("alice", "Hi.", profile_id="0", session_id="b")
        model.send("alice", "Again.", session_id="a")
        self.assertEqual(model.history("alice", session_id="a")["history"].__len__(), 4)
        self.assertEqual(model.history("alice", session_id="b")["history"].__len__(), 2)


class TestObserverPath(unittest.TestCase):
    def test_short_profile_runs_the_observer(self):
        model = make_model()
        result = model.send("carol", "Hi.", short_profile="Mara, 34, running on empty since spring.")
        self.assertTrue(result["model"]["observer_used"])
        self.assertEqual(result["session"]["profile"]["name"], "Mara")

    def test_observer_is_released_after_expansion_by_default(self):
        model = make_model()
        model.send("carol", "Hi.", short_profile="Mara, 34, exhausted.")
        self.assertIsNone(model._observer)

    def test_keep_both_retains_the_observer(self):
        model = make_model(keep_both_resident=True)
        model.send("carol", "Hi.", short_profile="Mara, 34, exhausted.")
        self.assertIsNotNone(model._observer)

    def test_expand_profile_can_be_called_standalone(self):
        model = make_model()
        result = model.expand_profile("Mara, 34, exhausted.", source_title="notes")
        self.assertIn("identity", result.rich_profile)
        self.assertEqual(result.rich_profile["_meta"]["source_title"], "notes")


class TestActorInputIsObserverOutput(unittest.TestCase):
    """Pins *which* profile the Actor role-plays: when stage 1 runs, the
    Actor's input is the Observer's output, not the profile as given."""

    def test_short_profile_actor_gets_the_expanded_profile(self):
        model = make_model()
        model.send("carol", "Hi.", short_profile="Mara, 34, running on empty since spring.")
        session = model.sessions["carol::default"]
        # The stub Observer invents this background; free text never contained it.
        self.assertIn("Works shift hours at a warehouse.", session.actor.profile["background"])
        self.assertIs(session.actor.rich_profile, session.actor.profile["_raw_profile"])
        self.assertTrue(session.actor.rich_profile["_meta"]["stage1_schema_adapted"])

    def test_expanded_profile_reaches_the_prompt(self):
        model = make_model()
        model.send("carol", "Hi.", short_profile="Mara, 34, exhausted.")
        system = model.sessions["carol::default"].actor.base_system_prompt
        self.assertIn("Works shift hours at a warehouse.", system)

    def test_rich_profile_is_expanded_by_default(self):
        model = make_model()
        result = model.send("dana", "Hi.", profile=load_example())
        self.assertTrue(result["model"]["observer_used"])
        self.assertTrue(model.sessions["dana::default"].actor.rich_profile["_meta"]["stage1_schema_adapted"])

    def test_expand_false_uses_an_already_expanded_profile_as_is(self):
        model = make_model()
        result = model.send("dana", "Hi.", profile=load_example(), expand=False)
        self.assertFalse(result["model"]["observer_used"])
        self.assertEqual(
            model.sessions["dana::default"].actor.profile["name"], "Sam"
        )

    def test_expand_true_routes_a_rich_profile_through_the_observer(self):
        model = make_model()
        result = model.send("dana", "Hi.", profile=load_example(), expand=True)
        self.assertTrue(result["model"]["observer_used"])
        session = model.sessions["dana::default"]
        self.assertTrue(session.actor.rich_profile["_meta"]["stage1_schema_adapted"])
        self.assertIn("Works shift hours at a warehouse.", session.actor.profile["background"])

    def test_expand_true_routes_a_builtin_profile_through_the_observer(self):
        model = make_model()
        result = model.send("dana", "Hi.", profile_id="0", expand=True)
        self.assertTrue(result["model"]["observer_used"])
        self.assertTrue(
            model.sessions["dana::default"].actor.rich_profile["_meta"]["stage1_schema_adapted"]
        )

    def test_expand_of_a_rich_profile_renders_it_as_stage1_input(self):
        # A structured profile is rendered to text before stage 1.
        model = make_model()
        model.expand_profile(load_example())
        observer_calls = model._observer.backend.calls
        user_message = observer_calls[-1][1]["content"]
        self.assertIn("Name: Sam", user_message)
        self.assertIn("Presenting problems:", user_message)

    def test_expand_of_a_rich_profile_inherits_its_source_title(self):
        model = make_model()
        result = model.expand_profile(load_example())
        self.assertEqual(result.rich_profile["_meta"]["source_title"], "Synthetic example profile")

    def test_rendered_profile_is_not_dumped_into_background(self):
        """Regression: `short_profile_text` is appended to `background` as one
        bullet. Feeding stage 1 a rich profile renders a multi-KB block, and
        passing that through would flatten it (newlines collapsed) into a single
        unreadable background entry that reaches the Actor's prompt."""
        model = make_model()
        rich = model.load_builtin_profile("0")["_raw_profile"]
        # The rendered block is genuinely large — that is what makes this a trap.
        self.assertGreater(len(demo_prompt.profile_to_short_text(rich)), 500)

        result = model.expand_profile(rich)
        longest = max(len(entry) for entry in result.rich_profile["background"])
        self.assertLess(longest, 500, "a rendered profile block leaked into background")

    def test_free_text_still_seeds_background(self):
        # The seed is correct behaviour for a genuine short note; only the dict
        # path suppresses it.
        model = make_model()
        note = "Mara, 34, running on empty since the spring restructure."
        result = model.expand_profile(note)
        self.assertIn(note, result.rich_profile["background"])

    def test_expanded_builtin_profile_keeps_the_prompt_compact(self):
        model = make_model()
        model.send("dana", "Hi.", profile_id="0", expand=True)
        actor = model.sessions["dana::default"].actor
        self.assertLess(len(actor.base_system_prompt), 8000)
        self.assertLess(max(len(b) for b in actor.rich_profile["background"]), 500)

    def test_expand_rejects_an_empty_source(self):
        model = make_model()
        with self.assertRaises(ValueError):
            model.expand_profile("   ")
        with self.assertRaises(ValueError):
            model.expand_profile({})


class TestHistoryResetEnd(unittest.TestCase):
    def test_history_returns_full_transcript(self):
        model = make_model()
        model.send("alice", "First.", profile_id="0")
        model.send("alice", "Second.")
        history = model.history("alice")["history"]
        self.assertEqual([turn["role"] for turn in history], ["user", "assistant", "user", "assistant"])
        self.assertEqual(history[0]["content"], "First.")
        self.assertEqual(history[2]["content"], "Second.")

    def test_history_on_unknown_session_raises(self):
        model = make_model()
        with self.assertRaises(KeyError):
            model.history("nobody")

    def test_reset_clears_transcript_but_keeps_profile(self):
        model = make_model()
        model.send("alice", "First.", profile_id="0")
        before = model.history("alice")["profile"]["profile_id"]
        model.reset("alice")
        self.assertEqual(model.history("alice")["history"], [])
        self.assertEqual(model.history("alice")["profile"]["profile_id"], before)

    def test_can_continue_after_reset_without_resupplying_profile(self):
        model = make_model()
        model.send("alice", "First.", profile_id="0")
        model.reset("alice")
        result = model.send("alice", "Starting over.")
        self.assertEqual(result["session"]["num_turns"], 1)

    def test_reset_on_unknown_session_raises(self):
        model = make_model()
        with self.assertRaises(KeyError):
            model.reset("nobody")

    def test_end_drops_the_session(self):
        model = make_model()
        model.send("alice", "First.", profile_id="0")
        self.assertEqual(model.end("alice")["status"], "ended")
        with self.assertRaises(KeyError):
            model.history("alice")

    def test_end_is_idempotent(self):
        model = make_model()
        self.assertEqual(model.end("ghost")["status"], "ended")


class TestProfileFingerprint(unittest.TestCase):
    """The API version compared profile_id only, so an edited custom profile that
    kept its name and source_title was silently ignored. These lock in the fix."""

    def test_edited_profile_with_same_name_restarts_the_session(self):
        model = make_model()
        original = load_example()
        model.send("dave", "Hi.", profile=original, expand=False)

        edited = copy.deepcopy(original)
        edited["presenting_problems"] = ["Completely different presenting problem"]
        # Same name and source_title => identical profile_id under the old scheme.
        self.assertEqual(edited["identity"]["name"], original["identity"]["name"])
        self.assertEqual(edited["identity"]["source_title"], original["identity"]["source_title"])

        result = model.send("dave", "Hi again.", profile=edited, expand=False)
        self.assertEqual(result["session"]["num_turns"], 1, "edited profile must restart the conversation")

    def test_edited_profile_actually_reaches_the_actor(self):
        model = make_model()
        original = load_example()
        model.send("dave", "Hi.", profile=original, expand=False)

        edited = copy.deepcopy(original)
        edited["presenting_problems"] = ["Completely different presenting problem"]
        model.send("dave", "Hi again.", profile=edited, expand=False)

        session = model.sessions["dave::default"]
        self.assertEqual(
            session.actor.profile["presenting_problems"], ["Completely different presenting problem"]
        )

    def test_identical_profile_continues_the_session(self):
        model = make_model()
        original = load_example()
        model.send("erin", "Hi.", profile=original)
        result = model.send("erin", "Still here.", profile=copy.deepcopy(original))
        self.assertEqual(result["session"]["num_turns"], 2, "identical profile must not restart")

    def test_fingerprint_is_stable_and_content_sensitive(self):
        original = load_example()
        edited = copy.deepcopy(original)
        edited["emotions"] = ["something else entirely"]
        self.assertEqual(profile_fingerprint(original), profile_fingerprint(copy.deepcopy(original)))
        self.assertNotEqual(profile_fingerprint(original), profile_fingerprint(edited))

    def test_switching_builtin_profiles_restarts(self):
        model = make_model()
        model.send("frank", "Hi.", profile_id="0", expand=False)
        result = model.send("frank", "Hi.", profile_id="1", expand=False)
        self.assertEqual(result["session"]["num_turns"], 1)


class TestStatusAndLifecycle(unittest.TestCase):
    def test_status_reports_sessions_and_config(self):
        model = make_model()
        model.send("alice", "Hi.", profile_id="0")
        status = model.status()
        self.assertTrue(status["ok"])
        self.assertEqual(status["active_sessions"], 1)
        self.assertEqual(status["backend"], "stub")

    def test_close_clears_sessions(self):
        model = make_model()
        model.send("alice", "Hi.", profile_id="0")
        model.close()
        self.assertEqual(model.sessions, {})

    def test_context_manager_closes(self):
        with make_model() as model:
            model.send("alice", "Hi.", profile_id="0")
        self.assertEqual(model.sessions, {})


class TestRetryLoopTerminates(unittest.TestCase):
    def test_persistent_refusal_does_not_loop_forever(self):
        """A backend that only ever refuses must still return, bounded by max_retries."""
        model = make_model()
        model._actor_backend = StubBackend(replies=["I don't know."])
        result = model.send("gary", "How are you?", profile_id="0")
        self.assertEqual(result["reply"], "I don't know.")

    def test_retry_count_is_bounded(self):
        model = make_model()
        backend = StubBackend(replies=["I don't know."])
        model._actor_backend = backend
        model.send("gary", "How are you?", profile_id="0")
        # 1 initial call + at most max_retries regenerations.
        self.assertLessEqual(len(backend.calls), 1 + model.config.actor.max_retries)


if __name__ == "__main__":
    unittest.main()
