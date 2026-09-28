import json
import unittest

from model_usage.angel.backends import _stub_observer_profile
from model_usage.angel.config import ObserverConfig
from model_usage.angel.observer import Observer
from model_usage.angel.observer_prompts import (
    SHORT_TO_JSON_SYSTEM_PROMPT,
    build_long_profile_messages,
    build_long_profile_user_prompt,
    clean_model_output,
    extract_first_json_object,
    parse_profile_json,
    strip_code_fences,
    strip_think_block,
)


class ScriptedBackend:
    """Returns queued strings in order, so retry behaviour is deterministic."""

    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0

    def generate(self, messages, **params):
        self.call_count += 1
        if not self.outputs:
            raise AssertionError("ScriptedBackend ran out of queued outputs")
        return self.outputs.pop(0)

    def unload(self):
        self.unloaded = True


VALID_JSON = json.dumps(_stub_observer_profile("Mara, 34."))


class TestPromptConstruction(unittest.TestCase):
    def test_short_text_is_embedded(self):
        prompt = build_long_profile_user_prompt("Mara, 34, exhausted.")
        self.assertIn("Mara, 34, exhausted.", prompt)

    def test_schema_and_diversity_present(self):
        prompt = build_long_profile_user_prompt("x")
        self.assertIn('"symptom_details"', prompt)
        self.assertIn("Diversity constraints", prompt)

    def test_messages_carry_the_system_prompt(self):
        messages = build_long_profile_messages("x")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SHORT_TO_JSON_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")


class TestJsonExtraction(unittest.TestCase):
    def test_strips_code_fences(self):
        self.assertEqual(strip_code_fences('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_strips_think_block(self):
        self.assertEqual(strip_think_block('<think>plan</think>{"a": 1}'), '{"a": 1}')

    def test_clean_handles_both(self):
        self.assertEqual(clean_model_output('<think>x</think>```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_extracts_object_from_surrounding_prose(self):
        self.assertEqual(extract_first_json_object('Here you go: {"a": 1} done'), '{"a": 1}')

    def test_handles_nested_objects(self):
        text = '{"a": {"b": {"c": 1}}}'
        self.assertEqual(extract_first_json_object(text), text)

    def test_brace_inside_string_does_not_truncate(self):
        # Upstream counted braces without string awareness, so a brace in the
        # prose cut the object short. Guard against that regressing.
        text = '{"note": "a { brace", "age": 34}'
        self.assertEqual(json.loads(extract_first_json_object(text))["age"], 34)

    def test_escaped_quote_inside_string(self):
        text = '{"note": "he said \\"hi\\" then left", "age": 7}'
        self.assertEqual(json.loads(extract_first_json_object(text))["age"], 7)

    def test_returns_none_without_json(self):
        self.assertIsNone(extract_first_json_object("no object here"))

    def test_parse_raises_on_missing_json(self):
        with self.assertRaises(ValueError):
            parse_profile_json("nothing to see")

    def test_parse_raises_on_malformed_json(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_profile_json('{"a": }')


class TestObserverExpand(unittest.TestCase):
    def make_observer(self, outputs, **config_kwargs):
        backend = ScriptedBackend(outputs)
        config = ObserverConfig(model_path="unused", **config_kwargs)
        return Observer(config, backend=backend), backend

    def test_expands_to_rich_profile(self):
        observer, backend = self.make_observer([VALID_JSON])
        result = observer.expand("Mara, 34, running on empty.", source_title="unit test")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(backend.call_count, 1)
        self.assertIn("identity", result.rich_profile)
        self.assertIn("presenting_problems", result.rich_profile)
        self.assertEqual(result.rich_profile["_meta"]["source_title"], "unit test")

    def test_keeps_raw_observer_output(self):
        observer, _ = self.make_observer([VALID_JSON])
        result = observer.expand("Mara, 34.")
        self.assertIn("symptom_details", result.observer_profile)
        self.assertTrue(result.raw_text)

    def test_retries_past_unparseable_output(self):
        observer, backend = self.make_observer(["sorry, no json", VALID_JSON])
        result = observer.expand("Mara, 34.")
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.errors), 1)

    def test_gives_up_after_max_attempts_with_diagnostics(self):
        observer, backend = self.make_observer(["bad", "still bad"], max_attempts=2)
        with self.assertRaises(ValueError) as ctx:
            observer.expand("Mara, 34.")
        self.assertEqual(backend.call_count, 2)
        self.assertIn("attempt 1", str(ctx.exception))
        self.assertIn("attempt 2", str(ctx.exception))

    def test_empty_short_text_rejected_before_any_generation(self):
        observer, backend = self.make_observer([VALID_JSON])
        with self.assertRaises(ValueError):
            observer.expand("   ")
        self.assertEqual(backend.call_count, 0)

    def test_unload_reaches_the_backend(self):
        observer, backend = self.make_observer([VALID_JSON])
        observer.unload()
        self.assertTrue(getattr(backend, "unloaded", False))


if __name__ == "__main__":
    unittest.main()
