import json
import unittest

from model_usage.angel.backends import StubBackend, _stub_observer_profile
from model_usage.angel.patient_profile import convert_rich_profile_to_internal, is_rich_profile_schema
from model_usage.angel.schema_adapter import (
    RICH_REQUIRED_FIELDS,
    adapt_observer_profile,
    age_level_from_identity,
    build_minimal_rich_profile,
    coerce_list,
    is_rich_schema,
    merge_lists,
)


class TestCoerceList(unittest.TestCase):
    def test_scalar_becomes_single_item(self):
        self.assertEqual(coerce_list("tired"), ["tired"])

    def test_none_and_empty_drop_out(self):
        self.assertEqual(coerce_list(None), [])
        self.assertEqual(coerce_list(""), [])
        self.assertEqual(coerce_list(["", None]), [])

    def test_nested_lists_flatten(self):
        self.assertEqual(coerce_list([["a"], ["b", ["c"]]]), ["a", "b", "c"])

    def test_whitespace_is_collapsed(self):
        self.assertEqual(coerce_list("a   b\n c"), ["a b c"])

    def test_dict_values_are_used(self):
        self.assertEqual(coerce_list({"x": "a", "y": "b"}), ["a", "b"])

    def test_non_string_scalar(self):
        self.assertEqual(coerce_list(31), ["31"])


class TestMergeLists(unittest.TestCase):
    def test_dedupes_case_insensitively_preserving_order(self):
        self.assertEqual(merge_lists("Tired", ["tired", "Numb"]), ["Tired", "Numb"])

    def test_skips_empty_sources(self):
        self.assertEqual(merge_lists(None, [], "a"), ["a"])


class TestAgeLevel(unittest.TestCase):
    def test_numeric_minor(self):
        self.assertEqual(age_level_from_identity({"age": 14}), "adolescent")

    def test_numeric_adult(self):
        self.assertEqual(age_level_from_identity({"age": 31}), "adult")

    def test_string_age_is_parsed(self):
        self.assertEqual(age_level_from_identity({"age": "15 years old"}), "adolescent")

    def test_missing_age_defaults_adult(self):
        self.assertEqual(age_level_from_identity({}), "adult")

    def test_unparseable_age_defaults_adult(self):
        self.assertEqual(age_level_from_identity({"age": "unknown"}), "adult")


class TestAdaptObserverProfile(unittest.TestCase):
    def setUp(self):
        self.observer = _stub_observer_profile("Mara, 34, running on empty since spring.")
        self.rich = adapt_observer_profile(
            self.observer, source_title="unit test", short_profile_text="short text here"
        )

    def test_output_has_every_rich_field(self):
        for field in RICH_REQUIRED_FIELDS:
            self.assertIn(field, self.rich, field)

    def test_output_passes_rich_schema_checks(self):
        self.assertTrue(is_rich_schema(self.rich))
        self.assertTrue(is_rich_profile_schema(self.rich))

    def test_output_converts_for_the_actor(self):
        # The real contract: whatever stage 1 produces must be loadable by stage 2.
        internal = convert_rich_profile_to_internal(self.rich, None)
        self.assertTrue(internal["name"])
        self.assertTrue(internal["presenting_problems"])
        self.assertTrue(internal["behavior_rules"])

    def test_nested_sections_are_flattened_into_lists(self):
        self.assertIn("Sunday evenings", self.rich["triggers"])
        self.assertIn("flat", self.rich["emotions"])
        self.assertIn("Screening calls", self.rich["behaviors"])

    def test_cognitive_material_reaches_hidden_state(self):
        self.assertIn("I let people down", self.rich["hidden_state"])

    def test_short_text_is_preserved_in_background(self):
        self.assertIn("short text here", self.rich["background"])

    def test_speaking_style_falls_back_when_absent(self):
        rich = adapt_observer_profile({"identity": {"name": "X"}}, source_title="t")
        self.assertEqual(rich["speaking_style"]["tone"], "natural and conversational")

    def test_disclosure_rules_are_flag_objects(self):
        self.assertIs(self.rich["disclosure_rules"]["reveal_gradually"], True)
        self.assertIsInstance(self.rich["disclosure_rules"]["topics_likely_late"], list)

    def test_observer_prose_rules_are_kept_not_dropped(self):
        # Stage 1 emits simulation_rules as prose; the rich schema wants flags.
        # The prose must survive somewhere rather than vanish.
        self.assertIn("observer_notes", self.rich["simulation_rules"])
        self.assertTrue(self.rich["simulation_rules"]["observer_notes"])
        self.assertIs(self.rich["simulation_rules"]["stay_in_character"], True)

    def test_empty_observer_dict_still_produces_valid_profile(self):
        rich = adapt_observer_profile({}, source_title="t", short_profile_text="some text")
        convert_rich_profile_to_internal(rich, None)  # must not raise

    def test_non_dict_input_rejected(self):
        with self.assertRaises(TypeError):
            adapt_observer_profile(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_list_shaped_hidden_state_is_accepted(self):
        observer = dict(self.observer)
        observer["hidden_state"] = ["a secret", "another"]
        rich = adapt_observer_profile(observer)
        self.assertIn("a secret", rich["hidden_state"])


class TestMinimalFallback(unittest.TestCase):
    def test_builds_usable_profile_from_text(self):
        rich = build_minimal_rich_profile("Adult with low mood.", source_title="fallback")
        internal = convert_rich_profile_to_internal(rich, None)
        self.assertEqual(internal["presenting_problems"], ["Adult with low mood."])

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            build_minimal_rich_profile("   ")


class TestStubObserverIsRealistic(unittest.TestCase):
    def test_stub_output_parses_as_json(self):
        backend = StubBackend()
        text = backend.generate(
            [
                {"role": "system", "content": "You are an expert clinical case-profile writer for research."},
                {"role": "user", "content": "Short patient description:\nMara, 34.\n\nGenerate a profile."},
            ]
        )
        parsed = json.loads(text)
        self.assertIn("identity", parsed)
        self.assertEqual(parsed["identity"]["name"], "Mara")
        self.assertEqual(parsed["identity"]["age"], 34)


if __name__ == "__main__":
    unittest.main()
