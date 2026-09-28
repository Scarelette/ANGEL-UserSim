import unittest

from model_usage.angel.postprocess import (
    FALLBACK_REPLY,
    cap_sentences,
    clean_reply,
    extract_state,
    is_refusal,
    looks_like_user_prompt_echo,
    normalize_format,
    postprocess_model_output,
    remove_duplicate_paragraphs,
    remove_role_leakage,
    remove_think_artifacts,
    too_similar,
    trim_truncated_ending,
)


class TestExtractState(unittest.TestCase):
    def test_pulls_state_and_strips_it(self):
        state, text = extract_state("<state>guarded</state>I'd rather not say.")
        self.assertEqual(state, "guarded")
        self.assertEqual(text, "I'd rather not say.")

    def test_absent_state_is_none(self):
        state, text = extract_state("Just tired.")
        self.assertIsNone(state)
        self.assertEqual(text, "Just tired.")


class TestThinkArtifacts(unittest.TestCase):
    def test_removes_closed_think_block(self):
        self.assertEqual(remove_think_artifacts("<think>plan</think>Hello."), "Hello.")

    def test_keeps_reply_after_dangling_close_tag(self):
        # A truncated opening tag must not swallow the actual reply.
        self.assertIn("Hello.", remove_think_artifacts("reasoning</think>Hello."))

    def test_empty_when_only_thinking(self):
        self.assertEqual(remove_think_artifacts("<think>only planning</think>"), "")


class TestRoleLeakage(unittest.TestCase):
    def test_strips_colon_prefix(self):
        self.assertEqual(remove_role_leakage("patient: I'm tired."), "I'm tired.")

    def test_strips_newline_prefix(self):
        self.assertEqual(remove_role_leakage("Assistant\n\nI'm tired."), "I'm tired.")

    def test_leaves_ordinary_text(self):
        self.assertEqual(remove_role_leakage("I saw my therapist."), "I saw my therapist.")


class TestNormalizeFormat(unittest.TestCase):
    def test_collapses_spaces_and_fixes_punctuation_gaps(self):
        self.assertEqual(normalize_format("I  am   fine ."), "I am fine.")

    def test_caps_blank_lines(self):
        self.assertEqual(normalize_format("a\n\n\n\nb"), "a\n\nb")


class TestDuplicateParagraphs(unittest.TestCase):
    def test_drops_case_insensitive_repeat(self):
        self.assertEqual(remove_duplicate_paragraphs("Same line.\n\nSAME LINE."), "Same line.")

    def test_keeps_distinct_paragraphs(self):
        self.assertEqual(remove_duplicate_paragraphs("One.\n\nTwo."), "One.\n\nTwo.")


class TestTrimTruncatedEnding(unittest.TestCase):
    def test_trims_dangling_fragment(self):
        self.assertEqual(trim_truncated_ending("I slept badly. Then I"), "I slept badly.")

    def test_leaves_complete_sentence(self):
        self.assertEqual(trim_truncated_ending("I slept badly."), "I slept badly.")

    def test_leaves_quote_closed_sentence(self):
        self.assertEqual(trim_truncated_ending('He said "fine."'), 'He said "fine."')

    def test_no_punctuation_at_all_is_kept(self):
        self.assertEqual(trim_truncated_ending("just tired"), "just tired")


class TestPromptEcho(unittest.TestCase):
    def test_detects_question_bank_echo(self):
        echo = "What worries you most? What triggers your anxiety? What helps you calm down?"
        self.assertTrue(looks_like_user_prompt_echo(echo))

    def test_many_questions_without_first_person(self):
        self.assertTrue(looks_like_user_prompt_echo("A? B? C? D? E?"))

    def test_normal_reply_is_not_echo(self):
        self.assertFalse(looks_like_user_prompt_echo("I worry about my job, mostly."))


class TestRefusalAndSimilarity(unittest.TestCase):
    def test_refusal_variants(self):
        for text in ("I don't know.", "I do not remember.", "I’m not sure.", "No idea.", ""):
            self.assertTrue(is_refusal(text), text)

    def test_real_answer_is_not_refusal(self):
        self.assertFalse(is_refusal("It started after the restructure in spring."))

    def test_near_duplicate_detected(self):
        prior = ["I just feel stuck and tired all the time lately"]
        self.assertTrue(too_similar("I feel stuck and tired all the time lately", prior))

    def test_short_text_never_similar(self):
        self.assertFalse(too_similar("Yeah.", ["Yeah."]))

    def test_distinct_text_not_similar(self):
        self.assertFalse(too_similar("My brother called on Tuesday.", ["I sleep badly most nights."]))


class TestCapSentences(unittest.TestCase):
    def test_caps_to_limit(self):
        self.assertEqual(cap_sentences("A. B. C. D. E.", max_sentences=2), "A. B.")

    def test_under_limit_untouched(self):
        self.assertEqual(cap_sentences("A. B.", max_sentences=4), "A. B.")


class TestCleanReplyChain(unittest.TestCase):
    def test_full_chain(self):
        raw = "<state>tense</state><think>plan</think>patient: I  slept badly . Then I"
        self.assertEqual(clean_reply(raw), "I slept badly.")

    def test_echo_becomes_fallback(self):
        raw = "What worries you most? What triggers your anxiety? What helps you calm down?"
        self.assertEqual(clean_reply(raw), FALLBACK_REPLY)

    def test_postprocess_returns_state_and_text(self):
        result = postprocess_model_output("<state>flat</state>Fine.")
        self.assertEqual(result["state"], "flat")
        self.assertEqual(result["clean_text"], "Fine.")

    def test_empty_input_is_safe(self):
        self.assertEqual(clean_reply(""), "")


if __name__ == "__main__":
    unittest.main()
