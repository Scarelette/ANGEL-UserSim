import unittest

from model_usage.angel.demo_prompt import cap_sentences, clean_reply, is_refusal, too_similar


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


class TestCleanReply(unittest.TestCase):
    def test_prefers_patient_block(self):
        self.assertEqual(clean_reply("<think>x</think><patient>I slept badly.</patient>"), "I slept badly.")

    def test_strips_think_and_tags(self):
        self.assertEqual(clean_reply("<think>plan</think> Honestly, not great."), "Honestly, not great.")

    def test_state_word_reply_keeps_only_the_speech(self):
        raw = "<state>anxious, replaying the midterm</state>\n<word>Probably my midterm. I got a D.</word>"
        self.assertEqual(clean_reply(raw), "Probably my midterm. I got a D.")

    def test_unclosed_word_block(self):
        self.assertEqual(clean_reply("<state>tense</state><word>I guess I'm just tired"), "I guess I'm just tired")

    def test_state_without_word_is_dropped(self):
        self.assertEqual(clean_reply("<state>guarded</state> Fine, I suppose."), "Fine, I suppose.")

    def test_caps_sentences(self):
        self.assertEqual(cap_sentences("One. Two. Three.", 2), "One. Two.")


if __name__ == "__main__":
    unittest.main()
