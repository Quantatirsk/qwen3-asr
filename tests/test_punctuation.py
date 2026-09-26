import unittest
from unittest.mock import Mock, patch

from app.services.asr.punctuation import restore_sentence_ending


class SentenceEndingTest(unittest.TestCase):
    def test_only_final_mark_is_copied_preserving_asr_characters(self) -> None:
        for text, restored, mark in (
            ("Qwen-3, GPT API costs 3.14", "Qwen, 3, Gpt api costs 3,14\u3002", "."),
            ("Can Qwen-3 use v1.2", "Can qwen 3 use v1,2\uff1f", "?"),
            ("\u4f60\u597d\uff0cGPT-4", "\u4f60\u597d\uff0c\uff0cGpt-4\u3002", "."),
            ("\u4f60\u597d", "\u4f60\u597d\u3002", "\u3002"),
        ):
            with (
                self.subTest(text=text),
                patch(
                    "app.services.asr.punctuation.get_punctuation_model",
                    return_value=Mock(generate=Mock(return_value=[{"text": restored}])),
                ),
            ):
                self.assertEqual(restore_sentence_ending(text), text + mark)

    def test_empty_and_existing_terminal_marks_skip_model(self) -> None:
        with patch("app.services.asr.punctuation.get_punctuation_model") as model:
            for text in (
                "",
                " \n",
                "Hello.",
                "Why?",
                'He said "yes."',
                "\u4f60\u597d\u3002\u201d",
                "Wait\u2026",
            ):
                with self.subTest(text=text):
                    self.assertEqual(restore_sentence_ending(text), text)
        model.assert_not_called()

    def test_missing_prediction_has_no_default_and_errors_propagate(self) -> None:
        model = Mock()
        with patch(
            "app.services.asr.punctuation.get_punctuation_model", return_value=model
        ):
            model.generate.return_value = [{"text": "hello"}]
            self.assertEqual(restore_sentence_ending("hello"), "hello")
            model.generate.return_value = []
            with self.assertRaises(RuntimeError):
                restore_sentence_ending("hello")
            model.generate.side_effect = ValueError("Model failed")
            with self.assertRaisesRegex(ValueError, "Model failed"):
                restore_sentence_ending("hello")


if __name__ == "__main__":
    unittest.main()
