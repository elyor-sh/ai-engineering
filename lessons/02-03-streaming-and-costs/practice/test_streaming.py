"""Проверки потока и условной цены. Настоящие API не вызываются."""

import io
import os
import runpy
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletionChunk

SOURCE = Path(__file__).with_name("01_streaming.py")
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SOURCE.is_file(), "Потребитель потока ещё не реализован"
        cls.experiment = runpy.run_path(str(SOURCE))

    def chunk(self, text=None, finish=None, usage=None, *, role=None, usage_only=False, **delta_fields):
        delta = dict(delta_fields)
        if text is not None:
            delta["content"] = text
        if role is not None:
            delta["role"] = role
        return ChatCompletionChunk.model_validate(
            {
                "id": "one-response-id",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [] if usage_only else [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": usage,
            }
        )

    def complete_chunks(self, usage=USAGE):
        chunks = [self.chunk(role="assistant"), self.chunk("Привет, "), self.chunk("Алекс!"), self.chunk(finish="stop")]
        if usage is not None:
            chunks.append(self.chunk(usage=usage, usage_only=True))
        return chunks

    def collect(self, chunks, on_text=None):
        return self.experiment["collect_stream"](chunks, on_text=on_text)

    def price(self, result):
        return self.experiment["estimate_cost"](result, Decimal(1), Decimal(5))

    def test_deltas_are_joined_without_stripping_spaces(self):
        callback = Mock()
        result = self.collect(self.complete_chunks(), on_text=callback)
        self.assertEqual(result.text, "Привет, Алекс!")
        self.assertTrue(result.accepted)
        self.assertEqual([call.args[0] for call in callback.call_args_list], ["Привет, ", "Алекс!"])

    def test_usage_only_event_after_stop_is_not_lost(self):
        result = self.collect(self.complete_chunks())
        self.assertEqual(result.usage.prompt_tokens, 100)
        self.assertEqual(result.usage.completion_tokens, 20)
        self.assertEqual(self.price(result), Decimal("0.0002"))

    def test_complete_text_without_usage_has_unknown_price(self):
        result = self.collect(self.complete_chunks(usage=None))
        self.assertTrue(result.accepted)
        self.assertIsNone(self.price(result))

    def test_eof_without_finish_reason_is_incomplete(self):
        result = self.collect([self.chunk("Черновик")])
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "incomplete")

    def test_length_is_not_accepted_but_can_have_reported_cost(self):
        result = self.collect([self.chunk("Обрезано"), self.chunk(finish="length"), self.chunk(usage=USAGE, usage_only=True)])
        self.assertFalse(result.accepted)
        self.assertTrue(result.transport_complete)
        self.assertEqual(self.price(result), Decimal("0.0002"))

    def test_interruption_preserves_partial_text_without_final_price(self):
        def broken():
            yield self.chunk("Часть")
            yield self.chunk(usage=USAGE, usage_only=True)
            raise ConnectionError("secret-fixture-must-not-be-logged")

        result = self.collect(broken())
        self.assertEqual(result.status, "error")
        self.assertEqual(result.text, "Часть")
        self.assertFalse(result.transport_complete)
        self.assertIsNone(self.price(result))
        self.assertNotIn("secret-fixture", result.error_kind or "")

    def test_user_cancellation_is_not_a_complete_answer(self):
        result = self.collect(self.complete_chunks(), on_text=Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(result.status, "cancelled")
        self.assertFalse(result.accepted)
        self.assertIsNone(self.price(result))

    def test_tools_and_refusal_are_unsupported_in_text_only_contract(self):
        tool = [{"index": 0, "id": "call-test", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
        cases = [
            [self.chunk(tool_calls=tool), self.chunk(finish="tool_calls")],
            [self.chunk(refusal="Учебный отказ"), self.chunk(finish="stop")],
        ]
        for chunks in cases:
            with self.subTest(chunks=len(chunks)):
                result = self.collect(chunks)
                self.assertEqual(result.status, "unsupported")
                self.assertFalse(result.accepted)

    def test_text_after_finish_is_a_protocol_error(self):
        result = self.collect([self.chunk("Начало"), self.chunk(finish="stop"), self.chunk("Лишнее")])
        self.assertEqual(result.status, "error")
        self.assertFalse(result.accepted)

    def test_usage_snapshots_replace_instead_of_sum(self):
        earlier = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
        result = self.collect([self.chunk("Текст", usage=earlier), self.chunk(finish="stop"), self.chunk(usage=USAGE, usage_only=True)])
        self.assertEqual(self.price(result), Decimal("0.0002"))

    def test_nested_cached_and_reasoning_counts_are_not_added_twice(self):
        usage = {**USAGE, "prompt_tokens_details": {"cached_tokens": 60}, "completion_tokens_details": {"reasoning_tokens": 10}}
        result = self.collect(self.complete_chunks(usage))
        self.assertEqual(self.price(result), Decimal("0.0002"))

    def test_reported_zero_and_unknown_are_distinct(self):
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.assertEqual(self.price(self.collect(self.complete_chunks(usage))), Decimal(0))
        self.assertIsNone(self.experiment["estimate_cost"](self.collect(self.complete_chunks()), None, None))

    def test_invalid_rates_are_rejected(self):
        result = self.collect(self.complete_chunks())
        for rate in [Decimal(-1), Decimal("NaN")]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                self.experiment["estimate_cost"](result, rate, Decimal(5))

    def test_history_commits_only_accepted_results_without_mutating_input(self):
        history = [("Раньше", "Ответ")]
        commit = self.experiment["commit_turn"]
        success = self.collect(self.complete_chunks())
        self.assertEqual(commit(history, "Привет", success), history + [("Привет", "Привет, Алекс!")])
        incomplete = self.collect([self.chunk("Часть")])
        self.assertEqual(commit(history, "Привет", incomplete), history)
        self.assertEqual(history, [("Раньше", "Ответ")])

    def test_stream_request_uses_streaming_and_optional_usage_request(self):
        client = Mock()
        stream = client.chat.completions.create.return_value
        stream.__enter__ = Mock(return_value=self.complete_chunks())
        stream.__exit__ = Mock(return_value=False)
        result = self.experiment["stream_request"](client, "test-model", "Привет", request_usage=True)
        self.assertTrue(result.accepted)
        request = client.chat.completions.create.call_args.kwargs
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        stream.__exit__.assert_called_once()

    def test_usage_option_is_omitted_unless_explicitly_requested(self):
        client = Mock()
        stream = client.chat.completions.create.return_value
        stream.__enter__ = Mock(return_value=self.complete_chunks())
        stream.__exit__ = Mock(return_value=False)
        self.experiment["stream_request"](client, "test-model", "Привет")
        self.assertNotIn("stream_options", client.chat.completions.create.call_args.kwargs)

    def test_stream_resource_is_closed_after_error_and_cancellation(self):
        def broken():
            yield self.chunk("Часть")
            raise ConnectionError("demo error")

        for events, callback in [(broken(), None), (self.complete_chunks(), Mock(side_effect=KeyboardInterrupt))]:
            with self.subTest(callback=callback is not None):
                client = Mock()
                stream = client.chat.completions.create.return_value
                stream.__enter__ = Mock(return_value=events)
                stream.__exit__ = Mock(return_value=False)
                result = self.experiment["stream_request"](client, "test-model", "Привет", on_text=callback)
                self.assertFalse(result.accepted)
                stream.__exit__.assert_called_once()

    def test_invalid_request_is_rejected_before_call(self):
        client = Mock()
        with self.assertRaises(ValueError):
            self.experiment["stream_request"](client, "test-model", " ")
        client.chat.completions.create.assert_not_called()

    def test_provider_resolution_requires_explicit_model_and_no_cross_provider_fallback(self):
        resolve = self.experiment["resolve_provider"]
        url, key, model = resolve("gemini", None, {"GEMINI_API_KEY": "fake-key", "MODEL_GEMINI": "chosen-model"})
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual((key, model), ("fake-key", "chosen-model"))
        self.assertEqual(resolve("local", "local-model", {})[0], "http://localhost:11434/v1")
        with self.assertRaises(ValueError):
            resolve("gemini", None, {"GEMINI_API_KEY": "fake-key"})

    def test_default_demo_has_no_client_or_env_access(self):
        output = io.StringIO()
        with (
            patch("sys.argv", [str(SOURCE)]),
            patch("openai.OpenAI") as client,
            patch("dotenv.load_dotenv") as env,
            redirect_stdout(output),
        ):
            runpy.run_path(str(SOURCE), run_name="__main__")
        client.assert_not_called()
        env.assert_not_called()
        self.assertIn("Без сети", output.getvalue())

    def test_live_branch_uses_bounded_client_and_never_prints_key(self):
        output = io.StringIO()
        with (
            patch("sys.argv", [str(SOURCE), "--live", "--provider", "gemini"]),
            patch.dict(os.environ, {"GEMINI_API_KEY": "secret-fixture-not-real", "MODEL_GEMINI": "test-model"}),
            patch("dotenv.load_dotenv"),
            patch("openai.OpenAI") as client,
            redirect_stdout(output),
        ):
            stream = client.return_value.__enter__.return_value.chat.completions.create.return_value
            stream.__enter__.return_value = self.complete_chunks(usage=None)
            runpy.run_path(str(SOURCE), run_name="__main__")
        self.assertEqual(client.call_args.kwargs["timeout"], 30.0)
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)
        self.assertNotIn("secret-fixture-not-real", output.getvalue())


if __name__ == "__main__":
    unittest.main()
