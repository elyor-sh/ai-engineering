"""Провайдер вправе не вернуть usage: пример первого вызова не должен падать."""

import io
import os
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from openai import APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletion


class FirstCallTests(unittest.TestCase):
    def setUp(self):
        self.before_request = ""
        self.client_options: dict[str, object] = {}

    def run_example(self, usage, error=None):
        response = ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Учебный ответ"}}],
                "usage": usage,
            }
        )
        output = io.StringIO()
        # Подменяем только границу внешнего API, ответ — настоящая структура SDK.
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "secret-fixture-not-a-real-key"}),
            patch("openai.OpenAI") as client,
            redirect_stdout(output),
        ):
            def complete(**kwargs):
                self.before_request = output.getvalue()
                if error is not None:
                    raise error
                return response

            client.return_value.chat.completions.create.side_effect = complete
            source = str(Path(__file__).with_name("01_first_call.py"))
            if error is None:
                runpy.run_path(source)
            else:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(source)
                self.assertEqual(stopped.exception.code, 1)
            self.client_options = dict(client.call_args.kwargs)
        return output.getvalue()

    def test_prints_tokens_when_usage_is_available(self):
        output = self.run_example({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        self.assertIn("вход 3 | выход 4 | всего 7", output)

    def test_prints_answer_without_usage(self):
        output = self.run_example(None)
        self.assertIn("Учебный ответ", output)
        self.assertIn("Провайдер не вернул счётчик токенов", output)

    def test_does_not_print_api_key(self):
        output = self.run_example(None)
        self.assertNotIn("secret-fixture-not-a-real-key", output)

    def test_announces_request_before_waiting_for_response(self):
        self.run_example(None)
        self.assertIn("Отправляю запрос", self.before_request)

    def test_uses_bounded_timeout_without_silent_retries(self):
        self.run_example(None)
        self.assertEqual(self.client_options.get("timeout"), 30.0)
        self.assertEqual(self.client_options.get("max_retries"), 0)

    def test_reports_timeout_without_traceback(self):
        output = self.run_example(None, error=APITimeoutError(request=Mock()))
        self.assertIn("тайм-аут", output.lower())
        self.assertNotIn("Traceback", output)

    def test_reports_service_unavailable_without_traceback(self):
        error = APIStatusError("service unavailable", response=Mock(status_code=503), body=None)
        output = self.run_example(None, error=error)
        self.assertIn("503", output)
        self.assertIn("недоступ", output.lower())
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
