"""Провайдер вправе не вернуть usage: пример первого вызова не должен падать."""

import io
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from openai.types.chat import ChatCompletion


class FirstCallTests(unittest.TestCase):
    def run_example(self, usage):
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
        with patch("openai.OpenAI") as client, redirect_stdout(output):
            client.return_value.chat.completions.create.return_value = response
            runpy.run_path(str(Path(__file__).with_name("01_first_call.py")))
        return output.getvalue()

    def test_prints_tokens_when_usage_is_available(self):
        output = self.run_example({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        self.assertIn("вход 3 | выход 4 | всего 7", output)

    def test_prints_answer_without_usage(self):
        output = self.run_example(None)
        self.assertIn("Учебный ответ", output)
        self.assertIn("Провайдер не вернул счётчик токенов", output)


if __name__ == "__main__":
    unittest.main()
