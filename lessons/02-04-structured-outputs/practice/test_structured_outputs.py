"""Локальные проверки схемы, Pydantic и границ проверки фактов. API не вызываются."""

import io
import json
import os
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion
from pydantic import ValidationError

SOURCE = Path(__file__).with_name("01_structured_outputs.py")
VALID = {"request_id": 28, "category": "access", "contact_email": None}


class StructuredOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SOURCE.is_file(), "Лаборатория структурированного вывода ещё не реализована"
        cls.experiment = runpy.run_path(str(SOURCE))

    def parse(self, data):
        return self.experiment["parse_draft"](json.dumps(data))

    def response(self, text: str | None = None, finish="stop", **message_fields):
        return ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(VALID) if text is None else text,
                            **message_fields,
                        },
                    }
                ],
                "usage": None,
            }
        )

    def test_schema_requires_nullable_email_and_forbids_extra_keys(self):
        schema = self.experiment["TicketDraft"].model_json_schema()
        self.assertEqual(set(schema["required"]), set(VALID))
        self.assertFalse(schema["additionalProperties"])
        self.assertIn({"type": "null"}, schema["properties"]["contact_email"]["anyOf"])

    def test_valid_draft_preserves_integer_and_null(self):
        draft = self.parse(VALID)
        self.assertEqual(draft.model_dump(), VALID)
        self.assertIsNone(draft.contact_email)

    def test_missing_nullable_field_is_still_rejected(self):
        with self.assertRaises(ValidationError):
            self.parse({"request_id": 28, "category": "access"})

    def test_extra_keys_are_not_silently_ignored(self):
        with self.assertRaises(ValidationError):
            self.parse({**VALID, "is_admin": True})

    def test_string_and_boolean_ids_are_not_coerced_to_integer(self):
        for value in ["28", True]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.parse({**VALID, "request_id": value})

    def test_invalid_category_is_rejected_but_unknown_is_allowed(self):
        with self.assertRaises(ValidationError):
            self.parse({**VALID, "category": "administrator"})
        self.assertEqual(self.parse({**VALID, "category": "unknown"}).category, "unknown")

    def test_json_with_markdown_or_prose_is_not_repaired_silently(self):
        for text in ["```json\n" + json.dumps(VALID) + "\n```", "Ответ: " + json.dumps(VALID)]:
            with self.subTest(text=text[:10]), self.assertRaises(json.JSONDecodeError):
                self.experiment["parse_draft"](text)

    def test_duplicate_keys_are_rejected_before_last_value_wins(self):
        text = '{"request_id":28,"request_id":82,"category":"access","contact_email":null}'
        with self.assertRaises(ValueError):
            self.experiment["parse_draft"](text)

    def test_nonstandard_json_constants_are_rejected(self):
        with self.assertRaises(ValueError):
            self.experiment["parse_draft"]('{"request_id":NaN,"category":"access","contact_email":null}')

    def test_json_root_must_be_an_object(self):
        with self.assertRaises(ValidationError):
            self.parse([VALID])

    def test_wrong_id_passes_structure_but_not_known_fact_check(self):
        draft = self.parse({**VALID, "request_id": 82})
        self.assertEqual(draft.request_id, 82)
        with self.assertRaises(ValueError):
            self.experiment["verify_known_facts"](draft, expected_request_id=28, expected_email=None)

    def test_invented_email_is_not_confirmed_by_being_a_string(self):
        draft = self.parse({**VALID, "contact_email": "fictional@example.test"})
        with self.assertRaises(ValueError):
            self.experiment["verify_known_facts"](draft, expected_request_id=28, expected_email=None)

    def test_limited_fact_check_does_not_claim_to_verify_category(self):
        draft = self.parse({**VALID, "category": "billing"})
        self.assertIsNone(self.experiment["verify_known_facts"](draft, expected_request_id=28, expected_email=None))

    def test_response_format_modes_are_distinct_and_unknown_mode_is_rejected(self):
        build = self.experiment["make_response_format"]
        self.assertIsNone(build("prompt"))
        self.assertEqual(build("json"), {"type": "json_object"})
        schema = build("schema")
        self.assertEqual(schema["type"], "json_schema")
        self.assertTrue(schema["json_schema"]["strict"])
        self.assertEqual(schema["json_schema"]["schema"], self.experiment["TicketDraft"].model_json_schema())
        with self.assertRaises(ValueError):
            build("unknown")

    def test_prompt_mode_omits_response_format(self):
        client = Mock()
        client.chat.completions.create.return_value = self.response()
        self.experiment["request_completion"](client, "test-model", "prompt")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("response_format", kwargs)
        self.assertFalse(kwargs["stream"])

    def test_schema_mode_sends_schema_without_hidden_retry(self):
        client = Mock()
        client.chat.completions.create.return_value = self.response()
        self.experiment["request_completion"](client, "test-model", "schema")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")

    def test_incomplete_refusal_tool_and_missing_choices_are_not_parsed_as_json(self):
        tool = [{"id": "call-test", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
        responses = [
            self.response('{"request_id":', "length"),
            self.response(refusal="Учебный отказ"),
            self.response(finish="tool_calls", tool_calls=tool),
            self.response(" "),
            self.response().model_copy(update={"choices": []}),
        ]
        for response in responses:
            with self.subTest(response=response.id), self.assertRaises(ValueError):
                self.experiment["extract_json_text"](response)

    def test_safe_validation_errors_do_not_include_input_values(self):
        with self.assertRaises(ValidationError) as caught:
            self.parse({**VALID, "category": "sensitive-fixture-value"})
        safe = self.experiment["safe_validation_errors"](caught.exception)
        self.assertIn("category", str(safe))
        self.assertNotIn("sensitive-fixture-value", str(safe))

    def test_invalid_request_is_rejected_before_network_boundary(self):
        client = Mock()
        with self.assertRaises(ValueError):
            self.experiment["request_completion"](client, "test-model", "schema", max_tokens=0)
        client.chat.completions.create.assert_not_called()

    def test_provider_resolution_needs_model_and_has_no_cloud_fallback(self):
        resolve = self.experiment["resolve_provider"]
        self.assertEqual(resolve("local", "local-model", {})[0], "http://localhost:11434/v1")
        url, key, model = resolve("gemini", None, {"GEMINI_API_KEY": "fake-key", "MODEL_GEMINI": "chosen-model"})
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual((key, model), ("fake-key", "chosen-model"))
        with self.assertRaises(ValueError):
            resolve("gemini", None, {"GEMINI_API_KEY": "fake-key"})

    def test_demo_and_schema_preview_have_no_client_or_env_access(self):
        for args in [[], ["--show-schema"]]:
            output = io.StringIO()
            with (
                patch("sys.argv", [str(SOURCE), *args]),
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
            patch("sys.argv", [str(SOURCE), "--live", "--provider", "gemini", "--mode", "schema"]),
            patch.dict(os.environ, {"GEMINI_API_KEY": "secret-fixture-not-real", "MODEL_GEMINI": "test-model"}),
            patch("dotenv.load_dotenv"),
            patch("openai.OpenAI") as client,
            redirect_stdout(output),
        ):
            client.return_value.__enter__.return_value.chat.completions.create.return_value = self.response()
            runpy.run_path(str(SOURCE), run_name="__main__")
        self.assertEqual(client.call_args.kwargs["timeout"], 30.0)
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)
        self.assertNotIn("secret-fixture-not-real", output.getvalue())


if __name__ == "__main__":
    unittest.main()
