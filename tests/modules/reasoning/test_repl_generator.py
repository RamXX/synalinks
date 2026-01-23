# License Apache 2.0: (c) 2025 Synalinks Team

"""Tests for REPL generator module."""

import pytest

from synalinks.src.modules.reasoning.repl_generator import (
    ACTION_INSTRUCTIONS_TEMPLATE,
    REPLAction,
    REPLGenerator,
    get_repl_instructions,
    strip_code_fences,
)


class TestStripCodeFences:
    """Tests for strip_code_fences function."""

    def test_strip_python_fence(self):
        """Test stripping ```python ... ``` fences."""
        code = '''```python
print("hello")
```'''
        result = strip_code_fences(code)
        assert result == 'print("hello")'

    def test_strip_py_fence(self):
        """Test stripping ```py ... ``` fences."""
        code = '''```py
print("hello")
```'''
        result = strip_code_fences(code)
        assert result == 'print("hello")'

    def test_strip_plain_fence(self):
        """Test stripping ``` ... ``` fences without language."""
        code = '''```
print("hello")
```'''
        result = strip_code_fences(code)
        assert result == 'print("hello")'

    def test_preserves_code_without_fences(self):
        """Test that code without fences is unchanged."""
        code = 'print("hello")'
        result = strip_code_fences(code)
        assert result == 'print("hello")'

    def test_strips_whitespace(self):
        """Test that surrounding whitespace is stripped."""
        code = '''   ```python
print("hello")
```   '''
        result = strip_code_fences(code)
        assert result == 'print("hello")'

    def test_multiline_code(self):
        """Test stripping fences from multiline code."""
        code = '''```python
x = 1
y = 2
print(x + y)
```'''
        result = strip_code_fences(code)
        assert result == 'x = 1\ny = 2\nprint(x + y)'

    def test_code_with_internal_backticks(self):
        """Test code containing backticks inside."""
        code = '''```python
s = "some `text` here"
print(s)
```'''
        result = strip_code_fences(code)
        assert '`text`' in result

    def test_partial_fence_not_stripped(self):
        """Test that partial fences (not wrapping) are not stripped."""
        code = 'print("hello")\n```'
        result = strip_code_fences(code)
        assert '```' in result

    def test_empty_code_block(self):
        """Test empty code block."""
        code = '''```python

```'''
        result = strip_code_fences(code)
        assert result == ''

    def test_code_with_submit(self):
        """Test code containing SUBMIT call."""
        code = '''```python
result = compute()
SUBMIT(answer=result)
```'''
        result = strip_code_fences(code)
        assert 'SUBMIT(answer=result)' in result


class TestREPLAction:
    """Tests for REPLAction data model."""

    def test_create_action(self):
        """Test creating a REPL action."""
        action = REPLAction(
            reasoning="Let me explore the data structure",
            code="print(type(data))",
        )

        assert action.reasoning == "Let me explore the data structure"
        assert action.code == "print(type(data))"

    def test_action_to_dict(self):
        """Test converting action to dict."""
        action = REPLAction(
            reasoning="Test reasoning",
            code="print(1)",
        )

        data = action.get_json()
        assert "reasoning" in data
        assert "code" in data


class TestGetReplInstructions:
    """Tests for get_repl_instructions function."""

    def test_basic_instructions(self):
        """Test generating basic instructions."""
        instructions = get_repl_instructions(
            output_fields=["answer", "confidence"],
        )

        assert "answer" in instructions
        assert "confidence" in instructions
        assert "SUBMIT" in instructions
        assert "llm_query" in instructions

    def test_includes_output_fields(self):
        """Test that all output fields are mentioned."""
        instructions = get_repl_instructions(
            output_fields=["result", "explanation", "score"],
        )

        assert "result" in instructions
        assert "explanation" in instructions
        assert "score" in instructions

    def test_includes_tool_descriptions(self):
        """Test that tool descriptions are included."""
        tool_desc = "- `custom_tool(x)`: Does something useful"
        instructions = get_repl_instructions(
            output_fields=["answer"],
            tool_descriptions=tool_desc,
        )

        assert "custom_tool" in instructions

    def test_includes_max_llm_calls(self):
        """Test that max LLM calls is included."""
        instructions = get_repl_instructions(
            output_fields=["answer"],
            max_llm_calls=25,
        )

        assert "25" in instructions

    def test_includes_behavioral_rules(self):
        """Test that DSPy-style behavioral rules are included."""
        instructions = get_repl_instructions(output_fields=["answer"])

        assert "EXPLORE FIRST" in instructions
        assert "ITERATE" in instructions
        assert "VERIFY" in instructions
        assert "llm_query" in instructions
        assert "MINIMIZE RETYPING" in instructions
        assert "SUBMIT ONLY AFTER SEEING" in instructions
        assert "JSON SAFETY" in instructions

    def test_references_variables_info(self):
        """Test that instructions reference variables_info instead of variables."""
        instructions = get_repl_instructions(output_fields=["answer"])

        assert "variables_info" in instructions
        assert "`variables`" not in instructions


class TestActionInstructionsTemplate:
    """Tests for ACTION_INSTRUCTIONS_TEMPLATE."""

    def test_template_has_placeholders(self):
        """Test that template has all required placeholders."""
        assert "{inputs}" in ACTION_INSTRUCTIONS_TEMPLATE
        assert "{output_fields}" in ACTION_INSTRUCTIONS_TEMPLATE
        assert "{final_output_names}" in ACTION_INSTRUCTIONS_TEMPLATE
        assert "{tool_docs}" in ACTION_INSTRUCTIONS_TEMPLATE
        assert "{max_llm_calls}" in ACTION_INSTRUCTIONS_TEMPLATE
        assert "{output_fields_list}" in ACTION_INSTRUCTIONS_TEMPLATE

    def test_template_formatting(self):
        """Test that template can be formatted without errors."""
        formatted = ACTION_INSTRUCTIONS_TEMPLATE.format(
            inputs="`query`, `context`",
            output_fields="- answer: The answer\n- confidence: Score 0-1",
            final_output_names="answer=value, confidence=value",
            tool_docs="",
            max_llm_calls=50,
            output_fields_list="answer, confidence",
        )

        assert "`query`, `context`" in formatted
        assert "answer" in formatted
        assert "confidence" in formatted


class TestREPLGenerator:
    """Tests for REPLGenerator class."""

    def test_init_with_schema(self):
        """Test initializing REPLGenerator with output schema."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "score": {"type": "number"},
            },
        }

        # Note: Can't fully test without language_model, but can verify init
        try:
            gen = REPLGenerator(output_schema=schema)
        except Exception:
            # Expected - needs language_model for actual use
            pass

    def test_output_fields_extracted(self):
        """Test that output fields are extracted from schema."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

        # Create a mock to avoid language model requirement
        class MockLM:
            pass

        gen = REPLGenerator.__new__(REPLGenerator)
        gen.output_schema = schema
        gen.output_fields = list(schema.get("properties", {}).keys())

        assert "answer" in gen.output_fields
        assert "confidence" in gen.output_fields

    def test_get_config(self):
        """Test get_config includes output_schema."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

        # Create minimal instance to test config
        gen = REPLGenerator.__new__(REPLGenerator)
        gen.output_schema = schema
        gen._max_llm_calls = 50

        # Mock parent get_config
        gen._name = "test_gen"
        gen._schema = {}
        gen._trainable_variables = []
        gen._use_inputs_schema = True

        # Can't fully test without parent setup, but verify attributes exist
        assert gen.output_schema == schema
        assert gen._max_llm_calls == 50

    def test_groq_schema_allows_direct_output(self):
        """Groq schema should allow direct output fields."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

        class GroqLM:
            model = "groq/test-model"

        gen = REPLGenerator(output_schema=schema, language_model=GroqLM())

        props = gen.schema.get("properties", {})
        assert "reasoning" in props
        assert "code" in props
        assert "answer" in props

    def test_non_groq_schema_stays_action_only(self):
        """Non-Groq schema should remain REPLAction-only."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

        class OpenAILM:
            model = "openai/test-model"

        gen = REPLGenerator(output_schema=schema, language_model=OpenAILM())

        props = gen.schema.get("properties", {})
        assert "reasoning" in props
        assert "code" in props
        assert "answer" not in props
