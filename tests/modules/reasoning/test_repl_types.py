# License Apache 2.0: (c) 2025 Synalinks Team

"""Tests for REPL types."""

import pytest
from pydantic import Field as PydanticField

from synalinks.src.modules.reasoning.repl_types import (
    REPLEntry,
    REPLHistory,
    REPLVariable,
)


class TestREPLVariable:
    """Tests for REPLVariable data model."""

    def test_create_from_string_value(self):
        """Test creating REPLVariable from a string value."""
        variable = REPLVariable.from_value(
            name="text",
            value="Hello, world!",
        )

        assert variable.name == "text"
        assert variable.type_name == "str"
        assert variable.desc == ""
        assert variable.constraints == ""
        assert variable.total_length == 13
        assert variable.preview == "Hello, world!"

    def test_create_from_dict_value(self):
        """Test creating REPLVariable from a dict value."""
        variable = REPLVariable.from_value(
            name="data",
            value={"key": "value", "count": 42},
        )

        assert variable.name == "data"
        assert variable.type_name == "dict"
        assert "key" in variable.preview
        assert "value" in variable.preview

    def test_create_from_list_value(self):
        """Test creating REPLVariable from a list value."""
        variable = REPLVariable.from_value(
            name="items",
            value=[1, 2, 3, 4, 5],
        )

        assert variable.name == "items"
        assert variable.type_name == "list"
        assert "[" in variable.preview

    def test_preview_truncation(self):
        """Test that preview is truncated for long values."""
        long_text = "a" * 1000
        variable = REPLVariable.from_value(
            name="long_text",
            value=long_text,
            preview_chars=100,
        )

        assert variable.total_length == 1000
        assert len(variable.preview) == 103  # 100 chars + "..."
        assert variable.preview.endswith("...")

    def test_preview_no_truncation_for_short_values(self):
        """Test that short values are not truncated."""
        short_text = "short"
        variable = REPLVariable.from_value(
            name="short",
            value=short_text,
            preview_chars=100,
        )

        assert variable.preview == "short"
        assert not variable.preview.endswith("...")

    def test_with_field_info_description(self):
        """Test extracting description from field info."""

        class MockFieldInfo:
            description = "This is a user query"
            json_schema_extra = None

        variable = REPLVariable.from_value(
            name="query",
            value="test query",
            field_info=MockFieldInfo(),
        )

        assert variable.desc == "This is a user query"

    def test_with_field_info_constraints(self):
        """Test extracting constraints from field info."""

        class MockFieldInfo:
            description = "A number"
            json_schema_extra = {"constraints": "Must be positive"}

        variable = REPLVariable.from_value(
            name="count",
            value=42,
            field_info=MockFieldInfo(),
        )

        assert variable.desc == "A number"
        assert variable.constraints == "Must be positive"

    def test_skips_placeholder_descriptions(self):
        """Test that placeholder descriptions like ${name} are skipped."""

        class MockFieldInfo:
            description = "${query}"
            json_schema_extra = None

        variable = REPLVariable.from_value(
            name="query",
            value="test",
            field_info=MockFieldInfo(),
        )

        assert variable.desc == ""

    def test_format_output(self):
        """Test format() produces expected output."""
        variable = REPLVariable(
            name="data",
            type_name="dict",
            desc="Input data",
            constraints="Non-empty",
            total_length=100,
            preview='{"key": "value"}',
        )

        formatted = variable.format()

        assert "Variable: `data`" in formatted
        assert "Type: dict" in formatted
        assert "Description: Input data" in formatted
        assert "Constraints: Non-empty" in formatted
        assert "Total length: 100 characters" in formatted
        assert "Preview:" in formatted
        assert '{"key": "value"}' in formatted

    def test_immutability(self):
        """Test that REPLVariable is immutable."""
        variable = REPLVariable.from_value(name="x", value="test")

        with pytest.raises(Exception):  # pydantic ValidationError
            variable.name = "y"


class TestREPLEntry:
    """Tests for REPLEntry data model."""

    def test_create_entry(self):
        """Test creating a REPL entry."""
        entry = REPLEntry(
            iteration=0,
            reasoning="Let me explore the data",
            code="print(len(data))",
            stdout="100",
            error=None,
        )

        assert entry.iteration == 0
        assert entry.reasoning == "Let me explore the data"
        assert entry.code == "print(len(data))"
        assert entry.stdout == "100"
        assert entry.error is None

    def test_entry_with_error(self):
        """Test creating an entry with an error."""
        entry = REPLEntry(
            iteration=1,
            reasoning="Try to access element",
            code="data[1000]",
            stdout="",
            error="IndexError: list index out of range",
        )

        assert entry.error == "IndexError: list index out of range"

    def test_entry_default_values(self):
        """Test default values for optional fields."""
        entry = REPLEntry(
            iteration=0,
            reasoning="Test",
            code="pass",
        )

        assert entry.stdout == ""
        assert entry.error is None

    def test_format_with_output(self):
        """Test format() with stdout."""
        entry = REPLEntry(
            iteration=0,
            reasoning="Check the type",
            code="print(type(x))",
            stdout="<class 'int'>",
        )

        formatted = entry.format()

        assert "=== Step 1 ===" in formatted  # 0-indexed becomes 1-indexed
        assert "Reasoning: Check the type" in formatted
        assert "print(type(x))" in formatted
        assert "<class 'int'>" in formatted

    def test_format_with_error(self):
        """Test format() with error."""
        entry = REPLEntry(
            iteration=0,
            reasoning="Divide",
            code="1/0",
            stdout="",
            error="ZeroDivisionError: division by zero",
        )

        formatted = entry.format()

        assert "[Error]" in formatted
        assert "ZeroDivisionError" in formatted

    def test_format_truncates_long_output(self):
        """Test format() truncates long output."""
        long_output = "x" * 10000
        entry = REPLEntry(
            iteration=0,
            reasoning="Long output",
            code="print('x' * 10000)",
            stdout=long_output,
        )

        formatted = entry.format(max_output_chars=100)

        assert "truncated" in formatted
        assert "10,000" in formatted

    def test_immutability(self):
        """Test that REPLEntry is immutable."""
        entry = REPLEntry(iteration=0, reasoning="Test", code="pass")

        with pytest.raises(Exception):  # pydantic ValidationError
            entry.iteration = 1


class TestREPLHistory:
    """Tests for REPLHistory data model."""

    def test_create_empty_history(self):
        """Test creating empty history."""
        history = REPLHistory()

        assert len(history) == 0
        assert list(history) == []

    def test_append_entry_returns_new_instance(self):
        """Test that append returns a new instance (immutability)."""
        history1 = REPLHistory()

        history2 = history1.append(
            iteration=0,
            reasoning="First step",
            code="print('hello')",
            stdout="hello",
        )

        # Original should be unchanged
        assert len(history1) == 0

        # New history should have the entry
        assert len(history2) == 1
        assert history2.entries[0].reasoning == "First step"

    def test_append_multiple_entries(self):
        """Test appending multiple entries."""
        history = REPLHistory()

        history = history.append(
            iteration=0,
            reasoning="First",
            code="x = 1",
            stdout="",
        )
        history = history.append(
            iteration=1,
            reasoning="Second",
            code="print(x)",
            stdout="1",
        )
        history = history.append(
            iteration=2,
            reasoning="Third",
            code="y = x + 1",
            stdout="",
        )

        assert len(history) == 3
        assert history.entries[0].reasoning == "First"
        assert history.entries[1].reasoning == "Second"
        assert history.entries[2].reasoning == "Third"

    def test_format_for_prompt_empty(self):
        """Test formatting empty history."""
        history = REPLHistory()
        formatted = history.format_for_prompt()

        assert formatted == "You have not interacted with the REPL environment yet."

    def test_format_for_prompt_with_entries(self):
        """Test formatting history with entries."""
        history = REPLHistory()

        history = history.append(
            iteration=0,
            reasoning="First step",
            code="x = 1",
            stdout="",
        )
        history = history.append(
            iteration=1,
            reasoning="Second step",
            code="print(x)",
            stdout="1",
        )

        formatted = history.format_for_prompt()

        assert "=== Step 1 ===" in formatted
        assert "=== Step 2 ===" in formatted
        assert "First step" in formatted
        assert "Second step" in formatted
        assert "x = 1" in formatted
        assert "print(x)" in formatted

    def test_format_for_prompt_with_error(self):
        """Test formatting history with error entry."""
        history = REPLHistory()

        history = history.append(
            iteration=0,
            reasoning="Divide by zero",
            code="1/0",
            stdout="",
            error="ZeroDivisionError: division by zero",
        )

        formatted = history.format_for_prompt()

        assert "[Error]" in formatted
        assert "ZeroDivisionError" in formatted

    def test_get_last_entry(self):
        """Test getting the last entry."""
        history = REPLHistory()

        # Empty history
        assert history.get_last_entry() is None

        # With entries
        history = history.append(iteration=0, reasoning="First", code="pass")
        history = history.append(iteration=1, reasoning="Second", code="pass")

        last = history.get_last_entry()
        assert last is not None
        assert last.reasoning == "Second"

    def test_had_errors(self):
        """Test checking for errors in history."""
        history = REPLHistory()

        # No entries
        assert history.had_errors() is False

        # Entry without error
        history = history.append(
            iteration=0, reasoning="OK", code="pass", stdout=""
        )
        assert history.had_errors() is False

        # Entry with error
        history = history.append(
            iteration=1,
            reasoning="Error",
            code="1/0",
            error="ZeroDivisionError",
        )
        assert history.had_errors() is True

    def test_get_total_output_chars(self):
        """Test calculating total output characters."""
        history = REPLHistory()

        history = history.append(
            iteration=0, reasoning="A", code="pass", stdout="hello"
        )
        history = history.append(
            iteration=1, reasoning="B", code="pass", stdout="world"
        )

        assert history.get_total_output_chars() == 10

    def test_iteration(self):
        """Test iterating over history."""
        history = REPLHistory()

        for i in range(3):
            history = history.append(
                iteration=i,
                reasoning=f"Step {i}",
                code="pass",
            )

        # Test iteration
        iterations = list(history)
        assert len(iterations) == 3
        for i, entry in enumerate(iterations):
            assert entry.iteration == i
            assert entry.reasoning == f"Step {i}"

    def test_bool_empty(self):
        """Test bool conversion for empty history."""
        history = REPLHistory()
        assert bool(history) is False

    def test_bool_non_empty(self):
        """Test bool conversion for non-empty history."""
        history = REPLHistory()
        history = history.append(iteration=0, reasoning="Test", code="pass")
        assert bool(history) is True

    def test_to_trajectory(self):
        """Test converting history to trajectory list."""
        history = REPLHistory()

        history = history.append(
            iteration=0,
            reasoning="Step one",
            code="x = 1",
            stdout="",
            error=None,
        )
        history = history.append(
            iteration=1,
            reasoning="Step two",
            code="print(x)",
            stdout="1",
            error=None,
        )

        trajectory = history.to_trajectory()

        assert len(trajectory) == 2
        assert trajectory[0] == {
            "iteration": 0,
            "reasoning": "Step one",
            "code": "x = 1",
            "stdout": "",
            "error": None,
        }
        assert trajectory[1] == {
            "iteration": 1,
            "reasoning": "Step two",
            "code": "print(x)",
            "stdout": "1",
            "error": None,
        }

    def test_immutability(self):
        """Test that REPLHistory is immutable."""
        history = REPLHistory()

        with pytest.raises(Exception):  # pydantic ValidationError
            history.entries = []
