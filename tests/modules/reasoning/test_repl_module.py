# License Apache 2.0: (c) 2025 Synalinks Team

"""Tests for RLM (Recursive Language Model)."""

import pytest

from synalinks.src.backend import DataModel, Field, JsonDataModel
from synalinks.src.interpreters.native import NativePythonInterpreter
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.reasoning.repl_module import RLM


class MockLanguageModel:
    """Mock language model for testing RLM."""

    def __init__(self, responses: list):
        """Initialize with a list of responses to return in order."""
        self.responses = responses
        self.call_count = 0

    async def __call__(self, messages, schema=None, **kwargs):
        """Return the next response in the list."""
        if self.call_count >= len(self.responses):
            return None

        response = self.responses[self.call_count]
        self.call_count += 1
        return response

    def get_config(self):
        """Return configuration for serialization."""
        return {"responses": self.responses}

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        return cls(responses=config.get("responses", []))


class TestRLMInit:
    """Tests for RLM initialization."""

    def test_init_with_schema(self):
        """Test initialization with schema."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }
        lm = MockLanguageModel([])

        module = RLM(
            schema=schema,
            language_model=lm,
            max_iterations=5,
        )

        assert module.output_schema == schema
        assert module.output_fields == ["answer", "confidence"]
        assert module.max_iterations == 5

    def test_init_requires_schema_or_data_model(self):
        """Test that initialization requires schema or data_model."""
        lm = MockLanguageModel([])

        with pytest.raises(ValueError, match="Must provide schema or data_model"):
            RLM(language_model=lm)

    def test_init_requires_language_model(self):
        """Test that initialization requires language_model."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        with pytest.raises(ValueError, match="Must provide language_model"):
            RLM(schema=schema)

    def test_init_with_custom_interpreter(self):
        """Test initialization with custom interpreter."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        lm = MockLanguageModel([])
        interpreter = NativePythonInterpreter(max_output_chars=1000)

        module = RLM(
            schema=schema,
            language_model=lm,
            interpreter=interpreter,
        )

        assert module.interpreter is interpreter


class TestRLMExecution:
    """Tests for RLM execution."""

    @pytest.fixture
    def simple_schema(self):
        """Simple output schema."""
        return {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
        }

    @pytest.mark.asyncio
    async def test_submit_in_first_iteration(self, simple_schema):
        """Test SUBMIT on first iteration."""
        # Mock LM returns code that SUBMITs immediately
        lm = MockLanguageModel(
            [
                {
                    "reasoning": "I know the answer",
                    "code": 'SUBMIT(result="hello world")',
                }
            ]
        )

        module = RLM(
            schema=simple_schema,
            language_model=lm,
            max_iterations=5,
        )

        inputs = JsonDataModel(
            json={"query": "test"},
            schema={"type": "object"},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        assert result.get("result") == "hello world"

    @pytest.mark.asyncio
    async def test_multi_iteration_exploration(self, simple_schema):
        """Test multiple iterations before SUBMIT."""
        lm = MockLanguageModel(
            [
                # First iteration: explore
                {
                    "reasoning": "Let me check the data",
                    "code": 'count = len(query)\nprint(f"Query length: {count}")',
                },
                # Second iteration: compute
                {
                    "reasoning": "Now I'll compute the result",
                    "code": 'result_value = query.upper()\nprint(f"Result: {result_value}")',
                },
                # Third iteration: submit
                {
                    "reasoning": "Ready to submit",
                    "code": 'SUBMIT(result=result_value)',
                },
            ]
        )

        module = RLM(
            schema=simple_schema,
            language_model=lm,
            max_iterations=10,
        )

        inputs = JsonDataModel(
            json={"query": "test"},
            schema={"type": "object"},
            name="inputs",
        )

        result = await module(inputs)

        assert result is not None
        assert result.get("result") == "TEST"

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, simple_schema):
        """Test that max_iterations is respected."""
        # LM never SUBMITs
        lm = MockLanguageModel(
            [
                {"reasoning": "Step 1", "code": 'print("step 1")'},
                {"reasoning": "Step 2", "code": 'print("step 2")'},
                {"reasoning": "Step 3", "code": 'print("step 3")'},
                # This would be the extraction fallback
                {"result": "fallback"},
            ]
        )

        module = RLM(
            schema=simple_schema,
            language_model=lm,
            max_iterations=3,  # Only 3 iterations
        )

        inputs = JsonDataModel(
            json={"query": "test"},
            schema={"type": "object"},
            name="inputs",
        )

        # Should not raise, should use fallback extractor
        result = await module(inputs)
        assert result is not None

    @pytest.mark.asyncio
    async def test_return_history(self, simple_schema):
        """Test returning execution history."""
        lm = MockLanguageModel(
            [
                {"reasoning": "Explore", "code": 'print("exploring")'},
                {"reasoning": "Submit", "code": 'SUBMIT(result="done")'},
            ]
        )

        module = RLM(
            schema=simple_schema,
            language_model=lm,
            return_history=True,
        )

        inputs = JsonDataModel(
            json={"query": "test"},
            schema={"type": "object"},
            name="inputs",
        )

        result = await module(inputs)

        assert "_history" in result.get_json()
        history = result.get("_history")
        assert len(history) == 2
        assert history[0]["reasoning"] == "Explore"
        assert history[1]["reasoning"] == "Submit"

    @pytest.mark.asyncio
    async def test_llm_query_batched_call_count(self, simple_schema):
        """Test that batched LLM calls count once per prompt."""
        class DummyLM:
            async def __call__(self, messages, schema=None, **kwargs):
                return {"content": "ok"}

        lm = MockLanguageModel(
            [
                {
                    "reasoning": "Batch call",
                    "code": 'print(llm_query_batched(["a", "b"]))',
                },
                {"reasoning": "Submit", "code": 'SUBMIT(result="done")'},
            ]
        )

        module = RLM(
            schema=simple_schema,
            language_model=lm,
            sub_language_model=DummyLM(),
            max_llm_calls=2,
            return_history=True,
        )

        inputs = JsonDataModel(
            json={"query": "test"},
            schema={"type": "object"},
            name="inputs",
        )

        result = await module(inputs)
        history = result.get("_history")
        assert "Maximum LLM calls exceeded" not in history[0]["stdout"]

    @pytest.mark.asyncio
    async def test_llm_query_batched_limit_exceeded(self, simple_schema):
        """Test that batched LLM calls respect max_llm_calls."""
        class DummyLM:
            async def __call__(self, messages, schema=None, **kwargs):
                return {"content": "ok"}

        lm = MockLanguageModel(
            [
                {
                    "reasoning": "Batch call",
                    "code": 'print(llm_query_batched(["a", "b"]))',
                },
                {"reasoning": "Submit", "code": 'SUBMIT(result="done")'},
            ]
        )

        module = RLM(
            schema=simple_schema,
            language_model=lm,
            sub_language_model=DummyLM(),
            max_llm_calls=1,
            return_history=True,
        )

        inputs = JsonDataModel(
            json={"query": "test"},
            schema={"type": "object"},
            name="inputs",
        )

        result = await module(inputs)
        history = result.get("_history")
        assert "Maximum LLM calls exceeded" in history[0]["stdout"]


class TestRLMTypeCoercion:
    """Tests for output type coercion."""

    @pytest.mark.asyncio
    async def test_string_coercion(self):
        """Test that invalid string values yield errors and can retry."""
        class Answer(DataModel):
            answer: str = Field(description="Answer string")

        lm = MockLanguageModel(
            [
                {"reasoning": "Bad", "code": "SUBMIT(answer=42)"},
                {"reasoning": "Good", "code": 'SUBMIT(answer="42")'},
            ]
        )

        module = RLM(
            data_model=Answer,
            language_model=lm,
            max_iterations=2,
            return_history=True,
        )
        inputs = JsonDataModel(json={}, schema={"type": "object"}, name="inputs")

        result = await module(inputs)
        assert result.get("answer") == "42"
        history = result.get("_history")
        assert "Type Error" in history[0]["error"]

    @pytest.mark.asyncio
    async def test_number_coercion(self):
        """Test that string numbers are coerced to numbers."""
        class Value(DataModel):
            value: float = Field(description="Numeric value")

        lm = MockLanguageModel(
            [{"reasoning": "Test", "code": 'SUBMIT(value="3.14")'}]
        )

        module = RLM(data_model=Value, language_model=lm)
        inputs = JsonDataModel(json={}, schema={"type": "object"}, name="inputs")

        result = await module(inputs)
        assert result.get("value") == 3.14

    @pytest.mark.asyncio
    async def test_array_validation(self):
        """Test that array outputs validate via DataModel."""
        class Items(DataModel):
            items: list[str] = Field(description="List of items")

        lm = MockLanguageModel(
            [{"reasoning": "Test", "code": 'SUBMIT(items=["single"])'}]
        )

        module = RLM(data_model=Items, language_model=lm)
        inputs = JsonDataModel(json={}, schema={"type": "object"}, name="inputs")

        result = await module(inputs)
        assert result.get("items") == ["single"]

    @pytest.mark.asyncio
    async def test_validation_error_feedback(self):
        """Test that invalid DataModel values yield a type error and retry."""
        class Flag(DataModel):
            flag: bool = Field(description="Boolean flag")

        lm = MockLanguageModel(
            [
                {"reasoning": "Bad", "code": 'SUBMIT(flag="notabool")'},
                {"reasoning": "Good", "code": "SUBMIT(flag=False)"},
            ]
        )

        module = RLM(
            data_model=Flag,
            language_model=lm,
            max_iterations=2,
            return_history=True,
        )
        inputs = JsonDataModel(json={}, schema={"type": "object"}, name="inputs")

        result = await module(inputs)
        assert result.get("flag") is False
        history = result.get("_history")
        assert "Type Error" in history[0]["error"]


class TestRLMToolValidation:
    """Tests for tool name validation."""

    @pytest.mark.asyncio
    async def test_invalid_tool_name(self):
        """Test invalid tool identifiers are rejected."""
        async def noop(value: str) -> str:
            """No-op tool.

            Args:
                value (str): Input value.
            """
            return value

        tool = Tool(noop, name="bad-name")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        lm = MockLanguageModel([])

        with pytest.raises(ValueError, match="valid Python identifier"):
            RLM(schema=schema, language_model=lm, tools=[tool])

    @pytest.mark.asyncio
    async def test_reserved_tool_name(self):
        """Test reserved tool names are rejected."""
        async def noop(value: str) -> str:
            """No-op tool.

            Args:
                value (str): Input value.
            """
            return value

        tool = Tool(noop, name="llm_query")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        lm = MockLanguageModel([])

        with pytest.raises(ValueError, match="conflicts with built-in"):
            RLM(schema=schema, language_model=lm, tools=[tool])

    @pytest.mark.asyncio
    async def test_duplicate_tool_names(self):
        """Test duplicate tool names are rejected."""
        async def noop(value: str) -> str:
            """No-op tool.

            Args:
                value (str): Input value.
            """
            return value

        tool_a = Tool(noop, name="lookup")
        tool_b = Tool(noop, name="lookup")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        lm = MockLanguageModel([])

        with pytest.raises(ValueError, match="Duplicate tool name"):
            RLM(schema=schema, language_model=lm, tools=[tool_a, tool_b])


class TestRLMSerialization:
    """Tests for RLM serialization."""

    def test_get_config(self):
        """Test getting module configuration."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        lm = MockLanguageModel([])

        module = RLM(
            schema=schema,
            language_model=lm,
            max_iterations=10,
            max_llm_calls=25,
            return_history=True,
            name="test_repl",
        )

        config = module.get_config()

        assert config["schema"] == schema
        assert config["max_iterations"] == 10
        assert config["max_llm_calls"] == 25
        assert config["return_history"] is True
        assert config["name"] == "test_repl"
