# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.react_agent import ReACTAgent
from synalinks.src.programs import Program


class ReACTAgentTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_flow_with_one_action(self, mock_completion):
        class Query(DataModel):
            query: str

        class FinalAnswer(DataModel):
            answer: float

        async def calculate(expression: str):
            """Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as
                    '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                    parentheses, and spaces.
            """
            if not all(char in "0123456789+-*/(). " for char in expression):
                return {
                    "result": None,
                    "log": "Error: invalid characters in expression",
                }
            try:
                # Evaluate the mathematical expression safely
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                return {
                    "result": result,
                    "log": "Successfully executed",
                }
            except Exception as e:
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        decision_response = (
            """{"thinking": "I need to calculate the total number of apples by adding """
            """the initial amount to the additional apples given by my friend.", """
            """"choice": "calculate"}"""
        )

        inference_response = """{"expression": "12 + 15"}"""

        decision_response_1 = (
            """{"thinking": "Now I know the answer so I finished, """
            """so I select `finish`.", """
            """"choice": "finish"}"""
        )

        inference_response_1 = """{"answer": 27.0}"""

        mock_responses = [
            {"choices": [{"message": {"content": decision_response}}]},
            {"choices": [{"message": {"content": inference_response}}]},
            {"choices": [{"message": {"content": decision_response_1}}]},
            {"choices": [{"message": {"content": inference_response_1}}]},
        ]

        mock_completion.side_effect = mock_responses

        x0 = Input(data_model=Query)
        x1 = await ReACTAgent(
            data_model=FinalAnswer,
            language_model=language_model,
            functions=[calculate],
            max_iterations=3,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            Query(
                query=(
                    "You have a basket with 12 apples. "
                    "Your friend gives you 15 more apples. "
                    "How many apples do you have in total now?"
                )
            )
        )
        self.assertEqual(result.get("answer"), 27.0)
