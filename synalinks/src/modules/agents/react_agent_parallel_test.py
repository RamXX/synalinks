# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)Add commentMore actions

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.react_agent import ParallelReACTAgent
from synalinks.src.programs import Program


class ReACTAgentTest(testing.TestCase):
    @patch("litellm.completion")
    async def test_basic_flow_with_parallel_actions(self, mock_completion):
        class Query(DataModel):
            query: str

        class FinalAnswer(DataModel):
            answer_sum: float
            answer_wc: int

        async def calculate(expression: str):
            """
            Calculate a basic math expression (supports +, −, *, /).

            Args:
                expression (str): The expression to evaluate, e.g. "12 + 15".

            Returns:
                dict: {"result": float | None, "log": str}
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

        async def word_count(text: str):
            """
            Count the number of whitespace-separated words in *text*.

            Args:
                text (str): The string to analyse. Words are separated by whitespace.

            Returns:
                dict: {"result": int | None, "log": str}
            """
            if not isinstance(text, str) or not text.strip():
                return {
                    "result": None,
                    "log": "Error: Expression must be a non-empty string"
                }
            try:
                word_count = len(text.split())
                
                return {
                    "result": word_count,
                    "log": "Successfully executed"
                }
            except Exception as e:
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        decision_continue = (
            """{
                "thinking": "I’ll need to run some tools before I can answer.", 
                "choice": "continue"
            }"""
        )

        decision_tools = (
            """{
                "thinking": "I should add 12 + 15 and count the words in 'hello parallel agent'.", 
                "choices": ["calculate", "word_count"]
            }"""
        )

        inference_response_calc = """{"expression": "12 + 15"}"""
        inference_response_wc = """{"text": "hello parallel agent"}"""

        decision_continue_1 = (
            """{
                "thinking": "Now I know the answer so I finished, so I select `finish`.", 
                "choice": "finish"
            }"""
        )

        final_answer = """{"answer_sum": 27.0, "answer_wc": 3}"""

        mock_responses = [
            {"choices": [{"message": {"content": decision_continue}}]},
            {"choices": [{"message": {"content": decision_tools}}]},
            {"choices": [{"message": {"content": inference_response_calc}}]},
            {"choices": [{"message": {"content": inference_response_wc}}]},
            {"choices": [{"message": {"content": decision_continue_1}}]},
            {"choices": [{"message": {"content": final_answer}}]},
        ]
        mock_completion.side_effect = mock_responses

        x0 = Input(data_model=Query)
        x1 = await ParallelReACTAgent(
            data_model=FinalAnswer,
            language_model=language_model,
            functions=[calculate, word_count],
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
                    "How many apples in total? "
                    "And how many words are in 'hello parallel agent'?"
                )
            )
        )
        self.assertEqual(result.get("answer_sum"), 27.0)
        self.assertEqual(result.get("answer_wc"), 3)
