# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import time
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.react_agent import ParallelReACTAgent
from synalinks.src.programs import Program


class ParallelReACTAgentTest(testing.TestCase):
    @patch("litellm.completion")
    async def test_basic_flow_with_parallel_actions(self, mock_completion):
        trajectory_log = []

        class Request(DataModel):
            query: str

        class Response(DataModel):
            number_of_apples: int
            number_words_in_string: int

        async def calculate(expression: str):
            """Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as '2 + 2'.
                    The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.
            """
            trajectory_log.append({"function": "calculate", "event": "start", "timestamp": time.time()})

            await asyncio.sleep(0.1)

            if not all(char in "0123456789+-*/(). " for char in expression):
                trajectory_log.append({"function": "calculate", "event": "end", "timestamp": time.time()})
                return {
                    "result": None,
                    "status": "Error: invalid characters in expression",
                }

            try:
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                trajectory_log.append({"function": "calculate", "event": "end", "timestamp": time.time()})
                return {
                    "result": result,
                    "status": "Successfully executed",
                }
            except Exception as e:
                trajectory_log.append({"function": "calculate", "event": "end", "timestamp": time.time()})
                return {
                    "result": None,
                    "status": f"Error: {e}",
                }

        async def count_words_in_string(string: str):
            """Count the number of whitespace-separated words in a string.

            Args:
                string (str): The string to analyse with words separated by whitespace.
            """
            trajectory_log.append({"function": "count_words_in_string", "event": "start", "timestamp": time.time()})

            await asyncio.sleep(0.1)

            if not isinstance(string, str) or not string.strip():
                trajectory_log.append({"function": "count_words_in_string", "event": "end", "timestamp": time.time()})
                return {
                    "result": None,
                    "status": "Error: Expression must be a non-empty string"
                }

            try:
                result = len(string.split())
                trajectory_log.append({"function": "count_words_in_string", "event": "end", "timestamp": time.time()})
                
                return {
                    "result": result,
                    "status": "Successfully executed"
                }
            except Exception as e:
                trajectory_log.append({"function": "count_words_in_string", "event": "end", "timestamp": time.time()})
                return {
                    "result": None,
                    "status": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        decision_continue = (
            """
            {
                "thinking": "I'll need to run some tools before I can answer.", 
                "choice": "continue"
            }
            """.strip()
        )

        decision_tools = (
            """
            {
                "thinking": "I should add 12 + 15 and count the words in 'hello parallel agent'.", 
                "choices": ["calculate", "count_words_in_string"]
            }
            """.strip()
        )

        arguments_calculate = """{"expression": "12 + 15"}"""
        arguments_count_words_in_string = """{"string": "hello parallel agent"}"""

        decision_continue_2 = (
            """
            {
                "thinking": "I got some results, but better run the tools again to double-check.", 
                "choice": "continue"
            }
            """.strip()
        )

        decision_tools_2 = (
            """
            {
                "thinking": "I will run both tools again to verify the results.", 
                "choices": ["calculate", "count_words_in_string"]
            }
            """.strip()
        )

        decision_finish = (
            """
            {
                "thinking": "The results are consistent. I have nothing else to do, I will safely select `finish`.", 
                "choice": "finish"
            }
            """.strip()
        )

        response = """{"number_of_apples": 27, "number_words_in_string": 3}"""

        mock_trajectory = [
            # 1st turn
            {"choices": [{"message": {"content": decision_continue}}]},
            {"choices": [{"message": {"content": decision_tools}}]},
            {"choices": [{"message": {"content": arguments_calculate}}]},
            {"choices": [{"message": {"content": arguments_count_words_in_string}}]},
            
            # 2nd turn
            {"choices": [{"message": {"content": decision_continue_2}}]},
            {"choices": [{"message": {"content": decision_tools_2}}]},
            {"choices": [{"message": {"content": arguments_calculate}}]},
            {"choices": [{"message": {"content": arguments_count_words_in_string}}]},
            
            # response
            {"choices": [{"message": {"content": decision_finish}}]},
            {"choices": [{"message": {"content": response}}]},
        ]

        mock_completion.side_effect = mock_trajectory

        x0 = Input(data_model=Request)
        x1 = await ParallelReACTAgent(
            data_model=Response,
            language_model=language_model,
            functions=[calculate, count_words_in_string],
            max_iterations=3,
            return_inputs_with_trajectory=True
        )(x0)

        program = Program(inputs=x0, outputs=x1)

        response = await program(
            Request(
                query=(
                    "You have a basket containing 12 apples. "
                    "Your friend gives you another 15. "
                    "How many apples do you have in total? "
                    "And how many words are there in the phrase 'hello, parallel agent'? "
                    "Ensure that the rationale and results are consistent across iterations."
                )
            )
        )

        self.assertEqual(response.get("number_of_apples"), 27)
        self.assertEqual(response.get("number_words_in_string"), 3)

        self._assert_parallel_tool_calls(trajectory_log)

    def _assert_parallel_tool_calls(self, trajectory_log):
        trajectory = []
        turn = {}
        
        for entry in trajectory_log:
            function = entry["function"]
            event = entry["event"]
            timestamp = entry["timestamp"]
            
            if function not in turn:
                turn[function] = {}
            
            turn[function][event] = timestamp
            
            if (len(turn) == 2 and all("end" in events for events in turn.values())):
                trajectory.append(turn)
                turn = {}
        
        for i, turn in enumerate(trajectory):            
            calculate_start = turn["calculate"]["start"]
            calculate_end = turn["calculate"]["end"]
            count_words_in_string_start = turn["count_words_in_string"]["start"]
            count_words_in_string_end = turn["count_words_in_string"]["end"]
                        
            turn_start = max(calculate_start, count_words_in_string_start)
            turn_end = min(calculate_end, count_words_in_string_end)
            
            self.assertLess(
                turn_start, 
                turn_end, 
                f"tool calls are not executed in parallel in round {i + 1}."
            )
        
        self.assertEqual(len(trajectory), 2, "expected 2 rounds of parallel execution")
