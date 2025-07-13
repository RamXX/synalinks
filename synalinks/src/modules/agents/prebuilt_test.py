# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from enum import Enum
from typing import List
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel, Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.prebuilt import Agent, dynamic_enum_on_nested_property, ToolDecision as OGToolDecision
from synalinks.src.programs import Program


class AgentTest(testing.TestCase):
    def setUp(self):
        schema = OGToolDecision.get_schema()
        
        toolkit = ["calculate", "websearch"]

        self.dynamic_schema = dynamic_enum_on_nested_property(
            schema,
            "ToolChoice/properties/tool",
            toolkit,
            description="The name of the tool to run from the available toolkit."
        )

    @patch("litellm.acompletion")
    async def test_flow_with_parallel_searches(self, mock_completion):
        class I(DataModel):
            query: str

        class O(DataModel):
            company_info: str
            news_summary: str
            total_sources: int

        async def websearch(query: str):
            """
            Search the web for information on a given topic.

            Args:
                query (str): The search query to execute.

            Returns:
                dict: {"result": str | None, "sources": int, "log": str}
            """
            if not isinstance(query, str) or not query.strip():
                return {
                    "result": None,
                    "sources": 0,
                    "log": "Error: Query must be a non-empty string",
                }

            try:
                # Simulate web search results
                mock_results = {
                    "Tesla company information": {
                        "result": (
                            "Tesla, Inc. is an American electric vehicle and "
                            "clean energy company founded by Elon Musk in 2003. "
                            "Headquarters in Austin, Texas."
                        ),
                        "sources": 5,
                    },
                    "Tesla recent news": {
                        "result": (
                            "Tesla reported strong Q4 2024 earnings with record "
                            "deliveries. New Cybertruck production ramping up "
                            "successfully."
                        ),
                        "sources": 3,
                    },
                }

                # Find matching result
                for key, value in mock_results.items():
                    if key.lower() in query.lower():
                        return {
                            "result": value["result"],
                            "sources": value["sources"],
                            "log": "Search completed successfully",
                        }

                return {
                    "result": f"General search results for: {query}",
                    "sources": 2,
                    "log": "Search completed successfully",
                }

            except Exception as e:
                return {
                    "result": None,
                    "sources": 0,
                    "log": f"Error: {e}",
                }

        async def web_news_search(topic: str):
            """
            Search for recent news articles about a specific topic.

            Args:
                topic (str): The topic to search for in news sources.

            Returns:
                dict: {"result": str | None, "sources": int, "log": str}
            """
            if not isinstance(topic, str) or not topic.strip():
                return {
                    "result": None,
                    "sources": 0,
                    "log": "Error: Topic must be a non-empty string",
                }

            try:
                # Simulate news search results
                mock_news = {
                    "Tesla": {
                        "result": (
                            "Latest Tesla news: Stock price reaches new highs "
                            "following successful Cybertruck launch. Analysts "
                            "optimistic about 2025 outlook."
                        ),
                        "sources": 4,
                    }
                }

                for key, value in mock_news.items():
                    if key.lower() in topic.lower():
                        return {
                            "result": value["result"],
                            "sources": value["sources"],
                            "log": "News search completed successfully",
                        }

                return {
                    "result": f"Recent news about: {topic}",
                    "sources": 2,
                    "log": "News search completed successfully",
                }

            except Exception as e:
                return {
                    "result": None,
                    "sources": 0,
                    "log": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        tool_choices_response = (
            "{"
                '"reasoning": "I need to search for information about Tesla '
                    'to answer this question. I should search for both company '
                    'information and recent news to provide a comprehensive answer.", '
                '"choices": ['
                    '{"tool": "websearch", "purpose": "Get general company information about Tesla"}, '
                    '{"tool": "web_news_search", "purpose": "Get recent news and updates about Tesla"}, '
                    '{"tool": "web_news_search", "purpose": "Get news and updates about Tesla from last week"}'
                ']'
            "}"
        )

        inference_response_web = """{"query": "Tesla company information"}"""
        inference_response_news = """{"topic": "Tesla"}"""
        inference_response_news_2 = """{"topic": "Tesla last week"}"""

        tool_choices_response_1 = (
            "{"
                '"reasoning": "I now have both company information and recent news '
                    'about Tesla, so I can provide the final answer.", '
                '"choices": []'
            "}"
        )

        final_answer = (
            "{"
            '"company_info": "Tesla, Inc. is an American electric vehicle and '
            "clean energy company founded by Elon Musk in 2003. Headquarters in "
            'Austin, Texas.", '
            '"news_summary": "Latest Tesla news: Stock price reaches new highs '
            "following successful Cybertruck launch. Analysts optimistic about "
            '2025 outlook.", '
            '"total_sources": 13'
            "}"
        )

        mock_responses = [
            {"choices": [{"message": {"content": tool_choices_response}}]},
            {"choices": [{"message": {"content": inference_response_web}}]},
            {"choices": [{"message": {"content": inference_response_news}}]},
            {"choices": [{"message": {"content": inference_response_news_2}}]},
            {"choices": [{"message": {"content": tool_choices_response_1}}]},
            {"choices": [{"message": {"content": final_answer}}]},
        ]
        mock_completion.side_effect = mock_responses

        x0 = Input(data_model=I)
        x1 = await Agent(
            data_model=O,
            language_model=language_model,
            toolkit=[websearch, web_news_search],
            max_iterations=3,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            I(
                query=(
                    "I need comprehensive information about Tesla. "
                    "Can you provide company details, recent news, and also "
                    "any specific news from the past week?"
                )
            )
        )

        self.assertIn("Tesla, Inc.", result.get("company_info"))
        self.assertIn("Stock price reaches new highs", result.get("news_summary"))
        self.assertEqual(
            result.get("total_sources"), 13
        )

    def test_dynamic_enum_schema_on_nested_property(self):
        class Tool(str, Enum):
            """The name of the tool to run from the available toolkit."""
            calculate = "calculate"
            websearch = "websearch"

        class ToolChoice(DataModel):
            tool: Tool
            purpose: str = Field(
                description="A clear, specific explanation of what the tool should accomplish."
            )

        class ToolDecision(DataModel):
            reasoning: str = Field(
                description="A step-by-step analysis of the current state, what has been done, and what should be done next."
            )
            choices: List[ToolChoice] = Field(
                description="The array of tool choices to run in parallel with their specific purpose."
            )

        self.assertJsonEqual(json.dumps(self.dynamic_schema, indent=2), ToolDecision.prettify_schema())
