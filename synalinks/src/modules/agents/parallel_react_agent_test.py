# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.parallel_react_agent import ParallelReACTAgent
from synalinks.src.programs import Program


class SearchAgentTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_flow_with_parallel_searches(self, mock_completion):
        class Query(DataModel):
            query: str

        class SearchResult(DataModel):
            company_info: str
            news_summary: str
            total_sources: int

        async def web_search(query: str):
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

        async def news_search(topic: str):
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

        decision_continue = (
            """{ "thinking": "I need to search for information about Tesla"""
            """ to answer this question.","""
            """ "choice": "continue"}"""
        )

        decision_tools = (
            "{"
            '"thinking": "I should search for Tesla company information and recent '
            'news about Tesla to provide a comprehensive answer.", '
            '"choices": ["web_search", "news_search"]'
            "}"
        )

        inference_response_web = """{"query": "Tesla company information"}"""
        inference_response_news = """{"topic": "Tesla"}"""

        decision_continue_1 = (
            "{"
            '"thinking": "I now have both company information and recent news '
            'about Tesla, so I can provide the final answer.", '
            '"choice": "finish"'
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
            '"total_sources": 9'
            "}"
        )

        mock_responses = [
            {"choices": [{"message": {"content": decision_continue}}]},
            {"choices": [{"message": {"content": decision_tools}}]},
            {"choices": [{"message": {"content": inference_response_web}}]},
            {"choices": [{"message": {"content": inference_response_news}}]},
            {"choices": [{"message": {"content": decision_continue_1}}]},
            {"choices": [{"message": {"content": final_answer}}]},
        ]
        mock_completion.side_effect = mock_responses

        x0 = Input(data_model=Query)
        x1 = await ParallelReACTAgent(
            data_model=SearchResult,
            language_model=language_model,
            functions=[web_search, news_search],
            max_iterations=3,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            Query(
                query=(
                    "I need comprehensive information about Tesla. "
                    "Can you provide both company details and recent news?"
                )
            )
        )

        self.assertIn("Tesla, Inc.", result.get("company_info"))
        self.assertIn("Stock price reaches new highs", result.get("news_summary"))
        self.assertEqual(
            result.get("total_sources"), 9
        )  # 5 from web_search + 4 from news_search
