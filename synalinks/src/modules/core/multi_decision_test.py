# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)
import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.core.multi_decision import MultiDecision
from synalinks.src.programs import Program


class MultiDecisionTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_multi_decision(self, mock_completion):
        class Query(DataModel):
            query: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        expected_string = (
            """{"thinking": "This question asks which of the options are European """
            """countries. Looking at the choices: A) France is a European country, """
            """B) Japan is in Asia, C) Germany is also a European country, """
            """D) Brazil is in South America. So both A and C are correct","""
            """ "choices": ["A", "C"]}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await MultiDecision(
            question=(
                "Which of the following options are correct answers? "
                "(Select all that apply)"
            ),
            labels=["A", "B", "C", "D"],
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            Query(
                query=(
                    "Which of the following are European countries?"
                    " A) France B) Japan C) Germany D) Brazil"
                )
            )
        )
        self.assertEqual(result.get_json(), json.loads(expected_string))
