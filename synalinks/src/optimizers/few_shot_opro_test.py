# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Generator
from synalinks.src.modules import Input
from synalinks.src.optimizers import FewShotOPRO
from synalinks.src.programs import Program
from synalinks.src.rewards.cosine_similarity import CosineSimilarity
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.testing.test_utils import mock_completion_data


class FewShotOPROTest(testing.TestCase):
    @patch("litellm.aembedding")
    @patch("litellm.acompletion")
    async def test_few_shot_opro_training(self, mock_completion, mock_embedding):
        language_model = LanguageModel(model="ollama/mistral")

        embedding_model = EmbeddingModel(model="ollama/all-minilm")

        inputs = Input(data_model=Query)
        outputs = await Generator(
            language_model=language_model,
            data_model=AnswerWithRationale,
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="test_program",
            description="A test program",
        )

        program.compile(
            optimizer=FewShotOPRO(
                language_model=language_model,
            ),
            reward=CosineSimilarity(
                embedding_model=embedding_model,
            ),
        )

        (x_train, y_train), (x_test, y_test) = load_test_data()

        mock_responses = mock_completion_data()

        opro_response = (
            """{"instructions": ["Provide a concise and effective response for the """
            """given question.", "Include only one relevant answer with its rationale."""
            """", "Use exact terms from the question to match the predicted output.","""
            """ "Only include the final answer in the required format.","""
            """ "Do not add any additional information or explanations."]}"""
        )

        mock_responses.append(
            {"choices": [{"message": {"content": opro_response}}]},
        )

        mock_completion.side_effect = mock_responses

        expected_value = [0.0, 0.1, 0.2, 0.3]
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        _ = await program.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
        )

        program_vars = program.get_variable(index=0).get_json()
        self.assertTrue(len(program_vars["examples"]) > 0)
        self.assertTrue(len(program_vars["instructions"]) > 0)

    @patch("litellm.aembedding")
    @patch("litellm.acompletion")
    async def test_few_shot_opro_training_with_optimizer_training(
        self, mock_completion, mock_embedding
    ):
        language_model = LanguageModel(model="ollama/mistral")

        embedding_model = EmbeddingModel(model="ollama/all-minilm")

        inputs = Input(data_model=Query)
        outputs = await Generator(
            language_model=language_model,
            data_model=AnswerWithRationale,
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="test_program",
            description="A test program",
        )

        program.compile(
            optimizer=FewShotOPRO(
                language_model=language_model,
            ),
            reward=CosineSimilarity(
                embedding_model=embedding_model,
            ),
        )

        (x_train, y_train), (x_test, y_test) = load_test_data()

        mock_responses = mock_completion_data()

        opro_response = (
            """{"instructions": ["Provide a concise and effective response for the """
            """given question.", "Include only one relevant answer with its rationale."""
            """", "Use exact terms from the question to match the predicted output.","""
            """ "Only include the final answer in the required format.","""
            """ "Do not add any additional information or explanations."]}"""
        )

        mock_responses.append(
            {"choices": [{"message": {"content": opro_response}}]},
        )

        mock_completion.side_effect = mock_responses

        expected_value = [0.0, 0.1, 0.2, 0.3]
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        _ = await program.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            train_optimizer=True,
        )

        opro_program_vars = program.optimizer.trainable_variables
        self.assertTrue(len(opro_program_vars[0].get_json()["examples"]) > 0)
        self.assertTrue(len(opro_program_vars[0].get_json()["instructions"]) > 0)
