"""Training integration tests for RLM modules.

Tests that RecursiveGenerator and RecursiveChainOfThought properly integrate
with Synalinks training system (compile, fit, optimizers, rewards).
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from synalinks.src import optimizers
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.rlm.core.recursive_chain_of_thought import (
    RecursiveChainOfThought,
)
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator
from synalinks.src.programs import Program


class Query(DataModel):
    """Test query data model."""

    question: str = Field(description="The question")


class Answer(DataModel):
    """Test answer data model."""

    answer: str = Field(description="The answer")


class TrainingIntegrationTest(testing.TestCase):
    """Training integration tests for RLM modules."""

    def test_recursive_generator_has_trainable_state_variable(self):
        """RecursiveGenerator.trainable_variables includes state variable."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        gen = RecursiveGenerator(
            language_model=mock_lm, instructions="Custom instructions"
        )

        # Verify state variable exists
        self.assertIsNotNone(gen.state)

        # Verify state is in trainable_variables
        trainable_vars = gen.trainable_variables
        self.assertGreater(len(trainable_vars), 0)
        self.assertIn(gen.state, trainable_vars)

    def test_state_variable_contains_instructions_examples_seed_candidates(self):
        """State variable contains instructions, examples, and seed_candidates."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        examples = [({"input": "test1"}, {"output": "result1"})]
        seed_instructions = ["Seed 1", "Seed 2", "Seed 3"]

        gen = RecursiveGenerator(
            language_model=mock_lm,
            instructions="Main instructions",
            examples=examples,
            seed_instructions=seed_instructions,
        )

        # Verify state contains all required fields
        state_data = gen.state.get_json()
        self.assertIn("instructions", state_data)
        self.assertIn("examples", state_data)
        self.assertIn("seed_candidates", state_data)

        # Verify values
        self.assertEqual(state_data["instructions"], "Main instructions")
        self.assertEqual(len(state_data["examples"]), 1)
        self.assertEqual(len(state_data["seed_candidates"]), 3)

        # Verify seed_candidates structure
        for seed_candidate in state_data["seed_candidates"]:
            self.assertIn("instructions", seed_candidate)

    def test_trainable_false_disables_all_variables(self):
        """trainable=False properly disables all variables."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        # Create with trainable=False
        gen = RecursiveGenerator(language_model=mock_lm, trainable=False)

        # Verify no trainable variables
        self.assertEqual(len(gen.trainable_variables), 0)

        # Verify state variable is non-trainable
        non_trainable_vars = gen.non_trainable_variables
        self.assertIn(gen.state, non_trainable_vars)

    def test_trainable_property_setter_disables_variables(self):
        """Setting trainable=False after creation disables variables."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        # Create with trainable=True (default)
        gen = RecursiveGenerator(language_model=mock_lm)
        self.assertGreater(len(gen.trainable_variables), 0)

        # Disable training
        gen.trainable = False

        # Verify no trainable variables
        self.assertEqual(len(gen.trainable_variables), 0)

        # Verify state is now non-trainable
        self.assertIn(gen.state, gen.non_trainable_variables)

    async def test_training_mode_tracks_predictions_in_current_predictions(self):
        """Training mode (training=True) tracks predictions in current_predictions."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test-model"

        # Create mock client
        mock_client = AsyncMock()
        mock_client.get_usage_summary = MagicMock(
            return_value={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "calls": 1,
            }
        )
        # Return FINAL immediately
        mock_client.acompletion.return_value = 'FINAL({"answer": "test answer"})'

        gen = RecursiveGenerator(language_model=mock_lm, data_model=Answer)

        # Verify current_predictions starts empty
        self.assertEqual(len(gen.state.get("current_predictions")), 0)

        with patch(
            "synalinks.src.modules.rlm.core.recursive_generator.SynalinksLMClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            query = Query(question="What is the answer?")

            # Call with training=True
            _ = await gen(query.to_json_data_model(), training=True)

            # Verify prediction was tracked
            current_predictions = gen.state.get("current_predictions")
            self.assertEqual(len(current_predictions), 1)

            # Verify tracked prediction structure
            tracked = current_predictions[0]
            self.assertIn("inputs", tracked)
            self.assertIn("outputs", tracked)
            self.assertIn("reward", tracked)
            self.assertIsNone(tracked["reward"])  # Reward not yet computed

    async def test_training_mode_false_does_not_track_predictions(self):
        """training=False does not track predictions."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test-model"

        mock_client = AsyncMock()
        mock_client.get_usage_summary = MagicMock(
            return_value={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }
        )
        mock_client.acompletion.return_value = 'FINAL({"answer": "test"})'

        gen = RecursiveGenerator(language_model=mock_lm, data_model=Answer)

        with patch(
            "synalinks.src.modules.rlm.core.recursive_generator.SynalinksLMClient"
        ) as mock_client_class:
            mock_client_class.return_value = mock_client

            query = Query(question="Test?")

            # Call with training=False (default)
            await gen(query.to_json_data_model(), training=False)

            # Verify no predictions tracked
            self.assertEqual(len(gen.state.get("current_predictions")), 0)

    def test_examples_provided_at_init_stored_in_state(self):
        """Examples provided at init are stored in state."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        examples = [
            ({"question": "What is 1+1?"}, {"answer": "2"}),
            ({"question": "What is 2+2?"}, {"answer": "4"}),
        ]

        gen = RecursiveGenerator(language_model=mock_lm, examples=examples)

        # Verify examples stored in state
        state_examples = gen.state.get("examples")
        self.assertEqual(len(state_examples), 2)

        # Verify structure matches Prediction format
        for example in state_examples:
            self.assertIn("inputs", example)
            self.assertIn("outputs", example)
            self.assertIn("reward", example)

    async def test_program_can_compile_with_recursive_generator(self):
        """Program can compile() with RecursiveGenerator."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        x0 = Input(data_model=Query)
        x1 = await RecursiveGenerator(
            language_model=mock_lm, data_model=Answer, max_iterations=10
        )(x0)

        program = Program(
            inputs=x0, outputs=x1, name="test_rlm_program", description="Test RLM program"
        )

        # Compile with optimizer and reward
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(),
        )

        # Verify compilation succeeded
        self.assertTrue(program.compiled)
        self.assertIsNotNone(program.optimizer)
        self.assertIsNotNone(program.reward)

    async def test_seed_instructions_enable_omega_style_prompt_evolution(self):
        """Seed instructions enable OMEGA-style prompt evolution."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        seed_instructions = [
            "Approach 1: Think step by step",
            "Approach 2: Break down into sub-problems",
            "Approach 3: Use code to verify",
        ]

        gen = RecursiveGenerator(
            language_model=mock_lm, seed_instructions=seed_instructions
        )

        # Verify seed_candidates in state
        seed_candidates = gen.state.get("seed_candidates")
        self.assertEqual(len(seed_candidates), 3)

        # Verify format matches what optimizers expect
        for i, candidate in enumerate(seed_candidates):
            self.assertIn("instructions", candidate)
            self.assertEqual(candidate["instructions"], seed_instructions[i])

    async def test_program_get_state_tree_includes_recursive_generator_state(self):
        """Program.get_state_tree() includes RecursiveGenerator trainable state."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        x0 = Input(data_model=Query)
        x1 = await RecursiveGenerator(
            language_model=mock_lm,
            data_model=Answer,
            instructions="Test instructions",
        )(x0)

        program = Program(inputs=x0, outputs=x1, name="test_program")

        state_tree = program.get_state_tree()

        # Verify structure includes trainable_variables
        self.assertIn("trainable_variables", state_tree)
        self.assertIn("recursive_generator", state_tree["trainable_variables"])

        # Verify state variable is present
        gen_state = state_tree["trainable_variables"]["recursive_generator"]
        self.assertIn("state_recursive_generator", gen_state)

        # Verify state contains required fields
        state_data = gen_state["state_recursive_generator"]
        self.assertIn("instructions", state_data)
        self.assertIn("examples", state_data)
        self.assertIn("seed_candidates", state_data)
        self.assertEqual(state_data["instructions"], "Test instructions")

    async def test_recursive_chain_of_thought_inherits_training_support(self):
        """RecursiveChainOfThought inherits training support from RecursiveGenerator."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        cot = RecursiveChainOfThought(
            language_model=mock_lm,
            data_model=Answer,
            k=2,
            instructions="Think carefully",
        )

        # Verify has trainable state variable
        self.assertIsNotNone(cot.state)
        self.assertIn(cot.state, cot.trainable_variables)

        # Verify state contains required fields (inherited from RecursiveGenerator)
        state_data = cot.state.get_json()
        self.assertIn("instructions", state_data)
        self.assertIn("examples", state_data)
        self.assertIn("seed_candidates", state_data)

    async def test_recursive_chain_of_thought_trainable_false_works(self):
        """RecursiveChainOfThought respects trainable=False."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        cot = RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, trainable=False
        )

        # Verify no trainable variables
        self.assertEqual(len(cot.trainable_variables), 0)

        # Verify state is non-trainable
        self.assertIn(cot.state, cot.non_trainable_variables)

    async def test_recursive_chain_of_thought_can_compile_in_program(self):
        """Program can compile() with RecursiveChainOfThought."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "zai/glm-4.7"

        x0 = Input(data_model=Query)
        x1 = await RecursiveChainOfThought(
            language_model=mock_lm, data_model=Answer, k=3
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_cot_program",
            description="Test CoT program",
        )

        # Compile
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(),
        )

        # Verify compilation succeeded
        self.assertTrue(program.compiled)

    async def test_multi_model_training_setup(self):
        """Training works with multi-model setup (root + sub models)."""
        mock_lm_root = MagicMock(spec=LanguageModel)
        mock_lm_root.model = "zai/glm-4.7"
        mock_lm_sub = MagicMock(spec=LanguageModel)
        mock_lm_sub.model = "groq/openai/gpt-oss-20b"

        gen = RecursiveGenerator(
            language_model=mock_lm_root,
            sub_language_model=mock_lm_sub,
            instructions="Multi-model instructions",
            seed_instructions=["Seed A", "Seed B"],
        )

        # Verify trainable state variable created
        self.assertIsNotNone(gen.state)
        self.assertIn(gen.state, gen.trainable_variables)

        # Verify both models are preserved
        self.assertEqual(gen.language_model, mock_lm_root)
        self.assertEqual(gen.sub_language_model, mock_lm_sub)

        # Verify state includes seed instructions
        self.assertEqual(len(gen.state.get("seed_candidates")), 2)

    async def test_serialization_preserves_training_state(self):
        """get_config/from_config preserves training-related state."""
        mock_lm = MagicMock(spec=LanguageModel)
        mock_lm.model = "test-model"
        mock_lm.get_config.return_value = {"model": "test-model"}

        examples = [({"input": "q1"}, {"output": "a1"})]
        seed_instructions = ["Seed 1", "Seed 2"]

        gen1 = RecursiveGenerator(
            language_model=mock_lm,
            instructions="Original instructions",
            examples=examples,
            seed_instructions=seed_instructions,
        )

        # Serialize
        config = gen1.get_config()

        # Verify config contains training-related fields
        self.assertEqual(config["instructions"], "Original instructions")
        self.assertEqual(config["examples"], examples)
        self.assertEqual(config["seed_instructions"], seed_instructions)

        # Deserialize
        gen2 = RecursiveGenerator.from_config(config)

        # Verify training state preserved
        self.assertEqual(gen2.instructions, "Original instructions")
        self.assertEqual(len(gen2.state.get("examples")), 1)
        self.assertEqual(len(gen2.state.get("seed_candidates")), 2)
