# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import random
from typing import List

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Instructions
from synalinks.src.backend import Prediction
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.input_module import Input
from synalinks.src.optimizers.opro import OPROInputs
from synalinks.src.optimizers.optimizer import Optimizer
from synalinks.src.programs import Program
from synalinks.src.saving import serialization_lib


class FewShotOPROOptimizedVariables(DataModel):
    examples: List[Prediction] = []
    predictions: List[Prediction] = []
    instructions: Instructions
    instructions_candidates: List[Instructions] = []


@synalinks_export("synalinks.optimizers.FewShotOPRO")
class FewShotOPRO(Optimizer):
    """Sample randomly among the best examples to populate the LM's prompt to make it
        learn using Few Shot Learning while generating instructions with OPRO.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():
        # ... your program definition

        program.compile(
            reward=synalinks.rewards.ExactMatch(),
            optimizer=synalinks.optimizers.FewShotOPRO(
                language_model=language_model,
                k=3, # The number of examples to provide to the prompt
                k_best=10, # The number of best examples to select from
            ),
        )

        history = await program.fit(...)
    ```

    References:
        - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
        - [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)

    Args:
        language_model (LanguageModel): The language model to use.
        k (int): The number of examples to select (default 3) among the best predictions.
        k_best (int): The max number of best predictions/instructions to select from
            (default 10).
        program (Program): The program to use. Optional. If None create one at start.
        name (str): The name of the optimizer.
        description (str): The description of the optimizer.
    """

    def __init__(
        self,
        language_model=None,
        k=3,
        k_best=10,
        program=None,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
            data_model=FewShotOPROOptimizedVariables,
        )
        self.language_model = language_model
        self.k = k
        self.k_best = k_best
        self.program = program

    async def build(self, variables):
        if not self.program:
            opro_inputs = Input(data_model=OPROInputs)
            opro_outputs = await Generator(
                language_model=self.language_model,
                data_model=Instructions.out_mask(mask=["label", "reward"]),
                instructions=[
                    "Your task is to generate instructions that maximize rewards.",
                    "The reward ranges from 0.0 to 1.0",
                    "Below are some previous instructions candidates with their rewards.",
                    (
                        "Generate instructions that are different from all "
                        "the candidates instructions."
                    ),
                    (
                        "The instructions should be concise, effective and generally"
                        " applicable to all predictions below."
                    ),
                ],
            )(opro_inputs)

            self.program = Program(
                inputs=opro_inputs,
                outputs=opro_outputs,
                name="opro",
                description="OPRO Program",
            )
        self.built = True

    async def optimize(self, trainable_variable, reward=None, training=False):
        """Perform a backprop/optimization on a single variable."""
        # Reward backpropagation
        predictions = trainable_variable.get("predictions")
        backpropagated_predictions = []
        backprop_pred_nb = 0
        for p in predictions:
            if p["reward"] is None:
                p["reward"] = reward
                backprop_pred_nb += 1
            backpropagated_predictions.append(p)
        if backprop_pred_nb > 0:
            trainable_variable.update({"predictions": backpropagated_predictions})
            # Backpropagate instructions reward
            instructions_candidates = trainable_variable.get("instructions_candidates")
            instructions = trainable_variable.get("instructions")
            instructions.update({"reward": reward})
            instructions_candidates.append(instructions)
            trainable_variable.update(
                {"instructions_candidates": instructions_candidates}
            )
            # Get the k best predictions (sorted by reward)
            sorted_predictions = sorted(
                backpropagated_predictions,
                key=lambda x: x["reward"] if x["reward"] is not None else float("-inf"),
                reverse=True,
            )
            top_k_predictions = sorted_predictions[: self.k_best]
            if len(top_k_predictions) > self.k:
                selected_predictions = random.sample(top_k_predictions, self.k)
            else:
                selected_predictions = top_k_predictions
            # Get the k best instructions candidates (sorted by reward)
            sorted_instructions_candidates = sorted(
                instructions_candidates,
                key=lambda x: x["reward"] if x["reward"] is not None else float("-inf"),
                reverse=True,
            )
            top_k_instructions_candidates = sorted_instructions_candidates[: self.k_best]
            # Prepare inputs for OPRO
            inputs = OPROInputs(
                instructions_candidates=top_k_instructions_candidates,
                predictions=top_k_predictions,
            )
            new_instructions = await self.program(inputs, training=training)
            new_instructions_json = {
                "label": "Instructions",
                **new_instructions.get_json(),
                "reward": None,
            }
            trainable_variable.update(
                {
                    "instructions": new_instructions_json,
                    "examples": selected_predictions,
                }
            )

    async def finalize(self, trainable_variable):
        """Finalize the optimization of a single variable (cleanup/scaling etc.)."""
        trainable_variable.update({"predictions": [], "instructions_candidates": []})

    def get_config(self):
        config = {
            "k": self.k,
            "k_best": self.k_best,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        program_config = {
            "program": serialization_lib.serialize_synalinks_object(
                self.program,
            )
        }
        return {**config, **language_model_config, **program_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        program = serialization_lib.deserialize_synalinks_object(
            config.pop("program"),
        )
        return cls(language_model=language_model, program=program, **config)
