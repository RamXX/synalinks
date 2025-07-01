# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import random
from typing import List

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Prediction
from synalinks.src.optimizers.optimizer import Optimizer


class FewShotOptimizedVariable(DataModel):
    examples: List[Prediction] = []
    predictions: List[Prediction] = []


@synalinks_export("synalinks.optimizers.RandomFewShot")
class RandomFewShot(Optimizer):
    """Sample randomly among the best examples to populate the LM's prompt to make it
        learn using Few Shot Learning.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():
        # ... your program definition

        program.compile(
            reward=synalinks.rewards.ExactMatch(),
            optimizer=synalinks.optimizers.RandomFewShot(
                k=3, # The number of examples to provide to the prompt
                k_best=10, # The number of best examples to select from
            ),
        )

        history = await program.fit(...)
    ```

    References:
        - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

    Args:
        k (int): The number of examples to select (default 3) among the best predictions.
        k_best (int): The max number of best predictions to select from (default 10).
    """

    def __init__(
        self,
        k=3,
        k_best=10,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
            data_model=FewShotOptimizedVariable,
        )
        self.k = k
        self.k_best = k_best

    async def build(self, variables):
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
            trainable_variable.update({"examples": selected_predictions})

    async def finalize(self, trainable_variable):
        """Finalize the optimization of a single variable (cleanup/scaling etc.)."""
        trainable_variable.update({"predictions": []})

    def get_config(self):
        return {
            "k": self.k,
            "k_best": self.k_best,
            "name": self.name,
            "description": self.description,
        }
