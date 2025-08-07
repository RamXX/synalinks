import synalinks
import asyncio

synalinks.enable_logging()


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )


class ChainOfThought(synalinks.Program):
    """Useful to answer in a step by step manner.

    The first line of the docstring is provided as description for the program
    if not provided in the `super().__init__()`. In a similar way the name is
    automatically infered based on the class name if not provided.
    """

    def __init__(self, language_model=None):
        super().__init__()
        self.answer = synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=language_model,
        )

    async def call(self, inputs, training=False):
        x = await self.answer(inputs)
        return x

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)


language_model = synalinks.LanguageModel(
    model="ollama/mistral",
)


async def main():
    # Functional API

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chain_of_thought_functional",
        description="Useful to answer in a step by step manner.",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/first_programs",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    program = ChainOfThought(
        language_model=language_model,
    )
    await program.build(Query)

    synalinks.utils.plot_program(
        program,
        to_folder="examples/first_programs",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    program = synalinks.Sequential(
        [
            synalinks.Input(data_model=Query),
            ChainOfThought(language_model=language_model),
        ],
        name="chain_of_thought_sequential",
        description="Useful to answer in a step by step manner.",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/first_programs",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
