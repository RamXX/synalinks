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


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The correct answer",
    )


class Critique(synalinks.DataModel):
    critique: str = synalinks.Field(
        description="The critique of the answer",
    )


language_model = synalinks.LanguageModel(
    model="ollama/deepseek-r1",
)


async def main():
    inputs = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=[x1, x2],
        name="parallel_branches",
        description="Illustrate the use of parallel branching",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Decision(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="decision_making",
        description="Illustrate the decision making process",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    inputs = synalinks.Input(data_model=Query)
    (x1, x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=[x1, x2],
        name="conditional_branches",
        description="Illustrate the conditional branches",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    inputs = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    outputs = x1 + x2

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="concatenation",
        description="Illustrate the use of the concatenation",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    inputs = synalinks.Input(data_model=Query)
    (x1, x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        return_decision=False,
    )(inputs)
    x3 = inputs & x1
    x4 = inputs & x2
    x5 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(x3)
    x6 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(x4)
    x7 = x5 | x6
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(x7)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="logical_and",
        description="Illustrate the use of logical and",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    inputs = synalinks.Input(data_model=Query)
    (x1, x2) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking, language_model=language_model
            ),
        ],
        return_decision=False,
    )(inputs)
    outputs = x1 | x2

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="logical_or",
        description="Illustrate the use of logical or",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/control_flow",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
