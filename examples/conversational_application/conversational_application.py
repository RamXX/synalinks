import synalinks
import asyncio

from synalinks.backend import ChatMessages

synalinks.enable_logging()


async def main():
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    inputs = synalinks.Input(data_model=ChatMessages)
    outputs = await synalinks.Generator(
        language_model=language_model,
        prompt_template=synalinks.chat_prompt_template(),
        streaming=True,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_chatbot",
        description="A simple conversation application",
    )

    synalinks.utils.plot_program(
        program,
        to_folder="examples/conversational_application",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
