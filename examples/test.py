import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )

class FinalAnswer(synalinks.DataModel):
    answer: float = synalinks.Field(
        description="The correct final answer",
    )

async def main():

    async def calculate(expression: str):
        """Calculate the result of a mathematical expression.

        Args:
            expression (str): The mathematical expression to calculate, such as
                '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                parentheses, and spaces.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            return {
                "result": None,
                "log": "Error: invalid characters in expression",
            }
        try:
            # Evaluate the mathematical expression safely
            result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
            return {
                "result": result,
                "log": "Successfully executed",
            }
        except Exception as e:
            return {
                "result": None,
                "log": f"Error: {e}",
            }

    language_model = synalinks.LanguageModel(model="ollama/mistral")
    
    agent = synalinks.ReACTAgent(
        data_model=FinalAnswer,
        language_model=language_model,
        functions=[calculate],
        max_iterations=3,
    )
    
    x0 = synalinks.Input(data_model=Query)
    x1 = await agent(x0)

    program = synalinks.Program(
        inputs=x0,
        outputs=x1,
        name="math_agent",
        description="A math agent that can use a calculator",
    )
    
    synalinks.utils.plot_program(
        agent,
        show_schemas=False,
    )

if __name__ == "__main__":
    asyncio.run(main())