import synalinks
import asyncio

MAX_ITERATIONS = 5

@synalinks.utils.register_synalinks_serializable()
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
            "log": (
                    "Error: invalid characters in expression. "
                    "The expression can only contain numbers, operators (+, -, *, /),"
                    " parentheses, and spaces NOT letters."
                ),
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

async def main():

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    tools = [
        synalinks.Tool(calculate),
    ]

    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    outputs = await synalinks.FunctionCallingAgent(
        tools=tools,
        language_model=language_model,
        return_inputs_with_trajectory=True,
        autonomous=False,
    )(inputs)
    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A math agent",
    )

    input_messages = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="user",
                content="How much is 152648 + 485?",
            )
        ]
    )
    
    for i in range(MAX_ITERATIONS):
        
        response = await agent(input_messages)
        
        print("Assistant response (with trajectory):")
        print(response.prettify_json())
        
        assistant_message = response.get("messages")[-1]
        
        if not assistant_message.get("tool_calls"):
            break # We stop the loop if the agent didn't call any tool
        
        # Validate the tool calls arguments (with an UI or CLI)
        # Then re-inject the validated assistant response in the input_messages
        # The corresponding tools will be called by the agent
        # Here we assume everything is okay for the purpose of the demo ^^
        
        input_messages.messages.append(assistant_message)

if __name__ == "__main__":
    asyncio.run(main())