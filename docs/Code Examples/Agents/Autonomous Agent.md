# Autonomous Agents

Autonomous agents represent a significant advancement in AI system design, combining the power of language models with the ability to perform tasks autonomously. This tutorial will guide you through building an autonomous agent using Synalinks, capable of processing mathematical queries and returning numerical answers.

## Understanding the Foundation

Autonomous agents address a fundamental limitation of traditional systems by enabling them to perform tasks without constant human intervention. While language models excel at generating coherent text, they often require additional tools and logic to perform specific tasks autonomously. Autonomous agents bridge this gap by dynamically processing information and executing tasks based on predefined tools.

The architecture of an autonomous agent follows several core stages. The input stage captures the user's query or command. The processing stage uses predefined tools and logic to process the input and generate a response. Finally, the output stage returns the result to the user.

Synalinks streamlines this complex process through its modular architecture, allowing you to compose components with precision while maintaining flexibility for different use cases.

## Understanding Autonomous Agent Architecture

Synalinks simplifies the implementation of autonomous agents through its modular architecture, allowing you to compose components with precision and flexibility.

The foundation of any autonomous agent begins with defining your data models. These models structure how information flows through your pipeline and ensure consistency across components.


```python
import synalinks
import asyncio
import uuid

synalinks.enable_logging()


class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class NumericalFinalAnswer(synalinks.DataModel):
    final_answer: float = synalinks.Field(
        description="The correct final numerical answer",
    )


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

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalFinalAnswer,
        tools=tools,
        language_model=language_model,
        max_iterations=5,
        return_inputs_with_trajectory=True,
        autonomous=True,
    )(inputs)
    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A math agent",
    )

    input_query = Query(query="How much is 152648 + 485?")
    response = await agent(input_query)

    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
```

#### Result

```json
{
  "query": "How much is 152648 + 485?",
  "messages": [
    {
      "role": "assistant",
      "content": "I will calculate the sum of the given numbers.",
      "tool_call_id": null,
      "tool_calls": [
        {
          "id": "79e1bd1f-d7b3-4807-9508-1fbd2e1365ac",
          "name": "calculate",
          "arguments": {
            "expression": "152648 + 485"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": {
        "result": 153133.0,
        "log": "Successfully executed"
      },
      "tool_call_id": "79e1bd1f-d7b3-4807-9508-1fbd2e1365ac",
      "tool_calls": []
    },
    {
      "role": "assistant",
      "content": "Upon observing the input, I note that it contains a mathematical expression '152648 + 485'. I have already executed a tool call to calculate this sum in the previous step. To avoid unnecessary repetition and maintain efficiency, I will return an empty tool calls array.",
      "tool_call_id": null,
      "tool_calls": []
    }
  ],
  "final_answer": 153133
}
```

The `Query` and `NumericalFinalAnswer` data models serve as the input and output contracts for your autonomous agent. The `Query` model captures user questions, while the `NumericalFinalAnswer` model structures the system's responses.

This explicit modeling ensures type safety and makes your pipeline's behavior predictable.

The calculate function represents the core tool that the agent uses to perform mathematical calculations.
It evaluates a mathematical expression and returns the result, ensuring that the agent can process numerical queries accurately.

The `FunctionCallingAgent` component processes the user's query using the predefined tools and logic. It returns the result of the calculation and maintains the original input for downstream processing.

### Key Takeaways

- **Autonomous Task Execution**: Autonomous agents solve the fundamental problem of performing tasks without constant human intervention, enabling systems to process information and execute tasks dynamically.
- **Synalinks Modular Implementation**: The framework simplifies the development of autonomous agents through composable components like `FunctionCallingAgent`, allowing you to build sophisticated pipelines with clear data flow and maintainable architecture.
- **Explicit Data Model Contracts**: Using structured `Query` and `NumericalFinalAnswer` models ensures type safety and predictable behavior throughout your pipeline, preventing data inconsistencies and enabling reliable processing across all components.
- **Tool Integration**: The tutorial demonstrates how to integrate tools like the calculate function into your autonomous agent, providing a robust foundation for processing specific types of queries.
- **Dynamic Processing**: The example shows how autonomous agents can dynamically process information and execute tasks based on predefined tools and logic, enabling them to perform complex operations autonomously.