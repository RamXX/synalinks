# Interactive Agent

Interactive agents represent a significant advancement in AI system design, enabling dynamic interactions with users through iterative processing and validation of inputs. This tutorial will guide you through building an interactive agent using Synalinks, capable of processing mathematical queries and returning numerical answers through a series of interactions.

## Understanding the Foundation

Interactive agents address the need for dynamic and iterative processing of user inputs, allowing for more flexible and responsive AI systems. Unlike autonomous agents, interactive agents require user validation at each step, ensuring accuracy and relevance in their responses.

The architecture of an interactive agent follows several core stages. The input stage captures the user's query or command. The processing stage uses predefined tools and logic to process the input and generate a response. The validation stage requires user input to validate the tool calls and their arguments. Finally, the output stage returns the result to the user.

Interactive agents are transforming the landscape of AI by facilitating dynamic and iterative interactions with users. This guide will walk you through the process of creating an interactive agent using Synalinks, designed to handle mathematical queries and provide numerical answers through a series of user interactions.

## The Basics of Interactive Agents

Interactive agents are designed to process user inputs dynamically, allowing for more engaging and responsive AI systems. Unlike traditional models that operate in a static manner, interactive agents require user validation at each step, ensuring that the responses are accurate and contextually relevant.

The architecture of an interactive agent involves several key stages. Initially, the agent captures the user's input or query thought chat messages. This input is then processed using predefined tools and logic to generate a response. The agent then seeks user validation for the proposed actions or tool calls. Finally, the validated results are presented to the user.

Synalinks simplifies the creation of such interactive agents through its modular architecture, providing the flexibility to design and implement complex workflows with ease.

## Exploring Interactive Agent Architecture

Synalinks offers a streamlined approach to building interactive agents, thanks to its modular and flexible architecture. Each user interaction initiates a new cycle in the Directed Acyclic Graph (DAG), and tool calls are executed only after receiving user validation.

The foundation of an interactive agent lies in defining clear data models and tools. These models ensure that information flows seamlessly through the system, maintaining consistency and reliability.

```python

import synalinks
import asyncio

# Activate logging for monitoring interactions
synalinks.enable_logging()


# Define the calculation tool
@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Perform calculations based on a mathematical expression."""
    # Check for valid characters in the expression
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "result": None,
            "log": (
                "Invalid characters detected in the expression. "
                "Only numbers, operators (+, -, *, /), parentheses, and spaces are allowed."
            ),
        }
    try:
        # Safely evaluate the mathematical expression
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return {
            "result": result,
            "log": "Calculation successful",
        }
    except Exception as e:
        return {
            "result": None,
            "log": f"Calculation error: {e}",
        }

# Main function to configure and run the agent
async def main():
    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )

    # Define the tools available to the agent
    tools = [
        synalinks.Tool(calculate),
    ]

    # Set up the input structure using ChatMessages
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Create the interactive agent
    outputs = await synalinks.FunctionCallingAgent(
        tools=tools,
        language_model=language_model,
        return_inputs_with_trajectory=True,
        autonomous=False,
    )(inputs)

    # Define the agent program
    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="interactive_math_agent",
        description="An agent designed to handle mathematical queries interactively",
    )

    # Initialize the conversation with a user query
    input_messages = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="user",
                content="Calculate the sum of 152648 and 485.",
            )
        ]
    )

    # Process the conversation through multiple iterations
    for iteration in range(MAX_ITERATIONS):
        response = await agent(input_messages)
        print("Agent's response with trajectory:")
        print(response.prettify_json())

        assistant_message = response.get("messages")[-1]
        if not assistant_message.get("tool_calls"):
            break  # Exit the loop if no tool calls are made

        # Here, you would typically validate the tool calls and their arguments
        # through a user interface or command-line interaction.
        # For this example, we assume the validation is successful.
        input_messages.messages.append(assistant_message)

# Execute the main function
if __name__ == "__main__":
    asyncio.run(main())

```

The `ChatMessages` data model is central to the interactive agent, facilitating the exchange of messages between the user and the agent. This model ensures that the conversation remains structured and predictable.

The `FunctionCallingAgent` is responsible for processing user queries using the available tools. It generates responses and maintains the conversation history for context. Each user message triggers a new cycle in the DAG, with tool calls executed only after validation.

### Key Takeaways

- **Dynamic Interaction**: Interactive agents facilitate dynamic and iterative processing of user inputs, enabling more engaging and responsive AI systems.
- **Modular Design**: Synalinks modular architecture simplifies the development of interactive agents, allowing for the creation of sophisticated and maintainable workflows.
- **Structured Data Models**: The use of structured ChatMessages models ensures consistency and predictability in the conversation flow, preventing data inconsistencies.
- **Tool Integration and Validation**: The integration of tools, such as the calculate function, and the validation of their arguments provide a robust foundation for handling specific types of queries.
- **User-Driven Processing**: Interactive agents dynamically process information and execute tasks based on user interactions, enabling complex operations through a series of validated steps.