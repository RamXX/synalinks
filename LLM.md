# Synalinks

## What is Synalinks?

Synalinks is an adaptation of Keras 3 focused on neuro-symbolic systems and in-context reinforcement learning, an ensemble of techniques that enhance the LMs predictions and accuracy without changing the weights of the model. Synalinks is an open-source framework that makes it easy to create, evaluate, train, and deploy industry-standard Language Models (LMs) applications.

## Core Concept

The goal of Synalinks is to facilitate the rapid setup of simple applications while providing the flexibility for researchers and advanced users to develop sophisticated systems.

The main concept of Synalinks is that it allows developers to build workflows/agents using the Functional API inherited from Keras. Instead of describing a directed acyclic graph (DAG) of tensor computation, it describes a DAG of JSON computation. In this context, the JSON schema acts as a specification (similar to the tensor shape in deep learning frameworks) of the data. The whole graph is computed only using the specification when the program is built.

## Key Features

### JSON-Based Data Flow
- **Data Models**: Uses Pydantic-based `DataModel` classes with field descriptions for structured JSON processing
- **Schema Specification**: JSON schemas act as specifications for data flow, similar to tensor shapes in traditional deep learning
- **Type Safety**: Ensures data correctness through structured output validation

### Neuro-Symbolic Architecture
- **Symbolic Reasoning**: Combines neural network capabilities with symbolic logic and reasoning
- **In-Context Learning**: Enhances predictions without modifying model weights
- **Constrained Generation**: Ensures both format and content correctness through structured outputs

### Programming Paradigms
Synalinks offers four distinct ways to build applications:

1. **Functional API**: Chain modules using Input → Processing → Output patterns
2. **Subclassing Program**: Define modules in `__init__()` and implement structure in `call()`
3. **Mixed Approach**: Combine subclassing with Functional API using `build()` method
4. **Sequential API**: Stack single-input, single-output modules linearly

### Production-Ready Features
- **Versioning**: Each program is serializable into JSON so you can version it with git
- **REST API Deployment**: Compatible out-of-the-box with FastAPI
- **Async Optimization**: Automatically optimizes pipelines by detecting parallel processes
- **Multi-Provider Support**: Integrates with Ollama, OpenAI, Anthropic, Mistral, and Groq
- **Documentation Tools**: Built-in plotting and visualization for workflows and training history

## Core Components

### DataModel
```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )
```

### Language Models
```python
language_model = synalinks.LanguageModel(
    model="ollama_chat/deepseek-r1",
)
```

### Modules
- **Input**: Defines input data structure
- **Generator**: Processes data using language models
- **Program**: Orchestrates the entire workflow
- **Sequential**: Linear processing pipeline

### Training and Optimization
- **In-Context Reinforcement Learning**: Optimizes prompts without changing model weights
- **Rewards System**: Built-in evaluation metrics (e.g., ExactMatch)
- **Optimizers**: Various optimization strategies (e.g., RandomFewShot)
- **Batch Processing**: Supports batch training with configurable parameters

## Example Usage

### Basic Functional API
```python
import synalinks
import asyncio

async def main():
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="The user query")
    
    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(description="Step by step thinking")
        answer: float = synalinks.Field(description="Numerical answer")
    
    language_model = synalinks.LanguageModel(model="ollama_chat/deepseek-r1")
    
    x0 = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(x0)
    
    program = synalinks.Program(
        inputs=x0,
        outputs=x1,
        name="chain_of_thought",
        description="Step by step reasoning system",
    )
```

### Training Example
```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.RandomFewShot()
)

history = await program.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=10,
)
```

## Target Users

Synalinks is designed for a diverse range of users, from professionals and AI researchers to students, independent developers, and hobbyists. It is suitable for anyone who wants to learn about AI by building/composing blocks or build solid foundations for enterprise-grade products.

## Technical Foundation

This work has been done under the supervision of François Chollet, the author of Keras. The framework leverages:
- **Keras**: For graph-based computation backbone and API design
- **DSPy**: For modules and optimizers inspiration
- **Pydantic**: For backend data layer and validation

## Philosophy

Synalinks follows the principle of progressive disclosure of complexity: meaning that simple workflows should be quick and easy, while arbitrarily advanced ones should be possible via a clear path that builds upon what you've already learned.

The framework bridges the gap between neural networks and symbolic reasoning, creating more robust, interpretable, and controllable AI systems that maintain the flexibility of deep learning while adding the transparency and reliability of symbolic approaches.

## Resources

- **Documentation**: https://synalinks.github.io/synalinks/
- **GitHub**: https://github.com/SynaLinks/synalinks  
- **Website**: https://www.synalinks.com/

## Installation

```bash
uv pip install synalinks
uv run synalinks init
```
