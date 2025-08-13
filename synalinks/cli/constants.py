# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os


def get_fake_data(filename):
    from synalinks.cli import datasets

    folder_path = datasets.__file__[: -len("/__init__.py")]
    json_data = ""
    with open(os.path.join(folder_path, filename), "r") as f:
        json_data = f.read()
    return json_data


PROJECT_CONFIG_FILENAME = "synalinks_project.json"

TEMPLATE_CONFIG_FILENAME = "synalinks_template.json"

README_TEMPLATE = """# {{config.project_name}}
## {{config.project_description}}

![{{config.project_name}}]({{config.project_name}}.png)

![training_history](training_history.png)

# Install dependencies

To plot your programs, you will need graphviz

```shell
sudo apt update
sudo apt install graphviz
```

Create your virtual environment and install synalinks

```shell
cd {{config.project_name}}
uv venv
uv pip install synalinks
```

# Train your application

```shell
cd {{config.project_name}}
python3 scripts/train.py
```

# Serve your API

```shell
cd {{config.project_name}}
docker compose up --build
```

Note: Ollama models are great for experimenting/training but not to use in production.

Powered by [🧠🔗 - synalinks](https://github.com/SynaLinks/synalinks)

More information available in the [documentation](https://synalinks.github.io/synalinks/)

If you have questions, check out the [FAQ](https://synalinks.github.io/synalinks/FAQ/).

Join our [Discord](https://discord.gg/82nt97uXcM) to never miss any update!

Happy coding 😀!
"""

PYPROJECT_TEMPLATE = """[project]
name = "{{config.package_name}}"
description = "{{config.project_description}}"
readme = "README.md"
requires-python = ">=3.9"
dynamic = ["version"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3 :: Only",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]
dependencies = [
    "synalinks"
]
[tool.setuptools.dynamic]
version = {attr = "{{config.package_name}}.src.version.__version__"}

[tool.setuptools.packages.find]
include = ["{{config.package_name}}.src", "{{config.package_name}}.src.*"]
"""


GITIGNORE_TEMPLATE = """# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

# Environment variables
.env
"""

MAIN_TEMPLATE = """import os
import synalinks
import psutil
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
# Uncomment for streaming apps
# from fastapi.responses import StreamingResponse

# Import your input data model for your API
from {{config.package_name}}.src.data_models import Query
# Import your custom module
from {{config.package_name}}.src.modules import AnswerWithChainOfThought

# Load the .env variables
load_dotenv()

# Clear Synalinks context
synalinks.clear_session()

async def create_program():
    language_model = synalinks.LanguageModel(
        model=os.environ.get(
            "LANGUAGE_MODEL",
            "{{config.model_provider}}/{{config.language_model}}",
        ),
        api_base=os.environ.get("MODEL_API_BASE", None),
    )
        
    inputs = synalinks.Input(data_model=Query)
    outputs = await AnswerWithChainOfThought(
        language_model=language_model,
    )(inputs)

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="{{config.package_name}}",
        description="{{config.project_description}}",
    )
    
program = asyncio.run(create_program())

# Load your application variables (the ones optimized by the training)
program_variables_filepath = "checkpoint.program.variables.json"
if os.path.exists(program_variables_filepath):
    program.load_variables(program_variables_filepath)
    
# Setup FastAPI
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/system_check")
def system_check():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage("/")
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_info.percent,
        "disk_usage": disk_usage.percent,
    }


@app.post("/v1/{{config.package_name}}")
async def {{config.package_name}}(inputs: Query):
    result = await program(inputs)
    # Uncomment for streaming apps
    # return StreamingResponse(result, media_type="application/json") if result else None
    return result.get_json() if result else None
"""

DOCKERFILE_TEMPLATE = """FROM python:3.13

COPY ./requirements.txt ./requirements.txt

RUN apt update && apt install -y graphviz

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY . ./

CMD ["fastapi", "run", "{{config.package_name}}/src/main.py", "--port", "8000"]
"""

DOCKER_COMPOSE_TEMPLATE = """services:
  {{config.package_name}}:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
"""

VERSION_TEMPLATE = """
# Unique source of truth for the version number.
__version__ = "0.1.0"
"""

INIT_TEMPLATE = """from {{config.package_name}}.src.version import __version__
from {{config.package_name}}.src.main import main
"""

DATA_MODELS_INIT_TEMPLATE = """from {{config.package_name}}.src.data_models.query import Query
from {{config.package_name}}.src.data_models.answer import Answer
"""

MODULES_INIT_TEMPLATE = """from {{config.package_name}}.src.modules.answer_with_chain_of_thought import AnswerWithChainOfThought
"""

REQUIREMENTS_TEMPLATE = """fastapi[standard]
psutil
synalinks
"""

QUERY_TEMPLATE = """import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )
"""

ANSWER_TEMPLATE = """import synalinks

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The correct answer",
    )
"""

PROGRAM_TEMPLATE = """import synalinks
from {{config.package_name}}.src.data_models import Answer

class AnswerWithChainOfThought(synalinks.Program):
    \"""Answer step by step.
    
    Args:
        language_model (LanguageModel): The language model to use.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    \"""

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=None,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.ChainOfThought(
            data_model=Answer,
            language_model=self.language_model,
        )(inputs)
        
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )
"""

TRAIN_TEMPLATE = """#!/usr/bin/python3
import os
import synalinks
import asyncio
import json
import numpy as np
from {{config.package_name}}.src.data_models import Query
from {{config.package_name}}.src.data_models import Answer
from {{config.package_name}}.src.modules import AnswerWithChainOfThought

# Clear Synalinks context
synalinks.clear_session()

# The training is done in batch, the batch size specify the 
# number of parralel program execution performed to train 
# the application. The reward is averaged per batch yielding 
# a better estimation of your program success.
BATCH_SIZE=32

# The epochs refer to the number of time the whole dataset is
# proccessed. At the end of each epochs, the optimization is
# performed. So the epochs is the number of successive optimization.
EPOCHS=4

def load_dataset(
    input_data_model,
    output_data_model,
    x_train_filepath="datasets/x_train.json",
    y_train_filepath="datasets/y_train.json",
    x_test_filepath="datasets/x_test.json",
    y_test_filepath="datasets/y_test.json",
):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    with open(x_train_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            x_train.append(input_data_model(**data_point))
    
    with open(y_train_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            y_train.append(output_data_model(**data_point))
        
    with open(x_test_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            x_test.append(input_data_model(**data_point))
            
    with open(y_test_filepath, "r") as f:
        json_data = json.loads(f.read())
        for data_point in json_data:
            y_test.append(output_data_model(**data_point))
    
    # Convert the dataset into numpy arrays
    
    x_train = np.array(x_train, dtype="object")
    y_train = np.array(y_train, dtype="object")
    
    x_test = np.array(x_test, dtype="object")
    y_test = np.array(y_test, dtype="object")

    return (x_train, y_train), (x_test, y_test)

async def train_program():
    language_model = synalinks.LanguageModel(
        model=os.environ.get(
            "LANGUAGE_MODEL",
            "{{config.model_provider}}/{{config.language_model}}"
        ),
    )
    
    embedding_model = synalinks.EmbeddingModel(
        model=os.environ.get(
            "EMBEDDING_MODEL",
            "{{config.model_provider}}/{{config.embedding_model}}"
        ),
    )
    
    inputs = synalinks.Input(data_model=Query)
    outputs = await AnswerWithChainOfThought(
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="{{config.project_name}}",
        description="{{config.project_description}}",
    )
    
    embedding_model = synalinks.EmbeddingModel(
        model=os.environ.get(
            "EMBEDDING_MODEL",
            "{{config.model_provider}}/{{config.embedding_model}}",
        ),
    )
    
    synalinks.utils.plot_program(
        program,
        show_module_names=True,
        show_schemas=True,
    )
    
    checkpoint_filepath = "checkpoint.program.variables.json"
    
    program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_reward",
        save_variables_only=True,
        mode="max",
        save_best_only=True,
    )
    
    program.compile(
        reward=synalinks.rewards.CosineSimilarity(
            # Filter to keep only the `answer` field in order to compute the reward
            in_mask=["answer"],
            # The embedding model to use to compute the similarity
            embedding_model=embedding_model,
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[
            synalinks.metrics.F1Score(in_mask=["answer"]),
        ],
    )
    
    (x_train, y_train), (x_test, y_test) = load_dataset(Query, Answer)
    
    history = await program.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[program_checkpoint_callback],
    )
    
    synalinks.utils.plot_history(
        history,
    )
    
def main():
    asyncio.run(train_program())

if __name__ == "__main__":
    main()
"""

PROJECT_STRUCTURE_TEMPLATE = {
    "README.md": README_TEMPLATE,
    "pyproject.toml": PYPROJECT_TEMPLATE,
    "{{config.package_name}}": {
        "src": {
            "data_models": {
                "__init__.py": DATA_MODELS_INIT_TEMPLATE,
                "query.py": QUERY_TEMPLATE,
                "answer.py": ANSWER_TEMPLATE,
                "README.md": (
                    "For the list of built-in data models check the documentation "
                    "[here](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)",
                ),
            },
            "modules": {
                "__init__.py": MODULES_INIT_TEMPLATE,
                "answer_with_chain_of_thought.py": PROGRAM_TEMPLATE,
                "README.md": (
                    "For the list of built-in modules check the documentation "
                    "[here](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/)",
                ),
            },
            "tools": {
                "__init__.py": "",
                "README.md": "Here lies the tools for your agents",
            },
            "__init__.py": "",
            "version.py": VERSION_TEMPLATE,
            "main.py": MAIN_TEMPLATE,
        },
    },
    "scripts": {
        "train.py": TRAIN_TEMPLATE,
        "README.md": "Here lies your scripts",
    },
    "datasets": {
        "x_train.json": get_fake_data("x_train.json"),
        "y_train.json": get_fake_data("y_train.json"),
        "x_test.json": get_fake_data("x_test.json"),
        "y_test.json": get_fake_data("y_test.json"),
        "README.md": "Here lies your custom datasets",
    },
    "requirements.txt": REQUIREMENTS_TEMPLATE,
    ".gitignore": GITIGNORE_TEMPLATE,
    "Dockerfile": DOCKERFILE_TEMPLATE,
    "docker-compose.yml": DOCKER_COMPOSE_TEMPLATE,
}
