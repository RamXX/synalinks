# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

"""
We provide different backend-dependent `DataModel`s to use.

These data models provide I/O for chatbots, agents, rags, knowledge graphs etc.

The user can build new data models by inheriting from these base models.

The check functions works for every type of data models (by checking the schema)
e.g. `SymbolicDataModel`, `JsonDataModel`, `DataModel` or `Variable`.
"""

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from pydantic import Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export(
    [
        "synalinks.backend.GenericOutputs",
        "synalinks.GenericOutputs",
    ]
)
class GenericOutputs(DataModel):
    """A generic outputs"""

    outputs: Dict[str, Any] = Field(
        description="The outputs",
    )


@synalinks_export(
    [
        "synalinks.backend.GenericInputs",
        "synalinks.GenericInputs",
    ]
)
class GenericInputs(DataModel):
    """A generic inputs"""

    inputs: Dict[str, Any] = Field(
        description="The inputs",
    )


@synalinks_export(
    [
        "synalinks.backend.GenericIO",
        "synalinks.GenericIO",
    ]
)
class GenericIO(DataModel):
    """A pair of generic inputs/outputs"""

    inputs: Dict[str, Any] = Field(
        description="The inputs",
    )
    outputs: Dict[str, Any] = Field(
        description="The outputs",
    )


@synalinks_export(
    [
        "synalinks.backend.GenericResult",
        "synalinks.GenericResult",
    ]
)
class GenericResult(DataModel):
    """A generic result"""

    result: List[Any] = Field(
        description="The result",
    )


@synalinks_export(
    [
        "synalinks.backend.ChatRole",
        "synalinks.ChatRole",
    ]
)
class ChatRole(str, Enum):
    """The chat message roles"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@synalinks_export(
    [
        "synalinks.backend.ChatMessage",
        "synalinks.ChatMessage",
    ]
)
class ChatMessage(DataModel):
    """A chat message"""

    role: ChatRole
    content: str


@synalinks_export(
    [
        "synalinks.backend.is_chat_message",
        "synalinks.is_chat_message",
    ]
)
def is_chat_message(x):
    """Checks if the given data model is a chat message

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ChatMessage.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.ChatMessages",
        "synalinks.ChatMessages",
    ]
)
class ChatMessages(DataModel):
    """A list of chat messages"""

    messages: List[ChatMessage] = Field(
        description="The list of chat messages",
        default=[],
    )


@synalinks_export(
    [
        "synalinks.backend.is_chat_messages",
        "synalinks.is_chat_messages",
    ]
)
def is_chat_messages(x):
    """Checks if the given data model are chat messages

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ChatMessages.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Embedding",
    ]
)
class Embedding(DataModel):
    """An embedding vector"""

    embedding: List[float] = Field(
        description="The embedding vector",
        default=[],
    )


@synalinks_export(
    [
        "synalinks.backend.is_embedding",
        "synalinks.is_embedding",
    ]
)
def is_embedding(x):
    """Checks if the given data model is an embedding

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Embedding.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Embeddings",
        "synalinks.Embeddings",
    ]
)
class Embeddings(DataModel):
    """A list of embeddings"""

    embeddings: List[List[float]] = Field(
        description="The list of embedding vectors",
        default=[],
    )


@synalinks_export(
    [
        "synalinks.backend.is_embeddings",
        "synalinks.is_embeddings",
    ]
)
def is_embeddings(x):
    """Checks if the given data model are embeddings

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Embeddings.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Entity",
        "synalinks.Entity",
    ]
)
class Entity(DataModel):
    """An entity data model"""

    label: str = Field(
        description="The entity label",
    )


@synalinks_export(
    [
        "synalinks.backend.is_entity",
        "synalinks.is_entity",
    ]
)
def is_entity(x):
    """Checks if the given data model is an entity

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("label", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.EmbeddedEntity",
        "synalinks.EmbeddedEntity",
    ]
)
class EmbeddedEntity(Embedding, Entity):
    pass


@synalinks_export(
    [
        "synalinks.backend.is_embedded_entity",
        "synalinks.is_embedded_entity",
    ]
)
def is_embedded_entity(x):
    """Checks if the given data model is an embedded entity

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("label", None) and properties.get("embedding", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Relation",
        "synalinks.Relation",
    ]
)
class Relation(DataModel):
    """A relation model"""

    obj: Entity = Field(
        description="The object entity",
    )
    label: str = Field(
        description="The relation label",
    )
    subj: Entity = Field(
        description="The subject entity",
    )


@synalinks_export(
    [
        "synalinks.backend.is_relation",
        "synalinks.is_relation",
    ]
)
def is_relation(x):
    """Checks if is a relation model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if (
            properties.get("subj", None)
            and properties.get("label", None)
            and properties.get("obj", None)
        ):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Entities",
        "synalinks.Entities",
    ]
)
class Entities(DataModel):
    entities: List[Entity] = Field(
        description="The entities",
    )


@synalinks_export(
    [
        "synalinks.backend.is_entities",
        "synalinks.is_entities",
    ]
)
def is_entities(x):
    """Checks if is an entities model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("entities", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Relations",
        "synalinks.Relations",
    ]
)
class Relations(DataModel):
    relations: List[Relation] = Field(
        description="The relations",
    )


@synalinks_export(
    [
        "synalinks.backend.is_relations",
        "synalinks.is_relations",
    ]
)
def is_relations(x):
    """Checks if is an relations model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("relations", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.KnowledgeGraph",
        "synalinks.KnowledgeGraph",
    ]
)
class KnowledgeGraph(DataModel):
    entities: List[Entity] = Field(
        description="The entities",
    )
    relations: List[Relation] = Field(
        description="The relations",
    )


@synalinks_export(
    [
        "synalinks.backend.is_knowledge_graph",
        "synalinks.is_knowledge_graph",
    ]
)
def is_knowledge_graph(x):
    """Checks if is a knowledge graph model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("entities", None) and properties.get("relations", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Prediction",
        "synalinks.Prediction",
    ]
)
class Prediction(Entity, GenericIO):
    """The generator's prediction"""

    label: Literal["Prediction"] = "Prediction"
    reward: Optional[float] = Field(
        description="The prediction's reward",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.is_prediction",
        "synalinks.is_prediction",
    ]
)
def is_prediction(x):
    """Checks if the given data model is a prediction

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Prediction.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Instructions",
        "synalinks.Instructions",
    ]
)
class Instructions(Entity):
    """The generator's instructions"""

    label: Literal["Instructions"] = "Instructions"
    instructions: List[str] = Field(
        description="The list of instructions",
    )
    reward: Optional[float] = Field(
        description="The instruction's reward",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.is_instructions",
        "synalinks.is_instructions",
    ]
)
def is_instructions(x):
    """Checks if the given data model is an instructions data model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Instructions.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.SimilaritySearch",
        "synalinks.SimilaritySearch",
    ]
)
class SimilaritySearch(DataModel):
    entity_label: str = Field(
        description=("The label of the entity to look for(use `*` to match them all)"),
    )
    similarity_search: str = Field(
        description=("The natural language similarity query to match specific entities"),
    )


@synalinks_export(
    [
        "synalinks.backend.is_similarity_search",
        "synalinks.is_similarity_search",
    ]
)
def is_similarity_search(x):
    """Checks if is a similarity search data model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("entity_label", None) and properties.get(
            "similarity_search", None
        ):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.TripletSearch",
        "synalinks.TripletSearch",
    ]
)
class TripletSearch(DataModel):
    subject_label: str = Field(
        description=("The label of the subject entity (use `*` to match them all)"),
    )
    subject_similarity_search: str = Field(
        description=(
            "A short similarity query to match specific subjects"
            "(use `*` to match them all)",
        ),
    )
    relation_label: str = Field(
        description="The label of the relation to search for",
    )
    object_label: str = Field(
        description=("The label of the object entity (use `*` to match them all)"),
    )
    object_similarity_search: str = Field(
        description=(
            "A short similarity query to match specific objects"
            " (use `*` to match them all)"
        ),
    )


@synalinks_export(
    [
        "synalinks.backend.is_triplet_search",
        "synalinks.is_triplet_search",
    ]
)
def is_triplet_search(x):
    """Checks if is a triplet seach data model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if (
            properties.get("subject_label", None)
            and properties.get("subject_similarity_search", None)
            and properties.get("relation_label", None)
            and properties.get("object_label", None)
            and properties.get("object_similarity_search", None)
        ):
            return True
    return False
