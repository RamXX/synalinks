from synalinks.src.knowledge_bases.database_adapters.database_adapter import (
    DatabaseAdapter,
)
from synalinks.src.knowledge_bases.database_adapters.memgraph_adapter import (
    MemGraphAdapter,
)
from synalinks.src.knowledge_bases.database_adapters.neo4j_adapter import Neo4JAdapter

# from synalinks.src.knowledge_bases.database_adapters.kuzu_adapter import KuzuAdapter


def get(uri):
    if uri.startswith("neo4j"):
        return Neo4JAdapter
    elif uri.startswith("memgraph"):
        return MemGraphAdapter
    # elif uri.startswith("kuzu"):
    #     return KuzuAdapter
    else:
        raise ValueError(f"No database adapter found for {uri}")
