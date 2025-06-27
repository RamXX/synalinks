# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import os
import warnings
from typing import Any
from typing import Dict

import neo4j

from synalinks.src.backend import is_entity
from synalinks.src.backend import is_relation
from synalinks.src.backend import is_similarity_search
from synalinks.src.backend import is_triplet_search
from synalinks.src.backend.common.json_utils import out_mask_json
from synalinks.src.knowledge_bases.database_adapters import DatabaseAdapter
from synalinks.src.utils.naming import to_snake_case


class Neo4JAdapter(DatabaseAdapter):
    def __init__(
        self,
        index_name=None,
        entity_models=None,
        relation_models=None,
        embedding_model=None,
        metric="cosine",
        wipe_on_start=False,
    ):
        super().__init__(
            index_name=index_name,
            embedding_model=embedding_model,
        )
        self.db = os.getenv("NEO4J_DATABASE", "neo4j")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")

        self.embedding_model = embedding_model
        self.embedding_dim = len(
            asyncio.get_event_loop().run_until_complete(embedding_model(texts=["test"]))[
                "embeddings"
            ][0]
        )

        self.metric = metric

        if not entity_models:
            entity_models = []
        if not relation_models:
            relation_models = []
        self.entity_models = entity_models
        self.relation_models = relation_models

        if wipe_on_start:
            asyncio.get_event_loop().run_until_complete(
                self.query(
                    """
                    MATCH (n)
                    CALL (n) {
                        DETACH DELETE n
                    } IN TRANSACTIONS OF 10000 ROWS
                    """
                )
            )

        query = "\n".join(
            [
                "CREATE VECTOR INDEX `entity` IF NOT EXISTS",
                "FOR (n:Entity) ON n.embedding",
                "OPTIONS {indexConfig : {"
                " `vector.dimensions`: $dimension,"
                " `vector.similarity_function`: $similarityFunction"
                "}};",
            ]
        )
        params = {
            "dimension": self.embedding_dim,
            "similarityFunction": self.metric,
        }
        asyncio.get_event_loop().run_until_complete(
            self.query(query, params=params),
        )
        for entity_model in self.entity_models:
            node_label = self.sanitize_label(entity_model.get_schema().get("title"))
            index_name = to_snake_case(node_label)
            query = "\n".join(
                [
                    "CREATE VECTOR INDEX $indexName IF NOT EXISTS",
                    f"FOR (n:{node_label}) ON n.embedding",
                    "OPTIONS {indexConfig : {"
                    " `vector.dimensions`: $dimension,"
                    " `vector.similarity_function`: $similarityFunction"
                    "}};",
                ]
            )
            params = {
                "indexName": index_name,
                "dimension": self.embedding_dim,
                "similarityFunction": self.metric,
            }
            asyncio.get_event_loop().run_until_complete(
                self.query(query, params=params),
            )
        if self.entity_models:
            asyncio.get_event_loop().run_until_complete(
                self.query("CALL db.awaitIndexes(300)"),
            )

    async def query(self, query: str, params: Dict[str, Any] = None, **kwargs):
        driver = neo4j.GraphDatabase.driver(
            self.index_name, auth=(self.username, self.password)
        )
        result_list = []
        try:
            with driver.session(database=self.db) as session:
                if params:
                    result = session.run(query, **params, **kwargs)
                else:
                    result = session.run(query, **kwargs)
                for record in reversed(list(result)):
                    data = record.data()
                    data = out_mask_json(data, mask=["embedding", "embeddings"])
                    result_list.append(data)
                session.close()
        finally:
            driver.close()
        return result_list

    async def update(
        self,
        data_model,
        threshold=0.9,
    ):
        if is_relation(data_model):
            subj = data_model.get_nested_entity("subj")
            obj = data_model.get_nested_entity("obj")
            relation_label = data_model.get("label")
            subj_label = self.sanitize_label(subj.get("label"))
            subj_vector = subj.get("embedding")
            obj_label = self.sanitize_label(obj.get("label"))
            obj_vector = obj.get("embedding")

            if not subj_vector or not obj_vector:
                warnings.warn(
                    "No embedding found for `subj` or `obj`:"
                    " Entities and relations needs to be embedded. "
                    "Use `Embedding` module before `UpdateKnowledge`. "
                    "Skipping update."
                )
                return

            relation_properties = self.sanitize_properties(data_model.get_json())
            set_clauses = []
            for key in relation_properties.keys():
                if key not in ("subj", "obj", "label"):
                    set_clauses.append(f"r.{key} = ${key}")
            set_statement = "SET " + ", ".join(set_clauses) if set_clauses else ""

            query = "\n".join(
                [
                    "CALL db.index.vector.queryNodes($subjIndexName, 1, $subjVector)",
                    "YIELD node AS s, score AS subj_score",
                    "WHERE subj_score >= $threshold",
                    "",
                    "CALL db.index.vector.queryNodes($objIndexName, 1, $objVector)",
                    "YIELD node AS o, score AS obj_score",
                    "WHERE obj_score >= $threshold",
                    "",
                    f"MERGE (s)-[r:{relation_label}]->(o)",
                    set_statement
                    if set_statement
                    else "// No additional properties to set",
                ]
            )
            params = {
                "subjIndexName": to_snake_case(subj_label),
                "objIndexName": to_snake_case(obj_label),
                "threshold": threshold,
                "subjVector": subj_vector,
                "objVector": obj_vector,
            }
            await self.query(query, params=params)
        elif is_entity(data_model):
            properties = self.sanitize_properties(data_model.get_json())
            node_label = self.sanitize_label(data_model.get("label"))
            vector = data_model.get("embedding")

            if not vector:
                warnings.warn(
                    "Entities need to be embedded. "
                    "Make sure to use `Embedding` module before `UpdateKnowledge`. "
                    "Skipping update."
                )
                return

            query = "\n".join(
                [
                    "CALL db.index.vector.queryNodes($indexName, 1, $vector)",
                    "YIELD node, score",
                    "WITH node, score",
                    "WHERE score >= $threshold",
                    "WITH count(node) as existing_count",
                    "WHERE existing_count = 0",
                    f"CREATE (n:Entity:{node_label})",
                    f"SET {', '.join([f'n.{key} = ${key}' for key in properties.keys()])}",  # noqa E501
                ]
            )
            params = {
                "indexName": to_snake_case(node_label),
                "threshold": threshold,
                "vector": vector,
                **properties,
            }
            await self.query(query, params=params)
        else:
            raise ValueError(
                "The parameter `data_model` must be an `Entity` or `Relation` instance"
            )

    async def similarity_search(
        self,
        similarity_search,
        k=10,
        threshold=0.7,
    ):
        if not is_similarity_search(similarity_search):
            raise ValueError(
                "The `similarity_search` argument "
                "should be a `SimilaritySearch` data model"
            )
        text = similarity_search.get("similarity_search")
        entity_label = similarity_search.get("entity_label")
        vector = (await self.embedding_model(texts=[text]))["embeddings"][0]

        index_name = (
            to_snake_case(self.sanitize_label(entity_label))
            if entity_label != "*"
            else "entity"
        )

        query = "\n".join(
            [
                "CALL db.index.vector.queryNodes(",
                " $indexName,",
                " $numberOfNearestNeighbours,",
                " $vector) YIELD node AS node, score",
                "WHERE score >= $threshold",
                "RETURN {name: node.name, label: node.label} AS node, score",
                "LIMIT $numberOfNearestNeighbours",
            ]
        )
        params = {
            "indexName": index_name,
            "numberOfNearestNeighbours": k,
            "threshold": threshold,
            "vector": vector,
        }
        result = await self.query(query, params=params)
        return result

    async def triplet_search(
        self,
        triplet_search,
        k=10,
        threshold=0.7,
    ):
        if not is_triplet_search(triplet_search):
            raise ValueError(
                "The `triplet_search` argument should be a `TripletSearch` data model"
            )

        subject_label = triplet_search.get("subject_label")
        subject_label = (
            self.sanitize_label(subject_label) if subject_label != "*" else subject_label
        )
        subject_similarity_query = triplet_search.get("subject_similarity_query")
        relation_label = triplet_search.get("relation_label")
        relation_label = (
            self.sanitize_label(relation_label)
            if relation_label != "*"
            else relation_label
        )
        object_label = triplet_search.get("object_label")
        object_label = (
            self.sanitize_label(object_label) if object_label != "*" else object_label
        )
        object_similarity_query = triplet_search.get("object_similarity_query")

        params = {
            "numberOfNearestNeighbours": k,
            "threshold": threshold,
            "k": k,
        }

        query_lines = []
        where_conditions = []

        # Determine which searches we need
        has_subject_similarity = (
            subject_similarity_query and subject_similarity_query != "*"
        )
        has_object_similarity = (
            object_similarity_query and object_similarity_query != "*"
        )

        if has_subject_similarity and has_object_similarity:
            # Both subject and object have similarity search
            subject_vector = (
                await self.embedding_model(texts=[subject_similarity_query])
            )["embeddings"][0]
            object_vector = (
                await self.embedding_model(texts=[object_similarity_query])
            )["embeddings"][0]
            params["subjVector"] = subject_vector
            params["objVector"] = object_vector

            if subject_label != "*":
                params["subjIndexName"] = to_snake_case(subject_label)
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "$subjIndexName, $numberOfNearestNeighbours, $subjVector)"
                    )
                )
            else:
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "'entity', $numberOfNearestNeighbours, $subjVector)"
                    )
                )

            query_lines.extend(
                [
                    "YIELD node AS subj, score AS subj_score",
                    "WHERE subj_score >= $threshold",
                    "WITH collect({subj: subj, subj_score: subj_score}) AS subjects",
                ]
            )

            if object_label != "*":
                params["objIndexName"] = to_snake_case(object_label)
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "$objIndexName, $numberOfNearestNeighbours, $objVector)"
                    )
                )
            else:
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "'entity', $numberOfNearestNeighbours, $objVector)"
                    )
                )

            query_lines.extend(
                [
                    "YIELD node AS obj, score AS obj_score",
                    "WHERE obj_score >= $threshold",
                    "UNWIND subjects AS s",
                    "WITH s.subj AS subj, s.subj_score AS subj_score, obj, obj_score",
                ]
            )

            if relation_label != "*":
                query_lines.append(f"MATCH (subj)-[relation:{relation_label}]->(obj)")
            else:
                query_lines.append("MATCH (subj)-[relation]->(obj)")
            query_lines.append("WITH subj, subj_score, relation, obj, obj_score")

        elif has_subject_similarity:
            subject_vector = (
                await self.embedding_model(texts=[subject_similarity_query])
            )["embeddings"][0]
            params["subjVector"] = subject_vector

            if subject_label != "*":
                params["subjIndexName"] = to_snake_case(subject_label)
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "$subjIndexName, $numberOfNearestNeighbours, $subjVector)"
                    )
                )
            else:
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "'entity', $numberOfNearestNeighbours, $subjVector)"
                    )
                )

            query_lines.extend(
                [
                    "YIELD node AS subj, score AS subj_score",
                    "WHERE subj_score >= $threshold",
                ]
            )

            if relation_label != "*" and object_label != "*":
                query_lines.append(
                    f"MATCH (subj)-[relation:{relation_label}]->(obj:{object_label})"
                )
            elif relation_label != "*":
                query_lines.append(
                    f"MATCH (subj)-[relation:{relation_label}]->(obj:Entity)"
                )
            elif object_label != "*":
                query_lines.append(f"MATCH (subj)-[relation]->(obj:{object_label})")
            else:
                query_lines.append("MATCH (subj)-[relation]->(obj:Entity)")
            query_lines.append(
                "WITH subj, subj_score, relation, obj, 1.0 AS obj_score"
            )

        elif has_object_similarity:
            object_vector = (
                await self.embedding_model(texts=[object_similarity_query])
            )["embeddings"][0]
            params["objVector"] = object_vector

            if object_label != "*":
                params["objIndexName"] = to_snake_case(object_label)
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "$objIndexName, $numberOfNearestNeighbours, $objVector)"
                    )
                )
            else:
                query_lines.append(
                    (
                        "CALL db.index.vector.queryNodes("
                        "'entity', $numberOfNearestNeighbours, $objVector)"
                    )
                )

            query_lines.extend(
                ["YIELD node AS obj, score AS obj_score", "WHERE obj_score >= $threshold"]
            )

            if relation_label != "*" and subject_label != "*":
                query_lines.append(
                    f"MATCH (subj:{subject_label})-[relation:{relation_label}]->(obj)"
                )
            elif relation_label != "*":
                query_lines.append(
                    f"MATCH (subj:Entity)-[relation:{relation_label}]->(obj)"
                )
            elif subject_label != "*":
                query_lines.append(f"MATCH (subj:{subject_label})-[relation]->(obj)")
            else:
                query_lines.append("MATCH (subj:Entity)-[relation]->(obj)")
            query_lines.append(
                "WITH subj, 1.0 AS subj_score, relation, obj, obj_score"
            )

        else:
            # Build the full pattern match
            subj_pattern = (
                f"subj:{subject_label}" if subject_label != "*" else "subj:Entity"
            )
            rel_pattern = (
                f"relation:{relation_label}" if relation_label != "*" else "relation"
            )
            obj_pattern = (
                f"obj:{object_label}" if object_label != "*" else "obj:Entity"
            )

            query_lines.extend(
                [
                    f"MATCH ({subj_pattern})-[{rel_pattern}]->({obj_pattern})",
                    "WITH subj, 1.0 AS subj_score, relation, obj, 1.0 AS obj_score",
                ]
            )

        # Add geometric mean score calculation for triplets
        query_lines.append(
            (
                "WITH subj, subj_score, relation, obj, obj_score, "
                "sqrt(subj_score * obj_score) "
                "AS score"
            )
        )
        where_conditions.append("score >= $threshold")

        if where_conditions:
            query_lines.append(f"WHERE {' AND '.join(where_conditions)}")

        # Clean the node data to exclude embeddings before returning
        query_lines.extend(
            [
                "RETURN {name: subj.name, label: subj.label} AS subj,",
                "       type(relation) AS relation,",
                "       {name: obj.name, label: obj.label} AS obj,",
                "       score",
                "ORDER BY score DESC",
            ]
        )

        query_lines.append("LIMIT $k")
        query = "\n".join(query_lines)

        result = await self.query(query, params=params)
        return result