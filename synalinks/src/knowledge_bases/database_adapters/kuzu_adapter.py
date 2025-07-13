# import asyncio
# import warnings
# from itertools import product
# from typing import Any
# from typing import Dict

# import kuzu

# from synalinks.src.backend import is_entity
# from synalinks.src.backend import is_relation
# from synalinks.src.knowledge_bases.database_adapters import DatabaseAdapter
# from synalinks.src.utils.naming import to_snake_case


# class KuzuAdapter(DatabaseAdapter):
#     def __init__(
#         self,
#         uri=None,
#         entity_models=None,
#         relation_models=None,
#         embedding_model=None,
#         metric="cosine",
#         wipe_on_start=False,
#     ):
#         self.db_name = uri.replace("kuzu://", "")
#         self.db = kuzu.Database(self.db_name)

#         super().__init__(
#             uri=uri,
#             entity_models=entity_models,
#             relation_models=relation_models,
#             embedding_model=embedding_model,
#             metric=metric,
#             wipe_on_start=wipe_on_start,
#         )
#         try:
#             asyncio.get_event_loop().run_until_complete(
#                 self.query("INSTALL vector; LOAD vector;")
#             )
#         except Exception:
#             pass

#     def wipe_database(self):
#         asyncio.get_event_loop().run_until_complete(
#             self.query("MATCH (n) DETACH DELETE n;")
#         )

#     def create_vector_index(self):
#         """Create vector indexes"""
#         for entity_model in self.entity_models:
#             node_label = self.sanitize_label(entity_model.get_schema().get("title"))
#             index_name = to_snake_case(node_label)
#             ddl = self._json_schema_to_kuzu_node_table(entity_model.get_schema())
#             asyncio.get_event_loop().run_until_complete(self.query(ddl))
#             try:
#                 query = "\n".join(
#                     [
#                         "CALL CREATE_VECTOR_INDEX(",
#                         f"'{node_label}', '{index_name}', 'embedding');",
#                     ]
#                 )
#                 asyncio.get_event_loop().run_until_complete(self.query(query))
#             except RuntimeError:
#                 pass

#         for relation_model in self.relation_models:
#             ddl = self._json_schema_to_kuzu_rel_table(relation_model.get_schema())
#             asyncio.get_event_loop().run_until_complete(self.query(ddl))

#     async def query(self, query, params=None):
#         result_list = []
#         with kuzu.AsyncConnection(self.db) as conn:
#             result = await conn.execute(query, parameters=params)
#             result_list.append(result.get_as_df().to_json(orient="records"))
#         return result_list

#     def _json_schema_to_kuzu_node_table(self, schema: Dict[str, Any]) -> str:
#         """Convert JSON schema to Kuzu CREATE NODE TABLE statement"""
#         table_name = self.sanitize_label(schema.get("title", "Entity"))
#         properties = schema.get("properties", {})

#         columns = []
#         primary_key = None

#         # Add embedding column
#         columns.append(f"embedding FLOAT[{self.embedding_dim}]")

#         for prop_name, prop_def in properties.items():
#             prop_name = self.sanitize_property_name(prop_name)
#             kuzu_type = self._json_type_to_kuzu_type(prop_def)
#             if prop_name not in ["embedding"]:
#                 # Check if this should be the primary key
#                 if prop_name == "name":
#                     if not primary_key:
#                         primary_key = prop_name
#                         columns.append(f"{prop_name} {kuzu_type} PRIMARY KEY")
#                     else:
#                         columns.append(f"{prop_name} {kuzu_type}")
#                 else:
#                     columns.append(f"{prop_name} {kuzu_type}")

#         if not primary_key:
#             columns.insert(0, "id SERIAL PRIMARY KEY")

#         columns_str = ", ".join(columns)
#         ddl = f"CREATE NODE TABLE IF NOT EXISTS {table_name} ({columns_str})"
#         return ddl

#     def _json_schema_to_kuzu_rel_table(self, schema: Dict[str, Any]) -> str:
#         """Convert JSON schema to Kuzu CREATE REL TABLE statement"""
#         table_name = self.sanitize_label(schema.get("title", "Relation"))
#         properties = schema.get("properties", {})

#         columns = []
#         subj_labels = []
#         obj_labels = []

#         # Extract FROM and TO table information from schema if available
#         if "subj" in properties:
#             if "anyOf" in properties["subj"]:
#                 for union_type in properties["subj"]["anyOf"]:
#                     subj_labels.append(union_type["$ref"].replace("#/$defs/", ""))
#             else:
#                 subj_labels.append(properties["subj"]["$ref"].replace("#/$defs/", ""))

#         if "obj" in properties:
#             if "anyOf" in properties["obj"]:
#                 for union_type in properties["obj"]["anyOf"]:
#                     obj_labels.append(union_type["$ref"].replace("#/$defs/", ""))
#             else:
#                 obj_labels.append(properties["subj"]["$ref"].replace("#/$defs/", ""))

#         # Add other properties (excluding subj, obj)
#         for prop_name, prop_def in properties.items():
#             if prop_name not in ["subj", "obj"]:
#                 prop_name = self.sanitize_property_name(prop_name)
#                 kuzu_type = self._json_type_to_kuzu_type(prop_def)

#                 default_val = prop_def.get("default")
#                 if default_val is not None:
#                     if isinstance(default_val, str):
#                         columns.append(f"{prop_name} {kuzu_type} DEFAULT '{default_val}'")
#                     else:
#                         columns.append(f"{prop_name} {kuzu_type} DEFAULT {default_val}")
#                 else:
#                     columns.append(f"{prop_name} {kuzu_type}")

#         # Build the DDL
#         if columns:
#             columns_str = ", " + ", ".join(columns)
#         else:
#             columns_str = ""

#         from_to_clauses = []
#         for from_table, to_table in product(subj_labels, obj_labels):
#             from_to_clauses.append(f"FROM {from_table} TO {to_table}")
#         from_to_clauses = ", ".join(from_to_clauses)

#         ddl = (
#             f"CREATE REL TABLE IF NOT EXISTS {table_name}({from_to_clauses}{columns_str})"
#         )

#         return ddl

#     def _json_type_to_kuzu_type(self, prop_def: Dict[str, Any]) -> str:
#         """Convert JSON schema type to Kuzu data type"""
#         json_type = prop_def.get("type", "string")

#         type_mapping = {
#             "string": "STRING",
#             "integer": "INT64",
#             "number": "DOUBLE",
#             "boolean": "BOOLEAN",
#             "array": "STRING",  # Store as JSON string for simplicity
#             "object": "STRING",  # Store as JSON string
#         }
#         # Handle format specifications
#         if json_type == "string":
#             format_type = prop_def.get("format")
#             if format_type == "date":
#                 return "DATE"
#             elif format_type == "date-time":
#                 return "TIMESTAMP"
#             elif format_type == "time":
#                 return "TIME"

#         return type_mapping.get(json_type, "STRING")

#     async def update(
#         self,
#         data_model,
#         threshold=0.9,
#     ):
#         if is_relation(data_model):
#             subj = data_model.get_nested_entity("subj")
#             obj = data_model.get_nested_entity("obj")
#             relation_label = data_model.get("label")
#             subj_label = self.sanitize_label(subj.get("label"))
#             subj_vector = subj.get("embedding")
#             obj_label = self.sanitize_label(obj.get("label"))
#             obj_vector = obj.get("embedding")

#             if not subj_vector or not obj_vector:
#                 warnings.warn(
#                     "No embedding found for `subj` or `obj`:"
#                     " Entities and relations needs to be embedded. "
#                     "Use `Embedding` module before `UpdateKnowledge`. "
#                     "Skipping update."
#                 )
#                 return

#             relation_properties = self.sanitize_properties(data_model.get_json())
#             set_clauses = []
#             for key in relation_properties.keys():
#                 if key not in ("subj", "obj", "label"):
#                     set_clauses.append(f"r.{key} = ${key}")
#             set_statement = "SET " + ", ".join(set_clauses) if set_clauses else ""

#             query = "\n".join(
#                 [
#                     f"CALL QUERY_VECTOR_INDEX('{subj_label}', '{to_snake_case(subj_label)}', $subjVector, 1)",
#                     "YIELD node AS s, distance AS subj_score",
#                     "WITH s, subj_score",
#                     "WHERE subj_score >= $threshold",
#                     "",
#                     f"CALL QUERY_VECTOR_INDEX('{obj_label}', '{to_snake_case(obj_label)}', $objVector, 1)",
#                     "YIELD node AS o, distance AS obj_score",
#                     "WITH o, obj_score",
#                     "WHERE obj_score >= $threshold",
#                     "",
#                     f"MERGE (s)-[r:{relation_label}]->(o)",
#                     (
#                         set_statement
#                         if set_statement
#                         else "// No additional properties to set"
#                     ),
#                 ]
#             )
#             relation_properties.pop("subj")
#             relation_properties.pop("obj")
#             relation_properties.pop("label")
#             params = {
#                 "threshold": threshold,
#                 "subjVector": subj_vector,
#                 "objVector": obj_vector,
#                 **relation_properties,
#             }
#             await self.query(query, params=params)
#         elif is_entity(data_model):
#             properties = self.sanitize_properties(data_model.get_json())
#             node_label = self.sanitize_label(data_model.get("label"))
#             vector = data_model.get("embedding")

#             if not vector:
#                 warnings.warn(
#                     "Entities need to be embedded. "
#                     "Make sure to use `Embedding` module before `UpdateKnowledge`. "
#                     "Skipping update."
#                 )
#                 return

#             query = "\n".join(
#                 [
#                     f"CALL QUERY_VECTOR_INDEX('{node_label}', '{to_snake_case(node_label)}', $vector, 1)",
#                     "YIELD node, distance AS score",
#                     "WITH node, score",
#                     "WHERE score >= $threshold",
#                     "WITH count(node) as existing_count",
#                     "WHERE existing_count = 0",
#                     f"CREATE (n:{node_label})",
#                     f"SET {', '.join([f'n.{key} = ${key}' for key in properties.keys()])}",  # noqa E501
#                 ]
#             )
#             params = {
#                 "threshold": threshold,
#                 "vector": vector,
#                 **properties,
#             }
#             await self.query(query, params=params)
#         else:
#             raise ValueError(
#                 "The parameter `data_model` must be an `Entity` or `Relation` instance"
#             )

#     async def similarity_search(
#         self,
#         similarity_search,
#         k=10,
#         threshold=0.5,
#     ):
#         if not is_similarity_search(similarity_search):
#             raise ValueError(
#                 "The `similarity_search` argument "
#                 "should be a `SimilaritySearch` data model"
#             )
#         text = similarity_search.get("similarity_search")
#         entity_label = similarity_search.get("entity_label")
#         vector = (await self.embedding_model(texts=[text]))["embeddings"][0]

#         index_name = to_snake_case(self.sanitize_label(entity_label))

#         query = "\n".join(
#             [
#                 "CALL QUERY_VECTOR_INDEX(",
#                 " $indexName,",
#                 " $numberOfNearestNeighbours,",
#                 " $vector) YIELD node AS node, distance AS score",
#                 "WITH node, score",
#                 "WHERE score >= $threshold",
#                 "RETURN {name: node.name, label: node.label} AS node, score",
#                 "LIMIT $numberOfNearestNeighbours",
#             ]
#         )
#         params = {
#             "indexName": index_name,
#             "numberOfNearestNeighbours": k,
#             "threshold": threshold,
#             "vector": vector,
#         }
#         result = await self.query(query, params=params)
#         return result
