import synalinks

synalinks.enable_logging()

# Concatenation with Synalinks


class Query(synalinks.DataModel):
    query: str


class Answer(synalinks.DataModel):
    answer: str


# Synalinks operators works at a metaclass level
# In that case, the result is a `SymbolicDataModel`
# A `SymbolicDataModel` can be understand as a data
# specification/contract. It only contains a JSON schema
# and cannot be used for computation. It allow Synalinks
# to build directed acyclic graph (DAG) of computation
# from inputs and outputs, like the tensor shape
# in deep learning frameworks.

qa_pair = Query + Answer

assert isinstance(qa_pair, synalinks.SymbolicDataModel)

print(qa_pair.prettify_schema())
# {
#   "additionalProperties": false,
#   "properties": {
#     "query": {
#       "title": "Query",
#       "type": "string"
#     },
#     "answer": {
#       "title": "Answer",
#       "type": "string"
#     }
#   },
#   "required": [
#     "query",
#     "answer"
#   ],
#   "title": "Query",
#   "type": "object"
# }

# Once we concatenate two instanciated data models, the result
# is a JsonDataModel, a data model containing both a JSON schema and
# a JSON object containing the actual data.

qa_pair = Query(query="What is the French city of aeronautics and robotics?") + Answer(
    answer="Toulouse"
)

assert isinstance(qa_pair, synalinks.JsonDataModel)

print(qa_pair.prettify_json())
# {
#   "query": "What is the French city of aeronautics and robotics?",
#   "answer": "Toulouse"
# }
