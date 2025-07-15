import synalinks

# Logical AND with Synalinks

class Query(synalinks.DataModel):
    query: str
    
    
class Answer(synalinks.DataModel):
    answer: str
    
    
qa_pair = Query & Answer

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

# When performing an And operation with `None` the output is `None`
# You can see the logical And as a robust concatenation operation.

qa_pair = Query(query="Why is neuro-symbolic AI powering the next wave?") & None

assert isinstance(qa_pair, None)

# Here is the table summarizing the behavior

# Truth Table:

# | `x1`   | `x2`   | Logical And (`&`) |
# | ------ | ------ | ----------------- |
# | `x1`   | `x2`   | `x1 + x2`         |
# | `x1`   | `None` | `None`            |
# | `None` | `x2`   | `None`            |
# | `None` | `None` | `None`            |