import synalinks

synalinks.enable_logging()


class Query(synalinks.DataModel):
    query: str


# Concatenation with Synalinks (Part 2)

# What happen if you concatenate two data
# models with the same fields?
# When property names conflict, numerical suffixes are
# added to ensure uniqueness.

two_queries = Query + Query

print(two_queries.prettify_schema())

# {
#   "additionalProperties": false,
#   "properties": {
#     "query": {
#       "title": "Query",
#       "type": "string"
#     },
#     "query_1": {
#       "title": "Query 1",
#       "type": "string"
#     }
#   },
#   "required": [
#     "query",
#     "query_1"
#   ],
#   "title": "Query",
#   "type": "object"
# }

two_queries = Query(
    query="Why is neuro-symbolic systems powering the next AI wave?"
) + Query(query="Can you give a multiple of 5?")


print(two_queries.prettify_json())
# {
#   "query": "Why is neuro-symbolic systems powering the next AI wave?",
#   "query_1": "Can you give a multiple of 5?"
# }

# Now, what happen when you concatenate with `None`?
# An exception is raised!

failing_query = Query(query="Why is neuro-symbolic AI powering the next wave?") + None
# ValueError: Received x1=query='Why is neuro-symbolic AI powering the next wave?' and x2=None

# This behavior can be summarized with the following truth table:

# Truth Table:

# | `x1`   | `x2`   | Concat (`+`)      |
# | ------ | ------ | ----------------- |
# | `x1`   | `x2`   | `x1 + x2`         |
# | `x1`   | `None` | `Exception`       |
# | `None` | `x2`   | `Exception`       |
# | `None` | `None` | `Exception`       |
