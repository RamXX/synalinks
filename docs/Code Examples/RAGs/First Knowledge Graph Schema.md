# First Knowledge Graph Schema

A knowledge graph schema is like a blueprint that defines the structure and rules for organizing information in a graph format. Just as a database schema defines tables and their relationships, a knowledge graph schema defines entities (nodes) and relations (edges) that can exist in your graph.
This tutorial will teach you how to analyze information domains and translate them into structured schema definitions using the synalinks framework.

### Entities

Entities represent the "things" in your domain - people, places, objects, concepts, or events.

Each entity type has:

- A `label`: A unique identifier that distinguishes this entity type from all others?
- Properties: Attributes that capture the entity's characteristics.
- Descriptions: Clear, specific descriptions that guide the LMs for accurate data extraction and understanding.

When designing entities, consider both current needs and future extensibility. Properties should be atomic (single-valued) when possible, but flexible enough to accommodate variations in your data.

Example:

```python
import synalinks
from typing import Literal, List, Union

class City(synalinks.Entity):
    label: Literal["City"]
    name: str = synalinks.Field(
        description="The name of a city, such as 'Paris' or 'New York'.",
    )

class Country(synalinks.Entity):
    label: Literal["Country"]
    name: str = synalinks.Field(
        description="The name of a country, such as 'France' or 'Canada'.",
    )

class Place(synalinks.Entity):
    label: Literal["Place"]
    name: str = synalinks.Field(
        description="The name of a specific place, which could be a landmark, building, or any point of interest, such as 'Eiffel Tower' or 'Statue of Liberty'.",
    )

class Event(synalinks.Entity):
    label: Literal["Event"]
    name: str = synalinks.Field(
        description="The name of an event, such as 'Summer Olympics 2024' or 'Woodstock 1969'.",
    )
```

### Relations

Relations are the connective tissue of your knowledge graph, representing how entities interact, depend on, or relate to each other. They transform isolated data points into a rich, interconnected web of knowledge.

Each relations has:

- A subject (`subj`): The source entity of the relation.
- A label (`label`): The type of relationship.
- A target (`obj`): The target entity of the relation.
- Properties: Attributes that describe/enrich the relation.
- Descriptions: Clear explanations of what each property represents to help extraction.

Example:

```python
class IsCapitalOf(synalinks.Relation):
    subj: City = synalinks.Field(
        description="The city entity that serves as the capital.",
    )
    label: Literal["IsCapitalOf"]
    obj: Country = synalinks.Field(
        description="The country entity for which the city is the capital.",
    )


class IsCityOf(synalinks.Relation):
    subj: City = synalinks.Field(
        description="The city entity that is a constituent part of a country.",
    )
    label: Literal["IsCityOf"]
    obj: Country = synalinks.Field(
        description="The country entity that the city is part of.",
    )
    

class IsLocatedIn(synalinks.Relation):
    subj: Union[Place] = synalinks.Field(
        description="The place entity that is situated within a larger geographical area.",
    )
    label: Literal["IsLocatedIn"]
    obj: Union[City, Country] = synalinks.Field(
        description="The city or country entity where the place is geographically located.",
    )


class TookPlaceIn(synalinks.Relation):
    subj: Event = synalinks.Field(
        description="The event entity that occurred in a specific location.",
    )
    label: Literal["TookPlaceIn"]
    obj: Union[City, Country] = synalinks.Field(
        description="The city or country entity where the event occurred.",
    )
```

### Schema Design Strategy and Best Practices

Start with Domain Analysis
Before writing any code, invest time in understanding your domain thoroughly:

1. **Identify Core Concepts**: List the most important "things" in your domain
2. **Map Natural Relationships**: Observe how these concepts connect in real-world scenarios
3. **Consider Use Cases**: Think about the questions your knowledge graph should answer
4. **Plan for Growth**: Design schemas that can evolve with your understanding

### Balance Granularity and Usability

Finding the right level of detail is crucial:

- **Too Generic**: Loses important nuances and becomes less useful
- **Too Specific**: Creates maintenance overhead and reduces flexibility
- **Just Right**: Captures essential distinctions while remaining manageable

### Implement Iterative Refinement

Schema development is rarely a one-shot process, always:

- **Start Simple**: Begin with basic entities and core relationships
- **Test with Real Data**: Validate your schema against actual use cases
- **Identify Gaps**: Notice what your current schema cannot represent
- **Refine Gradually**: Add complexity only when justified by real needs
- **Document Decisions**: Keep track of why you made specific design choices

### Conclusion

Creating effective knowledge graph schemas is both an art and a science. Success comes from understanding your domain deeply, designing with use-case in mind, and remaining flexible as requirements evolve. Your schema serves as the foundation for all downstream applications—from search and recommendation systems to complex analytics and AI applications.

With these foundations in place, your knowledge graph schema will serve as a robust platform for organizing, connecting, and leveraging information in powerful new ways.

### Key Takeaways

- **Knowledge Graph Schema Basics**: A knowledge graph schema defines the structure and rules for organizing information in a graph format, consisting of entities (nodes) and relations (edges).

- **Entities**: Represent "things" in your domain such as people, places, objects, concepts, or events. Each entity type has a unique `label`, properties, and descriptions. Properties should be atomic and flexible to accommodate variations in data.

- **Relations**: Represent how entities interact or relate to each other. Each relation has a subject (`subj`), a label (`label`), a target (`obj`), properties, and descriptions.

- **Schema Design Strategy**: Start with a thorough domain analysis to identify core concepts and map natural relationships. Consider use cases and plan for future growth.

- **Balance Granularity and Usability**: Avoid being too generic or too specific; aim for a balance that captures essential distinctions while remaining manageable.

- **Iterative Refinement**: Begin with simple entities and core relationships. Test with real data, identify gaps, and refine gradually. Document design decisions for future reference.