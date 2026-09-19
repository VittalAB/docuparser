from typing import Any, Dict, List
from pydantic import BaseModel, Field


class EntityNode(BaseModel):
    id: str = Field(description="Unique node identifier (e.g. company:apple)")
    name: str = Field(description="Surface text or entity name")
    label: str = Field(description="Entity type/label (e.g. Company, Person)")
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationEdge(BaseModel):
    source: str = Field(description="Source Entity ID")
    target: str = Field(description="Target Entity ID")
    relation_type: str = Field(description="Relationship name (e.g. VISITED, OWNS)")
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    nodes: List[EntityNode] = Field(default_factory=list)
    edges: List[RelationEdge] = Field(default_factory=list)

    def to_cypher(self) -> List[str]:
        """Generates idempotent Cypher MERGE queries for graph stores (Neo4j, Memgraph)."""
        def escape_string(value: Any) -> str:
            return str(value).replace("\\", "\\\\").replace("'", "\\'")

        def escape_identifier(value: Any) -> str:
            return str(value).replace("`", "``")

        def property_assignments(variable: str, properties: Dict[str, Any]) -> str:
            if not properties:
                return ""
            pairs = ", ".join(
                f"{variable}.`{escape_identifier(key)}` = '{escape_string(value)}'"
                for key, value in properties.items()
            )
            return pairs

        queries = []
        for node in self.nodes:
            label_safe = escape_identifier(node.label.replace(" ", "_").capitalize())
            node_id = escape_string(node.id)
            clean_name = escape_string(node.name)
            queries.append(
                f"MERGE (n:`{label_safe}` {{id: '{node_id}'}}) "
                f"SET n.name = '{clean_name}'"
                f"{', ' + property_assignments('n', node.properties) if node.properties else ''}"
            )
        for edge in self.edges:
            rel_safe = escape_identifier(edge.relation_type.upper().replace(" ", "_"))
            query = (
                f"MATCH (s {{id: '{escape_string(edge.source)}'}}), "
                f"(t {{id: '{escape_string(edge.target)}'}}) "
                f"MERGE (s)-[r:`{rel_safe}`]->(t)"
            )
            if edge.properties:
                query += f" SET {property_assignments('r', edge.properties)}"
            queries.append(query)
        return queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges],
        }