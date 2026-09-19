import json
import re
from typing import List, Optional
from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient
from docuparser.models.graph import EntityNode, KnowledgeGraph, RelationEdge

try:
    import torch
    from gliner2 import AutoExtractor
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False


@ExtensionRegistry.register("gliner_graph")
class GLiNERGraphExtension(BaseExtension):
    """Extracts zero-shot entities using GLiNER 2 and resolves directed relations via LLM."""

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        model_name: str = "fastino/gliner2.5-small-v1",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.labels = labels or ["Organization", "Person", "Location", "Date", "Product"]
        self.model_name = model_name
        self.device = device
        self._extractor = None

    def _get_extractor(self):
        if not HAS_GLINER:
            raise ImportError("GLiNER2 not found. Install via: pip install 'docuparser[graph]'")
        if self._extractor is None:
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._extractor = AutoExtractor.from_pretrained(self.model_name, map_location=dev)
        return self._extractor

    def run(self, doc: ParsedDocument, llm: Optional[BaseLLMClient] = None) -> ParsedDocument:
        extractor = self._get_extractor()
        pred = extractor.extract_entities(doc.markdown, self.labels)

        nodes: List[EntityNode] = []
        seen = set()

        for label, items in pred.get("entities", {}).items():
            for item in items:
                if isinstance(item, str):
                    entity_text = item
                elif isinstance(item, dict):
                    entity_text = item.get("text", "")
                else:
                    continue
                if not isinstance(entity_text, str):
                    continue
                eid = f"{label.lower()}:{entity_text.strip().lower()}"
                if eid not in seen and entity_text.strip():
                    seen.add(eid)
                    nodes.append(EntityNode(id=eid, name=entity_text.strip(), label=label))

        edges: List[RelationEdge] = []
        if llm and len(nodes) >= 2:
            entity_list = [{"id": n.id, "name": n.name, "type": n.label} for n in nodes]
            prompt = (
                f"Given the document text and the list of detected entities, extract valid directed relationships between them.\n\n"
                f"Document Text:\n{doc.markdown[:5000]}\n\n"
                f"Entities:\n{json.dumps(entity_list, indent=2)}\n\n"
                f"Return a strict JSON array of objects with keys: 'source', 'target', 'relation_type'.\n"
                f"Where 'source' and 'target' MUST match the entity 'id' values listed above.\n"
                f"Output ONLY the raw JSON array."
            )
            try:
                raw_response = llm.generate(prompt)
                clean = re.sub(r"^```(?:json)?", "", raw_response.strip())
                clean = re.sub(r"```$", "", clean).strip()

                data = json.loads(clean)
                if isinstance(data, list):
                    valid_ids = {node.id for node in nodes}
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        source = item.get("source")
                        target = item.get("target")
                        relation_type = item.get("relation_type", "RELATED_TO")
                        if (
                            isinstance(source, str)
                            and isinstance(target, str)
                            and source in valid_ids
                            and target in valid_ids
                            and isinstance(relation_type, str)
                            and relation_type.strip()
                        ):
                            edges.append(
                                RelationEdge(
                                    source=source,
                                    target=target,
                                    relation_type=relation_type,
                                )
                            )
            except Exception:
                pass

        doc.artifacts["knowledge_graph"] = KnowledgeGraph(nodes=nodes, edges=edges)
        return doc