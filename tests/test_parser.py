from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient
from docuparser.llm.factory import create_llm_client
from docuparser.models.graph import EntityNode, RelationEdge, KnowledgeGraph
from docuparser.extensions.pii import PIIMaskingExtension
from docuparser.extensions.tables import TableExtractionExtension


# --- Mock Schema & Client ---
class SampleInvoiceItem(BaseModel):
    item: str
    amount: float


class DummyMockLLM(BaseLLMClient):
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if "schema" in prompt.lower() or "table" in prompt.lower():
            return '[{"item": "Hosting", "amount": 250.0}]'
        if "relationships" in prompt.lower():
            return '[{"source": "company:example", "target": "person:example_user", "relation_type": "LED_BY"}]'
        return "Executive Summary: Document parsed successfully."


# --- Tests ---

def test_llm_factory_adapters():
    """Verify factory adapts callables and custom clients."""
    # 1. Custom Callable
    client = create_llm_client(lambda p: "callable_output")
    assert client.generate("test") == "callable_output"

    # 2. Duck-typed LangChain mock (.invoke)
    mock_langchain = MagicMock()
    mock_langchain.invoke.return_value = MagicMock(content="langchain_output")
    client_lc = create_llm_client(mock_langchain)
    assert client_lc.generate("test") == "langchain_output"

    # 3. Dict config
    client_dict = create_llm_client({"base_url": "http://localhost:8000/v1", "model_name": "test-model"})
    assert client_dict.model_name == "test-model"


def test_pii_masking_extension():
    """Verify PII masking replaces sensitive patterns."""
    doc = ParsedDocument(
        file_path="dummy.pdf",
        markdown="Contact Example Person at user@example.com or +1-555-0100 with SSN 123-45-6789."
    )
    ext = PIIMaskingExtension()
    processed_doc = ext.run(doc)

    assert processed_doc.is_anonymized is True
    assert "user@example.com" not in processed_doc.markdown
    assert "123-45-6789" not in processed_doc.markdown


def test_table_extraction_extension():
    """Verify structured table extraction with Pydantic validation."""
    mock_df = MagicMock()
    mock_df.to_csv.return_value = "Item,Amount\nHosting,250.0"

    mock_table = MagicMock()
    mock_table.export_to_dataframe.return_value = mock_df

    doc = ParsedDocument(
        file_path="dummy.pdf",
        markdown="Sample content",
        tables=[mock_table]
    )

    llm = DummyMockLLM()
    ext = TableExtractionExtension(schema=SampleInvoiceItem)
    processed_doc = ext.run(doc, llm=llm)

    extracted = processed_doc.artifacts.get("structured_tables")
    assert len(extracted) == 1
    assert len(extracted[0]) == 1
    assert isinstance(extracted[0][0], SampleInvoiceItem)
    assert extracted[0][0].item == "Hosting"
    assert extracted[0][0].amount == 250.0


def test_knowledge_graph_cypher_generation():
    """Verify knowledge graph models produce valid Cypher queries."""
    kg = KnowledgeGraph(
        nodes=[
            EntityNode(id="company:example", name="Example Corp", label="Company"),
            EntityNode(id="person:example_user", name="Example User", label="Person"),
        ],
        edges=[
            RelationEdge(source="person:example_user", target="company:example", relation_type="LEADS")
        ]
    )
    cypher = kg.to_cypher()
    assert len(cypher) == 3
    assert "MERGE (n:`Company` {id: 'company:example'})" in cypher[0]
    assert "MERGE (s)-[r:`LEADS`]->(t)" in cypher[2]


def test_custom_extension_pipeline():
    """Verify Open-Closed extensibility via custom BaseExtension."""
    @ExtensionRegistry.register("word_counter")
    class WordCounterExtension(BaseExtension):
        def run(self, doc: ParsedDocument, llm=None) -> ParsedDocument:
            doc.artifacts["word_count"] = len(doc.markdown.split())
            return doc

    doc = ParsedDocument(file_path="sample.pdf", markdown="One two three four five")
    ext = ExtensionRegistry.get("word_counter")()
    result = ext.run(doc)

    assert result.artifacts["word_count"] == 5