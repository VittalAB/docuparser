# Docuparser

Docuparser is an extensible Python library for document intelligence. It uses
[Docling](https://github.com/docling-project/docling) for document conversion
and provides optional extensions for PII masking, embeddings, structured table
extraction, summaries, and knowledge graphs.

## Requirements

- Python 3.10 or newer
- A document supported by Docling
- An LLM endpoint for summaries, table extraction, and graph relationships
- Optional feature dependencies for embeddings, PII analysis, and graphs

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the core package:

```powershell
python -m pip install -e .
```

Install optional features as needed:

```powershell
python -m pip install -e ".[pii]"
python -m pip install -e ".[embeddings]"
python -m pip install -e ".[graph]"
python -m pip install -e ".[all]"
```

The package has a regex fallback for common PII patterns when the `pii` extra
is not installed. The `embeddings` and `graph` extensions require their
corresponding extras.

## LLM Configuration

Docuparser accepts an OpenAI-compatible configuration dictionary. Set the API
key in the environment rather than committing it to source control:

```powershell
$env:LLM_API_KEY = "your-api-key"
```

```python
from docuparser import Docuparser

parser = Docuparser(
    file_path="sample.pdf",
    llm={
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
    },
)
```

You can provide `base_url`, `model_name`, `api_key`, `temperature`, and
`extra_headers` for a compatible LLM endpoint. A custom `BaseLLMClient`,
callable, or compatible LangChain chat model can also be supplied.

## Basic Usage

Parsing is lazy when an extension or `get_markdown()` is first used. Call
`parse()` explicitly when you want to perform conversion up front.

```python
from docuparser import Docuparser

parser = Docuparser("sample.pdf")
parser.parse()
print(parser.get_markdown())
```

The parsed state is available through `parser.doc_context`:

- `markdown`: exported document text
- `raw_document`: the underlying Docling document
- `tables`: extracted raw tables
- `chunks`: generated chunks and vectors
- `artifacts`: extension outputs
- `metadata`: document metadata

## PII Masking

```python
from docuparser import Docuparser

parser = Docuparser("sample.pdf")
parser.mask_pii()
print(parser.get_markdown())
```

To restrict masking to selected entity types:

```python
parser.mask_pii(entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"])
```

Supported configured entity names include `PERSON`, `EMAIL_ADDRESS`,
`PHONE_NUMBER`, `CREDIT_CARD`, `CRYPTO`, `IP_ADDRESS`, and `US_SSN`.

## Embeddings

Install `.[embeddings]` first. The extension uses Docling's `HybridChunker`
and Sentence Transformers.

```python
from docuparser import Docuparser

parser = Docuparser("sample.pdf")
chunks = parser.to_embeddings(
    model_name="all-MiniLM-L6-v2",
    chunk_size=512,
    device="cpu",
)

for chunk in chunks:
    print(chunk["chunk_id"], len(chunk["vector"]))
```

The model may be downloaded the first time it is used. Use `device="cuda"`
in a CUDA-capable PyTorch environment for GPU execution.

## Structured Tables

Table extraction requires an LLM and a Pydantic schema:

```python
from pydantic import BaseModel
from docuparser import Docuparser


class InvoiceItem(BaseModel):
    item: str
    amount: float
    currency: str = "USD"


parser = Docuparser("invoice.pdf", llm={
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini",
})
tables = parser.get_clean_tables(InvoiceItem)

for table in tables:
    for row in table:
        print(row.model_dump())
```

Rows that fail Pydantic validation are skipped. Results are also available at
`parser.doc_context.artifacts["structured_tables"]`.

## Summaries

```python
summary = parser.get_summary(max_characters=12000)
print(summary)
```

This requires an LLM. Without one, the extension stores an explanatory
message in the `summary` artifact.

## Knowledge Graphs

Install `.[graph]` first. GLiNER extracts entities, and the configured LLM
identifies relationships between those entities.

```python
from docuparser import Docuparser

parser = Docuparser("report.pdf", llm={
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini",
})

graph = parser.get_knowledge_graph(
    labels=["Organization", "Person", "Location", "Product"]
)

print(graph.to_dict())
for query in graph.to_cypher():
    print(query)
```

`to_cypher()` returns `MERGE` and `MATCH` statements suitable for review or
submission to a compatible graph database. Review generated queries before
executing them against a database.

## Pipelines and Custom Extensions

Built-in extensions can be composed with `pipe()` or `run_pipeline()`:

```python
from docuparser import Docuparser

parser = Docuparser("sample.pdf", llm={
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini",
})
parser.run_pipeline(["pii_masking", "summary"])
print(parser.doc_context.artifacts["summary"])
```

Custom extensions implement `BaseExtension` and return the updated
`ParsedDocument`:

```python
from docuparser import BaseExtension, Docuparser, ParsedDocument


class WordCountExtension(BaseExtension):
    def run(self, doc: ParsedDocument, llm=None) -> ParsedDocument:
        doc.artifacts["word_count"] = len(doc.markdown.split())
        return doc


parser = Docuparser("sample.pdf")
parser.pipe(WordCountExtension())
print(parser.doc_context.artifacts["word_count"])
```

Registered extensions can be inspected with:

```python
from docuparser import ExtensionRegistry

print(ExtensionRegistry.list_available())
```

## Development

Run the test suite from the project directory:

```powershell
python -m pytest
```

Tests use mocks for external services where possible. Real LLM, Docling,
GLiNER, and embedding model downloads may require network access and system
resources.

## License

Docuparser is open-source software released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Copyright 2026.
