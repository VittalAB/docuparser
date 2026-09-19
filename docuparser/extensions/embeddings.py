from typing import Optional
from docling.chunking import HybridChunker
from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


@ExtensionRegistry.register("embeddings")
class EmbeddingsExtension(BaseExtension):
    """Chunks the document via Docling HybridChunker and generates vector embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.device = device
        self._engine = None

    def _get_engine(self):
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers is not installed. Install via: pip install 'docuparser[embeddings]'"
            )
        if self._engine is None:
            self._engine = SentenceTransformer(self.model_name, device=self.device)
        return self._engine

    def run(self, doc: ParsedDocument, llm: Optional[BaseLLMClient] = None) -> ParsedDocument:
        if not doc.raw_document:
            return doc

        chunker = HybridChunker(max_tokens=self.chunk_size)
        raw_chunks = list(chunker.chunk(doc.raw_document))

        engine = self._get_engine()
        processed = []
        texts = []

        for idx, chunk in enumerate(raw_chunks):
            txt = chunk.text if hasattr(chunk, "text") else str(chunk)
            meta = chunk.meta.export_json_dict() if hasattr(chunk, "meta") else {}
            processed.append({"chunk_id": idx, "text": txt, "metadata": meta, "vector": None})
            texts.append(txt)

        if texts:
            vectors = engine.encode(texts, show_progress_bar=False).tolist()
            for item, v in zip(processed, vectors):
                item["vector"] = v

        doc.chunks = processed
        return doc