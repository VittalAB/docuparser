from typing import Any, Callable, Dict, List, Optional, Type, Union
from dotenv import load_dotenv
from pydantic import BaseModel

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient
from docuparser.llm.factory import create_llm_client
from docuparser.models.graph import KnowledgeGraph

# Import built-in extensions so they auto-register in ExtensionRegistry
from docuparser.extensions.pii import PIIMaskingExtension
from docuparser.extensions.embeddings import EmbeddingsExtension
from docuparser.extensions.tables import TableExtractionExtension
from docuparser.extensions.graph import GLiNERGraphExtension
from docuparser.extensions.summary import SummaryExtension

load_dotenv()


class Docuparser:
    """
    Main orchestration class for Docuparser.
    Provides foundational parsing via Docling, extension chaining via .pipe(),
    and shorthand convenience methods.
    """

    def __init__(
        self,
        file_path: str,
        llm: Optional[Union[BaseLLMClient, Dict[str, Any], Any, Callable]] = None,
        **docling_kwargs,
    ):
        self.file_path = file_path
        self.llm = create_llm_client(llm) if llm else None

        pipeline_options = PdfPipelineOptions()
        for k, v in docling_kwargs.items():
            if hasattr(pipeline_options, k):
                setattr(pipeline_options, k, v)

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.doc_context: Optional[ParsedDocument] = None

    def parse(self) -> "Docuparser":
        """Executes the foundational Docling layout and OCR extraction."""
        conv_result = self.converter.convert(self.file_path)
        self.doc_context = ParsedDocument(
            file_path=self.file_path,
            raw_document=conv_result.document,
            markdown=conv_result.document.export_to_markdown(),
            tables=getattr(conv_result.document, "tables", []),
        )
        return self

    # ---------------------------------------------------------
    # Extensible Pipeline Engine (.pipe() / .run_pipeline())
    # ---------------------------------------------------------
    def pipe(self, extension: Union[BaseExtension, str], **kwargs) -> "Docuparser":
        """
        Executes a single extension and updates the document context.
        Accepts either an instantiated BaseExtension or a registered string name.
        """
        if not self.doc_context:
            self.parse()

        if isinstance(extension, str):
            ext_class = ExtensionRegistry.get(extension)
            ext_instance = ext_class(**kwargs)
        else:
            ext_instance = extension

        if not isinstance(ext_instance, BaseExtension):
            raise TypeError("extension must be a registered name or BaseExtension instance")

        self.doc_context = ext_instance.run(self.doc_context, llm=self.llm)
        return self

    def run_pipeline(self, extensions: List[Union[BaseExtension, str]]) -> ParsedDocument:
        """Executes an ordered list of extensions sequentially and returns the final context."""
        for ext in extensions:
            self.pipe(ext)
        if self.doc_context is None:
            raise RuntimeError("Pipeline completed without a parsed document context")
        return self.doc_context

    # ---------------------------------------------------------
    # Shorthand Convenience Methods
    # ---------------------------------------------------------
    def mask_pii(self, entities: Optional[List[str]] = None) -> "Docuparser":
        return self.pipe(PIIMaskingExtension(entities=entities))

    def get_knowledge_graph(
        self,
        labels: Optional[List[str]] = None,
        model_name: str = "fastino/gliner2.5-small-v1",
        device: Optional[str] = None,
    ) -> KnowledgeGraph:
        self.pipe(GLiNERGraphExtension(labels=labels, model_name=model_name, device=device))
        return self.doc_context.artifacts.get("knowledge_graph")

    def to_embeddings(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        device: str = "cpu",
    ) -> List[Dict[str, Any]]:
        self.pipe(EmbeddingsExtension(model_name=model_name, chunk_size=chunk_size, device=device))
        return self.doc_context.chunks

    def get_clean_tables(self, schema: Type[BaseModel]) -> List[List[BaseModel]]:
        self.pipe(TableExtractionExtension(schema=schema))
        return self.doc_context.artifacts.get("structured_tables", [])

    def get_summary(self, max_characters: int = 12000) -> str:
        self.pipe(SummaryExtension(max_characters=max_characters))
        return self.doc_context.artifacts.get("summary", "")

    def get_markdown(self) -> str:
        if not self.doc_context:
            self.parse()
        return self.doc_context.markdown

    def __getattr__(self, name: str) -> Any:
        """Passthrough for missing attributes to underlying Docling document."""
        if not self.doc_context:
            self.parse()
        if hasattr(self.doc_context.raw_document, name):
            return getattr(self.doc_context.raw_document, name)
        raise AttributeError(f"'Docuparser' object has no attribute '{name}'")