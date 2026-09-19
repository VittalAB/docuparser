from typing import Optional
from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient


@ExtensionRegistry.register("summary")
class SummaryExtension(BaseExtension):
    """Generates an executive, structured summary using the configured LLM."""

    def __init__(self, max_characters: int = 12000):
        super().__init__()
        self.max_characters = max_characters

    def run(self, doc: ParsedDocument, llm: Optional[BaseLLMClient] = None) -> ParsedDocument:
        if not llm:
            doc.artifacts["summary"] = "Summary unavailable: LLM client is not configured."
            return doc

        content = doc.markdown[:self.max_characters]
        prompt = (
            "Please provide a structured, executive summary of the following document. "
            "Highlight key findings, metrics, and action items:\n\n"
            f"{content}"
        )
        doc.artifacts["summary"] = llm.generate(prompt)
        return doc