import re
from typing import List, Optional
from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False


@ExtensionRegistry.register("pii_masking")
class PIIMaskingExtension(BaseExtension):
    """Masks PII using a detector when available and regex fallback otherwise."""

    def __init__(self, entities: Optional[List[str]] = None):
        super().__init__()
        self.entities = entities or [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "CREDIT_CARD", "CRYPTO", "IP_ADDRESS", "US_SSN",
        ]
        self.analyzer = AnalyzerEngine() if HAS_PRESIDIO else None
        self.anonymizer = AnonymizerEngine() if HAS_PRESIDIO else None

    def _regex_mask(self, text: str) -> str:
        patterns = {
            "EMAIL_ADDRESS": (r"[\w.\-+]+@[\w.\-]+\.\w+", "<EMAIL>"),
            "PHONE_NUMBER": (r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", "<PHONE>"),
            "US_SSN": (r"\b\d{3}-\d{2}-\d{4}\b", "<SSN>"),
            "IP_ADDRESS": (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP_ADDRESS>"),
            "CREDIT_CARD": (r"\b(?:\d[ -]?){13,19}\b", "<CREDIT_CARD>"),
            "CRYPTO": (r"\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b", "<CRYPTO>"),
        }
        for entity in self.entities:
            pattern = patterns.get(entity)
            if pattern:
                text = re.sub(pattern[0], pattern[1], text)
        return text

    def run(self, doc: ParsedDocument, llm: Optional[BaseLLMClient] = None) -> ParsedDocument:
        raw_text = doc.markdown
        text = raw_text

        if HAS_PRESIDIO and self.analyzer and self.anonymizer:
            results = self.analyzer.analyze(text=raw_text, entities=self.entities, language="en")
            if results:
                text = self.anonymizer.anonymize(text=raw_text, analyzer_results=results).text

        doc.markdown = self._regex_mask(text)

        doc.is_anonymized = True
        doc.chunks = []
        doc.artifacts.pop("structured_tables", None)
        doc.artifacts.pop("knowledge_graph", None)
        doc.artifacts.pop("summary", None)
        return doc