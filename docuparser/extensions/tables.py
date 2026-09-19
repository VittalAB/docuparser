import json
import re
from typing import List, Optional, Type
from pydantic import BaseModel, ValidationError
from docuparser.context import ParsedDocument
from docuparser.extension import BaseExtension, ExtensionRegistry
from docuparser.llm.base import BaseLLMClient


@ExtensionRegistry.register("tables")
class TableExtractionExtension(BaseExtension):
    """Extracts tables, converts them to CSV, and structures them into validated Pydantic models via LLM."""

    def __init__(self, schema: Type[BaseModel]):
        super().__init__()
        self.schema = schema

    def run(self, doc: ParsedDocument, llm: Optional[BaseLLMClient] = None) -> ParsedDocument:
        if not llm:
            raise ValueError("An LLM client is required for TableExtractionExtension.")

        clean_tables: List[List[BaseModel]] = []
        schema_str = json.dumps(self.schema.model_json_schema(), indent=2)

        for table in doc.tables:
            try:
                # Support both older and newer Docling dataframe export signatures
                try:
                    df = table.export_to_dataframe(doc=doc.raw_document)
                except TypeError:
                    df = table.export_to_dataframe()

                table_csv = df.to_csv(index=False)
            except Exception:
                continue

            prompt = (
                f"You are a strict data extraction engine.\n"
                f"Convert the following table CSV into a JSON array of objects conforming to this schema:\n"
                f"```json\n{schema_str}\n```\n\n"
                f"Table Data:\n{table_csv}\n\n"
                f"Requirements:\n"
                f"- Return ONLY the valid JSON array.\n"
                f"- Do not include markdown formatting or commentary."
            )

            try:
                response = llm.generate(prompt)
                clean = re.sub(r"^```(?:json)?", "", response.strip())
                clean = re.sub(r"```$", "", clean).strip()

                data = json.loads(clean)
                valid_rows = []
                if isinstance(data, list):
                    for row in data:
                        try:
                            valid_rows.append(self.schema.model_validate(row))
                        except ValidationError:
                            pass
                if valid_rows:
                    clean_tables.append(valid_rows)
            except Exception:
                continue

        doc.artifacts["structured_tables"] = clean_tables
        return doc