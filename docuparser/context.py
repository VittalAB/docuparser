from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ParsedDocument(BaseModel):
    """
    State container passed through the extension pipeline.
    Carries the evolving representations and artifacts of the parsed document.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_path: str
    raw_document: Optional[Any] = None  # Underlying Docling Document instance
    markdown: str = ""
    is_anonymized: bool = False
    tables: List[Any] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)