from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class SourceLocation(BaseModel):
    page: int
    paragraph_id: str
    location: str


class EvidenceChunk(BaseModel):
    chunk_id: str
    text: str
    pages: List[int]
    document_id: str
    document_title: str
    document_author: str
    document_year: str

    source_locations: List[SourceLocation] = []
    metadata: Dict[str, Any] = {}

    # retrieval debug
    distance: Optional[float] = None
    relevance_score: Optional[float] = None

    # citation convenience
    citation: Optional[str] = None
    full_citation: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    top_k: int
    answer: str
    citations: List[str]
    evidence: List[EvidenceChunk]


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    author: str
    year: str
    filename: str
    file_path: str
    pages: int
