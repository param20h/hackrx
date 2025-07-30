"""
Data models for the Document Intelligence System.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# --- HackRX API Models (for /hackrx/run endpoint) ---

class HackrxRunRequest(BaseModel):
    """Request model for /hackrx/run endpoint."""
    documents: str = Field(..., description="URL to the policy PDF or document")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackrxRunResponse(BaseModel):
    """Response model for /hackrx/run endpoint."""
    answers: List[str] = Field(..., description="Answers to the questions, in order")

# For advanced output (optional, for explainability)
class HackrxAnswerWithJustification(BaseModel):
    """Detailed answer with justification and source clauses."""
    answer: str = Field(..., description="The answer to the question")
    justification: Optional[str] = Field(default=None, description="Reasoning behind the answer")
    source_clauses: Optional[List[str]] = Field(default=None, description="Relevant clauses from the document")

# For entity extraction (NER/Parsing)
class ParsedQueryEntities(BaseModel):
    """Structured entities extracted from natural language queries."""
    age: Optional[int] = Field(default=None, description="Person's age in years")
    gender: Optional[str] = Field(default=None, description="Person's gender (male/female)")
    procedure: Optional[str] = Field(default=None, description="Medical procedure or treatment")
    location: Optional[str] = Field(default=None, description="City or location")
    policy_duration_months: Optional[int] = Field(default=None, description="Policy duration in months")


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    INSURANCE_CLAIM = "insurance_claim"
    CONTRACT_ANALYSIS = "contract_analysis"
    POLICY_CHECK = "policy_check"
    GENERAL = "general"


class DocumentType(str, Enum):
    """Types of documents the system can process."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    EMAIL = "email"


class DecisionType(str, Enum):
    """Types of decisions the system can make."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    REQUIRES_REVIEW = "requires_review"


class QueryInput(BaseModel):
    """Input model for natural language queries."""
    query: str = Field(..., description="Natural language query from user")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context information")
    query_type: Optional[QueryType] = Field(default=QueryType.GENERAL, description="Type of query")


class ExtractedEntity(BaseModel):
    """Represents an extracted entity from the query."""
    entity_type: str = Field(..., description="Type of entity (age, procedure, location, etc.)")
    value: str = Field(..., description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for extraction")


class StructuredQuery(BaseModel):
    """Structured representation of the parsed query."""
    original_query: str = Field(..., description="Original natural language query")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    intent: str = Field(..., description="Identified intent of the query")
    query_type: QueryType = Field(..., description="Categorized query type")


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the chunk")
    source_document: str = Field(..., description="Source document path")
    page_number: Optional[int] = Field(default=None, description="Page number in source document")
    chunk_index: int = Field(..., description="Index of chunk in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievedClause(BaseModel):
    """Represents a retrieved clause from documents."""
    clause_id: str = Field(..., description="Unique identifier for the clause")
    content: str = Field(..., description="Text content of the clause")
    source_document: str = Field(..., description="Source document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score to query")
    section: Optional[str] = Field(default=None, description="Section or chapter in document")
    clause_type: Optional[str] = Field(default=None, description="Type of clause (coverage, exclusion, etc.)")


class Decision(BaseModel):
    """Represents a decision made by the system."""
    decision_type: DecisionType = Field(..., description="Type of decision made")
    amount: Optional[float] = Field(default=None, description="Amount if applicable")
    currency: Optional[str] = Field(default="INR", description="Currency for amount")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")


class ClauseReference(BaseModel):
    """Reference to a specific clause that influenced the decision."""
    clause_id: str = Field(..., description="ID of the referenced clause")
    clause_text: str = Field(..., description="Relevant text from the clause")
    source_document: str = Field(..., description="Document containing the clause")
    influence_weight: float = Field(..., ge=0.0, le=1.0, description="How much this clause influenced the decision")


class SystemResponse(BaseModel):
    """Complete response from the document intelligence system."""
    query_id: str = Field(..., description="Unique identifier for this query")
    original_query: str = Field(..., description="Original natural language query")
    structured_query: StructuredQuery = Field(..., description="Parsed and structured query")
    decision: Decision = Field(..., description="Final decision made by the system")
    supporting_clauses: List[ClauseReference] = Field(default_factory=list, description="Clauses that support the decision")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")


class DocumentMetadata(BaseModel):
    """Metadata for a processed document."""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Type of document")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(default=None, description="Number of pages")
    processed_at: datetime = Field(default_factory=datetime.now, description="When document was processed")
    document_category: Optional[str] = Field(default=None, description="Category of document (policy, contract, etc.)")
    version: Optional[str] = Field(default=None, description="Document version")


class ProcessingStatus(BaseModel):
    """Status of document processing."""
    document_id: str = Field(..., description="Document identifier")
    status: str = Field(..., description="Processing status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Processing progress")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class ConfigSettings(BaseModel):
    """Configuration settings for the system."""
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    max_tokens: int = Field(default=4096, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for LLM")
    chunk_size: int = Field(default=1000, description="Size of document chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    top_k_results: int = Field(default=5, description="Number of top results to retrieve")
