"""
Semantic search module for retrieving relevant document chunks using embeddings.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Vector database and embedding imports (will be available after package installation)
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None


# Gemini API imports (Google Generative AI)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from models.schemas import DocumentChunk, RetrievedClause, StructuredQuery
from core.config import get_settings

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Handles semantic search and retrieval of relevant document chunks."""
    
    def __init__(self):
        """Initialize the semantic search system."""
        self.settings = get_settings()
        self.client = None
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        
        # Initialize Gemini client
        if genai and self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.client = genai
        
        # Initialize ChromaDB
        if chromadb:
            self._init_chroma_db()
        
        # Initialize local embedding model (fallback)
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load local embedding model: {e}")
    
    def _init_chroma_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.settings.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("documents")
                logger.info("Connected to existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"description": "Document intelligence system collection"}
                )
                logger.info("Created new ChromaDB collection")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Index document chunks for semantic search.
        
        Args:
            chunks: List of document chunks to index
            
        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            logger.error("ChromaDB collection not available")
            return False
        
        try:
            # Prepare data for indexing
            documents = [chunk.content for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    "source_document": chunk.source_document,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number or 0,
                    **chunk.metadata
                }
                metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(documents)
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully indexed {len(chunks)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def search(self, structured_query: StructuredQuery, top_k: int = None) -> List[RetrievedClause]:
        """
        Search for relevant clauses based on the structured query.
        
        Args:
            structured_query: Processed query with entities and intent
            top_k: Number of top results to return
            
        Returns:
            List of retrieved clauses ranked by relevance
        """
        if not self.collection:
            logger.error("ChromaDB collection not available")
            return []
        
        if top_k is None:
            top_k = self.settings.top_k_results
        
        try:
            # Create search query from structured query
            search_text = self._create_search_query(structured_query)
            logger.info(f"Searching with query: {search_text}")
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([search_text])
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to RetrievedClause objects
            retrieved_clauses = []
            
            if results['documents'] and len(results['documents']) > 0:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to relevance score (lower distance = higher relevance)
                    relevance_score = max(0.0, 1.0 - distance)
                    
                    clause = RetrievedClause(
                        clause_id=f"clause_{i}",
                        content=doc,
                        source_document=metadata.get('source_document', 'unknown'),
                        relevance_score=relevance_score,
                        section=metadata.get('section'),
                        clause_type=self._infer_clause_type(doc, structured_query)
                    )
                    retrieved_clauses.append(clause)
            
            logger.info(f"Retrieved {len(retrieved_clauses)} relevant clauses")
            return retrieved_clauses
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _create_search_query(self, structured_query: StructuredQuery) -> str:
        """
        Create an enhanced search query from the structured query.
        
        Args:
            structured_query: Structured query object
            
        Returns:
            Enhanced search query string
        """
        query_parts = [structured_query.original_query]
        
        # Add entity information
        for entity in structured_query.entities:
            if entity.entity_type == "medical_procedure":
                query_parts.append(f"medical procedure {entity.value}")
            elif entity.entity_type == "age":
                query_parts.append(f"age {entity.value} years")
            elif entity.entity_type == "location":
                query_parts.append(f"location {entity.value}")
            elif entity.entity_type == "policy_duration":
                query_parts.append(f"policy duration {entity.value} months")
        
        # Add intent-specific terms
        intent_terms = {
            "check_coverage": ["coverage", "eligible", "covered", "benefits"],
            "policy_inquiry": ["policy terms", "conditions", "rules"],
            "process_claim": ["claim", "reimbursement", "payout", "settlement"]
        }
        
        if structured_query.intent in intent_terms:
            query_parts.extend(intent_terms[structured_query.intent])
        
        return " ".join(query_parts)
    
    def _generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors or None if failed
        """
        # Try Gemini embeddings first
        if self.client:
            try:
                model = self.client.GenerativeModel(self.settings.embedding_model)
                embeddings = []
                for text in texts:
                    resp = model.embed_content(text)
                    if hasattr(resp, 'embedding'):
                        embeddings.append(resp.embedding)
                    else:
                        logger.warning(f"No embedding returned for text: {text[:30]}")
                if embeddings:
                    return embeddings
            except Exception as e:
                logger.warning(f"Gemini embedding failed, falling back to local model: {e}")
        
        # Fallback to local embedding model
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Local embedding failed: {e}")
        
        logger.error("No embedding method available")
        return None
    
    def _infer_clause_type(self, content: str, structured_query: StructuredQuery) -> Optional[str]:
        """
        Infer the type of clause based on content and query context.
        
        Args:
            content: Clause content
            structured_query: Original structured query
            
        Returns:
            Inferred clause type
        """
        content_lower = content.lower()
        
        # Coverage-related clauses
        coverage_keywords = ['covered', 'coverage', 'eligible', 'benefit', 'included']
        if any(keyword in content_lower for keyword in coverage_keywords):
            return "coverage"
        
        # Exclusion clauses
        exclusion_keywords = ['excluded', 'not covered', 'limitation', 'restriction']
        if any(keyword in content_lower for keyword in exclusion_keywords):
            return "exclusion"
        
        # Waiting period clauses
        waiting_keywords = ['waiting period', 'waiting time', 'grace period']
        if any(keyword in content_lower for keyword in waiting_keywords):
            return "waiting_period"
        
        # Premium/payment clauses
        payment_keywords = ['premium', 'payment', 'fee', 'cost']
        if any(keyword in content_lower for keyword in payment_keywords):
            return "payment"
        
        # Claim procedure clauses
        claim_keywords = ['claim', 'procedure', 'process', 'submit']
        if any(keyword in content_lower for keyword in claim_keywords):
            return "claim_procedure"
        
        return "general"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {"error": "Collection not available"}
        
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name,
                "embedding_function": "Gemini" if self.client else "SentenceTransformer"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            return False
        
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection("documents")
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"description": "Document intelligence system collection"}
            )
            logger.info("Cleared document collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
