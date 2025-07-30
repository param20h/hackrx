"""
Main orchestrator for the Document Intelligence System.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from .models import (
    QueryInput, SystemResponse, DocumentMetadata, ProcessingStatus
)
from .core import (
    DocumentProcessor, QueryProcessor, SemanticSearch, 
    DecisionEngine, get_settings, validate_settings
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIntelligenceSystem:
    """
    Main orchestrator for the Document Intelligence System.
    
    This class coordinates all components to process natural language queries
    against a collection of documents and provide structured responses.
    """
    
    def __init__(self):
        """Initialize the document intelligence system."""
        logger.info("Initializing Document Intelligence System...")
        
        # Validate configuration
        try:
            validate_settings()
            self.settings = get_settings()
            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        self.query_processor = QueryProcessor()
        self.semantic_search = SemanticSearch()
        self.decision_engine = DecisionEngine()
        
        logger.info("Document Intelligence System initialized successfully")
    
    def process_query(self, query: str, context: Optional[dict] = None) -> SystemResponse:
        """
        Process a natural language query and return a structured response.
        
        Args:
            query: Natural language query
            context: Optional additional context
            
        Returns:
            Structured system response with decision and justification
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Parse and structure the query
            query_input = QueryInput(query=query, context=context)
            structured_query = self.query_processor.process_query(query_input)
            
            # Step 2: Search for relevant clauses
            retrieved_clauses = self.semantic_search.search(structured_query)
            
            # Step 3: Make decision based on retrieved clauses
            response = self.decision_engine.make_decision(structured_query, retrieved_clauses)
            
            logger.info(f"Query processed successfully: {response.decision.decision_type}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def add_documents(self, document_paths: List[str]) -> List[ProcessingStatus]:
        """
        Add documents to the system for indexing.
        
        Args:
            document_paths: List of paths to documents to add
            
        Returns:
            List of processing statuses for each document
        """
        logger.info(f"Adding {len(document_paths)} documents to the system")
        
        statuses = []
        
        for doc_path in document_paths:
            try:
                logger.info(f"Processing document: {doc_path}")
                
                # Check if file exists
                if not os.path.exists(doc_path):
                    status = ProcessingStatus(
                        document_id=os.path.basename(doc_path),
                        status="failed",
                        progress=0.0,
                        error=f"File not found: {doc_path}"
                    )
                    statuses.append(status)
                    continue
                
                # Get document metadata
                metadata = self.document_processor.get_document_metadata(doc_path)
                
                # Process document into chunks
                chunks = self.document_processor.process_document(doc_path)
                
                # Index chunks for semantic search
                success = self.semantic_search.index_documents(chunks)
                
                if success:
                    status = ProcessingStatus(
                        document_id=metadata.document_id,
                        status="completed",
                        progress=1.0,
                        message=f"Successfully processed {len(chunks)} chunks"
                    )
                else:
                    status = ProcessingStatus(
                        document_id=metadata.document_id,
                        status="failed",
                        progress=0.5,
                        error="Failed to index document chunks"
                    )
                
                statuses.append(status)
                logger.info(f"Document {doc_path} processed: {status.status}")
                
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                status = ProcessingStatus(
                    document_id=os.path.basename(doc_path),
                    status="failed",
                    progress=0.0,
                    error=str(e)
                )
                statuses.append(status)
        
        successful_docs = len([s for s in statuses if s.status == "completed"])
        logger.info(f"Successfully added {successful_docs}/{len(document_paths)} documents")
        
        return statuses
    
    def add_documents_from_directory(self, directory_path: str) -> List[ProcessingStatus]:
        """
        Add all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processing statuses
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all supported document files
        supported_extensions = ['.pdf', '.docx', '.txt']
        document_paths = []
        
        for ext in supported_extensions:
            pattern = f"**/*{ext}"
            files = list(Path(directory_path).glob(pattern))
            document_paths.extend([str(f) for f in files])
        
        logger.info(f"Found {len(document_paths)} documents in {directory_path}")
        
        return self.add_documents(document_paths)
    
    def get_system_status(self) -> dict:
        """
        Get the current status of the system.
        
        Returns:
            Dictionary with system status information
        """
        try:
            collection_stats = self.semantic_search.get_collection_stats()
            
            return {
                "system_status": "operational",
                "configuration": {
                    "model_name": self.settings.model_name,
                    "embedding_model": self.settings.embedding_model,
                    "chunk_size": self.settings.chunk_size,
                    "top_k_results": self.settings.top_k_results
                },
                "collection_stats": collection_stats,
                "components": {
                    "document_processor": "ready",
                    "query_processor": "ready", 
                    "semantic_search": "ready" if collection_stats.get("total_chunks", 0) > 0 else "no_documents",
                    "decision_engine": "ready"
                }
            }
            
        except Exception as e:
            return {
                "system_status": "error",
                "error": str(e)
            }
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from the system.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Clearing all documents from the system")
        
        try:
            success = self.semantic_search.clear_collection()
            if success:
                logger.info("All documents cleared successfully")
            else:
                logger.error("Failed to clear documents")
            return success
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Intelligence System")
    parser.add_argument("--add-docs", type=str, help="Directory path to add documents from")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--clear", action="store_true", help="Clear all documents")
    
    args = parser.parse_args()
    
    # Initialize system
    system = DocumentIntelligenceSystem()
    
    if args.add_docs:
        print(f"Adding documents from: {args.add_docs}")
        statuses = system.add_documents_from_directory(args.add_docs)
        for status in statuses:
            print(f"Document {status.document_id}: {status.status}")
    
    elif args.query:
        print(f"Processing query: {args.query}")
        response = system.process_query(args.query)
        print(f"\\nDecision: {response.decision.decision_type}")
        if response.decision.amount:
            print(f"Amount: ₹{response.decision.amount:,.2f}")
        print(f"Confidence: {response.decision.confidence:.2f}")
        print(f"Reasoning: {response.decision.reasoning}")
        
        if response.supporting_clauses:
            print("\\nSupporting Clauses:")
            for clause in response.supporting_clauses:
                print(f"- {clause.source_document}: {clause.clause_text[:100]}...")
    
    elif args.status:
        status = system.get_system_status()
        print("System Status:")
        print(f"Status: {status['system_status']}")
        if 'collection_stats' in status:
            stats = status['collection_stats']
            print(f"Indexed chunks: {stats.get('total_chunks', 0)}")
        
    elif args.clear:
        print("Clearing all documents...")
        success = system.clear_documents()
        print("Documents cleared successfully" if success else "Failed to clear documents")
    
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()
