"""
Document processing module for handling various document types.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Document processing imports (will be available after package installation)
try:
    import PyPDF2
    from docx import Document as DocxDocument
except ImportError:
    PyPDF2 = None
    DocxDocument = None

from models.schemas import DocumentChunk, DocumentMetadata, DocumentType, ProcessingStatus

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles processing of various document types into structured chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a document and return chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension == '.docx':
            return self._process_docx(file_path)
        elif file_extension == '.txt':
            return self._process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Process PDF document."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing")
        
        chunks = []
        document_id = self._generate_document_id(file_path)
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    full_text += f"\\n\\nPage {page_num + 1}:\\n{page_text}"
                
                # Create chunks from full text
                text_chunks = self._create_text_chunks(full_text)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{i}",
                        content=chunk_text,
                        source_document=file_path,
                        chunk_index=i,
                        metadata={
                            "document_type": "pdf",
                            "total_pages": len(pdf_reader.pages),
                            "processing_method": "PyPDF2"
                        }
                    )
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
        
        return chunks
    
    def _process_docx(self, file_path: str) -> List[DocumentChunk]:
        """Process DOCX document."""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX processing")
        
        chunks = []
        document_id = self._generate_document_id(file_path)
        
        try:
            doc = DocxDocument(file_path)
            
            # Extract text from paragraphs
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            full_text = "\\n\\n".join(paragraphs)
            
            # Create chunks from full text
            text_chunks = self._create_text_chunks(full_text)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{i}",
                    content=chunk_text,
                    source_document=file_path,
                    chunk_index=i,
                    metadata={
                        "document_type": "docx",
                        "total_paragraphs": len(paragraphs),
                        "processing_method": "python-docx"
                    }
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            raise
        
        return chunks
    
    def _process_txt(self, file_path: str) -> List[DocumentChunk]:
        """Process TXT document."""
        chunks = []
        document_id = self._generate_document_id(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()
            
            # Create chunks from full text
            text_chunks = self._create_text_chunks(full_text)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{i}",
                    content=chunk_text,
                    source_document=file_path,
                    chunk_index=i,
                    metadata={
                        "document_type": "txt",
                        "file_size": os.path.getsize(file_path),
                        "processing_method": "direct_read"
                    }
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {str(e)}")
            raise
        
        return chunks
    
    def _create_text_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundaries first
                sentence_break = text.rfind('.', start, end)
                if sentence_break > start:
                    end = sentence_break + 1
                else:
                    # Fall back to word boundaries
                    word_break = text.rfind(' ', start, end)
                    if word_break > start:
                        end = word_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                break
        
        return chunks
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique ID for a document based on its path and content."""
        # Use file path and modification time for ID generation
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_document_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        stat = os.stat(file_path)
        file_extension = Path(file_path).suffix.lower()
        
        # Determine document type
        doc_type_map = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.txt': DocumentType.TXT
        }
        
        doc_type = doc_type_map.get(file_extension, DocumentType.TXT)
        
        # Get page count for PDFs
        page_count = None
        if file_extension == '.pdf' and PyPDF2:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    page_count = len(pdf_reader.pages)
            except Exception:
                pass
        
        return DocumentMetadata(
            document_id=self._generate_document_id(file_path),
            filename=Path(file_path).name,
            file_type=doc_type,
            file_size=stat.st_size,
            page_count=page_count
        )
