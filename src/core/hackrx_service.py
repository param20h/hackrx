"""
HackRX-specific service for processing documents and answering questions.
"""

import logging
import tempfile
import os
from typing import List, Dict, Any
import requests

try:
    import PyMuPDF as fitz
except ImportError:
    fitz = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..core.config import get_settings
from ..core.document_processor import DocumentProcessor
from ..core.semantic_search import SemanticSearch
from ..models import HackrxRunRequest, HackrxRunResponse

logger = logging.getLogger(__name__)


class HackrxService:
    """Service for handling HackRX competition requirements."""
    
    def __init__(self):
        """Initialize the HackRX service."""
        self.settings = get_settings()
        self.doc_processor = DocumentProcessor()
        self.semantic_search = SemanticSearch()
        
        # Initialize Gemini client
        if genai and self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.gemini_client = genai
        else:
            self.gemini_client = None
            logger.warning("Gemini API not configured")
    
    async def process_request(self, request: HackrxRunRequest) -> HackrxRunResponse:
        """
        Process a HackRX request: download document, extract text, answer questions.
        
        Args:
            request: HackRX request with document URL and questions
            
        Returns:
            HackRX response with answers
        """
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        
        # Download and extract document text
        document_text = await self._download_and_extract_document(request.documents)
        
        # Answer each question using the document text
        answers = []
        for question in request.questions:
            answer = await self._answer_question(question, document_text)
            answers.append(answer)
        
        return HackrxRunResponse(answers=answers)
    
    async def _download_and_extract_document(self, url: str) -> str:
        """
        Download document from URL and extract text.
        
        Args:
            url: URL to the document
            
        Returns:
            Extracted text from the document
        """
        try:
            # Download the document
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            # Extract text based on file type
            if temp_path.endswith('.pdf'):
                text = self._extract_pdf_text(temp_path)
            else:
                # For other formats, try document processor
                chunks = self.doc_processor.process_document(temp_path)
                text = "\\n\\n".join([chunk.content for chunk in chunks])
            
            # Clean up
            os.unlink(temp_path)
            
            logger.info(f"Extracted {len(text)} characters from document")
            return text
            
        except Exception as e:
            logger.error(f"Error downloading/extracting document: {e}")
            raise Exception(f"Failed to process document: {str(e)}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        if not fitz:
            raise ImportError("PyMuPDF is required for PDF processing")
        
        try:
            doc = fitz.open(file_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"Page {page_num + 1}:\\n{text}")
            
            doc.close()
            return "\\n\\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise Exception(f"Failed to extract PDF text: {str(e)}")
    
    async def _answer_question(self, question: str, document_text: str) -> str:
        """
        Answer a question using the document text and Gemini.
        
        Args:
            question: Question to answer
            document_text: Full document text
            
        Returns:
            Answer to the question
        """
        try:
            if not self.gemini_client:
                return "Gemini API not available for processing questions"
            
            # Create a focused prompt for question answering
            prompt = f"""
You are an expert insurance policy analyst. Based on the provided policy document, answer the following question accurately and concisely.

POLICY DOCUMENT:
{document_text[:8000]}  # Limit to first 8000 chars to avoid token limits

QUESTION: {question}

INSTRUCTIONS:
1. Read the policy document carefully
2. Find the specific information that answers the question
3. Provide a clear, accurate, and concise answer
4. If the information is not available in the document, say "Information not available in the provided document"
5. Use exact terms and numbers from the policy when possible
6. Keep the answer focused and direct

ANSWER:
"""
            
            # Use Gemini to generate answer
            model = self.gemini_client.GenerativeModel(self.settings.model_name)
            response = model.generate_content(prompt)
            
            answer = response.text.strip()
            
            # Clean up the answer
            if answer.startswith("ANSWER:"):
                answer = answer[7:].strip()
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {e}")
            return f"Error processing question: {str(e)}"
    
    def _chunk_document_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split document text into chunks for processing.
        
        Args:
            text: Full document text
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _search_relevant_chunks(self, question: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """
        Search for most relevant chunks to answer the question.
        
        Args:
            question: Question to search for
            chunks: List of document chunks
            top_k: Number of top chunks to return
            
        Returns:
            List of most relevant chunks
        """
        # Simple keyword-based search (can be enhanced with embeddings)
        question_lower = question.lower()
        scored_chunks = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = 0
            
            # Count keyword matches
            for word in question_lower.split():
                if len(word) > 3:  # Only consider meaningful words
                    score += chunk_lower.count(word)
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
