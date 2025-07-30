"""
Simple HackRX API server for document Q&A processing.
"""

import logging
import tempfile
import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel

# Add the src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Models for HackRX API
class HackrxRunRequest(BaseModel):
    """Request model for HackRX run endpoint."""
    documents: str  # URL to document
    questions: List[str]  # List of questions to answer

class HackrxRunResponse(BaseModel):
    """Response model for HackRX run endpoint."""
    answers: List[str]  # List of answers corresponding to questions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Intelligence System",
    description="LLM-powered document processing and Q&A system for HackRX",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Expected bearer token for HackRX competition
EXPECTED_TOKEN = "3d2f575e504dc0a556592a02b475556bf406c5b03f06b1d789b7f1f0ccf45730"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token."""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Initialize Gemini client
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if genai and gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    logger.info("Gemini API initialized")
else:
    logger.warning("Gemini API not configured - add GEMINI_API_KEY to environment")

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
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

def download_and_extract_document(url: str) -> str:
    """Download document from URL and extract text."""
    try:
        # Download the document
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Extract text
        text = extract_pdf_text(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        logger.info(f"Extracted {len(text)} characters from document")
        return text
        
    except Exception as e:
        logger.error(f"Error downloading/extracting document: {e}")
        raise Exception(f"Failed to process document: {str(e)}")

def answer_question(question: str, document_text: str) -> str:
    """Answer a question using the document text and Gemini."""
    try:
        if not genai or not gemini_api_key:
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
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
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

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Document Intelligence System API", "status": "active"}

@app.post("/hackrx/run", response_model=HackrxRunResponse)
async def hackrx_run(
    request: HackrxRunRequest,
    token: str = Depends(verify_token)
):
    """
    HackRX competition endpoint for document Q&A processing.
    
    Processes documents from URLs and answers questions about their content.
    Requires bearer token authentication.
    """
    try:
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        
        # Download and extract document text
        document_text = download_and_extract_document(request.documents)
        
        # Answer each question using the document text
        answers = []
        for question in request.questions:
            answer = answer_question(question, document_text)
            answers.append(answer)
        
        logger.info(f"Successfully processed HackRX request")
        return HackrxRunResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing HackRX request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )

@app.get("/examples")
async def get_example_queries():
    """Get example queries for testing the system."""
    return {
        "example_queries": [
            {
                "query": "What is the grace period for premium payment?",
                "description": "Insurance policy grace period inquiry"
            },
            {
                "query": "What is the waiting period for pre-existing diseases?",
                "description": "Pre-existing condition waiting period"
            },
            {
                "query": "Does this policy cover maternity expenses?",
                "description": "Maternity coverage inquiry"
            }
        ],
        "sample_request": {
            "documents": "https://example.com/policy.pdf",
            "questions": [
                "What is the grace period for premium payment?",
                "What is the waiting period for pre-existing diseases?"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
