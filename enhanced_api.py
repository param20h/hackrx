"""
Enhanced HackRX API server with multiple AI model support.
Supports: Google Gemini, OpenAI GPT, Ollama, Anthropic Claude
"""

import logging
import tempfile
import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel, Field
from enum import Enum

# Add the src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# AI Model imports
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Configuration Models
class AIModel(str, Enum):
    """Supported AI models."""
    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"

class AIConfig(BaseModel):
    """AI model configuration."""
    model_type: AIModel = AIModel.GEMINI
    model_name: str = "gemini-1.5-flash-8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama
    max_tokens: int = 4096
    temperature: float = 0.1

# Request/Response Models
class HackrxRunRequest(BaseModel):
    """Request model for HackRX run endpoint."""
    documents: str  # URL to document
    questions: List[str]  # List of questions to answer
    ai_config: Optional[AIConfig] = None  # Optional AI configuration override

class HackrxRunResponse(BaseModel):
    """Response model for HackRX run endpoint."""
    answers: List[str]  # List of answers corresponding to questions
    model_used: str  # Which AI model was used
    processing_time: Optional[float] = None  # Processing time in seconds

class ModelStatus(BaseModel):
    """Status of available AI models."""
    available_models: Dict[str, bool]
    current_model: str
    model_details: Dict[str, Any]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Document Intelligence System",
    description="Multi-AI-model document processing and Q&A system for HackRX",
    version="2.0.0"
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

# AI Model Management
class AIModelManager:
    """Manages multiple AI models and their configurations."""
    
    def __init__(self):
        self.models = {}
        self.current_model = AIModel.GEMINI
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize available AI models based on environment variables."""
        
        # Initialize Gemini
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if genai and gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                self.models[AIModel.GEMINI] = {
                    "client": genai,
                    "available": True,
                    "model_name": "gemini-1.5-flash-8b"
                }
                logger.info("✅ Gemini API initialized")
            except Exception as e:
                logger.error(f"❌ Gemini initialization failed: {e}")
                self.models[AIModel.GEMINI] = {"available": False, "error": str(e)}
        else:
            self.models[AIModel.GEMINI] = {"available": False, "error": "API key or library missing"}
        
        # Initialize OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai and openai_key:
            try:
                openai.api_key = openai_key
                self.models[AIModel.OPENAI] = {
                    "client": openai,
                    "available": True,
                    "model_name": "gpt-3.5-turbo"
                }
                logger.info("✅ OpenAI API initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
                self.models[AIModel.OPENAI] = {"available": False, "error": str(e)}
        else:
            self.models[AIModel.OPENAI] = {"available": False, "error": "API key or library missing"}
        
        # Initialize Ollama (local)
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if ollama:
            try:
                # Test if Ollama is running
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    self.models[AIModel.OLLAMA] = {
                        "client": ollama,
                        "available": True,
                        "base_url": ollama_url,
                        "model_name": "llama2"  # Default model
                    }
                    logger.info("✅ Ollama API initialized")
                else:
                    self.models[AIModel.OLLAMA] = {"available": False, "error": "Ollama not running"}
            except Exception as e:
                logger.error(f"❌ Ollama initialization failed: {e}")
                self.models[AIModel.OLLAMA] = {"available": False, "error": str(e)}
        else:
            self.models[AIModel.OLLAMA] = {"available": False, "error": "Ollama library missing"}
        
        # Initialize Claude
        claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic and claude_key:
            try:
                client = anthropic.Anthropic(api_key=claude_key)
                self.models[AIModel.CLAUDE] = {
                    "client": client,
                    "available": True,
                    "model_name": "claude-3-haiku-20240307"
                }
                logger.info("✅ Claude API initialized")
            except Exception as e:
                logger.error(f"❌ Claude initialization failed: {e}")
                self.models[AIModel.CLAUDE] = {"available": False, "error": str(e)}
        else:
            self.models[AIModel.CLAUDE] = {"available": False, "error": "API key or library missing"}
        
        # Set current model to first available
        for model_type in [AIModel.GEMINI, AIModel.OPENAI, AIModel.OLLAMA, AIModel.CLAUDE]:
            if self.models.get(model_type, {}).get("available", False):
                self.current_model = model_type
                break
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get status of all available models."""
        return {model.value: config.get("available", False) for model, config in self.models.items()}
    
    def answer_question(self, question: str, document_text: str, config: Optional[AIConfig] = None) -> str:
        """Answer a question using the specified or current AI model."""
        
        model_type = config.model_type if config else self.current_model
        
        if model_type not in self.models or not self.models[model_type].get("available"):
            return f"❌ {model_type.value} model not available"
        
        try:
            if model_type == AIModel.GEMINI:
                return self._answer_with_gemini(question, document_text, config)
            elif model_type == AIModel.OPENAI:
                return self._answer_with_openai(question, document_text, config)
            elif model_type == AIModel.OLLAMA:
                return self._answer_with_ollama(question, document_text, config)
            elif model_type == AIModel.CLAUDE:
                return self._answer_with_claude(question, document_text, config)
            else:
                return f"❌ Unsupported model type: {model_type}"
                
        except Exception as e:
            logger.error(f"Error with {model_type.value}: {e}")
            return f"❌ Error processing with {model_type.value}: {str(e)}"
    
    def _create_prompt(self, question: str, document_text: str) -> str:
        """Create a standardized prompt for all models."""
        return f"""You are an expert document analyst. Based on the provided document, answer the following question accurately and concisely.

DOCUMENT CONTENT:
{document_text[:8000]}

QUESTION: {question}

INSTRUCTIONS:
1. Read the document carefully
2. Find specific information that answers the question
3. Provide a clear, accurate, and concise answer
4. If information is not available, say "Information not available in the provided document"
5. Use exact terms and numbers from the document when possible
6. Keep the answer focused and direct

ANSWER:"""
    
    def _answer_with_gemini(self, question: str, document_text: str, config: Optional[AIConfig]) -> str:
        """Answer using Google Gemini."""
        model_name = config.model_name if config else self.models[AIModel.GEMINI]["model_name"]
        model = genai.GenerativeModel(model_name)
        
        prompt = self._create_prompt(question, document_text)
        response = model.generate_content(prompt)
        
        answer = response.text.strip()
        if answer.startswith("ANSWER:"):
            answer = answer[7:].strip()
        
        return answer
    
    def _answer_with_openai(self, question: str, document_text: str, config: Optional[AIConfig]) -> str:
        """Answer using OpenAI GPT."""
        model_name = config.model_name if config else self.models[AIModel.OPENAI]["model_name"]
        
        prompt = self._create_prompt(question, document_text)
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens if config else 1000,
            temperature=config.temperature if config else 0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def _answer_with_ollama(self, question: str, document_text: str, config: Optional[AIConfig]) -> str:
        """Answer using Ollama (local)."""
        model_name = config.model_name if config else self.models[AIModel.OLLAMA]["model_name"]
        
        prompt = self._create_prompt(question, document_text)
        
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        
        return response['message']['content'].strip()
    
    def _answer_with_claude(self, question: str, document_text: str, config: Optional[AIConfig]) -> str:
        """Answer using Anthropic Claude."""
        client = self.models[AIModel.CLAUDE]["client"]
        model_name = config.model_name if config else self.models[AIModel.CLAUDE]["model_name"]
        
        prompt = self._create_prompt(question, document_text)
        
        response = client.messages.create(
            model=model_name,
            max_tokens=config.max_tokens if config else 1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()

# Initialize AI model manager
ai_manager = AIModelManager()

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

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Enhanced Document Intelligence System API", 
        "status": "active",
        "version": "2.0.0",
        "available_models": ai_manager.get_available_models(),
        "current_model": ai_manager.current_model.value
    }

@app.get("/models", response_model=ModelStatus)
async def get_model_status():
    """Get status of all available AI models."""
    return ModelStatus(
        available_models=ai_manager.get_available_models(),
        current_model=ai_manager.current_model.value,
        model_details={
            model.value: config for model, config in ai_manager.models.items()
            if config.get("available", False)
        }
    )

@app.post("/hackrx/run", response_model=HackrxRunResponse)
async def hackrx_run(
    request: HackrxRunRequest,
    token: str = Depends(verify_token)
):
    """
    HackRX competition endpoint with multi-AI-model support.
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        
        # Download and extract document text
        document_text = download_and_extract_document(request.documents)
        
        # Determine which model to use
        model_type = request.ai_config.model_type if request.ai_config else ai_manager.current_model
        
        # Answer each question using the specified AI model
        answers = []
        for question in request.questions:
            answer = ai_manager.answer_question(question, document_text, request.ai_config)
            answers.append(answer)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully processed HackRX request in {processing_time:.2f}s using {model_type.value}")
        
        return HackrxRunResponse(
            answers=answers,
            model_used=model_type.value,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing HackRX request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )

@app.get("/examples")
async def get_example_queries():
    """Get example queries with multi-model options."""
    return {
        "example_queries": [
            {
                "query": "What is the policy number?",
                "description": "Extract policy identification number"
            },
            {
                "query": "What is the sum insured amount?",
                "description": "Find the coverage amount"
            },
            {
                "query": "What are the waiting periods?",
                "description": "Identify any waiting period clauses"
            }
        ],
        "sample_request": {
            "documents": "https://example.com/policy.pdf",
            "questions": [
                "What is the policy number?",
                "What is the sum insured amount?"
            ],
            "ai_config": {
                "model_type": "gemini",
                "model_name": "gemini-1.5-flash-8b",
                "temperature": 0.1
            }
        },
        "available_models": ai_manager.get_available_models()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
