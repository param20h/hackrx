"""
FastAPI application for the Document Intelligence System.
Implements the HackRX competition API requirements.
"""

import logging
from typing import List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from ..models import HackrxRunRequest, HackrxRunResponse
from ..core.hackrx_service import HackrxService

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

# Initialize the HackRX service
hackrx_service = HackrxService()


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
    
    Args:
        request: HackrX request with document URL and questions
        token: Bearer token for authentication
        
    Returns:
        HackrX response with answers to the questions
    """
    try:
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        
        # Process the request using the HackRX service
        response = await hackrx_service.process_request(request)
        
        logger.info(f"Successfully processed HackRX request")
        return response
        
    except Exception as e:
        logger.error(f"Error processing HackRX request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )


@app.get("/examples")
async def get_example_queries():
    """
    Get example queries for testing the system.
    
    Returns:
        List of example queries
    """
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
            },
            {
                "query": "What is the waiting period for cataract surgery?",
                "description": "Specific procedure waiting period"
            },
            {
                "query": "Are medical expenses for organ donors covered?",
                "description": "Organ donor coverage inquiry"
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
    from ..core.config import get_settings
    
    settings = get_settings()
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port,
        log_level="info"
    )
