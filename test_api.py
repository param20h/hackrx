"""
Test script for the HackRX API endpoint.
"""

import requests
import json

# API configuration
API_URL = "http://localhost:8000"
BEARER_TOKEN = "3d2f575e504dc0a556592a02b475556bf406c5b03f06b1d789b7f1f0ccf45730"

def test_hackrx_endpoint():
    """Test the /hackrx/run endpoint with a sample request."""
    
    # Test data - using a working sample PDF URL
    test_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",  # Working sample PDF
        "questions": [
            "What is this document about?",
            "What are the main topics covered?",
            "What type of document is this?"
        ]
    }
    
    # Headers with bearer token
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("Testing HackRX API endpoint...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        # Make the API call
        response = requests.post(
            f"{API_URL}/hackrx/run",
            headers=headers,
            json=test_request,
            timeout=60
        )
        
        print(f"\\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\\nResponse:")
            print(json.dumps(result, indent=2))
            
            print("\\nAnswers:")
            for i, answer in enumerate(result["answers"], 1):
                print(f"{i}. Q: {test_request['questions'][i-1]}")
                print(f"   A: {answer}\\n")
        else:
            print(f"Error: {response.text}")
            
    except requests.RequestException as e:
        print(f"Request failed: {e}")

def test_health_endpoint():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Health check: {response.status_code} - {response.json()}")
    except requests.RequestException as e:
        print(f"Health check failed: {e}")

if __name__ == "__main__":
    # Test health endpoint first
    test_health_endpoint()
    print("\\n" + "="*50 + "\\n")
    
    # Test the main endpoint
    test_hackrx_endpoint()
