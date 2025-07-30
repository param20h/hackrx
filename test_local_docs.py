"""
Test script for HackRX API using local test documents.
"""

import requests
import json
import os
from pathlib import Path

# API configuration
API_URL = "http://localhost:8000"
BEARER_TOKEN = "3d2f575e504dc0a556592a02b475556bf406c5b03f06b1d789b7f1f0ccf45730"

def test_with_local_docs():
    """Test the API using local test documents by serving them locally."""
    
    # First, let's test with a simple HTTP server for local files
    testdocs_path = Path("testdocs")
    
    if not testdocs_path.exists():
        print("❌ testdocs folder not found!")
        return
    
    pdf_files = list(testdocs_path.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in testdocs folder!")
        return
    
    print(f"📁 Found {len(pdf_files)} test documents:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # For local testing, we need to serve files via HTTP
    # Let's use a simple approach with file:// URLs or upload capability
    
    # Test with insurance-related questions
    insurance_questions = [
        "What is the policy number?",
        "What is the sum insured amount?",
        "What is the premium amount?",
        "What are the coverage details?",
        "What is the policy duration?",
        "Are there any waiting periods mentioned?",
        "What are the exclusions listed?",
        "Who is the policyholder?"
    ]
    
    print("\\n🧪 Insurance Policy Questions:")
    for i, q in enumerate(insurance_questions, 1):
        print(f"  {i}. {q}")
    
    # Since we can't directly test with local files without a local server,
    # let's create a function to upload and test
    return pdf_files, insurance_questions

def start_local_file_server():
    """Start a simple HTTP server to serve test documents."""
    import http.server
    import socketserver
    import threading
    import time
    
    PORT = 8001
    DIRECTORY = "testdocs"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def run_server():
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"📡 Serving testdocs at http://localhost:{PORT}")
            httpd.serve_forever()
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Give server time to start
    
    return f"http://localhost:{PORT}"

def test_api_health():
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_hackrx_with_document(doc_url, questions):
    """Test the HackRX endpoint with a specific document."""
    
    test_request = {
        "documents": doc_url,
        "questions": questions[:3]  # Test with first 3 questions
    }
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print(f"\\n🔄 Testing with document: {doc_url}")
    print(f"📝 Questions: {test_request['questions']}")
    
    try:
        response = requests.post(
            f"{API_URL}/hackrx/run",
            headers=headers,
            json=test_request,
            timeout=120  # Increased timeout for local processing
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Answers received:")
            
            for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers']), 1):
                print(f"\\n  Q{i}: {question}")
                print(f"  A{i}: {answer}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting HackRX API Tests with Local Documents\\n")
    
    # Check API health
    if not test_api_health():
        print("\\n💡 To start the API server, run:")
        print("   cd a:\\hackrx")
        print("   $env:GEMINI_API_KEY='your_api_key'")
        print("   python simple_api.py")
        return
    
    # Get test documents and questions
    pdf_files, questions = test_with_local_docs()
    
    # Start local file server for testing
    try:
        base_url = start_local_file_server()
        
        # Test with first document
        if pdf_files:
            doc_url = f"{base_url}/{pdf_files[0].name}"
            success = test_hackrx_with_document(doc_url, questions)
            
            if success:
                print("\\n🎉 Local document testing successful!")
            else:
                print("\\n😞 Local document testing failed!")
        
    except Exception as e:
        print(f"❌ Error setting up local server: {e}")
        print("\\n💡 Alternative: Upload your PDFs to a public URL and test with those")

if __name__ == "__main__":
    main()
