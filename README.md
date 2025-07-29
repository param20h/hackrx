# Document Intelligence System

A comprehensive LLM-powered system that processes natural language queries and retrieves relevant information from large unstructured documents such as policy documents, contracts, and emails.

## 🎯 Overview

The Document Intelligence System uses Large Language Models (LLMs) to:

1. **Parse and structure natural language queries** to identify key details such as age, procedure, location, and policy duration
2. **Search and retrieve relevant clauses** from documents using semantic understanding rather than simple keyword matching
3. **Evaluate retrieved information** to determine decisions such as approval status or payout amounts
4. **Return structured JSON responses** with decisions, amounts, and justifications mapped to specific document clauses

## 🚀 Features

- **Multi-format Document Processing**: Supports PDF, DOCX, and TXT files
- **Semantic Search**: Uses embeddings for intelligent document retrieval
- **Entity Extraction**: Identifies age, gender, medical procedures, locations, and policy details
- **Decision Engine**: Makes informed decisions based on policy clauses
- **REST API**: FastAPI-based web service for easy integration
- **Structured Responses**: JSON outputs with decisions, amounts, and clause references
- **Real-time Processing**: Handles queries in seconds with confidence scoring

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, FastAPI
- **LLM Integration**: OpenAI GPT models, LangChain
- **Vector Database**: ChromaDB for semantic search
- **Document Processing**: PyPDF2, python-docx
- **Embeddings**: OpenAI embeddings / Sentence Transformers
- **API**: FastAPI with automatic OpenAPI documentation

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM and embeddings)
- 4GB+ RAM (for local embedding models)

## 🔧 Installation

1. **Clone or download the project:**
   ```bash
   cd hackrx
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\\Scripts\\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   copy .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Quick Start

### 1. Run the Demo

```bash
python demo.py
```

This will:
- Initialize the system
- Process sample documents
- Run test queries
- Show example API usage

### 2. Start the Web API

```bash
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` for the interactive API documentation.

### 3. Command Line Usage

```bash
# Add documents
python src/main.py --add-docs ./data/documents

# Process a query
python src/main.py --query "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"

# Check system status
python src/main.py --status
```

## 📝 Usage Examples

### Sample Query
```
"46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
```

### Expected Response
```json
{
  "decision": {
    "decision_type": "APPROVED",
    "amount": 150000,
    "currency": "INR",
    "confidence": 0.85,
    "reasoning": "Knee surgery is covered under the policy. Patient meets age requirements. However, the 3-month policy duration may be subject to waiting period conditions."
  },
  "supporting_clauses": [
    {
      "clause_id": "coverage_knee_surgery",
      "clause_text": "Knee surgery including arthroscopy, meniscus repair, and knee replacement",
      "source_document": "health_insurance_policy.txt",
      "influence_weight": 0.92
    }
  ]
}
```

### API Examples

**Process Query:**
```bash
curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "46M knee surgery Pune 3-month policy"}'
```

**Upload Documents:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \\
     -F "files=@policy.pdf" \\
     -F "files=@contract.docx"
```

## 🏗️ Architecture

```
Document Intelligence System
├── Document Processing
│   ├── PDF/DOCX/TXT parsing
│   ├── Text chunking
│   └── Metadata extraction
├── Query Processing
│   ├── Entity extraction
│   ├── Intent classification
│   └── Query structuring
├── Semantic Search
│   ├── Embedding generation
│   ├── Vector similarity search
│   └── Relevance scoring
├── Decision Engine
│   ├── LLM-based analysis
│   ├── Rule-based fallback
│   └── Confidence scoring
└── API Layer
    ├── REST endpoints
    ├── File upload handling
    └── Response formatting
```

## 📂 Project Structure

```
hackrx/
├── src/
│   ├── core/                 # Core processing modules
│   │   ├── config.py         # Configuration management
│   │   ├── document_processor.py
│   │   ├── query_processor.py
│   │   ├── semantic_search.py
│   │   └── decision_engine.py
│   ├── models/               # Data models and schemas
│   │   └── schemas.py
│   ├── api/                  # FastAPI web service
│   │   └── app.py
│   ├── utils/                # Utility functions
│   │   └── helpers.py
│   └── main.py               # Main orchestrator
├── data/
│   └── documents/            # Sample documents
├── tests/                    # Test files
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md
```

## 🔧 Configuration

Key configuration options in `.env`:

```env
# LLM Settings
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Supported Use Cases

### Insurance Claims
- Coverage verification
- Claim amount estimation
- Policy term checking
- Waiting period analysis

### Contract Analysis
- Clause extraction
- Compliance checking
- Term interpretation
- Risk assessment

### Legal Documents
- Policy interpretation
- Regulation compliance
- Document comparison
- Decision support

## 🎯 Sample Queries

The system handles various query formats:

```
✅ "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
✅ "35F cataract surgery Mumbai 2 year policy"
✅ "Is hip replacement covered for 65-year-old in Delhi?"
✅ "Claim for appendectomy, patient age 28, policy 1 year old"
✅ "What is the coverage amount for bypass surgery in Chennai?"
```

## 🔍 Entity Recognition

The system extracts:
- **Age**: "46", "35-year-old", "28yo"
- **Gender**: "male", "female", "M", "F"
- **Medical Procedures**: "knee surgery", "cataract surgery", "appendectomy"
- **Locations**: Indian cities (Mumbai, Delhi, Pune, etc.)
- **Policy Duration**: "3-month policy", "2 year old"
- **Amounts**: Monetary values in various formats

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   ```
   Error: OPENAI_API_KEY is required but not set
   ```
   Solution: Add your API key to `.env` file

2. **Dependencies Missing**
   ```
   Import "package" could not be resolved
   ```
   Solution: Run `pip install -r requirements.txt`

3. **No Documents Found**
   ```
   No relevant clauses found
   ```
   Solution: Add documents to `data/documents/` directory

4. **Port Already in Use**
   ```
   Port 8000 is already in use
   ```
   Solution: Change `API_PORT` in `.env` or stop other services

### Performance Tips

- Use GPU for faster embedding generation
- Limit document chunk size for memory efficiency
- Configure `TOP_K_RESULTS` based on needs
- Use local embedding models to reduce API costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Run the demo script for examples
- Check logs for detailed error information

## 🚀 Next Steps

- Add support for more document formats
- Implement fine-tuned models for domain-specific tasks
- Add multi-language support
- Integrate with external insurance systems
- Add real-time collaboration features

---

Built for HackRX - Document Intelligence Challenge
