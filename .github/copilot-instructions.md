<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Document Intelligence System - Copilot Instructions

This is a Python-based Document Intelligence System that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents.

## Project Structure
- `src/` - Main source code
- `src/core/` - Core processing modules
- `src/models/` - Data models and schemas
- `src/api/` - FastAPI web service
- `src/utils/` - Utility functions
- `data/` - Document storage and database
- `tests/` - Test files

## Key Components
1. **Document Processing**: Parse PDFs, Word docs, and emails
2. **Query Processing**: Extract structured information from natural language
3. **Semantic Search**: Use embeddings for intelligent document retrieval
4. **LLM Integration**: OpenAI GPT models for decision making
5. **Response Generation**: Structured JSON outputs with justifications

## Coding Guidelines
- Use type hints and Pydantic models for data validation
- Follow async/await patterns for API endpoints
- Implement proper error handling and logging
- Use dependency injection for configuration
- Write comprehensive docstrings and comments
- Maintain separation of concerns between modules

## Domain Context
- Insurance claim processing and policy evaluation
- Contract analysis and compliance checking
- Legal document review and decision support
- HR policy interpretation

## Sample Use Cases
- "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
- Should return: Decision, Amount, Justification with clause references
