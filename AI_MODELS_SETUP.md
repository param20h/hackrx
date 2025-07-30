# AI Models Setup Guide

This guide helps you set up different AI models for the Document Intelligence System.

## 🤖 Supported AI Models

### 1. Google Gemini (Default) ✅
**Status**: Currently configured and working
**Advantages**: Fast, cost-effective, good for document analysis
**Setup**: Already done!

```bash
# Already in .env
GEMINI_API_KEY=AIzaSyCNG7todahvnjMQydvwm068gamA-CdkmDE
```

### 2. Ollama (Local/Self-hosted) 🏠
**Advantages**: 
- ✅ **FREE** - No API costs
- ✅ **Private** - Data stays local
- ✅ **No rate limits**
- ✅ **Works offline**

**Setup Ollama**:
```bash
# Install Ollama (Windows)
# Download from: https://ollama.com/download

# Or using winget
winget install Ollama.Ollama

# Start Ollama service
ollama serve

# Install a model (choose one)
ollama pull llama2        # 7B model (4GB RAM)
ollama pull llama2:13b    # 13B model (8GB RAM) 
ollama pull mistral       # 7B model (4GB RAM)
ollama pull codellama     # Good for code/documents

# Add to .env
OLLAMA_BASE_URL=http://localhost:11434
```

**Install Python client**:
```bash
pip install ollama
```

### 3. OpenAI GPT 💰
**Advantages**: Very high quality, well-tested
**Disadvantages**: Costs money per request

**Setup**:
```bash
pip install openai

# Add to .env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Anthropic Claude 💰
**Advantages**: Excellent for analysis, good safety
**Disadvantages**: Costs money per request

**Setup**:
```bash
pip install anthropic

# Add to .env  
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 🔧 Quick Setup for Ollama (Recommended)

Ollama is perfect for your use case because:
- **Free forever**
- **Works offline** 
- **No data privacy concerns**
- **Good performance for document Q&A**

### Step-by-step Ollama setup:

1. **Install Ollama**:
   ```bash
   # Download and install from https://ollama.com/download
   # Or use package manager
   winget install Ollama.Ollama
   ```

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Pull a model** (choose based on your RAM):
   ```bash
   # For 8GB+ RAM (recommended)
   ollama pull llama2:7b
   
   # For 16GB+ RAM (better quality)
   ollama pull llama2:13b
   
   # For document analysis (specialized)
   ollama pull mistral
   ```

4. **Install Python client**:
   ```bash
   pip install ollama
   ```

5. **Update .env**:
   ```bash
   OLLAMA_BASE_URL=http://localhost:11434
   ```

6. **Test with enhanced API**:
   ```bash
   python enhanced_api.py
   ```

## 🚀 Usage Examples

### Using different models in requests:

**Gemini (current)**:
```json
{
  "documents": "http://localhost:8001/BAJHLIP23020V012223.pdf",
  "questions": ["What is the policy number?"],
  "ai_config": {
    "model_type": "gemini",
    "model_name": "gemini-1.5-flash-8b"
  }
}
```

**Ollama (local)**:
```json
{
  "documents": "http://localhost:8001/BAJHLIP23020V012223.pdf", 
  "questions": ["What is the policy number?"],
  "ai_config": {
    "model_type": "ollama",
    "model_name": "llama2"
  }
}
```

**OpenAI**:
```json
{
  "documents": "http://localhost:8001/BAJHLIP23020V012223.pdf",
  "questions": ["What is the policy number?"],
  "ai_config": {
    "model_type": "openai", 
    "model_name": "gpt-3.5-turbo"
  }
}
```

## 📊 Model Comparison

| Model | Cost | Privacy | Quality | Speed | Offline |
|-------|------|---------|---------|-------|---------|
| **Gemini** | Low | Cloud | Good | Fast | ❌ |
| **Ollama** | FREE | Local | Good | Medium | ✅ |
| **OpenAI** | High | Cloud | Excellent | Fast | ❌ |
| **Claude** | High | Cloud | Excellent | Medium | ❌ |

## 🎯 Recommendation for HackRX

**For competition/demo**: Stick with **Gemini** (already working)
**For production/cost savings**: Use **Ollama** (free, local)
**For highest quality**: Use **OpenAI GPT-4**

## 🔍 Testing Your Setup

1. Start the enhanced API:
   ```bash
   python enhanced_api.py
   ```

2. Check available models:
   ```bash
   curl http://localhost:8000/models
   ```

3. Test with your documents:
   ```bash
   python test_local_docs.py
   ```
