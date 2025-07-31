🧩 Objective Recap
Build a system that:

Accepts a natural language query

Extracts structured info (age, treatment, policy terms, etc.)

Searches unstructured insurance documents

Returns a decision with justification and relevant clauses in JSON format

📌 Step-by-Step Approach
1. Input Understanding & Parsing
✅ Goal: Extract structured data from plain text queries
Example Input:
"46-year-old male, knee surgery in Pune, 3-month-old insurance policy"

Use:

GPT-4 (or spaCy/transformer) for Named Entity Recognition (NER)

Define entities like:

Age

Gender

Procedure (treatment)

Location

Policy duration

Output (Structured Data):

json
Copy
Edit
{
  "age": 46,
  "gender": "male",
  "procedure": "knee surgery",
  "location": "Pune",
  "policy_duration_months": 3
}
2. Document Ingestion & Embedding
✅ Goal: Ingest policy PDFs and prepare them for semantic search
Steps:

Convert PDF to plain text using PyMuPDF / pdfminer.six

Split text into chunks (~300–500 tokens)

Use sentence-transformers (e.g., all-MiniLM-L6-v2) to generate embeddings

Store them in a vector database like:

FAISS

ChromaDB

Weaviate

3. Semantic Clause Retrieval
✅ Goal: Use the user query to fetch relevant clauses
Steps:

Convert structured input (or original query) into a semantic query

Use similarity search on vector DB to retrieve top matching document chunks

Use LLM to summarize relevant clauses

4. Rule Evaluation & Decision Engine
✅ Goal: Match retrieved info with policy logic
Example Logic:

✅ Age ≤ 65 → eligible

✅ Knee surgery → check if covered

✅ 3-month policy → check if waiting period applies

✅ Covered in India → Yes if within geographic scope

Implement:

LLM or rule-based logic (custom logic tree)

Create rules that check structured input + retrieved text

5. Decision & JSON Output Generation
✅ Goal: Return interpretable decision with traceability
Use LLM prompt to generate:

json
Copy
Edit
{
  "decision": "Approved",
  "amount": "Up to sum insured",
  "justification": "Knee surgery is covered under Section C, Part A, I-1. The patient is 46 years old, and the policy covers ages up to 65. The policy has been active for 3 months, which satisfies the condition for non-pre-existing claims."
}
6. Optional UI / API
Use FastAPI for backend REST API

Simple frontend with HTML/React to enter query & display result

Optionally allow document uploads via UI

⚙ Tools Summary
Task	Tools / Models
PDF Parsing	PyMuPDF, pdfminer.six
Embeddings	sentence-transformers/all-MiniLM-L6-v2
Vector Search	FAISS / ChromaDB
Query Understanding	OpenAI GPT-4 / spaCy NER
Semantic Search	LangChain or direct search
Decision Generation	GPT-4 (prompt with rules + context)
Output Format	JSON with explanation
Backend/API	FastAPI / Flask
Frontend (optional)	React / plain HTML

🧠 Prompting Strategy (LLM)
You can use a prompt template like:

"Given the extracted details: {age}, {procedure}, {location}, and the following policy clauses: {retrieved_chunks}, determine if the claim is admissible. Respond with decision, amount covered, and specific clauses that justify the answer."

🏁 Final Output Example
json
Copy
Edit
{
  "decision": "Approved",
  "amount": "Covered up to INR 3,750,000",
  "justification": "The claim is approved as knee surgery is covered under Section C, Part A, Clause I-1. The insured is 46 years old, which is within the policy coverage age range. The policy duration is 3 months, which meets the required period for surgery coverage."
}✅ No-Subscription (Free) Stack
You can build the core of the system without any paid services, using open-source tools and free-tier models:

🔹 Input Query Parsing (NER / Entity Extraction)
✅ Free: Use spaCy, transformers (like bert-base-cased), or flair

❌ LLMs like GPT-4 or Claude (from OpenAI or Anthropic) need subscriptions if you use their APIs

🔹 Document Parsing & Embedding
✅ Free Tools:

PyMuPDF or pdfminer.six for PDF to text

sentence-transformers/all-MiniLM-L6-v2 via Hugging Face

FAISS or ChromaDB for vector storage and search

These are all free and run locally (or on Google Colab if you need GPU).

🔹 Semantic Search
✅ Free with sentence-transformers + FAISS

✅ You can run basic semantic search and clause retrieval entirely on your machine

🔹 Decision Engine (LLM)
❌ GPT-4 or GPT-3.5 API:

Paid if used via OpenAI API

GPT-3.5-turbo: ~$0.0015 per 1K tokens

GPT-4: ~$0.03–$0.06 per 1K tokens

✅ Free Alternative:

Use open-source LLMs (like Mistral, LLama 3, or Phi-3) via:

Ollama (runs locally)

Hugging Face Transformers + Text Generation Pipelines

Tip: On limited hardware, use quantized LLMs (e.g., mistral-7b.Q4_K_M.gguf with Ollama)

🔹 Optional: Frontend & Backend
✅ All frontend/backend tools (FastAPI, Flask, React, HTML, etc.) are free

🔒 When You Might Need a Paid Subscription
Use Case	Paid Option Needed?
Using GPT-4 / Claude / Gemini for parsing or generation	✅ Yes — paid API or ChatGPT Plus
Deploying to cloud server (AWS/GCP)	✅ Yes, unless on free tier
Accessing Hugging Face Inference API	❌ Free for many models, but paid for high usage or premium models

🛠 Recommendation for Hackathon
If you're building a hackathon MVP:

Stick to free local tools

Use GPT-3.5 or Claude via free credits (some platforms offer trial credits)

Only use GPT-4 if you already have ChatGPT Plus or API access📅 Day 1: Core Setup — Parsing & Document Ingestion
🎯 Goals:
Parse user queries → structured JSON

Ingest policy documents → text + embeddings

✅ 1. Query Parser (Input Handler)
Tools: spaCy + custom rules

Install spaCy and load pretrained model:

bash
Copy
Edit
pip install spacy
python -m spacy download en_core_web_sm
Extract:

Age → regex or spaCy ENT

Gender → rule-based (male/female)

Procedure → use Noun Phrases

Location → spaCy GPE

Policy duration → regex

✅ Output format:

json
Copy
Edit
{
  "age": 46,
  "gender": "male",
  "procedure": "knee surgery",
  "location": "Pune",
  "policy_duration": "3 months"
}
✅ 2. Document Ingestion
Tools: PyMuPDF + sentence-transformers

Use PyMuPDF (or pdfminer.six) to extract plain text from PDF:

python
Copy
Edit
import fitz  # PyMuPDF
doc = fitz.open("policy.pdf")
text = "\n".join([page.get_text() for page in doc])
Split the text into chunks (~200–300 words)

✅ Save to chunks.txt or a list

✅ 3. Text Embedding
Tools: sentence-transformers + FAISS

Install:

bash
Copy
Edit
pip install sentence-transformers faiss-cpu
Use:

python
Copy
Edit
from sentence_transformers import SentenceTransformer
import faiss
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
Store in FAISS:

python
Copy
Edit
index = faiss.IndexFlatL2(384)  # 384 for MiniLM
index.add(embeddings)
🔁 Save index and chunks to disk

📅 Day 2: Clause Retrieval + Reasoning Engine
🎯 Goals:
Retrieve semantically similar clauses to the query

Match rules to make a decision

✅ 1. Semantic Search Engine
Steps:

Take original query

Convert to embedding

Use FAISS to search for top 3–5 matching chunks

python
Copy
Edit
query_vector = model.encode([query])
D, I = index.search(query_vector, 5)
top_chunks = [chunks[i] for i in I[0]]
✅ Output: relevant clauses from the policy

✅ 2. Rule-based Decision Engine
Basic Rules (Python logic):

python
Copy
Edit
decision = "Rejected"
justification = []

if age <= 65:
    justification.append("Age within limit")
else:
    justification.append("Age exceeds policy limit")

if "knee surgery" in " ".join(top_chunks).lower():
    justification.append("Knee surgery is covered")
    decision = "Approved"
else:
    justification.append("Procedure not clearly covered")
✅ 3. JSON Output Generator
python
Copy
Edit
response = {
    "decision": decision,
    "amount": "Up to Sum Insured" if decision == "Approved" else "N/A",
    "justification": justification,
    "source_clauses": top_chunks
}
📅 Day 3: Integrate + Polish + Test + Prepare Demo
🎯 Goals:
Build CLI or simple UI

Validate output on real queries

Prepare submission materials

✅ 1. Minimal Backend (Optional)
Use FastAPI if demo is required via API:

bash
Copy
Edit
pip install fastapi uvicorn
python
Copy
Edit
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/query")
async def process_query(data: dict):
    # Run parser, search, rules, return JSON
    return response
Run:

bash
Copy
Edit
uvicorn main:app --reload
✅ 2. Local UI or Colab Demo (Optional)
Build basic HTML page (or use Streamlit if GUI)

Input box → POST to FastAPI or call Python function

Display final JSON decision with styled formatting

✅ 3. Test Cases
Prepare at least 5 varied queries, such as:

“30-year-old woman, liver transplant, Mumbai, 2-month policy”

“62M, heart bypass, Pune, 1-year policy”

Record results for each to show accuracy

✅ 4. Submission Materials
README with system architecture

Short video demo (2–3 mins)

PPT slides (Problem → Solution → Architecture → Demo → Results)

🚀 Final Output Sample
json
Copy
Edit
{
  "decision": "Approved",
  "amount": "Up to INR 3,750,000",
  "justification": [
    "Age within covered range (<= 65)",
    "Knee surgery covered under Section C, Part A",
    "Policy active for 3 months"
  ],
  "source_clauses": [
    "If You are advised Hospitalization... for orthopedic implants...",
    "Knee surgery is listed under Day Care Procedures...",
    "...policy covers up to INR 3,750,000"
  ]
}
🎁 Bonus (Optional Enhancements)
Add confidence score from similarity match

Use LangChain with local LLM like Mistral via Ollama

Export JSON output to file or audit log Problem Statement Sections Overview
Understanding all components of the comprehensive problem statement

Problem Statement Sections:
1
Instructions Section
General contest instructions and guidelines

2
Problem Statement
Core challenge description and objectives

3
System Architecture
Technical system design and data flow

4
Evaluation Criteria
How your solution will be judged

5
API Documentation
Complete API specifications and endpoints

6
Submission Instructions
How to deploy and submit your solution
Building Your API Application
Guidelines for creating your solution and API structure

Application Development Guidelines:
Required API Structure:
POST Endpoint Required:

/hackrx/run
Authentication:

Authorization: Bearer <api_key>
Request Format:

POST /hackrx/run
Content-Type: application/json
Accept: application/json
Authorization: Bearer <api_key>

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
Response Format:

{
"answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}
Hosting Requirements:
Public URL
Your API must be publicly accessible

Example: https://your-domain.com/api/v1/hackrx/run

HTTPS Required
Secure connection mandatory for submission

Example: SSL certificate required

Response Time
API should respond within reasonable time

Example: < 30 seconds timeoutRecommended Tech Stack:
FastAPI (Backend)
Pinecone (Vector DB)
GPT-4 (LLM)
PostgreSQL (Database)
Deployment Platforms:
• Heroku
• Vercel
• Railway
• AWS/GCP/Azure
• Render
• DigitalOcean
• Netlify Functions
• Your own server
What to Submit in Submissions
Understanding what you need to provide when making a submission

Required Fields:
Webhook URL
Your deployed /hackrx/run endpoint

Example: https://myapp.herokuapp.com/api/v1/hackrx/run

Description (Optional)
Brief tech stack description

Example: FastAPI + GPT-4 + Pinecone

Evaluation Flow:
Platform sends test request
Your API processes & responds
Platform evaluates answers
Results displayed instantly
Request Format:
POST /hackrx/run
Authorization: Bearer <api_key>
{ "documents": "https://example.com/policy.pdf", "questions": ["Question 1", "Question 2", "..."] }
Expected Response:
{ "answers": [ "Answer to question 1", "Answer to question 2", "..." ] }
Pre-Submission Checklist
✓ API is live & accessible
✓ HTTPS enabled
✓ Handles POST requests
✓ Returns JSON response
✓ Response time < 30s
✓ Tested with sample dataStep 3: Making Submissions & Understanding Documents
1
Navigate to Submissions Page
Access the submissions section to submit your webhook and explore documents

How to get there:
Click "Submit" button from dashboard, OR
Sidebar → Competition → Submissions
URL: /submissions
Go to Submissions
Submissions Page Access
Submissions Page Access
Navigation paths to reach submissions page

Click to zoomWebhook Submission Process
Submit your AI model's webhook URL for evaluation

Webhook Submission Form:
Webhook URL (Required)
Your deployed /hackrx/run endpoint that will receive test requests

Example: https://your-api.com/api/v1/hackrx/run

Submission Notes (Optional)
Brief explanation of your tech stack and approach

Example: FastAPI + GPT-4 + Pinecone vector search with RAG

Submission Process:
Enter your /hackrx/run webhook URL
Add optional tech stack notes (500 char limit)
Review submission preview
Click "Run" to submit and start evaluation
Webhook Submission Form
Webhook Submission Form
Screenshot of the webhook URL input and notes section

Click to zoomWhat Happens During Evaluation:
Platform sends POST requests to your /hackrx/run endpoint
Each request includes documents and questions array
Your API should return success status and processing details
Platform evaluates responses and generates scores
Pre-Submission Checklist:
API Requirements:
• /hackrx/run endpoint is live
• HTTPS enabled and accessible
• Bearer token authentication ready
• Handles POST requests correctly
Response Format:
• Returns valid JSON response
• Includes success status field
• Contains processing information
• Response time under 30 secondsDocuments Section Deep Dive
Understand available and locked documents, and how they unlock

Available Documents
• Document name and size
• Last modified date
• "View Document" button
• Direct PDF access
Locked Documents
• Hidden document names
• "??? KB" file size
• "Requires Webhook" status
• Unlock after successful submission
Documents Grid
Documents Grid
Available vs locked documents display

Click to zoom

Document Stats at Bottom
1
Available
2
Locked
3
TotalStep 4: Understanding My Submissions & All Metrics
1
Navigate to My Submissions
Access your submission history and detailed evaluation results

How to access:
From Submissions page → "My Submissions" button (top-right), OR
Sidebar → Competition → My Submissions
URL: /submissions/all
Go to My Submissions
My Submissions Navigation
My Submissions Navigation
Path to access submission history and results

Click to zoom
Main Submission Metrics Explained
Understand every metric displayed in your submission overview

Key Performance Indicators:
Overall Score
Your total performance percentage across all questions and documents

Example: 85% (Calculated from correct answers)

Accuracy Ratio
Number of correct answers out of total questions asked

Example: 17/20 (85% accuracy rate)

Average Response Time
Mean time taken to respond to API requests

Example: 2.5s (Lower is better)

Metrics Dashboard
Metrics Dashboard
Main submission metrics display with color coding

Click to zoom

Score Color Coding:
90%+ Excellent
75-89% Good
60-74% Fair
Below 60% Needs Improvement
Submission Details & Metadata
Additional information about your submission

Submission Timestamp
When your webhook was submitted and evaluated

Example: July 22, 2025, 2:00 PM

API Endpoint
The webhook URL that was tested

Example: https://your-api.com/api/v1/hackrx/run

Description/Notes
Any notes you provided with the submission

Example: Updated model with better prompting

Submission Metadata
Submission Metadata
Additional submission details and context

Click to zoomStep 5: Checking Your Leaderboard Ranking
1
Navigate to Leaderboard
Access the competition rankings to see how your team is performing

How to access:
Sidebar → Competition → Leaderboard
URL: /leaderboardFinding Your Team & Understanding Rankings
How to locate your team and interpret the ranking system

Locating Your Team:
Teams are displayed in rank order (top 10 shown)
Once logged in, your team will be highlighted
Judging Parameters:
Accuracy: Precision of query understanding and clause matching
Token Efficiency: Optimized LLM token usage and cost-effectiveness
Latency: Response speed and real-time performance
Reusability: Code modularity and extensibility




Q.1PROBLEM STATEMENT
Design an LLM-Powered Intelligent Query–Retrieval System that can process large documents and make contextual decisions. Your system should handle real-world scenarios in insurance, legal, HR, and compliance domains.

Input Requirements:

Process PDFs, DOCX, and email documents
Handle policy/contract data efficiently
Parse natural language queries
Technical Specifications:

Use embeddings (FAISS/Pinecone) for semantic search
Implement clause retrieval and matching
Provide explainable decision rationale
Output structured JSON responses
Sample Query:

"Does this policy cover knee surgery, and what are the conditions?"Q.2SYSTEM ARCHITECTURE & WORKFLOW
2.1) Design and implement the following system components:

1
Input Documents
PDF Blob URL

2
LLM Parser
Extract structured query

3
Embedding Search
FAISS/Pinecone retrieval

4
Clause Matching
Semantic similarity

5
Logic Evaluation
Decision processing

6
JSON Output
Structured responseQ.3EVALUATION PARAMETERS
3.1) Your solution will be evaluated based on the following criteria:

a
Accuracy
Precision of query understanding and clause matching

b
Token Efficiency
Optimized LLM token usage and cost-effectiveness

c
Latency
Response speed and real-time performance

d
Reusability
Code modularity and extensibility

e
Explainability
Clear decision reasoning and clause traceabilityQ.4RETRIEVAL SYSTEM API DOCUMENTATION
Base URL (Local Development):

http://localhost:8000/api/v1
Authentication:

Authorization: Bearer 3d2f575e504dc0a556592a02b475556bf406c5b03f06b1d789b7f1f0ccf45730
✅ Team token loaded successfully

API Endpoints Overview
POST
/hackrx/run
Run Submissions

Sample Upload Request:

POST /hackrx/run
Content-Type: application/json
Accept: application/json
Authorization: Bearer 3d2f575e504dc0a556592a02b475556bf406c5b03f06b1d789b7f1f0ccf45730

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}
Sample Response:

{
"answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}
Recommended Tech Stack:
FastAPI
Backend

Pinecone
Vector DB

GPT-4
LLM

PostgreSQL
Database