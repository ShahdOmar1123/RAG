A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about space and astronomy topics by retrieving relevant passages from a curated collection of space-related documents (NASA reports, astronomy textbooks, research papers) and generating grounded, hallucination-resistant answers.

## Overview

Standard LLMs can't answer questions about documents they weren't trained on, and often hallucinate confident-sounding but incorrect answers. This project solves that by combining **semantic retrieval** with **grounded generation**: the system retrieves the most relevant passages from the source documents first, then instructs the LLM to answer strictly from that retrieved context — never from its own memory.

## Features

- **Document ingestion with OCR** — handles both text-based and image-based/scanned PDFs using Tesseract OCR, so documents with no extractable text layer are still fully searchable
- **Semantic chunking & embedding** — documents are split into overlapping chunks and embedded using a HuggingFace sentence-transformer model
- **Fast vector search** — chunks are indexed with FAISS for efficient similarity search across the knowledge base
- **Grounded generation via Groq** — a Groq-hosted LLM generates answers strictly from retrieved context, minimizing hallucination
- **Configurable top-k retrieval** — the number of retrieved passages can be adjusted per query, supporting both simple factual lookups and complex multi-part questions
- **Interactive web interface** — a Gradio-based UI for asking questions and viewing source citations
- **Quantitative evaluation** — pipeline quality is measured using RAGAS across four standard RAG metrics

## Architecture

```
PDF documents
     │
     ▼
OCR (Tesseract) — extracts text from scanned/image-based pages
     │
     ▼
Chunking — RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
     │
     ▼
Embedding — sentence-transformers/all-MiniLM-L6-v2
     │
     ▼
FAISS vector index
     │
     ▼
Query → embed → retrieve top-k chunks
     │
     ▼
Groq LLM (openai/gpt-oss-120b) + grounded prompt → Answer
```

## Tools & Stack

| Component | Technology |
|---|---|
| Language | Python |
| Orchestration | LangChain |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| OCR | Tesseract / `pdf2image` |
| Interface | Gradio |
| Evaluation | RAGAS |

## Evaluation Results

The pipeline was evaluated on a 10-question test set with ground-truth answers taken directly from the source document, using [RAGAS](https://github.com/explodinggradients/ragas):

| Metric | Score | What it measures |
|---|---|---|
| **Context Precision** | 0.90 | How much of the retrieved context is actually relevant |
| **Context Recall** | 0.90 | Whether retrieval captured all the information needed to answer |
| **Answer Relevancy** | 0.84 | How directly the generated answer addresses the question |
| **Faithfulness** | 0.75 | How well the answer is grounded in the retrieved context (i.e. absence of hallucination) |

High context precision/recall scores indicate the retrieval stage (FAISS + embeddings) performs strongly. Faithfulness — while solid — is the area with the most room for improvement, and can be pushed higher with a stricter grounding prompt.

## Setup

### 1. Install dependencies

```bash
apt-get install -y poppler-utils tesseract-ocr
pip install langchain langchain-community langchain-classic langchain-groq \
    langchain-huggingface sentence-transformers faiss-cpu pypdf pytesseract pdf2image gradio
```

### 2. Set your Groq API key

Get a free key from [console.groq.com](https://console.groq.com/keys), then:

```python
import os
from getpass import getpass
os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API key: ")
```

> Never hardcode API keys in notebooks or commit them to version control.

### 3. Run the notebook

Open `Space_Astronomy_RAG_Assistant.ipynb` and run the cells in order:
1. Setup & installs
2. OCR + chunking
3. FAISS index
4. Groq LLM + RAG chain
5. Quick test
6. Gradio interface (for the interactive demo)

Evaluation (RAGAS) is provided as a separate, optional section — see the notebook for details on running it independently.

## Usage Example

```python
query = "What causes Jupiter's striped cloud appearance?"
result = qa.invoke({"query": query})

print(result["result"])
# Jupiter's clouds of ammonia crystals form distinct bands at different
# latitudes, produced by air flowing in different directions...

for doc in result["source_documents"]:
    print(doc.metadata["page"])
```

## Possible Extensions

- Hybrid search (semantic + keyword) for improved retrieval on exact terms and numbers
- Multi-document support with per-source filtering
- Chat history for follow-up questions
- Swap FAISS for ChromaDB if metadata filtering or frequent document updates are needed

## License

This project is for educational and portfolio purposes.
