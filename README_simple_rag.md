# Simple PDF RAG System

A minimal RAG (Retrieval-Augmented Generation) implementation for asking questions about a single PDF document. Perfect for coding workshops and demonstrations!

## Features

- 📄 Loads a single PDF file
- 🔍 Creates vector embeddings using OpenAI
- 💾 In-memory vector storage (no external database needed)
- ❓ Ask questions about the PDF content
- 📊 Shows source pages and similarity scores

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_simple_rag.txt
```

### 2. Set Up OpenAI API Key

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and replace `your_openai_api_key_here` with your actual API key

### 3. Run the Program

```bash
python simple_pdf_rag.py
```

## Usage Example

```
Enter the path to your PDF file: ./example.pdf

==================================================
PDF QA System Ready!
Type 'quit' to exit
==================================================

Your question: What is the main topic of this document?

Answer: The main topic of this document is...

Sources:
1. Page 1 (Score: 0.89)
   This document discusses the fundamental concepts...

--------------------------------------------------

Your question: quit
Goodbye!
```

## How It Works (Workshop Talking Points)

1. **PDF Loading**: Uses `SimpleDirectoryReader` to load the PDF
2. **Text Splitting**: Breaks document into chunks (1024 tokens each)
3. **Embedding**: Converts text chunks into vectors using OpenAI's embeddings
4. **Vector Store**: Stores embeddings in memory using LlamaIndex
5. **Query Processing**:
   - Converts question into embedding
   - Finds most similar text chunks
   - Sends relevant chunks + question to GPT-3.5
   - Returns answer with sources

## Key Components

- **VectorStoreIndex**: In-memory vector database
- **OpenAIEmbedding**: Text-to-vector conversion
- **SentenceSplitter**: Intelligent text chunking
- **Query Engine**: Handles question-answering

## Workshop Demo Ideas

1. Try different types of questions:
   - Factual: "What is defined on page 5?"
   - Summarization: "Summarize the key findings"
   - Comparative: "How do X and Y differ?"

2. Show similarity scores to explain relevance

3. Demonstrate limitations:
   - Questions about information not in the PDF
   - Ambiguous queries

## Troubleshooting

- **"File not found"**: Check the PDF path is correct
- **"API key error"**: Verify your OpenAI API key in `.env`
- **Memory issues**: For very large PDFs, consider reducing `chunk_size`

## Extensions (for advanced workshops)

- Add multiple PDF support
- Implement conversation memory
- Add different LLM options
- Export to a web interface with Streamlit
