# PDF Question-Answering System (RAG)
A Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural-language questions about the content of PDF documents and get accurate answers grounded in the actual document text.

## Overview
This project extracts text from PDF files, converts the text into vector embeddings, stores those embeddings in a vector database, and retrieves the most relevant sections when a question is asked. The retrieved sections are then passed to a large language model, which generates a final answer based only on that retrieved content.

## What it does
1. Extracts text from PDF documents using PyMuPDF, with an OCR fallback for scanned or image-based PDFs
2. Splits extracted text into overlapping chunks to preserve context across boundaries
3. Converts each chunk into a vector embedding using Sentence Transformers
4. Stores the embeddings in ChromaDB, a vector database, using cosine similarity for search
5. Retrieves the most relevant chunks for a given question through semantic search
6. Sends the retrieved chunks along with the question to an LLM through the Groq API, which generates the final answer

## Tech stack
- PyMuPDF for PDF text extraction
- pytesseract and pdf2image for OCR fallback on scanned PDFs
- LangChain text splitters for chunking
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- ChromaDB for vector storage and similarity search
- Groq API (Llama 3.3 70B) for answer generation

## How it works
PDF is converted to raw text, split into chunks of 500 characters with 50 characters of overlap, then each chunk is embedded into a 384-dimensional vector. When a question is asked, it is embedded the same way and compared against all stored vectors using cosine similarity. The closest matching chunks are retrieved and passed to the LLM as context, so the answer is grounded in the actual document rather than the model's general knowledge.

## Example usage
```python
rag_answer("What is Newton's first law of motion?")
```
This retrieves the most relevant chunks from the stored PDF content and returns an answer generated from that content.

## Project structure
- Text extraction from PDF files
- Text chunking with configurable size and overlap
- Embedding generation and storage in ChromaDB
- Metadata support for filtering search by source document
- Full retrieval and generation pipeline combining ChromaDB and Groq

## Possible improvements
- Add page number metadata for source citation
- Support queries across multiple documents at once
- Add a simple web interface
- Evaluate retrieval accuracy using a labeled question and answer test set

## Requirements
```
pymupdf
sentence-transformers
chromadb
langchain-text-splitters
pytesseract
pdf2image
groq
```
Note: A Groq API key is required and should be set as an environment variable rather than hardcoded in the notebook.
