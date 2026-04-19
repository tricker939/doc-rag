# Document QA RAG System

A web application that implements Retrieval-Augmented Generation (RAG) to allow users to query the content of uploaded documents using large language models. The application processes PDF and text files, generates vector embeddings, and provides a chat interface for answering questions based on the document context.

## Features

- Multi-format Support: Process PDF and TXT files.
- Document Processing: Uses LangChain to chunk and process text.
- Vector Storage: Uses FAISS for local, efficient similarity search.
- Question Answering: Uses Google Generative AI (Gemini) models to generate answers based on document context.
- Chat Management: Includes session-based chat history and export functionality.

## Technology Stack

- Frontend: Streamlit
- RAG Framework: LangChain
- Vector Database: FAISS
- Embeddings and Generation: Google Generative AI
- Package Management: uv

## Prerequisites

- Python 3.11 or higher
- Google API Key
- uv package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tricker939/doc-rag.git
cd doc-rag
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up the environment variables:
Create a `.env` file in the root directory and add your Google API key:
```env
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

### Local Deployment

Start the Streamlit application:
```bash
streamlit run app.py
```
The application will be accessible at `http://localhost:8501`.

### Docker Deployment

Using Docker Compose:
```bash
docker-compose up -d
```

Using Docker directly:
```bash
docker build -t doc-rag-system .
docker run -p 8501:8501 --env-file .env doc-rag-system
```

## Configuration

Model selection is available through the application sidebar. Supported models include gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, among others.

Internal indexing parameters can be modified in `app.py`:
- `CHUNK_SIZE = 1000`: Maximum character count per document chunk.
- `CHUNK_OVERLAP = 100`: Character overlap between adjacent chunks to preserve context.
- `RETRIEVER_K = 4`: Number of chunks retrieved for the generation prompt.

## Usage Guide

1. Open the application interface.
2. Provide your Google API key in the configuration sidebar.
3. Upload a document (PDF or TXT format).
4. Initiate document processing to split the text and create vector embeddings in the FAISS database.
5. Enter queries in the chat input. The system retrieves relevant context from the FAISS database and passes it to the selected model to formulate an answer.

## Limitations

- Supported file formats are currently restricted to PDF and TXT.
- The system processes text data only; OCR for image-based PDFs is not implemented.
- Memory consumption increases with document size due to local vector storage.

## License

This project is licensed under the MIT License.
