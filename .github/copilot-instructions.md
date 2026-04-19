# Document QA RAG System Guidelines

## Code Style
- Written in Python (>=3.12).
- Streamlit is used for the frontend UI. Always ensure UI updates correctly handle `st.session_state`.
- Follow standard Streamlit patterns for caching (`@st.cache_resource`) and session management.
- The project relies heavily on LangChain and Google Generative AI for RAG operations. Keep new implementations aligned with LangChain LCEL (LangChain Expression Language) where possible.

## Architecture
- **Frontend:** Streamlit (`app.py`). Handles file uploads (PDF/TXT), rendering chat history, and sidebar configuration.
- **Document Processing:** Uses `PyPDFLoader`/`TextLoader` -> `RecursiveCharacterTextSplitter`.
- **Vector Store:** Implements `FAISS` with `GoogleGenerativeAIEmbeddings`.
- **Generation Chain:** Uses LangChain's `RunnableParallel` combined with `ChatGoogleGenerativeAI` for context retrieval and streaming AI responses.

## Build and Test
- Uses `uv` for dependency management. To install or sync dependencies: `uv sync`.
- Ensure a `.env` file exists with `GOOGLE_API_KEY` before running locally.
- Run the application locally: `streamlit run app.py`.
- Docker is supported for containerized runs: `docker-compose up -d`.

## Conventions
- **Tooling:** Assume `uv` is the standard Python package manager instead of `pip` or `poetry`.
- **UI State:** Manage Streamlit's `st.rerun()` carefully to avoid infinite loops during state updates.
- **Supported Formats:** Currently only PDF and TXT are supported. Do not attempt to add other format parsers without explicitly expanding the LangChain loader configuration in `app.py`.
- **Documentation:** See `README.md` for complete configuration options, architecture diagrams, and the future roadmap.