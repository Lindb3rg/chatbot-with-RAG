 # RAG Chat Application
 
 A Retrieval-Augmented Generation (RAG) application that leverages LangChain, Chroma vector store, and OpenAI to build an interactive Q&A interface over a custom Markdown knowledge base, powered by a Gradio frontend.

 ## Features
 - Ingests and chunks Markdown documents from `knowledge-base/` directories
 - Generates embeddings with OpenAI and stores them in a Chroma vector database
 - Provides conversational retrieval with LangChain's `ConversationalRetrievalChain`
 - Visualizes vector embeddings in 2D via t-SNE and Plotly
 - Interactive chat UI powered by Gradio

 ## Getting Started

 ### Prerequisites
 - Python 3.8+
 - An OpenAI API key
 - (Optional) Specify a model via `OPENAI_MODEL` (default: `gpt-3.5-turbo`)

 ### Installation
 1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```
 2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate     # macOS/Linux
    env\\Scripts\\activate      # Windows
    ```
 3. Install dependencies:
    ```bash
    pip install -r requirements.txt  # if available
    # or install manually:
    pip install python-dotenv gradio langchain langchain-openai langchain-chroma chromadb plotly scikit-learn
    ```
 4. Configure environment variables:
    Create a `.env` file in the project root:
    ```ini
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL=gpt-3.5-turbo
    VECTOR_DB_NAME=chroma-db
    ```

 ## Usage
 1. Prepare documents, build embeddings, and persist the vector store:
    ```bash
    python app.py
    ```
 2. Launch the Gradio interface:
    ```bash
    python app.py
    ```
 3. Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser to start chatting.

 ## Notebook Example
 Explore `notebook.ipynb` for an end-to-end demonstration of document ingestion, vector store construction, and 2D embedding visualization.

 ## Project Structure
 ```
 ├── app.py                  # Main application (Gradio UI + RAG pipeline)
 ├── chroma_utils.py         # Document loading, chunking, and Chroma vector store
 ├── db_utils.py             # (Placeholder) Database utilities
 ├── langchain_utils.py      # (Placeholder) LangChain helper functions
 ├── utils/tools.py          # Logging utilities
 ├── knowledge-base/         # Markdown knowledge base folders
 ├── vector-db/              # Persistent Chroma vector database files
 └── notebook.ipynb          # Jupyter notebook example
 ```

 ## Customization
 - Add or update Markdown files under `knowledge-base/` to expand your knowledge base.
 - Tweak chunking parameters (`chunk_size`, `chunk_overlap`) in `chroma_utils.py`.
 - Modify prompt templates and chain configurations in `app.py`.

 ## Visualization
 Use the `VectorStore.show_vectors_2D()` method to inspect embedding clusters:
 ```python
 from chroma_utils import PrepareDocumentsFolder, VectorStore
 folders = ["knowledge-base/company", "knowledge-base/contracts", ...]
 prep = PrepareDocumentsFolder(folders)
 chunks = prep.create_chunks()
 vs = VectorStore(chunks, "chroma-db")
 vs.show_vectors_2D()
 ```

 ## Contributing
 Contributions, issues, and feature requests are welcome! Feel free to open a pull request.

 ## License
 This project is released under the MIT License. See [LICENSE](LICENSE) for details.