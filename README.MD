# Physics Tutor AI (RAG-based)

A **Retrieval-Augmented Generation (RAG)** powered Physics Tutor AI built on **LangChain LCEL**, **OpenStax College Physics 2e**, and an **open-source LLM (Mistral)**.

## Features
- CLI-based AI to user conversation
- **Textbook-grounded answers** using OpenStax College Physics 2e
- **Semantic retrieval** via FAISS vector store
- Open source LLM MistralAI
- Strategy-based document parsing (Content, Glossary, Summaries)
- Process and Application loggers

<br>
To run application:<br>

1. Clone the repository
```
git clone https://github.com/eddie-mar/Physics-Tutor-AI-Agent.git
```

2. Install dependencies
```
python3 -m venv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

3. Set app/.env variables:
```
START_URL=https://openstax.org/books/college-physics-2e/pages/1-introduction-to-science-and-the-realm-of-physics-physical-quantities-and-units
END_URL=https://openstax.org/books/college-physics-2e/pages/34-section-summary
VECTORSTORE=vectorstore
BUILD_VECTORSTORE=False
```
START_URL and END_URL will be the basis for scraping the web-based book. 
BUILD_VECTORSTORE=False -> rebuild embeddings
BUILD_VECTORSTORE=True  -> load existing FAISS index
Vectorstore must be built first during initial run of the app, the app will automatically detect if vectorstore does not exist and initiate build. If you want to update the vectorstore (updated content from the book), set build_vectorstore=True.

4. Run application
```
cd app/
python3 -m main.py
```




