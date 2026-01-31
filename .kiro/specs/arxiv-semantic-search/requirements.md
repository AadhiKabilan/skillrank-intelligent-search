# Requirements Document

## Introduction

This document specifies the requirements for building a complete, minimal, production-ready full-stack intelligent search system for a large research paper corpus. The system will provide LLM-powered semantic search over ArXiv research papers with support for complex queries, cross-document reasoning, and citation-grounded synthesis.

## Glossary

- **Search_System**: The complete intelligent search application
- **Data_Pipeline**: The ingestion component that processes ArXiv data
- **Search_API**: The FastAPI backend that handles search requests
- **Frontend**: The HTML/JavaScript user interface
- **Vector_Store**: The FAISS index containing paper embeddings
- **LLM_Synthesizer**: The OpenAI-powered component that generates responses
- **Paper_Chunk**: A ~800 character segment of paper content (title + summary)
- **Semantic_Query**: A natural language search query processed via embeddings

## Requirements

### Requirement 1: Data Ingestion Pipeline

**User Story:** As a system administrator, I want to process ArXiv research papers into a searchable format, so that users can perform semantic searches across the corpus.

#### Acceptance Criteria

1. WHEN the Data_Pipeline processes arxivData.json, THE System SHALL extract title and summary fields from each paper record
2. WHEN text is processed, THE Data_Pipeline SHALL chunk content into segments of approximately 800 characters each
3. WHEN chunks are created, THE Data_Pipeline SHALL compute embeddings using OpenAI embeddings API with fallback to sentence-transformers
4. WHEN embeddings are computed, THE Data_Pipeline SHALL normalize them for cosine similarity calculations
5. WHEN all embeddings are ready, THE Data_Pipeline SHALL build a FAISS IndexFlatIP index
6. WHEN indexing is complete, THE Data_Pipeline SHALL save index.faiss and meta.json with metadata aligned to vector positions
7. WHEN a --limit argument is provided, THE Data_Pipeline SHALL process only the specified number of papers for demo purposes
8. THE Data_Pipeline SHALL support scaling to 24,000+ papers without architectural changes

### Requirement 2: Search API Backend

**User Story:** As a developer, I want a FastAPI backend that handles search requests, so that the frontend can retrieve semantically relevant papers with LLM-synthesized responses.

#### Acceptance Criteria

1. WHEN the Search_API starts, THE System SHALL load the FAISS index and metadata into memory
2. WHEN a POST request is made to /search endpoint, THE Search_API SHALL accept JSON with "q" (query) and "k" (top-k) parameters
3. WHEN a search query is received, THE Search_API SHALL embed the query using the same embedding model as ingestion
4. WHEN the query is embedded, THE Search_API SHALL retrieve the top-k most similar chunks from the Vector_Store
5. WHEN chunks are retrieved, THE Search_API SHALL aggregate chunks across multiple papers for comprehensive coverage
6. WHEN context is prepared, THE LLM_Synthesizer SHALL call OpenAI ChatCompletion API to generate a synthesized response
7. WHEN generating responses, THE LLM_Synthesizer SHALL perform cross-document reasoning and categorize insights
8. WHEN responses are generated, THE LLM_Synthesizer SHALL include citations to source papers in the output
9. WHEN processing is complete, THE Search_API SHALL return JSON with "answer" (synthesized response) and "hits" (retrieved papers)
10. WHEN search requests are processed, THE System SHALL complete retrieval within 3 seconds for demo purposes

### Requirement 3: Frontend User Interface

**User Story:** As a researcher, I want a simple web interface to search papers and view results, so that I can quickly find relevant research with synthesized insights.

#### Acceptance Criteria

1. WHEN the Frontend loads, THE System SHALL display a search textarea, top-k selector, and submit button
2. WHEN a user submits a search, THE Frontend SHALL send a POST request to the Search_API with the query and k value
3. WHEN search results are received, THE Frontend SHALL display the LLM-synthesized answer prominently
4. WHEN displaying results, THE Frontend SHALL show retrieved papers with titles, summaries, and clickable arXiv links
5. THE Frontend SHALL use plain HTML and vanilla JavaScript without external frameworks
6. THE Frontend SHALL maintain minimal, readable styling without CSS frameworks

### Requirement 4: Semantic Search Capabilities

**User Story:** As a researcher, I want to perform semantic searches using natural language queries, so that I can find conceptually related papers beyond keyword matching.

#### Acceptance Criteria

1. WHEN a user submits a Semantic_Query, THE Search_System SHALL use embedding-based similarity rather than keyword matching
2. WHEN embeddings are compared, THE System SHALL use cosine similarity to rank relevance
3. WHEN multiple papers contain relevant information, THE LLM_Synthesizer SHALL synthesize insights across documents
4. WHEN generating responses, THE System SHALL categorize findings and provide coherent cross-document reasoning
5. WHEN citing sources, THE System SHALL reference specific papers that contributed to the synthesized answer

### Requirement 5: System Architecture and Performance

**User Story:** As a system architect, I want clean separation between retrieval and reasoning components, so that the system is maintainable and scalable.

#### Acceptance Criteria

1. THE System SHALL separate retrieval logic (FAISS) from reasoning logic (LLM) into distinct components
2. WHEN processing the full corpus, THE System SHALL handle 24,000+ papers without architectural modifications
3. WHEN running locally, THE System SHALL operate within VS Code development environment constraints
4. THE System SHALL use only the specified technology stack: Python, FastAPI, FAISS, OpenAI APIs, plain HTML/JS
5. THE System SHALL avoid heavy frameworks like LangChain to maintain simplicity and readability

### Requirement 6: Project Structure and Documentation

**User Story:** As a developer, I want clear project organization and documentation, so that I can understand and run the system quickly.

#### Acceptance Criteria

1. THE System SHALL organize code into /ingest.py, /app.py, /templates/index.html, /static/main.js, /requirements.txt, /.gitignore, and /README.md
2. WHEN documentation is provided, THE README.md SHALL include problem overview, architecture explanation, and step-by-step local setup instructions
3. WHEN explaining the system, THE README.md SHALL describe how semantic search works and how cross-document synthesis operates
4. WHEN providing setup instructions, THE README.md SHALL explain the demo subset versus full 24K scalability approach
5. THE System SHALL be understandable by an evaluator within 5 minutes of code review

### Requirement 7: Environment and Deployment

**User Story:** As a user, I want to run the system locally for demonstration, so that I can evaluate its capabilities without complex deployment.

#### Acceptance Criteria

1. WHEN setting up the system, THE User SHALL only need to set OPENAI_API_KEY environment variable
2. WHEN running ingestion, THE User SHALL execute "python ingest.py --limit 500" for demo setup
3. WHEN starting the server, THE User SHALL execute "uvicorn app:app" to launch the API
4. WHEN accessing the interface, THE User SHALL open a web browser to interact with the search system
5. THE System SHALL operate entirely within local VS Code execution environment
6. THE System SHALL support demonstration of complex queries with cross-document reasoning capabilities