# Implementation Plan: ArXiv Semantic Search System

## Overview

This implementation plan converts the ArXiv semantic search design into discrete coding tasks. The approach follows incremental development: data pipeline → search API → frontend → integration. Each task builds on previous work and includes comprehensive testing to ensure correctness properties are validated throughout development.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure with /templates, /static, /data folders
  - Create requirements.txt with FastAPI, FAISS, OpenAI, sentence-transformers, pytest-hypothesis
  - Create .gitignore for Python projects with data/ and .env exclusions
  - Create basic README.md with project overview and setup instructions
  - _Requirements: 6.1, 7.1_

- [ ] 2. Implement data ingestion pipeline (ingest.py)
  - [x] 2.1 Create command-line interface and argument parsing
    - Implement argparse for --limit, --input, --output parameters
    - Add help text and validation for command-line arguments
    - _Requirements: 1.7, 7.2_

  - [ ]* 2.2 Write property test for command-line limit enforcement
    - **Property 5: Command Line Limit Enforcement**
    - **Validates: Requirements 1.7**

  - [ ] 2.3 Implement ArXiv data loading and parsing
    - Create load_arxiv_data() function to read JSON file
    - Extract title, summary, author, year, tags, arxiv_links fields
    - Handle malformed JSON and missing fields gracefully
    - _Requirements: 1.1_

  - [ ]* 2.4 Write property test for data extraction consistency
    - **Property 1: Data Extraction Consistency**
    - **Validates: Requirements 1.1**

  - [ ] 2.5 Implement text chunking functionality
    - Create create_chunks() function for ~800 character segments
    - Combine title + summary for each paper
    - Ensure no chunk exceeds 1000 chars or is shorter than 100 chars (except final)
    - _Requirements: 1.2_

  - [ ]* 2.6 Write property test for text chunking boundaries
    - **Property 2: Text Chunking Boundaries**
    - **Validates: Requirements 1.2**

- [ ] 3. Implement embedding generation and FAISS indexing
  - [ ] 3.1 Create embedding generation with OpenAI API
    - Implement generate_embeddings() with text-embedding-ada-002
    - Add sentence-transformers fallback for missing API key
    - Normalize all embeddings to unit vectors for cosine similarity
    - Handle API rate limits and errors gracefully
    - _Requirements: 1.3, 1.4_

  - [ ]* 3.2 Write property test for embedding normalization
    - **Property 3: Embedding Normalization**
    - **Validates: Requirements 1.4**

  - [ ] 3.3 Implement FAISS index construction
    - Create build_faiss_index() using IndexFlatIP for cosine similarity
    - Add all normalized embeddings to index
    - Validate index size matches number of chunks
    - _Requirements: 1.5_

  - [ ] 3.4 Implement index and metadata persistence
    - Create save_index_and_metadata() to write index.faiss and meta.json
    - Ensure metadata positions align with vector indices
    - Include chunk information, paper references, and arXiv URLs
    - _Requirements: 1.6_

  - [ ]* 3.5 Write property test for index-metadata alignment
    - **Property 4: Index-Metadata Alignment**
    - **Validates: Requirements 1.6**

- [ ] 4. Checkpoint - Validate data pipeline
  - Run ingest.py with --limit 10 to test end-to-end pipeline
  - Verify index.faiss and meta.json are created correctly
  - Ensure all tests pass, ask the user if questions arise

- [ ] 5. Implement FastAPI search backend (app.py)
  - [x] 5.1 Create FastAPI application and startup logic
    - Initialize FastAPI app with CORS middleware
    - Implement load_search_index() to load FAISS index and metadata at startup
    - Add error handling for missing or corrupted index files
    - _Requirements: 2.1_

  - [ ] 5.2 Implement search endpoint request handling
    - Create POST /search endpoint accepting JSON with "q" and "k" parameters
    - Add request validation using Pydantic models
    - Return structured error responses for invalid requests
    - _Requirements: 2.2_

  - [ ]* 5.3 Write property test for API request-response format
    - **Property 6: API Request-Response Format**
    - **Validates: Requirements 2.2, 2.9**

  - [ ] 5.4 Implement query embedding and similarity search
    - Create embed_query() using same model as ingestion
    - Implement retrieve_similar_chunks() with FAISS search
    - Ensure query embeddings have same dimensionality as document embeddings
    - Return top-k results ranked by cosine similarity
    - _Requirements: 2.3, 2.4_

  - [ ]* 5.5 Write property test for query-document embedding consistency
    - **Property 7: Query-Document Embedding Consistency**
    - **Validates: Requirements 2.3**

  - [ ]* 5.6 Write property test for top-k retrieval accuracy
    - **Property 8: Top-K Retrieval Accuracy**
    - **Validates: Requirements 2.4**

- [ ] 6. Implement LLM synthesis and response generation
  - [ ] 6.1 Create result aggregation across papers
    - Implement aggregate_papers() to group chunks by paper
    - Ensure results include chunks from different papers when relevant
    - Prepare context for LLM with paper titles and summaries
    - _Requirements: 2.5_

  - [ ]* 6.2 Write property test for cross-paper aggregation
    - **Property 9: Cross-Paper Aggregation**
    - **Validates: Requirements 2.5**

  - [ ] 6.3 Implement OpenAI ChatCompletion integration
    - Create synthesize_response() using gpt-3.5-turbo
    - Design prompts for cross-document reasoning and citation generation
    - Handle API failures gracefully with fallback responses
    - Ensure responses include citations to source papers
    - _Requirements: 2.6, 2.8_

  - [ ]* 6.4 Write property test for citation accuracy
    - **Property 10: Citation Accuracy**
    - **Validates: Requirements 2.8, 4.5**

  - [ ] 6.5 Complete search endpoint response formatting
    - Return JSON with "answer" (synthesized response) and "hits" (papers)
    - Include relevance scores, query timing, and paper metadata
    - Add comprehensive error handling and logging
    - _Requirements: 2.9_

- [ ] 7. Checkpoint - Validate search API
  - Test search endpoint with sample queries
  - Verify LLM synthesis and citation generation
  - Ensure all tests pass, ask the user if questions arise

- [ ] 8. Implement frontend user interface
  - [ ] 8.1 Create HTML template (templates/index.html)
    - Design search form with textarea, top-k selector, submit button
    - Create results display areas for synthesized answer and paper hits
    - Add loading states and error message containers
    - Use semantic HTML without external CSS frameworks
    - _Requirements: 3.1_

  - [ ] 8.2 Implement JavaScript functionality (static/main.js)
    - Create submitSearch() for form handling and API communication
    - Implement displayResults() for rendering search responses
    - Add loading state management and error handling
    - Ensure proper formatting of paper cards with arXiv links
    - _Requirements: 3.2, 3.3, 3.4_

  - [ ]* 8.3 Write property test for frontend request format
    - **Property 11: Frontend Request Format**
    - **Validates: Requirements 3.2**

  - [ ]* 8.4 Write property test for result display completeness
    - **Property 12: Result Display Completeness**
    - **Validates: Requirements 3.3, 3.4**

  - [ ] 8.5 Add CSS styling and responsive design
    - Create minimal, clean styling without external frameworks
    - Implement responsive layout using CSS Grid and Flexbox
    - Add hover effects and visual feedback for better UX
    - Ensure proper contrast ratios and accessibility
    - _Requirements: 3.6_

- [ ] 9. Implement semantic search validation
  - [ ] 9.1 Add semantic search methodology verification
    - Ensure search uses embedding-based similarity not keyword matching
    - Validate cosine similarity ranking in search results
    - Add logging to confirm semantic search methodology
    - _Requirements: 4.1, 4.2_

  - [ ]* 9.2 Write property test for semantic search methodology
    - **Property 13: Semantic Search Methodology**
    - **Validates: Requirements 4.1, 4.2**

- [ ] 10. Integration and final wiring
  - [ ] 10.1 Create static file serving in FastAPI
    - Add static file mounting for /static directory
    - Implement GET / endpoint to serve index.html
    - Ensure proper MIME types for CSS and JavaScript files
    - _Requirements: 3.1_

  - [ ] 10.2 Add comprehensive error handling
    - Implement error handling across all components
    - Add structured error responses with appropriate HTTP status codes
    - Create user-friendly error messages for frontend display
    - Add retry logic for external API calls
    - _Requirements: Error Handling section_

  - [ ]* 10.3 Write integration tests for end-to-end workflows
    - Test complete pipeline from ingestion through search to display
    - Validate error scenarios and recovery mechanisms
    - Test with various query types and edge cases

- [ ] 11. Documentation and deployment preparation
  - [ ] 11.1 Complete README.md documentation
    - Add problem overview and architecture explanation
    - Include step-by-step local setup instructions
    - Explain semantic search and cross-document synthesis concepts
    - Document demo subset vs full 24K scalability approach
    - _Requirements: 6.2, 6.3, 6.4_

  - [ ] 11.2 Create example usage and demo scripts
    - Add sample queries demonstrating complex search capabilities
    - Create demo dataset preparation instructions
    - Include troubleshooting guide for common issues
    - _Requirements: 7.6_

- [ ] 12. Final checkpoint - Complete system validation
  - Run full ingestion with --limit 500 for demo
  - Start uvicorn server and test all functionality
  - Validate complex queries with cross-document reasoning
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout development
- Property tests validate universal correctness properties from the design
- Unit tests validate specific examples and edge cases
- The implementation follows clean architecture with separated concerns