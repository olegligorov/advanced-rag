# Implementation Plan: FastAPI RAG Server + React TypeScript UI

## Project Overview
Transform the Jupyter notebook advanced RAG implementation into a production-ready FastAPI server with a React TypeScript frontend for querying Kubernetes documentation.

---

## Architecture

### Backend: FastAPI REST API
- **Framework**: FastAPI + Uvicorn
- **Core Components**:
  - Document loading & semantic chunking (on startup)
  - Hybrid retrieval (Vector + BM25)
  - RRF fusion
  - Cross-encoder re-ranking
  - LLM generation with citations
  - Response caching (optional don't do for now)

### Frontend: React TypeScript SPA
- **Framework**: React + TypeScript + Axios
- **Features**:
  - Query input form
  - Loading states
  - Display answer with citations
  - Show retrieved documents (collapsible)
  - Error handling

---

## Backend Implementation Plan

### Phase 1: Project Structure
```
kubernetes_advanced_rag/
├── backend/
│   ├── main.py                # FastAPI server entry point
│   ├── config.py              # Configuration constants
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py        # Semantic chunking logic
│   │   ├── retrieval.py       # Hybrid retrieval (Vector + BM25 + RRF)
│   │   ├── reranking.py       # Cross-encoder re-ranking
│   │   └── generation.py      # LLM generation
│   ├── models/
│   │   └── rag_pipeline.py    # Main RAG orchestrator
│   ├── requirements.txt
│   └── .env                   # Environment variables
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── public/
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── QueryForm.tsx
│       │   ├── AnswerDisplay.tsx
│       │   └── DocumentList.tsx
│       ├── services/
│       │   └── api.ts
│       └── types/
│           └── index.ts
└── k8s_data/                  # Existing data directory
```

### Phase 2: Backend Core Components

#### 2.1: `config.py` - Configuration
```python
# Model configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3"

# Retrieval parameters
RETRIEVAL_K = 25
RERANK_TOP_N = 5
CHUNK_PERCENTILE = 95

# Paths
DATA_PATH = "../k8s_data/concepts"
```

#### 2.2: `rag/chunking.py` - Semantic Chunking
- Port `create_semantic_chunks()` from notebook
- Remove visualization code
- Return only chunks (no distances/threshold)
- Add logging for chunk statistics

#### 2.3: `rag/retrieval.py` - Hybrid Retrieval
- Port FAISS vector retriever initialization
- Port BM25 retriever initialization
- Port `rrf()` function with optimizations
- Create `HybridRetriever` class with `.retrieve(query)` method

#### 2.4: `rag/reranking.py` - Re-ranking
- Port `hf_rerank()` function
- Add automatic device detection
- Return documents with scores
- Add error handling for model loading

#### 2.5: `rag/generation.py` - LLM Generation
- Port Ollama LLM initialization
- Port prompt template
- Create context formatting function
- Return answer + metadata (sources, scores)

#### 2.6: `models/rag_pipeline.py` - Orchestrator
```python
class RAGPipeline:
    def __init__(self):
        # Initialize all components (load models, build indices)
        pass

    def query(self, question: str) -> dict:
        # 1. Hybrid retrieval
        # 2. Re-ranking
        # 3. LLM generation
        # Return: {answer, sources, retrieval_time, generation_time}
        pass
```

#### 2.7: `main.py` - FastAPI Server
- Initialize RAGPipeline on startup event (load once)
- Create `/api/query` endpoint (POST)
- Create `/api/health` endpoint (GET)
- Add CORS middleware
- Add error handling with HTTPException
- Use Pydantic models for request/response validation

**API Endpoints**:
```
POST /api/query
Request: { "question": "How to configure Pod limits?" }
Response: {
  "answer": "...",
  "sources": [
    {"rank": 1, "source": "file.md", "snippet": "...", "score": 6.07}
  ],
  "metadata": {
    "retrieval_time_ms": 234,
    "generation_time_ms": 1523,
    "total_time_ms": 1757
  }
}

GET /api/health
Response: { "status": "ok", "models_loaded": true }
```

---

## Frontend Implementation Plan

### Phase 3: React Application

#### 3.1: Project Setup
- Create React TypeScript app: `npx create-react-app frontend --template typescript`
- Install dependencies: `axios`, `react-markdown` (for rendering citations)
- Configure CORS (FastAPI handles this)

#### 3.2: `types/index.ts` - TypeScript Interfaces
```typescript
export interface QueryRequest {
  question: string;
}

export interface Source {
  rank: number;
  source: string;
  snippet: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  metadata: {
    retrieval_time_ms: number;
    generation_time_ms: number;
    total_time_ms: number;
  };
}
```

#### 3.3: `services/api.ts` - API Client
```typescript
import axios from 'axios';
import { QueryRequest, QueryResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export const queryRAG = async (question: string): Promise<QueryResponse> => {
  const response = await axios.post<QueryResponse>(
    `${API_BASE_URL}/api/query`,
    { question } as QueryRequest
  );
  return response.data;
};
```

#### 3.4: `components/QueryForm.tsx`
- Input field for question with proper typing
- Submit button
- Loading spinner during request
- Sample questions (pills/buttons)
- Form validation

#### 3.5: `components/AnswerDisplay.tsx`
- Display formatted answer
- Render citations as clickable links/chips
- Show metadata (response time, model info)
- Copy answer button
- Type-safe props

#### 3.6: `components/DocumentList.tsx`
- Collapsible section: "Retrieved Documents"
- Display top 5 documents with:
  - Rank
  - Source file name
  - Relevance score
  - Content snippet (truncated)
- Type-safe Source interface

#### 3.7: `App.tsx` - Main Component
- State management with proper TypeScript typing
- Layout: Header + QueryForm + AnswerDisplay + DocumentList
- Error handling UI with typed error states

---

## Implementation Order

### Step 1: Backend Foundation (Do First)
1. Create directory structure
2. Create `config.py` with all constants
3. Create `requirements.txt`:
   ```
   fastapi
   uvicorn[standard]
   pydantic
   langchain
   langchain-community
   langchain-huggingface
   sentence-transformers
   faiss-cpu
   unstructured
   markdown
   python-dotenv
   ```

### Step 2: Port RAG Components
4. Implement `rag/chunking.py`
5. Implement `rag/retrieval.py`
6. Implement `rag/reranking.py`
7. Implement `rag/generation.py`

### Step 3: RAG Pipeline
8. Implement `models/rag_pipeline.py`
9. Add initialization logic (load docs, create indices)
10. Add query method with timing

### Step 4: FastAPI Server
11. Implement `main.py` with async endpoints
12. Add Pydantic models for request/response
13. Add startup event for model loading
14. Add error handling with HTTPException
15. Test with curl/Postman or FastAPI docs

### Step 5: Frontend
16. Create React TypeScript app structure
17. Define TypeScript interfaces in `types/index.ts`
18. Implement API service in `services/api.ts`
19. Implement QueryForm component
20. Implement AnswerDisplay component
21. Implement DocumentList component
22. Integrate in App.tsx
23. Add styling (CSS/Tailwind/Material-UI)

### Step 6: Integration & Testing
24. Test end-to-end flow
25. Handle edge cases (empty results, errors)
26. Add loading states and error messages
27. Type checking with `tsc --noEmit`
28. Performance optimization (if needed)

---

## Technical Decisions

### Model Loading Strategy
**Decision**: Load all models once on FastAPI startup event (not per-request)
**Rationale**:
- Embedding model, FAISS index, BM25 index, cross-encoder, LLM are expensive to load
- First request would take 30+ seconds otherwise
- Trade-off: Higher memory usage, but much faster queries
- Use `@app.on_event("startup")` to initialize RAG pipeline

### Document Indexing Strategy
**Decision**: Build indices on startup from files
**Rationale**:
- 163 markdown files → ~1200 chunks (from notebook)
- Takes ~30 seconds to load + chunk + index
- Alternative: Serialize indices to disk and load (future optimization)

### Frontend State Management
**Decision**: Use React useState with TypeScript (no Redux/Context)
**Rationale**:
- Simple app with minimal state
- Avoid over-engineering
- TypeScript provides compile-time safety without runtime overhead

### Error Handling
**Decision**: Graceful degradation
- If Ollama is down → return error message
- If re-ranking fails → return fused results without re-ranking
- If retrieval fails → return empty results with clear error

---

## Performance Considerations

### Expected Latency (Estimates)
- Hybrid retrieval: ~200-500ms
- Re-ranking (5 docs): ~100-300ms
- LLM generation: ~2-5 seconds (depends on Ollama, answer length)
- **Total**: ~2.5-6 seconds per query

### Optimization Opportunities (Future)
1. Cache query embeddings (for repeated queries)
2. Serialize FAISS/BM25 indices to disk
3. Async processing (retrieval + re-ranking in parallel)
4. Batch re-ranking if needed
5. Use faster LLM or API-based LLM (GPT-3.5/4)

---

## Testing Strategy

### Backend Testing
- Test each module independently:
  - Chunking: Verify chunk count and sizes
  - Retrieval: Test with sample query
  - Re-ranking: Verify score ordering
  - Generation: Test prompt formatting
- Integration test: Full pipeline with known query

### Frontend Testing
- Manual testing with various queries
- Edge cases: empty query, very long query, special characters
- UI responsiveness on different screen sizes

---

## Environment Variables (.env)

### Backend
```
HOST=0.0.0.0
PORT=8000
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
```

### Frontend
```
REACT_APP_API_URL=http://localhost:8000
```

---

## Documentation Requirements

### README.md (Create After Implementation)
- Installation instructions
- Setup guide (Ollama, models, dependencies)
- Usage examples
- API documentation
- Screenshots

### Comments in Code
- Docstrings for all functions
- Inline comments for complex logic (RRF, semantic chunking)
- Type hints where possible

---

## Known Limitations & Future Work

### Current Scope (MVP)
- ✅ Single-query interface
- ✅ No user authentication
- ✅ No query history
- ✅ No feedback mechanism
- ✅ No analytics/logging to database

### Future Enhancements (Out of Scope)
- Multi-turn conversations (chat history)
- User accounts and saved queries
- Query analytics dashboard
- A/B testing different RAG configurations
- Evaluation metrics endpoint
- Admin panel for re-indexing

---

## Dependencies Check

### Prerequisites
- Python 3.9+
- Node.js 18+ (for TypeScript support)
- Ollama installed and running (`ollama run llama3`)
- Git

### Model Downloads (Automatic on First Run)
- `all-MiniLM-L6-v2` (~90MB)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB)
- `llama3` via Ollama (~4GB)

---

## Success Criteria

### Backend
- [ ] FastAPI server starts without errors
- [ ] All models load successfully on startup event
- [ ] `/api/query` returns valid response in <10 seconds
- [ ] Proper error handling with HTTPException
- [ ] Pydantic models validate request/response

### Frontend
- [ ] React TypeScript app renders without errors
- [ ] No TypeScript compilation errors
- [ ] Query submission works with proper typing
- [ ] Answer displays with proper formatting
- [ ] Sources are clickable/expandable
- [ ] Loading states work correctly

### Integration
- [x] Frontend communicates with backend
- [x] CORS configured correctly
- [x] End-to-end query flow works
- [x] Error messages displayed to user

---

## Timeline Estimate (No Time Promises!)

**Note**: This is for planning purposes only, actual time may vary.

- Backend core components: Multiple sessions
- FastAPI server: One session
- Frontend TypeScript setup: One session
- Frontend components: Multiple sessions
- Integration & testing: One session
- Styling & polish: One session

**Total**: Plan for several focused work sessions

---

## Next Steps

1. Review this plan
2. Clarify any questions about architecture
3. Start with Step 1 (Backend Foundation)
4. Implement incrementally, testing each component
5. Keep notebook as reference but don't copy-paste blindly
6. Refactor as needed based on production requirements

---

## Questions to Consider

1. Should we add rate limiting to the API?
2. Do we want to support multiple LLM models (switchable)?
3. Should documents be re-indexable without server restart?
4. Do we want a "feedback" button on answers (for evaluation data)?
5. Should we add query history in the UI (client-side)?

Let me know if you want to proceed with implementation or adjust the plan!
