# System Architecture - SFDA Cosmetics Inspector Assistant

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         المفتش الميداني                         │
│                      (Field Inspector - User)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Gradio Web Interface                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Source Selection:                                       │   │
│  │  ○ لوائح التجميل (Regulations)                          │   │
│  │  ○ محظورات التجميل (Banned Substances)                  │   │
│  │  ○ الكل (All Sources)                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Chat Interface (Arabic)                                │   │
│  │  User: "ما هي المادة الرابعة؟"                          │   │
│  │  Bot: [Streaming Response with Sources]                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SFDAChatbot (Main Logic)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Query Analysis                                       │   │
│  │     - ArabicArticleParser: Extract article numbers       │   │
│  │     - Query classification (specific vs general)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  2. Route Decision                                       │   │
│  │     ┌─────────────┐        ┌──────────────┐             │   │
│  │     │ Article #?  │───Yes──│ Direct Fetch │             │   │
│  │     └─────────────┘        └──────────────┘             │   │
│  │           │                                              │   │
│  │          No                                              │   │
│  │           │                                              │   │
│  │           ▼                                              │   │
│  │     ┌─────────────┐                                      │   │
│  │     │ RAG Pipeline│                                      │   │
│  │     └─────────────┘                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Vector Store (ChromaDB)                                 │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  Embedding Model: multilingual-e5-large            │  │   │
│  │  │  Collection: sfda_collection                       │  │   │
│  │  │  Documents: ~500+ chunks                           │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  │  Filter by Source:                                       │   │
│  │  - Regulations: {"category": "regulation"}              │   │
│  │  - Banned: {"category": "banned"}                       │   │
│  │  - All: No filter                                        │   │
│  │                                                          │   │
│  │  Similarity Search (Top K=8)                             │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Context Building                                        │   │
│  │  - Top 3 retrieved documents                             │   │
│  │  - Format with source labels                             │   │
│  │  - Limit to 1400 chars per doc                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LLM (OpenRouter - DeepSeek Chat)                        │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │ System Prompt:                                     │  │   │
│  │  │ "أنت مساعد امتثال. لا تضف معلومات خارجية.       │  │   │
│  │  │  اذكر المصادر دائماً."                            │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  │  Streaming Response                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Post-Processing                                         │   │
│  │  - TextFormatter: Clean Arabic text                      │   │
│  │  - Add source footer                                     │   │
│  │  - Remove repeated characters                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────────┐
│  Knowledge Base  │
│   (knowledge/)   │
└────────┬─────────┘
         │
         │ Ingestion Phase
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│              Document Loaders                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │   Excel    │  │    JSON    │  │    PDF     │         │
│  │  (Banned)  │  │ (Articles) │  │  (Regs)    │         │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘         │
│        │               │               │                 │
│        └───────────────┴───────────────┘                 │
│                        │                                 │
│                        ▼                                 │
│          ┌──────────────────────────┐                    │
│          │   Text Processing        │                    │
│          │   - Normalization        │                    │
│          │   - Arabic cleaning      │                    │
│          │   - Chunking (1000/150)  │                    │
│          └──────────┬───────────────┘                    │
└───────────────────────┼──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  Embedding Generation    │
         │  (multilingual-e5-large) │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │     ChromaDB             │
         │  - Persist to disk       │
         │  - Metadata filtering    │
         │  - Similarity search     │
         └──────────────────────────┘
```

## Component Architecture

### 1. Frontend Layer (Gradio UI)

```python
┌──────────────────────────────────────┐
│        Gradio Interface              │
│  ┌────────────────────────────────┐  │
│  │  Source Radio Buttons          │  │
│  │  - لوائح التجميل (PDF)         │  │
│  │  - محظورات التجميل             │  │
│  │  - الكل                        │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │  ChatInterface Component       │  │
│  │  - Textbox (Arabic RTL)        │  │
│  │  - Streaming messages          │  │
│  │  - Example queries             │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### 2. Application Layer (Python Backend)

```
SFDAChatbot (Main Controller)
├── ArabicArticleParser
│   ├── AR_WORD_TO_NUM dictionary
│   ├── normalize_article_to_num()
│   └── extract_article_number()
│
├── TextFormatter
│   ├── clean_repeated_characters()
│   ├── merge_spaced_arabic_letters()
│   └── pretty_arabic_text()
│
├── SourceDisplayManager
│   ├── display_source_name()
│   └── sources_footer_once()
│
└── Core Methods
    ├── get_article_doc() - Direct article retrieval
    ├── build_retriever() - Configure search filter
    ├── build_knowledge() - Format context
    └── stream_response() - Main chat logic
```

### 3. Data Layer (Vector Store)

```
ChromaDB
├── Collection: sfda_collection
├── Documents Structure:
│   ├── Regulations (category: "regulation")
│   │   ├── page_content: "المادة 4\n\n[text]"
│   │   └── metadata: {
│   │         "source": "لوائح التجميل",
│   │         "article": "4",
│   │         "category": "regulation",
│   │         "type": "pdf"
│   │       }
│   │
│   ├── Banned Substances (category: "banned")
│   │   ├── page_content: "[محظورات...][text]"
│   │   └── metadata: {
│   │         "source": "محظورات التجميل",
│   │         "category": "banned",
│   │         "type": "excel"
│   │       }
│   │
│   └── Other Sources (category: "generic_json")
│
└── Search Methods:
    ├── similarity_search() - Semantic search
    ├── Filtered search - By category/article
    └── Top-K retrieval (K=8)
```

## Request-Response Flow

### Example 1: Direct Article Query

```
User Query: "ما هي المادة الرابعة؟"

1. Query enters stream_response()
   └─> ArabicArticleParser.extract_article_number("ما هي المادة الرابعة؟")
       └─> Returns: "4"

2. Direct article fetch
   └─> get_article_doc("4")
       └─> vector_store.similarity_search(
             query="المادة 4",
             k=3,
             filter={"article": "4", "category": "regulation"}
           )
       └─> Returns: Document with Article 4

3. Format output
   └─> format_article_output(doc)
       └─> "**نص المادة (4) من لوائح التجميل**\n\n[text]"

4. Add sources
   └─> sources_footer_once([doc])
       └─> "\n\n**المصدر:** لوائح التجميل"

5. Stream to user
```

### Example 2: RAG Query

```
User Query: "ما هي متطلبات التسجيل؟"

1. Query enters stream_response()
   └─> ArabicArticleParser.extract_article_number("ما هي متطلبات التسجيل؟")
       └─> Returns: None (no article number)

2. RAG Pipeline
   └─> build_retriever(source_choice)
       └─> retriever with filter {"category": "regulation"}

3. Retrieve documents
   └─> retriever.invoke("query: ما هي متطلبات التسجيل؟")
       └─> Returns top 8 documents

4. Build context
   └─> build_knowledge(top_docs[:3])
       └─> Formatted context string

5. Generate answer
   └─> llm.stream([HumanMessage(prompt)])
       └─> Stream chunks to user

6. Add sources
   └─> sources_footer_once(top_docs)

7. Stream complete response
```

## Configuration Management

```
config.py
├── Environment Variables (.env)
│   ├── OPENROUTER_API_KEY
│   ├── EMBEDDING_MODEL
│   ├── LLM_MODEL
│   ├── CHROMA_PATH
│   └── ...
│
├── Validation
│   └── validate_config()
│       ├── Check API keys
│       ├── Verify paths exist
│       └── Validate parameters
│
└── Constants
    ├── CHUNK_SIZE = 1000
    ├── CHUNK_OVERLAP = 150
    ├── RETRIEVAL_K = 8
    └── BATCH_SIZE = 2000
```

## Evaluation Pipeline

```
evaluation.py
├── SFDAEvaluator
│   ├── load_test_queries() - Load test set
│   │
│   ├── evaluate_retrieval() - Precision/Recall/F1
│   │   ├── Expected articles
│   │   ├── Retrieved articles
│   │   └── Calculate metrics
│   │
│   ├── evaluate_citation() - Citation accuracy
│   │   ├── Check source footer
│   │   ├── Extract mentioned articles
│   │   └── Verify against retrieved
│   │
│   ├── generate_answer() - Full query test
│   │
│   └── generate_report() - Markdown report
│       ├── Aggregate metrics
│       ├── Query-level results
│       └── Recommendations
```

## Key Design Decisions

### 1. **No Chunking for Regulations**
- **Reason**: Articles are atomic units, splitting them breaks context
- **Implementation**: Regulation documents kept whole, only generic JSON/Excel chunked

### 2. **Dual Query Strategy**
- **Article Queries**: Direct fetch with metadata filtering
- **General Queries**: RAG with semantic search
- **Reason**: Faster and more accurate for specific article requests

### 3. **Arabic-First Design**
- **Word-to-Number mapping**: Handles "الرابعة" → "4"
- **Text normalization**: Removes tatweel, merges spaced letters
- **RTL interface**: Proper Arabic text display

### 4. **Source Attribution**
- **Mandatory citations**: Every answer includes source footer
- **Metadata tracking**: Source file, article number, category
- **Reason**: Trust and verifiability for field inspectors

### 5. **Streaming Responses**
- **UX benefit**: Users see partial answers immediately
- **Implementation**: LLM streaming + Gradio ChatInterface
- **Reason**: Better perceived performance

## Deployment Architecture (Optional)

```
┌─────────────────────────────────────────┐
│         Streamlit Cloud / Render        │
│  ┌───────────────────────────────────┐  │
│  │  Docker Container                 │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Python 3.9                 │  │  │
│  │  │  + Dependencies             │  │  │
│  │  │  + ChromaDB (local)         │  │  │
│  │  │  + Gradio App               │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└────────────┬────────────────────────────┘
             │
             │ HTTPS
             │
             ▼
      ┌──────────────┐
      │   End Users  │
      │ (Inspectors) │
      └──────────────┘
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Gradio 5.39 | Web UI with Arabic support |
| **Backend** | Python 3.9+ | Application logic |
| **LLM** | DeepSeek Chat (via OpenRouter) | Answer generation |
| **Embeddings** | multilingual-e5-large | Arabic semantic search |
| **Vector DB** | ChromaDB 0.4.24 | Document retrieval |
| **Framework** | LangChain 0.2.1 | RAG orchestration |
| **Config** | python-dotenv | Environment management |
| **Evaluation** | Custom + Ragas (optional) | Quality metrics |

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **First Query Latency** | ~5-10s | Model loading time |
| **Subsequent Queries** | ~2-3s | Cached models |
| **Retrieval Time** | <500ms | ChromaDB similarity search |
| **LLM Generation** | ~1-2s | Streaming response |
| **Embedding Model Size** | 1.12 GB | multilingual-e5-large |
| **Vector Store Size** | ~100-500 MB | Depends on documents |
| **Memory Usage** | ~2-4 GB | Models + vector store |

## Security Considerations

1. **API Key Management**: Environment variables, not hardcoded
2. **Input Validation**: Query length limits, sanitization
3. **Source Attribution**: All answers cite sources to prevent misinformation
4. **No External Data**: Responses strictly from indexed documents
5. **Audit Trail**: Logging for debugging and monitoring

---

**Document Version**: 1.0
**Last Updated**: 2026-01-30
**Prepared for**: سدايا AI Bootcamp Capstone Project
