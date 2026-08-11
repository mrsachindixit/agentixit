# LangChain Component Architecture - Detailed Ecosystem Diagram

> Based on the official LangChain documentation: [Component Architecture](https://docs.langchain.com/oss/python/langchain/component-architecture)

![LangChain ecosystem diagram](../images/langchain-ecosystem.png)
*Figure: LangChain component ecosystem — orchestration, memory, generation, tools, retrieval, embedding, and input processing layers*

<details>
<summary>Mermaid Source</summary>

```mermaid
graph TB
    subgraph ORCH["🎯 ORCHESTRATION LAYER"]
        direction TB
        AGENTS["🤖 Agents<br/><i>create_agent · ReAct Loop</i>"]
        MIDDLEWARE["⚙️ Middleware<br/><i>@before_model · @after_model<br/>@wrap_tool_call · @wrap_model_call<br/>@dynamic_prompt</i>"]
        MULTIAGENT["👥 Multi-Agent Systems<br/><i>Supervisor Agent<br/>Specialist Agents · Subgraphs</i>"]
        AGENTS --> MIDDLEWARE
        AGENTS --> MULTIAGENT
    end

    subgraph MEMORY["🧠 MEMORY LAYER"]
        direction TB
        STM["📝 Short-Term Memory<br/><i>AgentState · Message History<br/>Custom State Schema</i>"]
        LTM["💾 Long-Term Memory<br/><i>BaseStore · InMemoryStore<br/>PostgresStore</i>"]
        CHECKPOINTER["🔖 Checkpointers<br/><i>InMemorySaver · PostgresSaver<br/>SQLite · Azure Cosmos DB</i>"]
        MEMSTRAT["📋 Memory Strategies<br/><i>Trim Messages<br/>Delete Messages<br/>SummarizationMiddleware</i>"]
        STM --- CHECKPOINTER
        STM --- MEMSTRAT
    end

    subgraph GEN["🤖 GENERATION LAYER"]
        direction TB
        MODELS["💬 Chat Models<br/><i>invoke · stream · batch</i>"]
        PROVIDERS["☁️ Model Providers<br/><i>OpenAI · Anthropic · Google Gemini<br/>Azure · AWS Bedrock · Ollama<br/>HuggingFace · Mistral · Cohere</i>"]
        CAPABILITIES["✨ Model Capabilities<br/><i>Tool Calling · Structured Output<br/>Multimodal · Reasoning<br/>Prompt Caching · Server-Side Tools</i>"]
        STRUCTURED["📐 Structured Output<br/><i>ProviderStrategy<br/>ToolStrategy<br/>Pydantic · TypedDict · JSON Schema</i>"]
        MODELS --> PROVIDERS
        MODELS --> CAPABILITIES
        MODELS --> STRUCTURED
    end

    subgraph TOOLS_LAYER["🔧 TOOLS LAYER"]
        direction TB
        TOOLS["🛠️ Tools<br/><i>@tool Decorator · ToolNode<br/>Static Tools · Dynamic Tools</i>"]
        TOOLCTX["📊 Tool Runtime Context<br/><i>State · Context · Store<br/>Stream Writer · Config</i>"]
        PREBUILT["📦 Prebuilt Tools & Toolkits<br/><i>Web Search · Code Interpreter<br/>Database Access · APIs</i>"]
        TOOLRETURN["↩️ Tool Returns<br/><i>String · Object · Command</i>"]
        TOOLS --> TOOLCTX
        TOOLS --> PREBUILT
        TOOLS --> TOOLRETURN
    end

    subgraph RETRIEVAL["🔍 RETRIEVAL LAYER"]
        direction TB
        RETRIEVERS["🔎 Retrievers<br/><i>Vector Retrievers · Web Retrievers<br/>BM25 · Wikipedia · Arxiv<br/>Tavily · Amazon Kendra<br/>Elasticsearch · Azure AI Search</i>"]
        RAG["📚 RAG Architectures<br/><i>2-Step RAG · Agentic RAG<br/>Hybrid RAG</i>"]
        RETRIEVERS --> RAG
    end

    subgraph EMBED["🔢 EMBEDDING & STORAGE LAYER"]
        direction TB
        EMBMODELS["🧮 Embedding Models<br/><i>OpenAI · Azure OpenAI<br/>Google · Ollama · Cohere<br/>Voyage AI · HuggingFace<br/>Mistral · NVIDIA · Nomic</i>"]
        VECSTORES["🗄️ Vector Stores<br/><i>Chroma · FAISS · Pinecone<br/>Qdrant · Milvus · PGVector<br/>Elasticsearch · Weaviate<br/>MongoDB Atlas · Astra DB<br/>InMemoryVectorStore</i>"]
        EMBMODELS -->|"vectors"| VECSTORES
    end

    subgraph INPUT["📥 INPUT PROCESSING LAYER"]
        direction TB
        DOCLOADERS["📄 Document Loaders<br/><i><b>Web:</b> URLs · Sitemap · Firecrawl<br/><b>PDF:</b> PyPDF · PyMuPDF · PDFPlumber<br/><b>Cloud:</b> S3 · GCS · Azure Blob<br/><b>Apps:</b> Notion · Slack · GitHub<br/><b>Files:</b> CSV · JSON · HTML</i>"]
        SPLITTERS["✂️ Text Splitters<br/><i>Recursive Character<br/>Token-based · Semantic<br/>Markdown · HTML · Code</i>"]
        DOCLOADERS -->|"Documents"| SPLITTERS
    end

    %% Cross-layer interactions
    INPUT -->|"chunked docs"| EMBED
    EMBED -->|"similarity search"| RETRIEVAL
    RETRIEVAL -->|"relevant context"| GEN
    GEN -->|"reasoning engine"| ORCH
    TOOLS_LAYER -->|"tool results"| GEN
    GEN -->|"tool calls"| TOOLS_LAYER
    ORCH -->|"manages"| MEMORY
    MEMORY -->|"context"| GEN
    RETRIEVAL -.->|"retrieval tool"| TOOLS_LAYER

    %% Styling
    classDef orchStyle fill:#4A90D9,stroke:#2C5F8A,color:#fff,stroke-width:2px
    classDef memStyle fill:#9B59B6,stroke:#6C3483,color:#fff,stroke-width:2px
    classDef genStyle fill:#E74C3C,stroke:#A93226,color:#fff,stroke-width:2px
    classDef toolStyle fill:#F39C12,stroke:#B7770D,color:#fff,stroke-width:2px
    classDef retStyle fill:#27AE60,stroke:#1E8449,color:#fff,stroke-width:2px
    classDef embStyle fill:#2980B9,stroke:#1A5276,color:#fff,stroke-width:2px
    classDef inpStyle fill:#16A085,stroke:#0E6655,color:#fff,stroke-width:2px

    class ORCH orchStyle
    class MEMORY memStyle
    class GEN genStyle
    class TOOLS_LAYER toolStyle
    class RETRIEVAL retStyle
    class EMBED embStyle
    class INPUT inpStyle
```

</details>

---

## Component Categories Reference

### 1. Models (Generation Layer)
| Aspect | Details |
|--------|---------|
| **Core Interface** | `init_chat_model()`, `BaseChatModel` |
| **Key Methods** | `invoke()`, `stream()`, `batch()`, `batch_as_completed()` |
| **Providers** | OpenAI, Anthropic, Google Gemini, Azure, AWS Bedrock, Ollama, HuggingFace, Mistral, Cohere |
| **Capabilities** | Tool Calling, Structured Output, Multimodal (image/audio/video), Reasoning, Prompt Caching, Server-Side Tools |
| **Structured Output** | `ProviderStrategy` (native), `ToolStrategy` (via tool calling), Pydantic / TypedDict / JSON Schema |
| **Advanced** | Model Profiles, Configurable Models, Rate Limiting, Log Probabilities, Token Usage Tracking |
| **Parameters** | `model`, `api_key`, `temperature`, `max_tokens`, `timeout`, `max_retries` |

### 2. Tools
| Aspect | Details |
|--------|---------|
| **Creation** | `@tool` decorator, Pydantic `args_schema`, JSON Schema |
| **Types** | Static Tools, Dynamic Tools (state-filtered, runtime-registered) |
| **Execution** | `ToolNode` (parallel execution, error handling, state injection) |
| **Runtime Context** | `ToolRuntime` → State, Context, Store, Stream Writer, Config, Tool Call ID |
| **Return Types** | String, Object (dict), `Command` (state updates) |
| **Error Handling** | `@wrap_tool_call` middleware, `handle_tool_errors` on ToolNode |
| **Prebuilt** | Web Search, Code Interpreter, Database Access, APIs, Server-side tools |
| **Routing** | `tools_condition` for conditional graph edges |

### 3. Agents (Orchestration Layer)
| Aspect | Details |
|--------|---------|
| **Core** | `create_agent()` → LangGraph-based runtime |
| **Pattern** | ReAct loop (Reasoning + Acting) |
| **Model Selection** | Static (fixed model) or Dynamic (middleware-based routing) |
| **Tool Management** | Static tools, Dynamic tools (state/context/runtime filtered) |
| **System Prompt** | Static string, `SystemMessage`, Dynamic via `@dynamic_prompt` middleware |
| **Structured Output** | `response_format` → `ProviderStrategy` / `ToolStrategy` |
| **Streaming** | `stream()` with `stream_mode="values"` |
| **Multi-Agent** | Supervisor + Specialist agents, named subgraphs |

### 4. Memory
| Aspect | Details |
|--------|---------|
| **Short-Term** | `AgentState` (messages + custom fields), thread-scoped |
| **Long-Term** | `BaseStore` → `InMemoryStore`, `PostgresStore` (namespace/key pattern) |
| **Checkpointers** | `InMemorySaver`, `PostgresSaver`, SQLite, Azure Cosmos DB |
| **Strategies** | Trim Messages, Delete Messages, `SummarizationMiddleware` |
| **Access Points** | Tools (`ToolRuntime`), Prompts (`@dynamic_prompt`), Middleware (`@before_model`, `@after_model`) |

### 5. Retrievers
| Aspect | Details |
|--------|---------|
| **Interface** | Returns `Document` objects given string query |
| **Custom Index** | `AmazonKnowledgeBasesRetriever`, `ElasticsearchRetriever`, `AzureAISearchRetriever`, `NVIDIARetriever` |
| **External Index** | `ArxivRetriever`, `TavilySearchAPIRetriever`, `WikipediaRetriever`, BM25 |
| **From Vector Stores** | All vector stores can be cast to retrievers via `.as_retriever()` |
| **RAG Architectures** | 2-Step RAG, Agentic RAG, Hybrid RAG |

### 6. Document Processing (Input Layer)
| Aspect | Details |
|--------|---------|
| **Loaders** | Web (URLs, Sitemap, Firecrawl, Spider), PDF (PyPDF, PyMuPDF, PDFPlumber, PDFMiner), Cloud (S3, GCS, Azure Blob, Google Drive, OneDrive, SharePoint), Apps (Notion, Slack, GitHub, Figma, Trello), Files (CSV, JSON, HTML, Unstructured, Docling) |
| **Text Splitters** | Recursive Character, Token-based, Semantic, Markdown, HTML, Code |
| **Interface** | `load()`, `lazy_load()` → `Document(page_content, metadata)` |

### 7. Vector Stores (Embedding & Storage Layer)
| Aspect | Details |
|--------|---------|
| **Interface** | `add_documents()`, `delete()`, `similarity_search()` |
| **Implementations** | Chroma, FAISS, Pinecone, Qdrant, Milvus, PGVector, Elasticsearch, Weaviate, MongoDB Atlas, Astra DB, InMemoryVectorStore, Azure Cosmos DB, CockroachDB |
| **Embedding Models** | OpenAI, Azure OpenAI, Google, Ollama, Cohere, Voyage AI, HuggingFace, Mistral, NVIDIA, Nomic |
| **Similarity Metrics** | Cosine Similarity, Euclidean Distance, Dot Product |
| **Features** | Metadata filtering, HNSW indexing, Embedding caching (`CacheBackedEmbeddings`) |

---

## Data Flow Summary

```
Input Sources → Document Loaders → Text Splitters → Embedding Models → Vector Stores
                                                                            ↓
User Query → Embedding Model → Query Vector → Retriever → Relevant Context → Chat Model
                                                                                ↓
                                                            Agent (ReAct Loop) ↔ Tools
                                                                    ↓
                                                            Memory (Short/Long-Term)
                                                                    ↓
                                                              Final Response
```
