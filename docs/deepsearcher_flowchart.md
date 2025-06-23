# DeepSearcher Flowchart: From `main.py` to RAG Query Resolution


## 1. Application Entry and Initialization Flow

```mermaid
flowchart TD
    A[Start: main.py] --> B[FastAPI initialization]
    B --> C["init_config(config: Configuration)"]

    C -->|"Creates and initializes"| D["ModuleFactory(config)"]
    D --> E1["create_llm()"]
    D --> E2["create_embedding()"]
    D --> E3["create_vector_db()"]
    D --> E4["create_file_loader()"] 
    D --> E5["create_web_crawler()"]
    
    E1 -->|"Returns: BaseLLM"| F1[OpenAI LLM]
    E2 -->|"Returns: BaseEmbedding"| F2[OpenAI Embedding]
    E3 -->|"Returns: BaseVectorDB"| F3[FAISS Vector DB]
    E4 -->|"Returns: BaseLoader"| F4[File Loader]
    E5 -->|"Returns: BaseCrawler"| F5[Web Crawler]
    
    subgraph "RAG Agent Initialization"
        F1 & F2 & F3 --> G1["DeepSearch(llm, embedding_model, vector_db, max_iter, route_collection, text_window_splitter)"]
        F1 & F2 & F3 --> G2["ChainOfRAG(llm, embedding_model, vector_db, max_iter, route_collection, text_window_splitter)"]
        F1 & F2 & F3 --> G3["NaiveRAG(llm, embedding_model, vector_db, top_k, route_collection, text_window_splitter)"]
        G1 & G2 --> H["RAGRouter(llm, rag_agents=[DeepSearch, ChainOfRAG])"]
    end

    C -->|"Initialized global variables"| I[FastAPI app ready]
```

## 2. Document Loading Flow

```mermaid
flowchart TD
    A["/load-files/ Endpoint"] -->|"Input: paths, collection_name, collection_description, batch_size"| B["load_files()"]
    B -->|"Calls"| C["load_from_local_files(paths_or_directory, collection_name, collection_description, batch_size)"]
    
    D["/load-website/ Endpoint"] -->|"Input: urls, collection_name, collection_description, batch_size"| E["load_website()"]
    E -->|"Calls"| F["load_from_website(urls, collection_name, collection_description, batch_size)"]
    
    subgraph "Knowledge Base Loading Process"
        C --> G["vector_db.init_collection(dim, collection, description, force_new_collection)"]
        G -->|"Creates FAISS index"| H["FAISS index initialized"]
        
        C --> I["file_loader.load_file(path) or file_loader.load_directory(path)"]
        I -->|"Returns: List[Document]"| J["all_docs"]
        
        F --> K["web_crawler.crawl_urls(urls, **kwargs)"]
        K -->|"Returns: List[Document]"| J
        
        J -->|"Input: documents, chunk_size, chunk_overlap"| L["split_docs_to_chunks(documents, chunk_size, chunk_overlap)"]
        L -->|"Uses"| M["_sentence_window_split(split_docs, original_document, offset)"]
        M -->|"Returns: List[Chunk]"| N["chunks"]
        
        N -->|"Input: chunks, batch_size"| O["embedding_model.embed_chunks(chunks, batch_size)"]
        O -->|"Returns: List[Chunk] with embeddings"| P["embedded_chunks"]
        
        P -->|"Input: collection, chunks"| Q["vector_db.insert_data(collection, chunks)"]
        Q -->|"Adds vectors to FAISS index"| R["Knowledge base updated"]
    end
```

## 3. Query Execution Flow

```mermaid
flowchart TD
    A["/query/ Endpoint"] -->|"Input: original_query, max_iter"| B["perform_query()"]
    B -->|"Calls"| C["query(original_query, max_iter)"]
    
    C -->|"Gets default_searcher"| D["default_searcher.query(original_query, max_iter)"]
    D -->|"default_searcher is RAGRouter"| E["RAGRouter.query(original_query, max_iter)"]
    
    subgraph "RAG Agent Selection Process"
        E -->|"Calls"| F["RAGRouter._route(original_query)"]
        F -->|"Prompt to LLM"| G["llm.chat(router_prompt)"]
        G -->|"Returns: selected_agent_index"| H["Select DeepSearch or ChainOfRAG"]
        H -->|"Returns: Tuple[RAGAgent, token_count]"| I["selected_agent, n_token_router"]
    end
    
    E -->|"With selected agent"| J["selected_agent.query(original_query, max_iter)"]
    J -->|"Returns: Tuple[str, List[RetrievalResult], int]"| K["answer, retrieved_results, n_token_retrieval"]
    
    E -->|"Returns: Tuple[str, List[RetrievalResult], int]"| L["answer, retrieved_results, n_token_router + n_token_retrieval"]
    C -->|"Returns to API"| M["{'result': result_text, 'consume_token': consume_token}"]
```

## 4. DeepSearch Agent Workflow

```mermaid
flowchart TD
    A["DeepSearch.query(query, **kwargs)"] -->|"Calls"| B["DeepSearch.retrieve(query, **kwargs)"]
    B -->|"Uses asyncio"| C["asyncio.run(async_retrieve(query, **kwargs))"]
    
    C -->|"First step"| D["_generate_sub_queries(original_query)"]
    D -->|"Prompt to LLM"| E["llm.chat(SUB_QUERY_PROMPT)"]
    E -->|"Returns: List[str], token_count"| F["sub_queries, used_token"]
    
    C -->|"For each query in sub_queries"| G["_search_chunks_from_vectordb(query, sub_queries)"]
    
    subgraph "Collection Selection"
        G -->|"If route_collection=True"| H["collection_router.invoke(query, dim)"]
        H -->|"Returns: List[str], int"| I["selected_collections, n_token_route"]
    end
    
    subgraph "Vector Search Process"
        G -->|"For each collection"| J["embedding_model.embed_query(query)"]
        J -->|"Returns: List[float]"| K["query_vector"]
        K -->|"Input: collection, vector, query_text"| L["vector_db.search_data(...)"]
        L -->|"Returns: List[RetrievalResult]"| M["retrieved_results"]
    end
    
    subgraph "Result Reranking"
        G --> N["For each result"]
        N -->|"Input: query, result"| O["llm.chat(RERANK_PROMPT)"]
        O -->|"If YES"| P["Keep result"]
        O -->|"If NO"| Q["Discard result"]
    end
    
    G -->|"Returns: List[RetrievalResult], int"| R["all_retrieved_results, consume_tokens"]
    
    C -->|"If not final iteration"| S["_generate_gap_queries(original_query, all_sub_queries, all_search_res)"]
    S -->|"Prompt to LLM"| T["llm.chat(REFLECT_PROMPT)"]
    T -->|"Returns: List[str], int"| U["sub_gap_queries, consumed_token"]
    
    C -->|"After all iterations"| V["deduplicate_results(all_search_res)"]
    V -->|"Returns: List[RetrievalResult], int, dict"| W["all_search_res, total_tokens, additional_info"]
    
    A -->|"After retrieve"| X["llm.chat(SUMMARY_PROMPT)"]
    X -->|"Returns: str"| Y["final_answer"]
    A -->|"Returns: Tuple[str, List[RetrievalResult], int]"| Z["final_answer, all_retrieved_results, n_token_retrieval + chat_response.total_tokens"]
```

## 5. ChainOfRAG Agent Workflow

```mermaid
flowchart TD
    A["ChainOfRAG.query(query, **kwargs)"] -->|"Calls"| B["ChainOfRAG.retrieve(query, **kwargs)"]
    
    subgraph "Iterative Chain Process"
        B -->|"For each iteration"| C["_reflect_get_subquery(query, intermediate_contexts)"]
        C -->|"Prompt to LLM"| D["llm.chat(FOLLOWUP_QUERY_PROMPT)"]
        D -->|"Returns: str, int"| E["followup_query, n_token0"]
        
        E -->|"With followup query"| F["_retrieve_and_answer(followup_query)"]
        F -->|"Select collections"| G["collection_router.invoke(query, dim)"]
        G -->|"Returns: List[str], int"| H["selected_collections, n_token_route"]
        
        H -->|"For each collection"| I["vector_db.search_data(collection, vector, query_text)"]
        I -->|"Returns: List[RetrievalResult]"| J["retrieved_results"]
        
        F -->|"Generate answer"| K["llm.chat(INTERMEDIATE_ANSWER_PROMPT)"]
        K -->|"Returns: str, List[RetrievalResult], int"| L["intermediate_answer, retrieved_results, consume_tokens"]
        
        L -->|"Find supporting docs"| M["_get_supported_docs(retrieved_results, query, intermediate_answer)"]
        M -->|"Returns: List[RetrievalResult], int"| N["supported_retrieved_results, token_usage"]
        
        B -->|"If early_stopping=True"| O["_check_has_enough_info(query, intermediate_contexts)"]
        O -->|"Returns: bool, int"| P["has_enough_info, token_usage"]
    end
    
    B -->|"Returns: List[RetrievalResult], int, dict"| Q["all_retrieved_results, token_usage, {'intermediate_context': intermediate_contexts}"]
    
    A -->|"Generate final answer"| R["llm.chat(FINAL_ANSWER_PROMPT)"]
    R -->|"Returns: str"| S["final_answer"]
    A -->|"Returns: Tuple[str, List[RetrievalResult], int]"| T["final_answer, all_retrieved_results, n_token_retrieval + chat_response.total_tokens"]
```

## 6. NaiveRAG Agent Workflow

```mermaid
flowchart TD
    A["NaiveRAG.query(query, **kwargs)"] -->|"Calls"| B["NaiveRAG.retrieve(query, **kwargs)"]
    
    subgraph "Simple Retrieval Process"
        B -->|"If route_collection=True"| C["collection_router.invoke(query, dim)"]
        C -->|"Returns: List[str], int"| D["selected_collections, n_token_route"]
        
        D -->|"For each collection"| E["embedding_model.embed_query(query)"]
        E -->|"Returns: List[float]"| F["query_vector"]
        
        F -->|"Input: collection, vector, top_k"| G["vector_db.search_data(collection, query_vector, top_k=max(top_k/len(selected_collections), 1))"]
        G -->|"Returns: List[RetrievalResult]"| H["retrieved_results"]
    end
    
    B -->|"Returns: List[RetrievalResult], int, dict"| I["all_retrieved_results, consume_tokens, {}"]
    
    A -->|"Format chunks"| J["For each chunk, get text or wider_text"]
    J -->|"Format for prompt"| K["mini_chunk_str"]
    
    K -->|"Generate summary"| L["llm.chat(SUMMARY_PROMPT)"]
    L -->|"Returns: ChatResponse"| M["final_answer"]
    
    A -->|"Returns: Tuple[str, List[RetrievalResult], int]"| N["final_answer, all_retrieved_results, n_token_retrieval + char_response.total_tokens"]
```

## 7. FAISS Vector Database Operations

```mermaid
flowchart TD
    A["FAISSDB.init_collection(dim, collection, description, force_new_collection)"] --> B{"force_new_collection or self.index is None?"}
    B -->|"Yes"| C["Create new FAISS index: faiss.IndexFlatL2(dim)"]
    C --> D["self.texts = [], self.references = [], self.metadatas = [], self.embeddings = []"]
    D --> E["self.save()"]
    B -->|"No"| F["Use existing index"]
    
    G["FAISSDB.insert_data(collection, chunks)"] --> H["Extract vectors from chunks"]
    H --> I["self.index.add(vectors_np)"]
    I --> J["Store text, reference, metadata, embedding"]
    J --> K["self.save()"]
    
    L["FAISSDB.search_data(collection, vector, top_k)"] --> M{"self.index is None or empty?"}
    M -->|"Yes"| N["Return []"]
    M -->|"No"| O["self.index.search(query, top_k)"]
    O --> P["Convert results to RetrievalResult objects"]
    P -->|"Returns: List[RetrievalResult]"| Q["results"]
    
    R["FAISSDB.save()"] --> S["faiss.write_index(self.index, INDEX_FILE)"]
    S --> T["pickle.dump(metadata to META_FILE)"]
    
    U["FAISSDB.load()"] --> V["faiss.read_index(INDEX_FILE)"]
    V --> W["Load metadata from META_FILE"]
```

## 8. OpenAI Embedding Operations

```mermaid
flowchart TD
    A["OpenAIEmbedding.embed_query(text)"] --> B["Check if Azure or regular OpenAI"]
    B --> C["client.embeddings.create(input=[text], model=model)"]
    C -->|"Returns: List[float]"| D["embedding vector"]
    
    E["OpenAIEmbedding.embed_documents(texts)"] --> F["Check if Azure or regular OpenAI"]
    F --> G["client.embeddings.create(input=texts, model=model)"]
    G -->|"Returns: List[List[float]]"| H["list of embedding vectors"]
    
    I["OpenAIEmbedding.embed_chunks(chunks, batch_size)"] --> J["Extract texts from chunks"]
    J --> K["Process in batches of batch_size"]
    K --> L["self.embed_documents(batch_texts)"]
    L --> M["Update chunks with embeddings"]
    M -->|"Returns: List[Chunk]"| N["embedded chunks"]
```

## 9. Document Splitting Operations

```mermaid
flowchart TD
    A["split_docs_to_chunks(documents, chunk_size, chunk_overlap)"] --> B["Create RecursiveCharacterTextSplitter"]
    B --> C["For each document"]
    C --> D["text_splitter.split_documents([doc])"]
    D -->|"Returns: List[Document]"| E["split_docs"]
    
    E --> F["_sentence_window_split(split_docs, doc, offset=300)"]
    F --> G["For each split doc"]
    G --> H["Find position in original document"]
    H --> I["Extract wider context around split"]
    I --> J["Create Chunk object"]
    J -->|"Returns: List[Chunk]"| K["chunks with context windows"]
    
    C -->|"After processing all documents"| L["all_chunks"]
    L -->|"Returns: List[Chunk]"| M["final chunks for embedding"]
```

## Complete End-to-End Query Flow: From User Request to Response

```mermaid
flowchart TD
    A["User Query: /query/ endpoint"] -->|"Input: original_query, max_iter"| B["perform_query()"]
    B --> C["query(original_query, max_iter)"]
    C --> D["RAGRouter.query(original_query, max_iter)"]
    
    D --> E["RAGRouter._route(original_query)"]
    E --> F["llm.chat(ROUTER_PROMPT)"]
    F -->|"Returns: selected_agent"| G["selected_agent = DeepSearch or ChainOfRAG"]
    
    G --> H{"Selected agent?"}
    H -->|"DeepSearch"| I1["DeepSearch.query(original_query, max_iter)"]
    H -->|"ChainOfRAG"| I2["ChainOfRAG.query(original_query, max_iter)"]
    
    subgraph "DeepSearch Process"
        I1 --> J1["DeepSearch.retrieve(original_query)"]
        J1 --> K1["_generate_sub_queries(original_query)"]
        K1 --> L1["For each sub-query and iteration:"]
        L1 --> M1["_search_chunks_from_vectordb(query, sub_queries)"]
        M1 --> N1["collection_router.invoke(query, dim)"]
        N1 --> O1["vector_db.search_data(collection, vector, query_text)"]
        O1 --> P1["Re-rank results with LLM"]
        L1 --> Q1["_generate_gap_queries(original_query, all_sub_queries, all_search_res)"]
        J1 -->|"Returns: List[RetrievalResult], int, dict"| R1["all_search_res, total_tokens, additional_info"]
        
        R1 --> S1["llm.chat(SUMMARY_PROMPT)"]
        S1 -->|"Returns: final answer"| T1["final_answer"]
    end
    
    subgraph "ChainOfRAG Process"
        I2 --> J2["ChainOfRAG.retrieve(original_query)"]
        J2 --> K2["For each iteration:"]
        K2 --> L2["_reflect_get_subquery(query, intermediate_contexts)"]
        L2 --> M2["_retrieve_and_answer(followup_query)"]
        M2 --> N2["collection_router.invoke(query, dim)"]
        N2 --> O2["vector_db.search_data(collection, vector, query_text)"]
        O2 --> P2["_get_supported_docs(retrieved_results, query, intermediate_answer)"]
        K2 --> Q2["_check_has_enough_info(query, intermediate_contexts)"]
        J2 -->|"Returns: List[RetrievalResult], int, dict"| R2["all_retrieved_results, token_usage, additional_info"]
        
        R2 --> S2["llm.chat(FINAL_ANSWER_PROMPT)"]
        S2 -->|"Returns: final answer"| T2["final_answer"]
    end
    
    T1 & T2 -->|"Returns to RAGRouter"| U["answer, retrieved_results, token_usage"]
    U --> V["Return to API: {'result': result_text, 'consume_token': consume_token}"]
```

## Legend and Function Details

### Key Components

1. **FastAPI**: Web framework that handles HTTP requests
2. **Configuration**: Manages settings and component initialization
3. **ModuleFactory**: Creates instances of LLMs, embeddings, etc.
4. **RAGRouter**: Routes queries to appropriate RAG agents
5. **DeepSearch**: Complex RAG agent for comprehensive information retrieval
6. **ChainOfRAG**: RAG agent that decomposes queries into iterative steps
7. **NaiveRAG**: Simple RAG agent for basic retrieval operations
8. **FAISSDB**: Vector database for storing and searching embeddings
9. **OpenAIEmbedding**: Generates embeddings using OpenAI's API

### Key Function Inputs/Outputs

| Function | Inputs | Outputs | Description |
|---------|--------|---------|-------------|
| `init_config()` | `config: Configuration` | None (sets globals) | Initializes all system components |
| `load_from_local_files()` | `paths_or_directory, collection_name, collection_description, batch_size` | None | Loads documents from files into vector DB |
| `load_from_website()` | `urls, collection_name, collection_description, batch_size` | None | Loads documents from websites into vector DB |
| `query()` | `original_query: str, max_iter: int` | `Tuple[str, List[RetrievalResult], int]` | Main query function that routes to RAG agents |
| `RAGRouter._route()` | `query: str` | `Tuple[RAGAgent, int]` | Selects best RAG agent for query |
| `DeepSearch.retrieve()` | `original_query: str, **kwargs` | `Tuple[List[RetrievalResult], int, dict]` | Retrieves documents using sub-queries and reflection |
| `ChainOfRAG.retrieve()` | `query: str, **kwargs` | `Tuple[List[RetrievalResult], int, dict]` | Retrieves documents using iterative queries |
| `split_docs_to_chunks()` | `documents: List[Document], chunk_size: int, chunk_overlap: int` | `List[Chunk]` | Splits documents into smaller chunks |
| `embed_chunks()` | `chunks: List[Chunk], batch_size: int` | `List[Chunk]` | Adds embeddings to document chunks |
| `vector_db.search_data()` | `collection: str, vector: List[float], top_k: int` | `List[RetrievalResult]` | Searches for similar vectors in database |
