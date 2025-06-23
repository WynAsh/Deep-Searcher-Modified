# DeepSearcher: Detailed Execution Flow

This document traces the complete execution flow of DeepSearcher, starting from `main.py`, with a focus on the RAG implementation and knowledge base querying. Each function is documented with its file path, purpose, and implementation details.

## Table of Contents
1. [Application Startup Flow](#1-application-startup-flow)
2. [Knowledge Base Loading Flow](#2-knowledge-base-loading-flow)
3. [Query Execution Flow](#3-query-execution-flow)
4. [RAG Implementation Details](#4-rag-implementation-details)
5. [Vector Database Operations](#5-vector-database-operations)

## 1. Application Startup Flow

### 1.1 Entry Point: `main.py`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\main.py`

The application starts in `main.py`, which:
1. Imports necessary modules
2. Initializes the configuration
3. Sets up FastAPI routes
4. Starts the web server when executed

```python
# Initial configuration setup
config = Configuration()
init_config(config)
```

### 1.2 Configuration Initialization: `init_config()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\configuration.py`

This function creates and initializes all necessary components:
```python
def init_config(config: Configuration):
    global module_factory, llm, embedding_model, file_loader, vector_db, web_crawler, default_searcher, naive_rag
    
    # Create module factory based on configuration
    module_factory = ModuleFactory(config)
    
    # Initialize components
    llm = module_factory.create_llm()               # OpenAI
    embedding_model = module_factory.create_embedding()  # OpenAI embeddings
    file_loader = module_factory.create_file_loader()
    web_crawler = module_factory.create_web_crawler()
    vector_db = module_factory.create_vector_db()   # FAISS
    
    # Initialize RAG components
    default_searcher = RAGRouter(
        llm=llm,
        rag_agents=[
            DeepSearch(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True,
            ),
            ChainOfRAG(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True,
            ),
        ],
    )
    naive_rag = NaiveRAG(
        llm=llm,
        embedding_model=embedding_model,
        vector_db=vector_db,
        top_k=10,
        route_collection=True,
        text_window_splitter=True,
    )
```

### 1.3 Component Creation: `ModuleFactory.create_*`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\configuration.py`

The ModuleFactory creates instances of each component based on configuration:
```python
def _create_module_instance(self, feature: FeatureType, module_name: str):
    class_name = self.config.provide_settings[feature]["provider"]
    module = __import__(module_name, fromlist=[class_name])
    class_ = getattr(module, class_name)
    return class_(**self.config.provide_settings[feature]["config"])

def create_llm(self) -> BaseLLM:
    # Creates OpenAI LLM instance
    return self._create_module_instance("llm", "deepsearcher.llm")

def create_embedding(self) -> BaseEmbedding:
    # Creates OpenAI embedding instance
    return self._create_module_instance("embedding", "deepsearcher.embedding")

def create_vector_db(self) -> BaseVectorDB:
    # Creates FAISS vector database instance
    return self._create_module_instance("vector_db", "deepsearcher.vector_db")
```

## 2. Knowledge Base Loading Flow

### 2.1 Loading Files: `load_files()` API Endpoint
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\main.py`

This FastAPI endpoint loads files into the vector database:
```python
@app.post("/load-files/")
def load_files(paths, collection_name, collection_description, batch_size):
    try:
        load_from_local_files(
            paths_or_directory=paths,
            collection_name=collection_name,
            collection_description=collection_description,
            batch_size=batch_size,
        )
        return {"message": "Files loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2.2 Local File Loading: `load_from_local_files()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\offline_loading.py`

This function processes files and adds them to the vector database:
```python
def load_from_local_files(paths_or_directory, collection_name=None, collection_description=None, force_new_collection=False, chunk_size=1500, chunk_overlap=100, batch_size=256):
    # Get components from configuration
    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader
    
    # Initialize or get collection name
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    
    # Initialize collection in vector database
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )
    
    # Process paths
    if isinstance(paths_or_directory, str):
        paths_or_directory = [paths_or_directory]
    all_docs = []
    
    # Load documents from all paths
    for path in tqdm(paths_or_directory, desc="Loading files"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File or directory '{path}' does not exist.")
        if os.path.isdir(path):
            docs = file_loader.load_directory(path)
        else:
            docs = file_loader.load_file(path)
        all_docs.extend(docs)
    
    # Split documents into chunks
    chunks = split_docs_to_chunks(
        all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Generate embeddings for chunks
    chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
    
    # Insert chunks into vector database
    vector_db.insert_data(collection=collection_name, chunks=chunks)
```

### 2.3 Vector Database Initialization: `init_collection()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\vector_db\faiss_db.py`

This function initializes a FAISS index for the collection:
```python
def init_collection(self, dim, collection=None, description="", force_new_collection=False, *args, **kwargs):
    # Set collection name if provided
    if collection is not None:
        self.collection_name = collection
        
    # Create new collection if needed
    if force_new_collection or self.index is None or self.dim != dim:
        # Initialize FAISS index with specified dimensions
        self.index = faiss.IndexFlatL2(dim)
        self.dim = dim
        
        # Initialize storage arrays
        self.texts = []
        self.references = []
        self.metadatas = []
        self.embeddings = []
        
        log.color_print(f"Created FAISS collection [{self.collection_name}] with dim={dim}")
        self.save()  # Save to disk
```

### 2.4 Generating Embeddings: `embed_chunks()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\embedding\openai_embedding.py`

This function generates embeddings for document chunks using OpenAI:
```python
def embed_chunks(self, chunks, batch_size=256):
    if not chunks:
        return []
    
    text_batch = []
    processed_chunks = []
    
    # Process chunks in batches
    for chunk in chunks:
        text_batch.append(chunk.text)
        
        if len(text_batch) == batch_size:
            # Get embeddings from OpenAI API
            embeddings = self.client.embeddings.create(
                input=text_batch, model=self.model
            ).data
            
            # Add embeddings to chunks
            for i, embedding in enumerate(embeddings):
                chunks[len(processed_chunks) + i].embedding = embedding.embedding
            
            processed_chunks.extend(chunks[len(processed_chunks):len(processed_chunks) + len(text_batch)])
            text_batch = []
    
    # Process remaining chunks
    if text_batch:
        embeddings = self.client.embeddings.create(
            input=text_batch, model=self.model
        ).data
        
        for i, embedding in enumerate(embeddings):
            chunks[len(processed_chunks) + i].embedding = embedding.embedding
        
        processed_chunks.extend(chunks[len(processed_chunks):len(processed_chunks) + len(text_batch)])
    
    return processed_chunks
```

### 2.5 Inserting Data: `insert_data()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\vector_db\faiss_db.py`

This function inserts embedded chunks into the FAISS index:
```python
def insert_data(self, collection=None, chunks=None, batch_size=256, *args, **kwargs):
    if collection is not None:
        self.collection_name = collection
    
    # Extract vectors from chunks
    vectors = [np.array(chunk.embedding, dtype=np.float32) for chunk in chunks]
    if not vectors:
        return
    
    # Stack vectors and add to FAISS index
    vectors_np = np.stack(vectors)
    self.index.add(vectors_np)
    
    # Store text and metadata
    self.texts.extend([chunk.text for chunk in chunks])
    self.references.extend([chunk.reference for chunk in chunks])
    self.metadatas.extend([chunk.metadata for chunk in chunks])
    self.embeddings.extend([chunk.embedding for chunk in chunks])
    
    # Save to disk
    self.save()
```

## 3. Query Execution Flow

### 3.1 Query Endpoint: `perform_query()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\main.py`

This FastAPI endpoint handles query requests:
```python
@app.get("/query/")
def perform_query(
    original_query: str = Query(...),
    max_iter: int = Query(3),
):
    try:
        # Execute query and get results
        result_text, _, consume_token = query(original_query, max_iter)
        return {"result": result_text, "consume_token": consume_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.2 Query Function: `query()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\online_query.py`

This function is the main entry point for querying the knowledge base:
```python
def query(original_query: str, max_iter: int = 3) -> Tuple[str, List[RetrievalResult], int]:
    # Get default searcher from configuration
    default_searcher = configuration.default_searcher
    
    # Execute query using the default searcher
    return default_searcher.query(original_query, max_iter=max_iter)
```

### 3.3 RAG Router: `RAGRouter.query()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\rag_router.py`

This function routes the query to the appropriate RAG agent:
```python
def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
    # Select the most appropriate RAG agent for the query
    agent, n_token_router = self._route(query)
    
    # Execute query using the selected agent
    answer, retrieved_results, n_token_retrieval = agent.query(query, **kwargs)
    
    # Return results with total token usage
    return answer, retrieved_results, n_token_router + n_token_retrieval
```

### 3.4 Agent Selection: `RAGRouter._route()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\rag_router.py`

This function selects the most appropriate RAG agent:
```python
def _route(self, query: str) -> Tuple[RAGAgent, int]:
    # Prepare agent descriptions for the LLM prompt
    description_str = "\n".join(
        [f"[{i + 1}]: {description}" for i, description in enumerate(self.agent_descriptions)]
    )
    
    # Create prompt to select agent
    prompt = RAG_ROUTER_PROMPT.format(query=query, description_str=description_str)
    
    # Ask LLM to select the best agent
    chat_response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
    
    # Parse response to get agent index
    try:
        selected_agent_index = int(self.llm.remove_think(chat_response.content)) - 1
    except ValueError:
        # Fallback for LLMs that explain their choice
        selected_agent_index = int(self.find_last_digit(self.llm.remove_think(chat_response.content))) - 1
    
    # Get selected agent
    selected_agent = self.rag_agents[selected_agent_index]
    
    # Log selection and return
    log.color_print(f"<think> Select agent [{selected_agent.__class__.__name__}] to answer the query [{query}] </think>\n")
    return self.rag_agents[selected_agent_index], chat_response.total_tokens
```

## 4. RAG Implementation Details

### 4.1 DeepSearch: `DeepSearch.query()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\deep_search.py`

This function executes a query using the DeepSearch RAG agent:
```python
def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
    # Retrieve relevant documents
    all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
    
    # Handle case with no results
    if not all_retrieved_results or len(all_retrieved_results) == 0:
        return f"No relevant information found for query '{query}'.", [], n_token_retrieval
    
    # Get sub-queries used during retrieval
    all_sub_queries = additional_info["all_sub_queries"]
    
    # Prepare text from retrieved chunks
    chunk_texts = []
    for chunk in all_retrieved_results:
        if self.text_window_splitter and "wider_text" in chunk.metadata:
            chunk_texts.append(chunk.metadata["wider_text"])
        else:
            chunk_texts.append(chunk.text)
    
    # Create summary prompt with retrieved information
    summary_prompt = SUMMARY_PROMPT.format(
        question=query,
        mini_questions=all_sub_queries,
        mini_chunk_str=self._format_chunk_texts(chunk_texts),
    )
    
    # Generate answer using LLM
    chat_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
    
    # Return answer, results, and token usage
    return (
        self.llm.remove_think(chat_response.content),
        all_retrieved_results,
        n_token_retrieval + chat_response.total_tokens,
    )
```

### 4.2 Document Retrieval: `DeepSearch.retrieve()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\deep_search.py`

This function retrieves relevant documents using an async implementation:
```python
def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
    # Use asyncio to run the async version
    return asyncio.run(self.async_retrieve(original_query, **kwargs))

async def async_retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
    max_iter = kwargs.pop("max_iter", self.max_iter)
    
    # Log the query
    log.color_print(f"<query> {original_query} </query>\n")
    
    # Initialize results storage
    all_search_res = []
    all_sub_queries = []
    total_tokens = 0
    
    # Generate sub-queries for the original query
    sub_queries, used_token = self._generate_sub_queries(original_query)
    total_tokens += used_token
    
    # Handle case with no sub-queries
    if not sub_queries:
        log.color_print("No sub queries were generated by the LLM. Exiting.")
        return [], total_tokens, {}
    
    # Log generated sub-queries
    log.color_print(f"<think> Break down the original query into new sub queries: {sub_queries}</think>\n")
    
    # Add sub-queries to collection
    all_sub_queries.extend(sub_queries)
    sub_gap_queries = sub_queries
    
    # Iterative retrieval process
    for iter in range(max_iter):
        log.color_print(f">> Iteration: {iter + 1}\n")
        search_res_from_vectordb = []
        
        # Create and execute search tasks in parallel
        search_tasks = [
            self._search_chunks_from_vectordb(query, sub_gap_queries)
            for query in sub_gap_queries
        ]
        search_results = await asyncio.gather(*search_tasks)
        
        # Process results
        for result in search_results:
            search_res, consumed_token = result
            total_tokens += consumed_token
            search_res_from_vectordb.extend(search_res)
        
        # Deduplicate results and add to collection
        search_res_from_vectordb = deduplicate_results(search_res_from_vectordb)
        all_search_res.extend(search_res_from_vectordb)
        
        # Exit if max iterations reached
        if iter == max_iter - 1:
            log.color_print("<think> Exceeded maximum iterations. Exiting. </think>\n")
            break
        
        # Generate new queries to fill gaps
        log.color_print("<think> Reflecting on the search results... </think>\n")
        sub_gap_queries, consumed_token = self._generate_gap_queries(
            original_query, all_sub_queries, all_search_res
        )
        total_tokens += consumed_token
        
        # Exit if no new queries generated
        if not sub_gap_queries or len(sub_gap_queries) == 0:
            log.color_print("<think> No new search queries were generated. Exiting. </think>\n")
            break
        
        # Log and add new queries
        log.color_print(f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n")
        all_sub_queries.extend(sub_gap_queries)
    
    # Final deduplication and return
    all_search_res = deduplicate_results(all_search_res)
    additional_info = {"all_sub_queries": all_sub_queries}
    return all_search_res, total_tokens, additional_info
```

### 4.3 Vector Database Search: `_search_chunks_from_vectordb()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\deep_search.py`

This function searches the vector database for relevant chunks:
```python
async def _search_chunks_from_vectordb(self, query: str, sub_queries: List[str]):
    consume_tokens = 0
    
    # Determine which collections to search
    if self.route_collection:
        selected_collections, n_token_route = self.collection_router.invoke(
            query=query, dim=self.embedding_model.dimension
        )
    else:
        selected_collections = self.collection_router.all_collections
        n_token_route = 0
    consume_tokens += n_token_route
    
    # Initialize results storage
    all_retrieved_results = []
    
    # Generate query vector
    query_vector = self.embedding_model.embed_query(query)
    
    # Search each selected collection
    for collection in selected_collections:
        log.color_print(f"<search> Search [{query}] in [{collection}]...  </search>\n")
        
        # Retrieve initial results based on vector similarity
        retrieved_results = self.vector_db.search_data(
            collection=collection, vector=query_vector, query_text=query
        )
        
        # Skip if no results
        if not retrieved_results or len(retrieved_results) == 0:
            log.color_print(f"<search> No relevant document chunks found in '{collection}'! </search>\n")
            continue
        
        # Filter results using LLM reranking
        accepted_chunk_num = 0
        references = set()
        for retrieved_result in retrieved_results:
            # Ask LLM if chunk is relevant to query
            chat_response = self.llm.chat(
                messages=[{
                    "role": "user",
                    "content": RERANK_PROMPT.format(
                        query=[query] + sub_queries,
                        retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",
                    ),
                }]
            )
            consume_tokens += chat_response.total_tokens
            
            # Parse response
            response_content = self.llm.remove_think(chat_response.content).strip()
            
            # Add to results if LLM says YES
            if "YES" in response_content and "NO" not in response_content:
                all_retrieved_results.append(retrieved_result)
                accepted_chunk_num += 1
                references.add(retrieved_result.reference)
        
        # Log results
        if accepted_chunk_num > 0:
            log.color_print(f"<search> Accept {accepted_chunk_num} document chunk(s) from references: {list(references)} </search>\n")
        else:
            log.color_print(f"<search> No document chunk accepted from '{collection}'! </search>\n")
    
    return all_retrieved_results, consume_tokens
```

### 4.4 Sub-query Generation: `_generate_sub_queries()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\deep_search.py`

This function breaks down the original query into sub-queries:
```python
def _generate_sub_queries(self, original_query: str) -> Tuple[List[str], int]:
    # Ask LLM to break down the query
    chat_response = self.llm.chat(
        messages=[{
            "role": "user", 
            "content": SUB_QUERY_PROMPT.format(original_query=original_query)
        }]
    )
    
    # Extract and parse response
    response_content = self.llm.remove_think(chat_response.content)
    
    # Convert string representation of list to actual list
    return self.llm.literal_eval(response_content), chat_response.total_tokens
```

### 4.5 Gap Query Generation: `_generate_gap_queries()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\deep_search.py`

This function generates additional queries to fill knowledge gaps:
```python
def _generate_gap_queries(self, original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]) -> Tuple[List[str], int]:
    # Create prompt to identify knowledge gaps
    reflect_prompt = REFLECT_PROMPT.format(
        question=original_query,
        mini_questions=all_sub_queries,
        mini_chunk_str=self._format_chunk_texts([chunk.text for chunk in all_chunks])
        if len(all_chunks) > 0
        else "NO RELATED CHUNKS FOUND.",
    )
    
    # Ask LLM to generate new queries
    chat_response = self.llm.chat([{"role": "user", "content": reflect_prompt}])
    
    # Extract and parse response
    response_content = self.llm.remove_think(chat_response.content)
    
    # Convert string representation of list to actual list
    return self.llm.literal_eval(response_content), chat_response.total_tokens
```

## 5. Vector Database Operations

### 5.1 Vector Search: `search_data()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\vector_db\faiss_db.py`

This function searches the FAISS vector database:
```python
def search_data(self, collection=None, vector=None, query_text="", top_k=5, **kwargs):
    # Update collection if specified
    if collection is not None:
        self.collection_name = collection
    
    # Initialize results
    results = []
    
    # Check if index exists and has data
    if self.index is None or self.index.ntotal == 0:
        return results
    
    # Convert vector if needed
    if isinstance(vector, list):
        vector = np.array(vector, dtype=np.float32)
    
    # Execute search
    D, I = self.index.search(np.array([vector], dtype=np.float32), top_k)
    
    # Process results
    for idx, dist in zip(I[0], D[0]):
        if idx < len(self.texts):
            # Create retrieval result with metadata
            results.append(
                RetrievalResult(
                    text=self.texts[idx],
                    reference=self.references[idx],
                    metadata=self.metadatas[idx],
                    score=float(1.0 / (1.0 + dist)),
                )
            )
    
    return results
```

### 5.2 Query Embedding: `embed_query()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\embedding\openai_embedding.py`

This function generates embeddings for query strings:
```python
def embed_query(self, query: str) -> List[float]:
    # Generate embedding using OpenAI API
    embedding = self.client.embeddings.create(
        input=query, model=self.model
    ).data[0].embedding
    
    return embedding
```

### 5.3 LLM Chat Function: `chat()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\llm\openai_llm.py`

This function handles communication with OpenAI's LLM:
```python
def chat(self, messages):
    try:
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        
        # Extract and return results
        return ChatResponse(
            content=response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,
        )
    except Exception as e:
        # Handle API errors
        error_message = f"OpenAI API error: {str(e)}"
        return ChatResponse(
            content=f"Error: {error_message}",
            total_tokens=0,
        )
```

### 5.4 Collection Router: `invoke()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\collection_router.py`

This function determines which collections to search for a query:
```python
def invoke(self, query: str, dim: int = None) -> Tuple[List[str], int]:
    # Initialize collections registry if needed
    if not self.collections_registry:
        self._initialize_collections_registry(dim=dim)
    
    # If only one collection exists, return it
    if len(self.collections_registry) <= 1:
        return list(self.collections_registry.keys()), 0
    
    # Generate embedding for query
    query_embedding = self.embedding_model.embed_query(query) if self.embedding_model else None
    
    # Create routing prompt
    collections_description = "\n".join(
        [f"{i+1}. Collection: {name}, desc: {desc.description}" 
         for i, (name, desc) in enumerate(self.collections_registry.items())]
    )
    prompt = COLLECTION_ROUTING_PROMPT.format(
        query=query, collections_description=collections_description
    )
    
    # Ask LLM which collections to search
    chat_response = self.llm.chat(messages=[{"role": "user", "content": prompt}])
    selected_indices = self._parse_indices(chat_response.content)
    
    # Get collection names from indices
    selected_collections = [
        list(self.collections_registry.keys())[i-1] 
        for i in selected_indices 
        if 1 <= i <= len(self.collections_registry)
    ]
    
    # Return selected collections and token usage
    return selected_collections, chat_response.total_tokens
```

### 5.5 Deduplication: `deduplicate_results()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\vector_db\base.py`

This function removes duplicate retrieval results:
```python
def deduplicate_results(retrieved_results: List[RetrievalResult]) -> List[RetrievalResult]:
    # Initialize set to track unique texts
    unique_texts = set()
    unique_results = []
    
    # Process each result
    for result in retrieved_results:
        # Skip if text already processed
        if result.text in unique_texts:
            continue
        
        # Add to unique set and results
        unique_texts.add(result.text)
        unique_results.append(result)
    
    return unique_results
```

### 5.6 Answer Formatting: `_format_chunk_texts()`
**File Path**: `c:\Users\Ashwin\Documents\deep-searcher\deepsearcher\agent\deep_search.py`

This function formats chunks for inclusion in prompts:
```python
def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
    chunk_str = ""
    for i, chunk in enumerate(chunk_texts):
        chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
    return chunk_str
```
