# Deep Searcher Codebase Documentation

This documentation provides a detailed explanation of the Deep Searcher codebase, starting from the main entry point and following the workflow of function invocations. Each function and method is documented with its purpose, implementation details, and context within the overall system.

## Table of Contents

1. [Main Entry Point (main.py)](#main-entry-point)
2. [Configuration System](#configuration-system)
3. [Offline Loading Process](#offline-loading-process)
4. [Online Query Process](#online-query-process)
5. [Agent System](#agent-system)
6. [Embedding System](#embedding-system)
7. [Vector Database System](#vector-database-system)
8. [Document Processing](#document-processing)

## Main Entry Point

The main entry point for the Deep Searcher application is `main.py`, which sets up a FastAPI server to expose the system's functionality as API endpoints.

### `main.py`

```python
if __name__ == "__main__":
    # Parse command line arguments and run the FastAPI server
```

**Invoked from**: Command line or script execution
**Purpose**: Initializes the FastAPI application and starts the server

**Implementation**:
- Parses command line arguments for enabling CORS
- Configures CORS middleware if requested
- Starts the uvicorn server on port 8000, binding to all interfaces (0.0.0.0)

### `init_config(config)`

**Invoked from**: `main.py` during application startup
**Purpose**: Initializes the global configuration and creates instances of all required components

**Args**:
- `config` (Configuration): The configuration object to use for initialization

**Implementation**:
- Initializes global variables for different components:
  - `module_factory`: Factory for creating component instances
  - `llm`: Language model for text generation
  - `embedding_model`: Model for creating vector embeddings
  - `file_loader`: For loading documents from files
  - `web_crawler`: For loading documents from websites
  - `vector_db`: For storing and retrieving document embeddings
  - `default_searcher`: Main RAG router for intelligent query handling
  - `naive_rag`: Simple RAG implementation for basic queries

**Side effects**:
- Creates and initializes all main components of the system
- Sets up RAG agents with appropriate configuration

## API Endpoints

### `set_provider_config(request)`

**Invoked from**: HTTP POST requests to `/set-provider-config/`
**Purpose**: Configure a specific provider for a feature (e.g., setting OpenAI as the LLM provider)

**Args**:
- `request` (ProviderConfigRequest): Contains feature name, provider name, and configuration parameters

**Returns**:
- Dictionary with success message and updated configuration
- HTTP 500 error if configuration fails

**Implementation**:
- Updates the configuration with provided settings
- Reinitializes the components with the new configuration using `init_config(config)`

### `load_files(paths, collection_name, collection_description, batch_size)`

**Invoked from**: HTTP POST requests to `/load-files/`
**Purpose**: Load files from the local filesystem into the vector database

**Args**:
- `paths` (Union[str, List[str]]): File paths or directories to load
- `collection_name` (str, optional): Name for the collection
- `collection_description` (str, optional): Description for the collection
- `batch_size` (int, optional): Batch size for processing

**Returns**:
- Dictionary with success message
- HTTP 500 error if loading fails

**Implementation**:
- Delegates to `load_from_local_files` in `offline_loading.py` with provided parameters

### `load_website(urls, collection_name, collection_description, batch_size)`

**Invoked from**: HTTP POST requests to `/load-website/`
**Purpose**: Load content from websites into the vector database

**Args**:
- `urls` (Union[str, List[str]]): URLs of websites to load
- `collection_name` (str, optional): Name for the collection
- `collection_description` (str, optional): Description for the collection
- `batch_size` (int, optional): Batch size for processing

**Returns**:
- Dictionary with success message
- HTTP 500 error if loading fails

**Implementation**:
- Delegates to `load_from_website` in `offline_loading.py` with provided parameters

### `perform_query(original_query, max_iter)`

**Invoked from**: HTTP GET requests to `/query/`
**Purpose**: Query the system with a user question and get an answer

**Args**:
- `original_query` (str): The user's question or query
- `max_iter` (int, optional): Maximum number of iterations for reflection

**Returns**:
- Dictionary with query result and token consumption
- HTTP 500 error if query fails

**Implementation**:
- Delegates to `query` in `online_query.py` with provided parameters

## Configuration System

### `Configuration` class (`configuration.py`)

**Invoked from**: `main.py` during initialization
**Purpose**: Manages configuration settings for the system components

**Methods**:

#### `__init__(config_path)`

**Purpose**: Initialize the configuration object
**Args**:
- `config_path` (str): Path to the configuration YAML file

**Implementation**:
- Loads configuration from the specified YAML file
- Sets up provider settings, query settings, and load settings

#### `load_config_from_yaml(config_path)`

**Invoked from**: `Configuration.__init__`
**Purpose**: Load configuration data from a YAML file

**Args**:
- `config_path` (str): Path to the YAML file

**Returns**:
- Dictionary containing the loaded configuration data

**Implementation**:
- Opens and reads the YAML file
- Parses YAML content into a Python dictionary

#### `set_provider_config(feature, provider, provider_configs)`

**Invoked from**: `set_provider_config` endpoint
**Purpose**: Update provider configuration for a specific feature

**Args**:
- `feature` (FeatureType): Feature to configure (e.g., 'llm', 'embedding')
- `provider` (str): Provider name (e.g., 'openai', 'deepseek')
- `provider_configs` (dict): Configuration parameters

**Implementation**:
- Validates the feature name
- Updates the provider and configuration settings for the feature

#### `get_provider_config(feature)`

**Purpose**: Retrieve configuration for a specific feature

**Args**:
- `feature` (FeatureType): Feature to retrieve (e.g., 'llm', 'embedding')

**Returns**:
- Dictionary containing provider and configuration settings

**Implementation**:
- Validates the feature name
- Returns the stored provider and configuration settings

### `ModuleFactory` class (`configuration.py`)

**Invoked from**: `init_config` in `configuration.py`
**Purpose**: Creates instances of various components based on configuration settings

**Methods**:

#### `__init__(config)`

**Purpose**: Initialize the factory with configuration
**Args**:
- `config` (Configuration): Configuration object

#### `_create_module_instance(feature, module_name)`

**Invoked from**: Factory creation methods (`create_llm`, etc.)
**Purpose**: Generic method to create component instances

**Args**:
- `feature` (FeatureType): Feature type (e.g., 'llm', 'embedding')
- `module_name` (str): Module name to import from

**Returns**:
- Instance of the specified component

**Implementation**:
- Dynamically imports the module and class based on configuration
- Instantiates the class with the appropriate configuration parameters

#### Factory Methods (`create_llm`, `create_embedding`, etc.)

**Invoked from**: `init_config`
**Purpose**: Create specific component instances

**Returns**:
- Instance of the requested component type

**Implementation**:
- Delegates to `_create_module_instance` with appropriate parameters

## Offline Loading Process

### `load_from_local_files(paths_or_directory, collection_name, collection_description, force_new_collection, chunk_size, chunk_overlap, batch_size)`

**Invoked from**: `load_files` endpoint in `main.py`
**Purpose**: Process and load documents from local files or directories into the vector database

**Args**:
- `paths_or_directory` (Union[str, List[str]]): Paths to files or directories to load
- `collection_name` (str, optional): Name for the collection
- `collection_description` (str, optional): Description for the collection
- `force_new_collection` (bool): Whether to force creation of a new collection
- `chunk_size` (int): Size of each text chunk
- `chunk_overlap` (int): Overlap between consecutive chunks
- `batch_size` (int): Batch size for embedding

**Implementation**:
1. Initializes vector database collection
2. Processes each path:
   - If directory, loads all files using `file_loader.load_directory`
   - If file, loads using `file_loader.load_file`
3. Splits documents into chunks with context using `split_docs_to_chunks`
4. Generates embeddings for chunks using `embedding_model.embed_chunks`
5. Inserts data into vector database using `vector_db.insert_data`

### `load_from_website(urls, collection_name, collection_description, force_new_collection, chunk_size, chunk_overlap, batch_size, **crawl_kwargs)`

**Invoked from**: `load_website` endpoint in `main.py`
**Purpose**: Process and load documents from websites into the vector database

**Args**:
- `urls` (Union[str, List[str]]): URLs to crawl
- `collection_name` (str, optional): Name for the collection
- `collection_description` (str, optional): Description for the collection
- `force_new_collection` (bool): Whether to force creation of a new collection
- `chunk_size` (int): Size of each text chunk
- `chunk_overlap` (int): Overlap between consecutive chunks
- `batch_size` (int): Batch size for embedding
- `**crawl_kwargs`: Additional keyword arguments for the web crawler

**Implementation**:
1. Initializes vector database collection
2. Crawls specified URLs using `web_crawler.crawl_urls`
3. Splits documents into chunks with context using `split_docs_to_chunks`
4. Generates embeddings for chunks using `embedding_model.embed_chunks`
5. Inserts data into vector database using `vector_db.insert_data`

## Document Processing

### `split_docs_to_chunks(documents, chunk_size, chunk_overlap)`

**Invoked from**: `load_from_local_files` and `load_from_website`
**Purpose**: Split documents into smaller, overlapping chunks with context windows

**Args**:
- `documents` (List[Document]): Documents to split
- `chunk_size` (int): Maximum size of each chunk
- `chunk_overlap` (int): Number of characters to overlap between chunks

**Returns**:
- List[Chunk]: List of text chunks with context windows

**Implementation**:
1. Creates a `RecursiveCharacterTextSplitter` with specified parameters (RecursiveCharacterTestSplitter is inbuit function )
2. For each document:
   - Splits it into smaller pieces using the text splitter
   - Creates context windows for each piece using `_sentence_window_split`
3. Returns the combined list of chunks

### `_sentence_window_split(split_docs, original_document, offset)`

**Invoked from**: `split_docs_to_chunks`
**Purpose**: Create chunks with context windows by including text before and after each split piece (basically, implement the overlap function, by default 100 characters in this case, to include prior context for a chunk)

**Args**:
- `split_docs` (List[Document]): Documents that have been split
- `original_document` (Document): The original document before splitting
- `offset` (int): Number of characters to include before and after

**Returns**:
- List[Chunk]: List of chunks with context windows

**Implementation**:
1. For each split document:
   - Locates its position in the original document
   - Creates a wider text window by including text before and after (uses 100 characters padding for a chunk)
   - Creates a Chunk object with the document text and wider context
2. Returns the list of Chunk objects

## Online Query Process

### `query(original_query, max_iter)`

**Invoked from**: `perform_query` endpoint in `main.py`
**Purpose**: Process a query and generate an answer using the default searcher

**Args**:
- `original_query` (str): The user's question
- `max_iter` (int): Maximum number of iterations for search refinement

**Returns**:
- Tuple containing generated answer, retrieval results, and token usage

**Implementation**:
- Delegates to `default_searcher.query` with the provided parameters (implemented in rag_router.py for rag agent related operations)

### `retrieve(original_query, max_iter)`

**Invoked from**: None (available for direct API usage)
**Purpose**: Retrieve relevant information without generating an answer

**Args**:
- `original_query` (str): The user's question
- `max_iter` (int): Maximum number of iterations for search refinement

**Returns**:
- Tuple containing retrieval results, empty list, and token usage

**Implementation**:
- Delegates to `default_searcher.retrieve` with the provided parameters

### `naive_retrieve(query, collection, top_k)`

**Invoked from**: None (available for direct API usage)
**Purpose**: Perform a simple retrieval without advanced processing

**Args**:
- `query` (str): The user's question
- `collection` (str, optional): Collection to search in
- `top_k` (int): Maximum number of results

**Returns**:
- List of retrieval results

**Implementation**:
- Delegates to `naive_rag.retrieve` with the provided parameters

### `naive_rag_query(query, collection, top_k)`

**Invoked from**: None (available for direct API usage)
**Purpose**: Query using the naive RAG approach

**Args**:
- `query` (str): The user's question
- `collection` (str, optional): Collection to search in
- `top_k` (int): Maximum number of results

**Returns**:
- Tuple containing generated answer and retrieval results

**Implementation**:
- Delegates to `naive_rag.query` with the provided parameters

## Agent System

### `BaseAgent` and `RAGAgent` classes (`agent/base.py`). 
 Most functions of RAGAgent are implmented in rag_router.py, including all the afore mentioned abstracted functions. RAG_Routed_prompt is also included in rag_router.py, where the prompt is used to select which RAG agent to be used for the given query.

**Purpose**: Abstract base classes for all agent implementations

#### `RAGAgent.retrieve(query, **kwargs)`

**Purpose**: Abstract method for retrieving documents
**Returns**: Tuple of retrieval results, token usage, and metadata

#### `RAGAgent.query(query, **kwargs)`

**Purpose**: Abstract method for generating answers
**Returns**: Tuple of generated answer, retrieval results, and token usage

### `RAGRouter` class (`agent/rag_router.py`)

**Invoked from**: `configuration.py` during initialization
**Purpose**: Routes queries to the most appropriate RAG agent

**Methods**:

#### `__init__(llm, rag_agents, agent_descriptions)`

**Purpose**: Initialize the router with available agents

#### `_route(query)`

**Invoked from**: `RAGRouter.retrieve` and `RAGRouter.query`
**Purpose**: Determine which agent should handle the query

**Args**:
- `query` (str): The user's question

**Returns**:
- Tuple of selected agent and token usage

**Implementation**:
1. Formats agent descriptions into a prompt
2. Uses the LLM to select the most appropriate agent
3. Returns the selected agent and token usage

#### `retrieve(query, **kwargs)` and `query(query, **kwargs)`

**Invoked from**: `online_query.py` functions
**Purpose**: Route the query to the appropriate agent and process it

**Implementation**:
1. Uses `_route` to select an agent
2. Delegates the query to that agent
3. Returns the results along with total token usage

### `CollectionRouter` class (`agent/collection_router.py`)

**Invoked from**: RAG agent initialization
**Purpose**: Determines which vector database collections should be searched for a query

**Methods**:

#### `__init__(llm, vector_db, dim, **kwargs)`

**Purpose**: Initialize the router with the vector database

#### `invoke(query, dim, **kwargs)`

**Invoked from**: RAG agent retrieve methods
**Purpose**: Determine which collections are relevant for the query

**Args**:
- `query` (str): The user's question
- `dim` (int): Embedding dimension

**Returns**:
- Tuple of selected collection names and token usage

**Implementation**:
1. Gets collection information from the vector database
2. For a single collection, returns it immediately
3. For multiple collections, uses the LLM to select relevant ones
4. Always includes the default collection if it exists
5. Returns unique collection names and token usage

### `NaiveRAG` class (`agent/naive_rag.py`)

**Invoked from**: `configuration.py` during initialization
**Purpose**: Simple RAG implementation without complex processing

**Methods**:

#### `__init__(llm, embedding_model, vector_db, top_k, route_collection, text_window_splitter, **kwargs)`

**Purpose**: Initialize the naive RAG agent

#### `retrieve(query, **kwargs)`

**Invoked from**: `naive_retrieve` in `online_query.py`
**Purpose**: Retrieve relevant documents for the query

**Implementation**:
1. Uses collection router to select collections (if enabled)
2. For each collection:
   - Embeds the query using the embedding model
   - Searches the vector database for relevant documents
3. Deduplicates results and returns them with token usage

#### `query(query, **kwargs)`

**Invoked from**: `naive_rag_query` in `online_query.py`
**Purpose**: Generate an answer based on retrieved documents

**Implementation**:
1. Retrieves relevant documents using `retrieve`
2. Formats the documents into a prompt
3. Uses the LLM to generate an answer based on the documents
4. Returns the answer, retrieved documents, and token usage

### `DeepSearch` class (`agent/deep_search.py`)

**Invoked from**: `configuration.py` during initialization
**Purpose**: Advanced RAG implementation with query decomposition and reflection. The deep_search.py also contains the prompts for asking for sub queries, prompt to rerank the previous prompt to determine if the retrieved chunk is helpful to the reponse. It also contains the prompt for  reflecting whether the prompt requires more reflection or additional queries. It also has the prompt requesting the summarization of all the chunks and their sub queries as well.

**Methods**:

#### `__init__(llm, embedding_model, vector_db, max_iter, route_collection, text_window_splitter, **kwargs)`

**Purpose**: Initialize the deep search agent

#### `_generate_sub_queries(original_query)`

**Invoked from**: `async_retrieve` at the start of retrieval process
**Purpose**: Break down a complex query into simpler sub-queries for more focused searching

**Args**:
- `original_query` (str): The user's question

**Returns**:
- Tuple of sub-queries list and token usage

**Implementation**:
- Creates a prompt using SUB_QUERY_PROMPT template with the original query
- Sends the prompt to the LLM via `llm.chat`
- Removes thinking sections from the response using `llm.remove_think`
- Parses the response string into a Python list using `llm.literal_eval`
- Returns both the list of sub-queries and token usage information
- If decomposition is unnecessary, returns a list with only the original query

#### `_search_chunks_from_vectordb(query, sub_queries)`

**Invoked from**: `async_retrieve` (called in parallel for each sub-query)
**Purpose**: Search for relevant document chunks for a query and filter them using LLM

**Args**:
- `query` (str): The search query
- `sub_queries` (List[str]): Related sub-queries

**Returns**:
- Tuple of retrieved results and token usage

**Implementation**:
1. Uses `collection_router.invoke` to determine relevant collections for the query
2. Generates query vector using `embedding_model.embed_query`
3. For each selected collection:
   - Logs the search operation
   - Retrieves initial results using `vector_db.search_data`
   - For each retrieved chunk:
     - Creates a reranking prompt using RERANK_PROMPT template
     - Sends prompt to LLM with the query and chunk text
     - Parses response to check for "YES" (accept) or "NO" (reject)
     - Adds accepted chunks to results and tracks their references
   - Logs acceptance statistics for the collection
4. Returns all accepted chunks and accumulated token usage

#### `_generate_gap_queries(original_query, all_sub_queries, all_chunks)`

**Invoked from**: `async_retrieve`
**Purpose**: Generate additional queries to fill knowledge gaps

**Args**:
- `original_query` (str): The user's question
- `all_sub_queries` (List[str]): All previous sub-queries
- `all_chunks` (List[RetrievalResult]): All retrieved chunks so far

**Returns**:
- Tuple of additional queries list and token usage

**Implementation**:
- Creates a structured prompt using REFLECT_PROMPT template
- Includes the original query, all previous sub-queries, and retrieved chunks
- Formats chunk texts using `_format_chunk_texts`
- Sends prompt to LLM to analyze knowledge gaps
- Uses `literal_eval` to convert the LLM's string response to an actual list
- Returns the generated gap queries and token usage

#### `retrieve(original_query, **kwargs)` and `async_retrieve(original_query, **kwargs)`

**Invoked from**: `query` method and indirectly from `online_query.py`
**Purpose**: Perform iterative deep search to retrieve comprehensive information

**Implementation**:
1. `retrieve` delegates to async implementation using `asyncio.run`
2. `async_retrieve` performs the full iterative retrieval process:
   - Logs the original query for debugging
   - Generates initial sub-queries using `_generate_sub_queries`
   - Tracks all sub-queries and search results
   - For each iteration (up to max_iter):
     - Creates async search tasks for each sub-query
     - Executes all tasks in parallel using `asyncio.gather`
     - Processes results and accumulates token usage
     - Deduplicates results using `deduplicate_results`
     - Generates gap queries using `_generate_gap_queries`
     - Stops if no new queries are generated or max iterations reached
   - Returns all retrieved results, total token usage, and additional metadata

#### `_format_chunk_texts(chunk_texts)`

**Invoked from**: `_generate_gap_queries` and `query`
**Purpose**: Format chunk texts for inclusion in prompts

**Args**:
- `chunk_texts` (List[str]): List of text chunks to format

**Returns**:
- Formatted string with chunk identifiers

**Implementation**:
- Creates a structured representation of chunks
- Wraps each chunk with identifiers (`<chunk_{i}>\n{chunk}\n</chunk_{i}>\n`)
- Combines them into a single string for use in prompts

#### `query(query, **kwargs)`

**Invoked from**: `online_query.py` through `default_searcher.query`
**Purpose**: Generate a comprehensive answer based on deep search results

**Args**:
- `query` (str): The user's question
- `**kwargs`: Additional parameters, typically including `max_iter`

**Returns**:
- Tuple containing the answer text, retrieved results, and token usage

**Implementation**:
1. Calls `retrieve`/`async_retrieve` to get relevant documents
2. Handles case with no results by returning an appropriate message
3. Extracts sub-queries from the additional_info metadata
4. Prepares chunk texts for the prompt:
   - Uses wider context windows when available for better understanding
   - Falls back to regular text when wider context isn't available
5. Creates a summary prompt using SUMMARY_PROMPT template:
   - Includes original question
   - Includes all sub-queries used during retrieval
   - Includes formatted chunk texts from `_format_chunk_texts`
6. Sends prompt to LLM to generate comprehensive answer
7. Removes thinking sections from response
8. Returns the final answer, retrieved results, and total token usage

## LLM System

### `BaseLLM` class (`llm/base.py`)

**Purpose**: Abstract base class for language model implementations

**Methods**:

#### `chat(messages)`

**Invoked from**: Various agent methods, especially during query and retrieval operations
**Purpose**: Send messages to a language model and receive a response

**Args**:
- `messages` (List[Dict]): List of message dictionaries with role and content

**Returns**:
- ChatResponse object containing the generated content and token usage

**Implementation**:
- Calls the underlying LLM API (e.g., OpenAI)
- Handles potential API errors
- Returns a structured response with content and token count

#### `remove_think(content)`

**Invoked from**: Various agent methods when processing LLM responses
**Purpose**: Remove thinking/reasoning portions from LLM responses

**Args**:
- `content` (str): The raw content from the LLM

**Returns**:
- Cleaned text with thinking sections removed

#### `literal_eval(response_content)`

**Invoked from**: Methods like `_generate_sub_queries` and `_generate_gap_queries`
**Purpose**: Safely parse and evaluate string representations of Python objects

**Args**:
- `response_content` (str): String representation of a Python object from LLM

**Returns**:
- The evaluated Python object (e.g., list, dict)

## Embedding System

### `BaseEmbedding` class (`embedding/base.py`)

**Purpose**: Abstract base class for embedding model implementations

**Methods**:

#### `embed_query(text)`

**Invoked from**: Search operations in RAG agents
**Purpose**: Generate an embedding vector for a single query text

**Args**:
- `text` (str): The text to generate an embedding for

**Returns**: 
- List of floats representing the embedding vector

**Implementation**:
- For OpenAI implementation, calls the embeddings API
- Returns the embedding vector as a list of floats

#### `embed_documents(texts)`

**Invoked from**: `embed_chunks` method
**Purpose**: Generate embedding vectors for multiple document texts

**Args**:
- `texts` (List[str]): List of texts to generate embeddings for

**Returns**:
- List of embedding vectors (List[List[float]])

**Implementation**:
- Default implementation calls `embed_query` for each text
- Implementations may override with more efficient batch processing

#### `embed_chunks(chunks, batch_size)`

**Invoked from**: `load_from_local_files` and `load_from_website` in `offline_loading.py`
**Purpose**: Generate embedding vectors for a list of Chunk objects

**Args**:
- `chunks` (List[Chunk]): Chunks to embed
- `batch_size` (int): Number of chunks to process at once

**Returns**:
- List of Chunk objects with embeddings added

**Implementation**:
1. Extracts text from each chunk
2. Processes texts in batches to create embeddings
3. Updates each chunk with its embedding
4. Returns the updated chunks

#### `dimension` property

**Purpose**: Get the dimensionality of the embeddings
**Returns**: Number of dimensions in the embedding vectors

## Vector Database System

### `BaseVectorDB` class (implied from usage)

**Purpose**: Abstract base class for vector database implementations

**Methods**:

#### `init_collection(dim, collection, description, force_new_collection)`

**Invoked from**: `load_from_local_files` and `load_from_website` in `offline_loading.py`
**Purpose**: Initialize a collection in the vector database

**Args**:
- `dim` (int): Dimension size for the vector embeddings
- `collection` (str, optional): Name of the collection to initialize
- `description` (str, optional): Description of the collection
- `force_new_collection` (bool): Whether to force creation of a new collection

**Implementation**:
- Sets the collection name if provided
- Creates a new FAISS index if needed or force is true
- Initializes the index with the specified dimensions
- Saves the index to disk

#### `insert_data(collection, chunks)`

**Invoked from**: `load_from_local_files` and `load_from_website` in `offline_loading.py`
**Purpose**: Insert chunks with embeddings into the vector database

**Args**:
- `collection` (str, optional): Name of the collection to insert into
- `chunks` (List[Chunk]): List of chunks with embeddings to insert
- `batch_size` (int, optional): Number of chunks to process at once

**Implementation**:
- Sets the collection name if provided
- Extracts embedding vectors from chunks
- Stacks vectors and adds them to the FAISS index
- Stores text content, references, and metadata
- Saves the updated index to disk

#### `search_data(collection, vector, top_k, query_text)`

**Invoked from**: RAG agent implementations
**Purpose**: Search for similar vectors in the database

**Args**:
- `collection` (str): Collection to search in
- `vector` (List[float]): Query vector
- `top_k` (int): Maximum number of results
- `query_text` (str): Original query text

**Returns**:
- List of RetrievalResult objects

**Implementation**:
- Sets the collection name if provided
- Checks if the index exists and has data
- Converts the vector to the required format if needed
- Executes the search using the FAISS index
- Processes results and constructs RetrievalResult objects with scores
- Returns the list of retrieval results

#### `list_collections(dim)`

**Invoked from**: `CollectionRouter.invoke`
**Purpose**: List available collections with the given dimension

**Args**:
- `dim` (int): Dimension size to filter collections by

**Returns**: 
- List of collection information objects with names and descriptions

**Implementation**:
- Scans the vector database for collections matching the dimension
- Returns collection metadata including names and descriptions

#### `deduplicate_results(retrieved_results)`

**Invoked from**: Various RAG agent methods including `DeepSearch.async_retrieve`
**Purpose**: Remove duplicate retrieval results from the search results

**Args**:
- `retrieved_results` (List[RetrievalResult]): List of retrieval results

**Returns**:
- List of deduplicated RetrievalResult objects

**Implementation**:
- Tracks already seen texts with a set
- Filters out results with duplicate text content
- Returns only the unique results

## Implementation Workflow

The Deep Searcher system follows a comprehensive workflow that can be divided into three main phases: Configuration, Offline Loading, and Online Querying.

### 1. Configuration Phase

Before any operations can occur, the system undergoes configuration:

1. **System Initialization**:
   - The FastAPI application (`main.py`) initializes and loads the default configuration from `config.yaml`
   - `init_config(config)` is called to create component instances

2. **Component Creation**:
   - `ModuleFactory` dynamically instantiates components based on configuration:
     - LLM provider (e.g., OpenAI)
     - Embedding model
     - File loader and web crawler
     - Vector database implementation

3. **RAG Agent Setup**:
   - The `default_searcher` (RAGRouter) is initialized with specialized agents:
     - `DeepSearch` for complex, multi-faceted queries
     - `ChainOfRAG` for iterative refinement queries
   - The `naive_rag` agent is set up for simple queries

### 2. Offline Loading Phase

When a user wants to add content to the knowledge base, the following steps occur:

1. **Content Source Processing**:
   - From API endpoints (`/load-files/` or `/load-website/`), requests are routed to appropriate handlers
   - For files: `load_from_local_files()` processes file paths or directories
   - For websites: `load_from_website()` uses the web crawler to fetch content

2. **Document Processing**:
   - Documents are loaded using specialized loaders (from `file_loader` or `web_crawler`)
   - Raw documents are converted into a standardized format with metadata

3. **Chunking & Context Window Creation**:
   - Documents are split into manageable chunks using `split_docs_to_chunks()`
   - `RecursiveCharacterTextSplitter` breaks documents into segments of specified size (default 1500 chars)
   - Context windows are created using `_sentence_window_split()` to preserve context
   - Each chunk maintains its original wider context (typically +/- 300 chars)

4. **Embedding Generation**:
   - Chunks are processed in batches (default 256) by `embedding_model.embed_chunks()`
   - Each chunk's text is converted into a high-dimensional vector representation
   - Vectors capture semantic meaning for later similarity search

5. **Vector Database Storage**:
   - Vector database is initialized with appropriate collections
   - Embedded chunks are inserted into the database using `vector_db.insert_data()`
   - Metadata and references are preserved alongside vectors for retrieval

### 3. Online Querying Phase

When a user submits a query, a sophisticated process occurs to generate accurate answers:

1. **Query Reception**:
   - User query is received via the `/query/` endpoint
   - Request is forwarded to `query()` in `online_query.py`

2. **Agent Selection**:
   - `RAGRouter._route()` analyzes the query to determine which specialized agent is most appropriate
   - Agent selection uses LLM to match query characteristics with agent capabilities
   - Appropriate agent (e.g., `DeepSearch` or `NaiveRAG`) is selected

3. **Query Processing** (varies by agent):
   
   **For DeepSearch**:
   - **Query Decomposition**: Original query is broken down into sub-queries using LLM via `_generate_sub_queries()`
   - **Parallel Search**: Each sub-query is processed in parallel to find relevant document chunks
   - **Relevance Filtering**: Retrieved chunks are filtered by relevance using LLM-based reranking
   - **Iterative Refinement**: 
     - System identifies knowledge gaps via `_generate_gap_queries()`
     - Additional queries are generated to fill these gaps
     - Process repeats for up to `max_iter` iterations (default 3)
   - **Result Consolidation**: All relevant chunks are deduplicated and combined

   **For NaiveRAG**:
   - **Direct Search**: Query is embedded and used to search vector database
   - **Collection Routing**: Relevant collections are selected via `CollectionRouter.invoke()`
   - **Simple Retrieval**: Top results are retrieved without complex refinement

4. **Answer Generation**:
   - Retrieved document chunks are formatted with appropriate context
   - Wider context windows are used when available for better understanding
   - Prompt is constructed with original query and relevant chunks
   - LLM generates comprehensive answer based on retrieved information

5. **Response Delivery**:
   - Final answer, along with token usage statistics, is returned to the user
   - Results are formatted and sent as API response

This sophisticated workflow enables the Deep Searcher system to handle complex queries by leveraging several advanced techniques:

1. **Query Understanding** through decomposition into manageable sub-questions
2. **Comprehensive Information Retrieval** via parallel and iterative search processes
3. **Contextual Preservation** through text window splitting and wider context storage
4. **Intelligent Routing** to select the most appropriate agent for each query type
5. **Knowledge Gap Identification** and targeted follow-up searching
6. **Coherent Answer Synthesis** by providing LLM with relevant context from multiple sources

The combination of these techniques allows the system to provide detailed, accurate answers to complex questions while efficiently managing computational resources through targeted search and retrieval.
