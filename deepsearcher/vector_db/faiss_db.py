import os
import pickle
import numpy as np
import faiss
from typing import List, Optional, Union
from deepsearcher.loader.splitter import Chunk
from deepsearcher.utils import log
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult

INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.pkl"

class FAISSDB(BaseVectorDB):
    """Persistent FAISS vector DB implementation."""

    def __init__(self, default_collection: str = "deepsearcher", **kwargs):
        super().__init__(default_collection)
        self.index = None
        self.dim = None
        self.texts = []
        self.references = []
        self.metadatas = []
        self.embeddings = []
        self.collection_name = default_collection
        # Try to load if files exist
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.load()

    def init_collection(
        self,
        dim: int,
        collection: Optional[str] = None,
        description: Optional[str] = "",
        force_new_collection: bool = False,
        *args,
        **kwargs,
    ):
        if collection is not None:
            self.collection_name = collection
        if force_new_collection or self.index is None or self.dim != dim:
            self.index = faiss.IndexFlatL2(dim)
            self.dim = dim
            self.texts = []
            self.references = []
            self.metadatas = []
            self.embeddings = []
            log.color_print(f"Created FAISS collection [{self.collection_name}] with dim={dim}")
            self.save()

    def insert_data(
        self,
        collection: Optional[str],
        chunks: List[Chunk],
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        if collection is not None:
            self.collection_name = collection
        vectors = [np.array(chunk.embedding, dtype=np.float32) for chunk in chunks]
        if not vectors:
            return
        vectors_np = np.stack(vectors)
        self.index.add(vectors_np)
        self.texts.extend([chunk.text for chunk in chunks])
        self.references.extend([chunk.reference for chunk in chunks])
        self.metadatas.extend([chunk.metadata for chunk in chunks])
        self.embeddings.extend([chunk.embedding for chunk in chunks])
        self.save()

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.array, List[float]],
        top_k: int = 5,
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query = np.array(vector, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(query, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append(
                RetrievalResult(
                    embedding=self.embeddings[idx],
                    text=self.texts[idx],
                    reference=self.references[idx],
                    metadata=self.metadatas[idx],
                    score=float(dist),
                )
            )
        return results

    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        return [CollectionInfo(self.collection_name, "Persistent FAISS collection")]

    def clear_db(self, *args, **kwargs):
        self.index = None
        self.texts = []
        self.references = []
        self.metadatas = []
        self.embeddings = []
        self.dim = None
        self.save()

    def save(self):
        if self.index is not None and self.dim is not None:
            faiss.write_index(self.index, INDEX_FILE)
            with open(META_FILE, "wb") as f:
                pickle.dump({
                    "texts": self.texts,
                    "references": self.references,
                    "metadatas": self.metadatas,
                    "embeddings": self.embeddings,
                    "dim": self.dim,
                    "collection_name": self.collection_name,
                }, f)

    def load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                meta = pickle.load(f)
                self.texts = meta["texts"]
                self.references = meta["references"]
                self.metadatas = meta["metadatas"]
                self.embeddings = meta["embeddings"]
                self.dim = meta["dim"]
                self.collection_name = meta.get("collection_name", self.collection_name) 