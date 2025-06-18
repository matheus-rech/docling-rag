"""
Core module for vector storage and retrieval using FAISS and SentenceTransformers.

This module provides the `VectorStore` class, which is responsible for:
- Generating text embeddings using SentenceTransformer models.
- Building an in-memory FAISS index for efficient similarity search.
- Searching for relevant text chunks based on a query.
- Calculating similarity scores between a query and text chunks.
- Persisting (saving and loading) the FAISS index and associated text chunks
  to/from disk using `faiss.write_index`/`faiss.read_index` and `pickle`.

Its primary use case within the associated applications (FastAPI, Streamlit) is
for per-document processing, where an index is built for a single uploaded document
to perform searches on it. The persistence features (`save_index`, `load_index`)
provide foundational capabilities for extending this to multi-document scenarios,
such as creating and querying a persistent, collective knowledge base.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Any
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Base class for exceptions in VectorStore."""
    pass

class VectorStoreInitializationError(VectorStoreError):
    """Custom exception for errors during VectorStore initialization."""
    pass

class VectorStore:
    """
    Manages text embeddings, FAISS indexing, and similarity search.

    This class encapsulates the functionality to:
    - Load a SentenceTransformer model for creating text embeddings.
    - Build an in-memory FAISS index from a list of text chunks.
    - Perform similarity searches on the index to find relevant chunks.
    - Calculate similarity scores for chunks against a query.
    - Save the FAISS index and its associated chunks to disk.
    - Load a previously saved index and chunks from disk.

    It's primarily designed for in-memory indexing of text data derived from
    a single document at a time in its current application context, but the
    save/load functionality allows for persistence and potential reuse.

    Attributes:
        encoder (SentenceTransformer): The loaded sentence embedding model.
        model_name (str): The name of the SentenceTransformer model used.
        index (faiss.Index | None): The FAISS index, None if not built or loaded.
        chunks (List[Any]): The list of chunks currently indexed. Expected to be
                            objects with a `.text` attribute.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStore with a specified SentenceTransformer model.

        Args:
            model_name (str): The name or path of the SentenceTransformer model
                              to use (e.g., "all-MiniLM-L6-v2"). Defaults to
                              "all-MiniLM-L6-v2".

        Raises:
            VectorStoreInitializationError: If the SentenceTransformer model
                                           cannot be loaded (e.g., model not found,
                                           network issues, insufficient memory).
        """
        try:
            # Attempt to load the SentenceTransformer model
            self.encoder = SentenceTransformer(model_name)
            self.model_name = model_name
        except OSError as e:
            logger.error("Failed to load SentenceTransformer model '%s' from disk: %s", model_name, str(e))
            raise VectorStoreInitializationError(f"Failed to load SentenceTransformer model '{model_name}' from disk: {e}") from e
        except Exception as e: # Catching other potential exceptions from SentenceTransformer
            logger.error("An unexpected error occurred while loading SentenceTransformer model '%s': %s", model_name, str(e))
            raise VectorStoreInitializationError(f"An unexpected error occurred while loading SentenceTransformer model '{model_name}': {e}") from e
        self.index = None
        self.chunks = []
    
    def build_index(self, chunks: List[Any]) -> faiss.Index | None:
        """
        Builds or rebuilds the FAISS index from a list of text chunks.

        Args:
            chunks (List[Any]): A list of chunk objects. Each object is expected
                                to have a `.text` attribute containing the string
                                to be indexed.

        Returns:
            faiss.Index | None: The built FAISS index if successful, otherwise None.
                                Returns None if input `chunks` is empty or if an error
                                occurs during text extraction, embedding generation,
                                or FAISS index creation.
        """
        # Handle empty or None chunks input
        if not chunks:
            logger.warning("Input chunks are empty or None. Index will not be built.")
            self.index = None
            self.chunks = []
            return None

        self.chunks = chunks
        try:
            # Extract text from chunks; requires chunks to have a '.text' attribute
            texts = [chunk.text for chunk in chunks]
        except AttributeError as e: # Catch error if a chunk object lacks '.text'
            logger.error("Error extracting text from chunks: %s. Ensure chunks have a '.text' attribute.", str(e))
            return None
        except TypeError as e: # Handle cases where chunks might not be iterable or subscriptable as expected
            logger.error("Error processing chunks: %s. Ensure chunks are iterable and processable.", str(e))
            return None

        try:
            # Create embeddings using the SentenceTransformer model
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
        except Exception as e: # Catching exceptions from sentence-transformer library during encoding
            logger.error("Error encoding texts with SentenceTransformer: %s", str(e))
            return None
        
        try:
            # Build FAISS index (IndexFlatL2 is a common choice for dense vector similarity)
            dimension = embeddings.shape[1] # Dimensionality of embeddings
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings) # Add embeddings to the FAISS index
        except AttributeError as e: # Handles cases where embeddings might be None or not a numpy array
            logger.error("Failed to build FAISS index due to invalid embeddings (e.g., None or wrong type): %s", str(e))
            return None
        except faiss.FaissException as e: # Catching specific FAISS library exceptions
            logger.error("FAISS index creation or adding embeddings failed: %s", str(e))
            return None
        except Exception as e: # Catching other potential errors during FAISS operations
            logger.error("An unexpected error occurred during FAISS index operations: %s", str(e))
            return None

        return self.index
    
    def search(self, query: str, chunks: List[Any], k: int = 5) -> List[Any]:
        """
        Searches for the top-k relevant chunks for a given query.

        If an index is not already built, or if the provided `chunks` differ
        from the currently indexed ones, this method will attempt to build a
        new index from the provided `chunks` before searching.

        Args:
            query (str): The search query string.
            chunks (List[Any]): The list of chunks to search within. If different
                                from indexed chunks, triggers a rebuild.
            k (int): The number of top relevant chunks to retrieve. Defaults to 5.

        Returns:
            List[Any]: A list of the top-k relevant chunk objects. Returns an
                       empty list if the index cannot be built, if the search
                       fails, or if no relevant chunks are found.
        """
        # Build or rebuild index if it's not present or if new chunks are provided
        if self.index is None or self.chunks != chunks:
            # Log if an existing index is being replaced due to new chunks
            if self.index is not None and self.chunks != chunks: # Check if an index exists but chunks differ
                logger.warning("New chunks provided for search differ from the currently indexed ones. Rebuilding index for new chunks.")

            # Attempt to build the index; if it fails, cannot proceed with search
            if self.build_index(chunks) is None:
                logger.error("Index building failed. Cannot proceed with search.")
                return []
        
        # Ensure index is valid before proceeding (e.g. build_index might have failed silently if not checked)
        if self.index is None:
            logger.error("Index is not available. Cannot proceed with search.")
            return []

        try:
            # Encode the query to its vector representation
            query_embedding = self.encoder.encode([query])
        except Exception as e: # Handle errors during query encoding
            logger.error("Error encoding query with SentenceTransformer: %s", str(e))
            return []
        
        try:
            # Perform search on the FAISS index
            distances, indices = self.index.search(query_embedding, k)
        except faiss.FaissException as e: # Handle FAISS-specific search errors
            logger.error("FAISS search failed: %s", str(e))
            return []
        except Exception as e: # Handle other unexpected search errors
            logger.error("An unexpected error occurred during FAISS search: %s", str(e))
            return []
        
        # Retrieve and return the relevant chunks based on search indices
        try:
            relevant_chunks = [self.chunks[i] for i in indices[0]]
            return relevant_chunks
        except IndexError: # Handle cases where indices might be out of bounds for self.chunks
            logger.error("Error accessing chunks with search indices. Indices might be out of bounds.")
            return []
    
    def get_similarity_scores(self, query: str, chunks: List[Any]) -> List[float]:
        """
        Calculates similarity scores for all provided chunks against a query.

        Similar to `search`, this method will build/rebuild the index if necessary.
        The similarity is typically derived from the distance (e.g., L2 distance).

        Args:
            query (str): The query string.
            chunks (List[Any]): The list of chunks to score. If different from
                                indexed chunks, triggers a rebuild.

        Returns:
            List[float]: A list of similarity scores (0-1 range, higher is more similar).
                         Returns an empty list on failure (e.g., index build error,
                         search error, calculation error).
        """
        # Build or rebuild index if necessary
        if self.index is None or self.chunks != chunks:
            if self.index is not None and self.chunks != chunks: # Check if an index exists but chunks differ
                logger.warning("New chunks provided for similarity scoring differ from the currently indexed ones. Rebuilding index for new chunks.")
            if self.build_index(chunks) is None:
                logger.error("Index building failed. Cannot proceed with getting similarity scores.")
                return []

        # Ensure index is valid before proceeding
        if self.index is None:
            logger.error("Index is not available. Cannot proceed with getting similarity scores.")
            return []
        
        try:
            # Encode the query
            query_embedding = self.encoder.encode([query])
        except Exception as e: # Handle query encoding errors
            logger.error("Error encoding query with SentenceTransformer for similarity scores: %s", str(e))
            return []

        try:
            # Search against all indexed chunks to get distances
            # Note: self.chunks here refers to the chunks currently in the index.
            # If the input `chunks` argument led to a rebuild, self.chunks would be updated.
            distances, _ = self.index.search(query_embedding, len(self.chunks))
        except faiss.FaissException as e: # Handle FAISS search errors
            logger.error("FAISS search for similarity scores failed: %s", str(e))
            return []
        except Exception as e: # Handle other unexpected search errors
            logger.error("An unexpected error occurred during FAISS search for similarity scores: %s", str(e))
            return []
        
        # Check if search returned valid distances
        if distances is None or distances.size == 0:
            logger.warning("Search returned no distances. Cannot calculate similarity scores.")
            return []

        # Convert distances to similarity scores (e.g., 1 / (1 + distance))
        try:
            similarities = 1 / (1 + distances[0]) # distances[0] because query is singular
            return similarities.tolist()
        except IndexError: # If distances[0] is out of bounds
            logger.error("Distances array is not in the expected format for similarity calculation.")
            return []
        except Exception as e: # Other potential math errors during similarity calculation
            logger.error("Error calculating similarity scores from distances: %s", str(e))
            return []

    def save_index(self, index_path: str, chunks_path: str) -> bool:
        """
        Saves the current FAISS index and its associated chunks to disk.

        The FAISS index is saved using `faiss.write_index`, and the list of
        chunks is saved using `pickle.dump`. Directories for paths will be
        created if they don't exist.

        Args:
            index_path (str): The file path where the FAISS index will be saved.
            chunks_path (str): The file path where the list of chunks will be saved.

        Returns:
            bool: True if both the index and chunks were saved successfully,
                  False otherwise.
        """
        if self.index is None:
            logger.warning("No index to save. Build or load an index first.")
            return False

        try:
            # Create parent directory for index_path if it doesn't exist
            index_dir = os.path.dirname(index_path)
            if index_dir: # Check if index_dir is not an empty string (path is in current dir)
                os.makedirs(index_dir, exist_ok=True)

            faiss.write_index(self.index, index_path) # Save the FAISS index object
            logger.info("FAISS index saved successfully to %s", index_path)
        except (IOError, faiss.FaissException) as e: # Handle IO or FAISS errors during save
            logger.error("Failed to save FAISS index to %s: %s", index_path, str(e))
            return False
        except Exception as e: # Catch any other unexpected errors during index saving
            logger.error("An unexpected error occurred while saving FAISS index to %s: %s", index_path, str(e))
            return False

        try:
            # Create parent directory for chunks_path if it doesn't exist
            chunks_dir = os.path.dirname(chunks_path)
            if chunks_dir: # Check if chunks_dir is not an empty string
                os.makedirs(chunks_dir, exist_ok=True)

            # Save chunks using pickle
            with open(chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)
            logger.info("Chunks saved successfully to %s", chunks_path)
        except (IOError, pickle.PickleError) as e: # Handle IO or Pickle errors during chunk saving
            logger.error("Failed to save chunks to %s: %s", chunks_path, str(e))
            return False
        except Exception as e: # Catch any other unexpected errors during chunk saving
            logger.error("An unexpected error occurred while saving chunks to %s: %s", chunks_path, str(e))
            return False

        return True

    def load_index(self, index_path: str, chunks_path: str) -> bool:
        """
        Loads a FAISS index and its associated chunks from disk.

        The FAISS index is loaded using `faiss.read_index`, and the chunks
        are loaded using `pickle.load`. The SentenceTransformer model specified
        during `__init__` must be compatible with the embeddings used to create
        the loaded index for meaningful search results.

        Args:
            index_path (str): The file path from which to load the FAISS index.
            chunks_path (str): The file path from which to load the list of chunks.

        Returns:
            bool: True if both the index and chunks were loaded successfully,
                  False otherwise.
        """
        # Check if specified paths exist before attempting to load
        if not os.path.exists(index_path):
            logger.warning("FAISS index file not found at %s. Cannot load index.", index_path)
            return False
        if not os.path.exists(chunks_path):
            logger.warning("Chunks file not found at %s. Cannot load chunks.", chunks_path)
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            logger.info("FAISS index loaded successfully from %s", index_path)
        except (IOError, faiss.FaissException) as e: # Handle IO or FAISS errors during load
            logger.error("Failed to load FAISS index from %s: %s", index_path, str(e))
            self.index = None # Ensure index is None if loading failed
            return False
        except Exception as e: # Catch any other unexpected errors during index loading
            logger.error("An unexpected error occurred while loading FAISS index from %s: %s", index_path, str(e))
            self.index = None # Ensure index is None on unexpected error
            return False

        try:
            # Load chunks using pickle
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info("Chunks loaded successfully from %s", chunks_path)
        except (IOError, pickle.PickleError) as e: # Handle IO or Pickle errors during chunk loading
            logger.error("Failed to load chunks from %s: %s", chunks_path, str(e))
            self.chunks = [] # Ensure chunks are empty if loading failed
            # If chunks fail to load, the loaded index might be inconsistent or unusable
            # with the (now empty) chunks list. Clearing the index for safety.
            self.index = None
            logger.warning("Index has been cleared due to failure in loading associated chunks.")
            return False
        except Exception as e: # Catch any other unexpected errors during chunk loading
            logger.error("An unexpected error occurred while loading chunks from %s: %s", chunks_path, str(e))
            self.chunks = [] # Ensure chunks are empty
            self.index = None # Clear index for safety
            logger.warning("Index has been cleared due to an unexpected error in loading associated chunks.")
            return False

        # Final check to ensure the encoder (from __init__) is available.
        # A loaded index is not very useful without an encoder for queries.
        if self.encoder is None:
            # This case should ideally not be reached if __init__ completed successfully,
            # as self.encoder is vital. If it's None here, it indicates a severe issue
            # possibly post-initialization or if __init__ logic changes.
            logger.error("Sentence encoder is not available. Index may not be usable for new operations. Please initialize VectorStore correctly.")
            # Depending on strictness, you might return False here or even raise an error,
            # as a VectorStore without an encoder might not be fully operational for all methods.
            # However, __init__ is designed to raise VectorStoreInitializationError if encoder loading fails.

        return True

# Note on Multi-Document Usage:
# While this VectorStore includes `save_index` and `load_index` for persistence,
# its current integration in the applications (FastAPI, Streamlit) is primarily
# for processing individual documents on-the-fly. To use it as a persistent,
# multi-document database that is queried collectively, application logic
# would need to be adapted to manage a persistent index (e.g., load one at startup,
# provide ways to add new documents to it, and query the collective set).
# The `save_index` and `load_index` methods provide foundational capabilities for such extensions.