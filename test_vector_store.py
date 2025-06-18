import os
import shutil
import logging
from core.vector_store import VectorStore, VectorStoreInitializationError, VectorStoreError

# Configure basic logging for testing feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a simple mock chunk class for testing
class MockChunk:
    def __init__(self, text, id):
        self.text = text
        self.id = id

    def __repr__(self):
        return f"MockChunk(id={self.id}, text='{self.text[:20]}...')"

    # For self.chunks != chunks comparison to work as expected if objects are re-created
    def __eq__(self, other):
        if not isinstance(other, MockChunk):
            return NotImplemented
        return self.id == other.id and self.text == other.text

# Test data
SAMPLE_CHUNKS = [
    MockChunk("This is the first document chunk.", 1),
    MockChunk("Another piece of text for testing.", 2),
    MockChunk("The quick brown fox jumps over the lazy dog.", 3),
]
EMPTY_CHUNKS = []
MALFORMED_CHUNKS = ["just a string", "another string"] # Not MockChunk objects

TEST_INDEX_PATH = "test_artifacts/faiss.index"
TEST_CHUNKS_PATH = "test_artifacts/chunks.pkl"
TEST_MODEL_NAME_INVALID = "this-model-does-not-exist-hopefully"

def cleanup_test_artifacts():
    if os.path.exists("test_artifacts"):
        shutil.rmtree("test_artifacts")
    logger.info("Cleaned up test artifacts.")

def setup_test_artifacts_dir():
    if not os.path.exists("test_artifacts"):
        os.makedirs("test_artifacts")
    logger.info("Created test_artifacts directory.")

def test_initialization():
    logger.info("--- Testing Initialization ---")
    vs_default = VectorStore() # Default model
    assert vs_default.encoder is not None, "Encoder should be initialized with default model"
    assert vs_default.model_name == "all-MiniLM-L6-v2", "Default model name not set"
    logger.info("Default initialization OK.")

    vs_custom = VectorStore(model_name="sentence-transformers/all-MiniLM-L6-v2") # Explicit valid model
    assert vs_custom.encoder is not None, "Encoder should be initialized with custom valid model"
    assert vs_custom.model_name == "sentence-transformers/all-MiniLM-L6-v2", "Custom model name not set"
    logger.info("Custom model initialization OK.")

    try:
        VectorStore(model_name=TEST_MODEL_NAME_INVALID)
        assert False, "Should have raised VectorStoreInitializationError for invalid model"
    except VectorStoreInitializationError:
        logger.info(f"Correctly caught VectorStoreInitializationError for invalid model: {TEST_MODEL_NAME_INVALID}")
    except Exception as e:
        assert False, f"Wrong exception type for invalid model: {e}"
    logger.info("Invalid model name handling OK.")


def test_build_index():
    logger.info("--- Testing Build Index ---")
    vs = VectorStore()

    # Valid chunks
    index = vs.build_index(SAMPLE_CHUNKS)
    assert index is not None, "Index should be built with valid chunks"
    assert vs.index is not None, "Internal index should be set"
    assert len(vs.chunks) == len(SAMPLE_CHUNKS), "Internal chunks not stored correctly"
    logger.info("Building index with valid chunks OK.")

    # Empty chunks
    index_empty = vs.build_index(EMPTY_CHUNKS)
    assert index_empty is None, "Index should be None for empty chunks"
    assert vs.index is None, "Internal index should be None after empty chunks"
    logger.info("Building index with empty chunks OK (expected warning in logs).")

    # Malformed chunks
    index_malformed = vs.build_index(MALFORMED_CHUNKS)
    assert index_malformed is None, "Index should be None for malformed chunks"
    logger.info("Building index with malformed chunks OK (expected error in logs).")

    # Test with chunks that have no .text attribute (implicitly via malformed)
    class NoTextChunk:
        pass
    index_no_text = vs.build_index([NoTextChunk(), NoTextChunk()])
    assert index_no_text is None, "Index should be None for chunks without .text"
    logger.info("Building index with chunks missing .text attribute OK (expected error in logs).")


def test_search():
    logger.info("--- Testing Search ---")
    vs = VectorStore()

    # Search before index built
    results_pre_build = vs.search("query", SAMPLE_CHUNKS) # build_index will be called internally
    assert len(results_pre_build) > 0, "Search should build index and return results"
    logger.info("Search with implicit build OK.")

    vs_no_index = VectorStore()
    # Simulate build_index failure by providing bad chunks
    results_fail_build = vs_no_index.search("query", MALFORMED_CHUNKS)
    assert results_fail_build == [], "Search should return empty list if index build fails"
    logger.info("Search with failed index build returns empty list OK.")

    # Search with valid query
    vs.build_index(SAMPLE_CHUNKS) # Explicit build
    results_valid = vs.search("first document", SAMPLE_CHUNKS, k=1)
    assert len(results_valid) == 1, "Search should return k results"
    assert results_valid[0].text == "This is the first document chunk.", "Search returned incorrect chunk"
    logger.info("Search with valid query OK.")


def test_persistence():
    logger.info("--- Testing Persistence ---")
    vs_save = VectorStore()
    vs_save.build_index(SAMPLE_CHUNKS)
    assert vs_save.index is not None

    # Save
    setup_test_artifacts_dir()
    save_success = vs_save.save_index(TEST_INDEX_PATH, TEST_CHUNKS_PATH)
    assert save_success, "Save index should succeed"
    assert os.path.exists(TEST_INDEX_PATH), "FAISS index file not created"
    assert os.path.exists(TEST_CHUNKS_PATH), "Chunks pickle file not created"
    logger.info("Saving index and chunks OK.")

    # Load
    vs_load = VectorStore()
    load_success = vs_load.load_index(TEST_INDEX_PATH, TEST_CHUNKS_PATH)
    assert load_success, "Load index should succeed"
    assert vs_load.index is not None, "Index not loaded"
    assert len(vs_load.chunks) == len(SAMPLE_CHUNKS), "Chunks not loaded correctly"
    # Verify content by searching
    results_loaded = vs_load.search("lazy dog", SAMPLE_CHUNKS, k=1) # Pass original SAMPLE_CHUNKS to trigger comparison if needed
    assert len(results_loaded) == 1, "Search on loaded index failed"
    assert results_loaded[0].text == "The quick brown fox jumps over the lazy dog.", "Search on loaded index returned incorrect chunk"
    logger.info("Loading index and chunks and then searching OK.")

    # Test loading non-existent index
    vs_load_fail = VectorStore()
    load_fail_success = vs_load_fail.load_index("non_existent.index", "non_existent.pkl")
    assert not load_fail_success, "Loading non-existent index should fail"
    logger.info("Handling of loading non-existent index OK.")

    # Test search with different chunks after loading (should rebuild)
    logger.info("Testing search with different chunks after loading (expect rebuild warning)...")
    new_chunks = [MockChunk("A completely new document.", 4)]
    # The following search call should trigger a rebuild because `new_chunks` are different from `vs_load.chunks` (which are SAMPLE_CHUNKS)
    # And the warning "New chunks provided... Rebuilding index..." should appear in logs.
    results_rebuild = vs_load.search("new document", new_chunks, k=1)
    assert len(results_rebuild) == 1, "Search after rebuild with new chunks failed"
    assert results_rebuild[0].text == "A completely new document.", "Search after rebuild returned incorrect chunk"
    assert vs_load.chunks == new_chunks, "Internal chunks not updated after rebuild" # vs_load.chunks should now be new_chunks
    logger.info("Search with different chunks after loading (triggering rebuild) OK.")


if __name__ == "__main__":
    try:
        cleanup_test_artifacts() # Clean before tests start
        test_initialization()
        test_build_index()
        test_search()
        test_persistence()
    finally:
        cleanup_test_artifacts() # Clean up after tests
    logger.info("All VectorStore tests completed.")
