from chunking_evaluation.chunking import FixedTokenChunker, RecursiveTokenChunker, ClusterSemanticChunker, LLMSemanticChunker, KamradtModifiedChunker
from custom_chunking import GeminiSyntheticEvaluation
from chunking_evaluation.utils import openai_token_count
from utils import get_gemini_embedding
from chromadb.utils.embedding_functions import EmbeddingFunction
import pandas as pd

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """
        Tạo embedding cho một list văn bản.
        
        Args:
            texts (list[str]): Danh sách các câu/văn bản.
        
        Returns:
            list[list[float]]: Danh sách vector embedding.
        """
        embeddings = get_gemini_embedding(texts)
        return embeddings

corpora_paths = [
    'cleaned_data/page_501_600.txt',
    'cleaned_data/page_601_700.txt',
    'cleaned_data/page_701_800.txt',
    'cleaned_data/page_801_900.txt',
    'cleaned_data/page_901_1000.txt',
    'cleaned_data/page_1001_1100.txt',
    'cleaned_data/page_1101_1123.txt',
]
queries_csv_path = 'data_test_chunking.csv'

ef = GeminiEmbeddingFunction()

eval = GeminiSyntheticEvaluation(
    corpora_paths=corpora_paths,
    queries_csv_path=queries_csv_path,
    chroma_db_path= "/Users/nhatnamdo/chroma_data",

)
# eval.generate_queries_and_excerpts(approximate_excerpts=True, num_rounds=1, queries_per_corpus=50)

chunkers = [
    RecursiveTokenChunker(chunk_size=800, chunk_overlap=400, length_function=openai_token_count),
    RecursiveTokenChunker(chunk_size=400, chunk_overlap=200, length_function=openai_token_count),
    RecursiveTokenChunker(chunk_size=400, chunk_overlap=0, length_function=openai_token_count),
    RecursiveTokenChunker(chunk_size=200, chunk_overlap=0, length_function=openai_token_count),
    ClusterSemanticChunker(embedding_function=ef, max_chunk_size=400, length_function=openai_token_count),
]

results = []
for i, chunker in enumerate(chunkers):
    result = eval.run(chunker, ef, retrieve=5)
    chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else 0
    chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else 0
    result['chunker'] = chunker.__class__.__name__ + f"_{chunk_size}_{chunk_overlap}"
    results.append(result)
    print(f"Completed {i+1}/{len(chunkers)}: {chunker.__class__.__name__} with chunk size {chunk_size} and chunk overlap {chunk_overlap}")
df = pd.DataFrame(results)
df.to_csv('gemini_chunking_evaluation_results.csv', index=False)
