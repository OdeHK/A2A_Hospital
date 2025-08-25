from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, VECTOR_SIZE

class VectorDB:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=600)
        self.collection_name = QDRANT_COLLECTION_NAME

    def create_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    def upsert_points(self, embeddings, texts):
        points = [
            PointStruct(id=idx, vector=emb, payload={"content": text})
            for idx, (emb, text) in enumerate(zip(embeddings, texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, limit=5):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )