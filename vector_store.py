import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from fastembed import TextEmbedding
from config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_model = TextEmbedding(model_name)

    def __call__(self, texts: Documents) -> Embeddings:
        return [vector.tolist() for vector in self.embedding_model.embed(texts)]

    @staticmethod
    def name() -> str:
        return "fastembed_dense"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "CustomEmbeddingFunction":
        return CustomEmbeddingFunction(model_name=config["model_name"])


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=CustomEmbeddingFunction()
    )