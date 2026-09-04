from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embedding(text: str):
    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()