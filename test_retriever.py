from retriever import search_document


document = {
    "chunks": [
        {
            "page": 1,
            "text": "This document describes the company background and history.",
            "words": [],
            "embedding": []
        },
        {
            "page": 2,
            "text": "The company generated revenue of 4.2 million dollars in 2024.",
            "words": [],
            "embedding": []
        },
        {
            "page": 3,
            "text": "The company expanded its international operations.",
            "words": [],
            "embedding": []
        }
    ]
}


# Create embeddings for the test chunks
from embeddings import create_embedding

for chunk in document["chunks"]:
    chunk["embedding"] = create_embedding(chunk["text"])


query = "What was the company revenue in 2024?"

results = search_document(
    document,
    query,
    top_k=3
)


for result in results:
    print(
        f"Page {result['page']} | Score: {result['score']:.4f}"
    )
    print(result["text"])
    print()