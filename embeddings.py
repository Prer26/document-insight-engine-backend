import re
import math
import hashlib


EMBEDDING_DIMENSION = 512


def _tokenize(text: str):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _hash_index(token: str) -> int:
    digest = hashlib.md5(
        token.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little"
    ) % EMBEDDING_DIMENSION


def create_embedding(text: str):
    """
    Create a lightweight deterministic text embedding.

    Uses hashed word and bigram features instead of a
    heavyweight neural embedding model, making the
    backend suitable for low-memory deployment.
    """

    tokens = _tokenize(text)

    vector = [0.0] * EMBEDDING_DIMENSION

    if not tokens:
        return vector

    # Word features
    for token in tokens:
        index = _hash_index(token)
        vector[index] += 1.0

    # Bigram features improve phrase matching
    for first, second in zip(tokens, tokens[1:]):
        bigram = first + "_" + second
        index = _hash_index(bigram)
        vector[index] += 0.5

    # L2 normalization
    magnitude = math.sqrt(
        sum(value * value for value in vector)
    )

    if magnitude == 0:
        return vector

    return [
        value / magnitude
        for value in vector
    ]
