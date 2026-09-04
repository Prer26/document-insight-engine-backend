import re
import numpy as np

from embeddings import create_embedding


def normalize_text(text: str) -> str:
    """
    Convert text into a consistent format for matching.
    """

    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two embeddings.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def keyword_score(query: str, text: str) -> float:
    """
    Calculate how many meaningful query words
    appear in the text.
    """

    query_words = set(
        normalize_text(query).split()
    )

    text_words = set(
        normalize_text(text).split()
    )

    # Common question words that don't help retrieval
    stop_words = {
        "what",
        "are",
        "is",
        "was",
        "were",
        "the",
        "a",
        "an",
        "for",
        "of",
        "in",
        "on",
        "to",
        "and",
        "or",
        "how",
        "why",
        "when",
        "where",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should"
    }

    query_words = {
        word
        for word in query_words
        if word not in stop_words
    }

    if not query_words:
        return 0.0

    matching_words = (
        query_words.intersection(text_words)
    )

    return len(matching_words) / len(query_words)


def requirement_boost(
    query: str,
    text: str
) -> float:
    """
    Give a small boost to passages that look like
    requirement sections when the user asks about
    requirements.
    """

    query_words = normalize_text(
        query
    ).split()

    requirement_intent = any(
        word in query_words
        for word in [
            "requirement",
            "requirements",
            "required",
            "must"
        ]
    )

    if not requirement_intent:
        return 0.0

    text_words = set(
        normalize_text(text).split()
    )

    requirement_words = {
        "requirement",
        "requirements",
        "required",
        "must",
        "should",
        "work",
        "show",
        "stream",
        "highlight",
        "support",
        "documents",
        "pages"
    }

    matches = (
        requirement_words.intersection(
            text_words
        )
    )

    if not matches:
        return 0.0

    return min(
        len(matches) * 0.03,
        0.15
    )


def search_document(
    document: dict,
    query: str,
    top_k: int = 8
):
    """
    Hybrid document retrieval.

    Combines:

    1. Semantic similarity
    2. Keyword matching
    3. Requirement-intent boosting

    Final score:

        65% semantic
        25% keyword
        10% requirement boost
    """

    # -----------------------------------------
    # Create embedding for the user's question
    # -----------------------------------------

    query_embedding = create_embedding(
        query
    )

    results = []

    # -----------------------------------------
    # Search through stored chunks
    # -----------------------------------------

    for chunk in document.get(
        "chunks",
        []
    ):

        embedding = chunk.get(
            "embedding"
        )

        # Skip chunks without embeddings
        if not embedding:
            continue

        text = chunk.get(
            "text",
            ""
        )

        if not text.strip():
            continue

        # -----------------------------------------
        # Semantic similarity
        # -----------------------------------------

        semantic = cosine_similarity(
            query_embedding,
            embedding
        )

        # -----------------------------------------
        # Keyword matching
        # -----------------------------------------

        keyword = keyword_score(
            query,
            text
        )

        # -----------------------------------------
        # Requirement boost
        # -----------------------------------------

        boost = requirement_boost(
            query,
            text
        )

        # -----------------------------------------
        # Hybrid score
        # -----------------------------------------

        final_score = (
            0.65 * semantic
            + 0.25 * keyword
            + 0.10 * boost
        )

        results.append({
            "page": chunk["page"],
            "text": text,
            "score": float(
                final_score
            ),
            "semantic_score": float(
                semantic
            ),
            "keyword_score": float(
                keyword
            ),
            "requirement_boost": float(
                boost
            ),
            "words": chunk.get(
                "words",
                []
            )
        })

    # -----------------------------------------
    # Sort highest score first
    # -----------------------------------------

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # -----------------------------------------
    # Return multiple supporting passages
    # -----------------------------------------

    return results[:top_k]