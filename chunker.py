import re
from embeddings import create_embedding


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text while preserving readable content.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(
    document: dict,
    words_per_chunk: int = 180,
    overlap: int = 60
):
    """
    Create overlapping chunks from PDF pages.

    Example:
        Chunk 1 -> words 0-179
        Chunk 2 -> words 120-299
        Chunk 3 -> words 240-419

    The overlap prevents important sentences or requirements
    from being cut between chunks.
    """

    chunks = []

    if overlap >= words_per_chunk:
        raise ValueError(
            "overlap must be smaller than words_per_chunk"
        )

    step = words_per_chunk - overlap

    for page in document.get("pages", []):

        words = page.get("words", [])

        if not words:
            continue

        page_number = page.get("page")

        for start in range(0, len(words), step):

            chunk_words = words[
                start:start + words_per_chunk
            ]

            if not chunk_words:
                continue

            text = " ".join(
                word.get("text", "")
                for word in chunk_words
            )

            chunk_text = normalize_text(text)

            if not chunk_text:
                continue

            embedding = create_embedding(chunk_text)

            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "words": chunk_words,
                "embedding": embedding
            })

            # Stop once we reach the end of the page.
            if start + words_per_chunk >= len(words):
                break

    return chunks