import re


def normalize_text(text: str) -> str:
    """
    Normalize text so small differences in spaces,
    line breaks, and capitalization don't prevent matching.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_highlight(page_words, target_text):
    """
    Find the words belonging to target_text and return
    their coordinates.
    """

    target_words = normalize_text(target_text).split()

    if not target_words:
        return []

    normalized_page_words = [
        normalize_text(word["text"])
        for word in page_words
    ]

    for start in range(len(normalized_page_words)):
        end = start + len(target_words)

        if end > len(normalized_page_words):
            break

        candidate = normalized_page_words[start:end]

        if candidate == target_words:

            matched_words = page_words[start:end]

            x0 = min(word["x0"] for word in matched_words)
            y0 = min(word["y0"] for word in matched_words)
            x1 = max(word["x1"] for word in matched_words)
            y1 = max(word["y1"] for word in matched_words)

            return [{
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0
            }]

    return []