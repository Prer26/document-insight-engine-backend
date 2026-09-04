import fitz


def process_pdf(file_path: str):
    document = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        words = []

        for word in page.get_text("words"):
            x0, y0, x1, y1, word_text = word[:5]

            words.append({
                "text": word_text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            })

        pages.append({
            "page": page_number,
            "text": text,
            "words": words
        })

    document.close()

    return {
        "page_count": len(pages),
        "pages": pages
    }