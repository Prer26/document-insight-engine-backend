from chunker import create_chunks


document = {
    "pages": [
        {
            "page": 1,
            "text": "",
            "words": [
                {
                    "text": "The",
                    "x0": 10,
                    "y0": 10,
                    "x1": 20,
                    "y1": 20
                },
                {
                    "text": "company",
                    "x0": 25,
                    "y0": 10,
                    "x1": 60,
                    "y1": 20
                },
                {
                    "text": "generated",
                    "x0": 65,
                    "y0": 10,
                    "x1": 110,
                    "y1": 20
                }
            ]
        }
    ]
}


chunks = create_chunks(document, words_per_chunk=2)

print("Number of chunks:", len(chunks))

for chunk in chunks:
    print("\nPage:", chunk["page"])
    print("Text:", chunk["text"])
    print("Words:", len(chunk["words"]))