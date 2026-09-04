from highlight import find_highlight


page_words = [
    {
        "text": "The",
        "x0": 100,
        "y0": 200,
        "x1": 120,
        "y1": 215
    },
    {
        "text": "company",
        "x0": 125,
        "y0": 200,
        "x1": 190,
        "y1": 215
    },
    {
        "text": "generated",
        "x0": 195,
        "y0": 200,
        "x1": 270,
        "y1": 215
    },
    {
        "text": "$4.2",
        "x0": 275,
        "y0": 200,
        "x1": 310,
        "y1": 215
    },
    {
        "text": "million",
        "x0": 315,
        "y0": 200,
        "x1": 370,
        "y1": 215
    }
]


target = "The company generated $4.2 million"


result = find_highlight(page_words, target)

print(result)