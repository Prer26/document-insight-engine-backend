import os

from groq import Groq
from dotenv import load_dotenv


# Load .env
load_dotenv()


# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def build_prompts(question: str, sources: list):
    """
    Build the system and user prompts from retrieved document sources.
    """

    context_parts = []

    for index, source in enumerate(sources, start=1):
        context_parts.append(
            f"[Source {index} | Page {source['page']}]\n"
            f"{source['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question ONLY using the provided document sources.

Rules:

1. Do not invent information.
2. If the answer is not present in the sources,
   clearly say that the document does not contain
   enough information to answer.
3. Cite supporting sources using:
   [Source 1], [Source 2], etc.
4. Give a clear and concise answer.
5. If multiple sources support the answer,
   cite all relevant sources.
"""

    user_prompt = f"""
Question:

{question}

Document sources:

{context}

Answer the question using ONLY the sources above.
"""

    return system_prompt, user_prompt


def generate_answer(
    question: str,
    sources: list
):
    """
    Generate a complete answer using the retrieved
    document sources.
    """

    if not sources:
        return (
            "I couldn't find enough information "
            "in the document to answer this question."
        )

    system_prompt, user_prompt = build_prompts(
        question,
        sources
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    return response.choices[0].message.content


def stream_answer(
    question: str,
    sources: list
):
    """
    Stream the answer from Groq chunk by chunk.
    """

    if not sources:
        yield (
            "I couldn't find enough information "
            "in the document to answer this question."
        )
        return

    system_prompt, user_prompt = build_prompts(
        question,
        sources
    )

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000,
        stream=True
    )

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta and delta.content:
            yield delta.content