from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Any
import shutil
import uuid

from llm import generate_answer
from chunker import create_chunks
from retriever import search_document
from highlight import find_highlight
from pdf_processor import process_pdf
from document_store import save_document, load_document


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="FindMe API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://document-insight-engine-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "FindMe backend is running",
        "status": "ok",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# HELPER: DETECT NO-ANSWER RESPONSES
# ============================================================

def answer_is_unsupported(answer: str) -> bool:
    """
    Detect whether the LLM is saying that the document
    does not contain enough information to answer.

    This is important because retrieval may still return
    semantically similar chunks even when the document
    does not actually answer the user's question.

    When an unsupported answer is detected, the API returns:
        sources = []
        no_answer = True
        noAnswer = True
    """

    if not answer:
        return True

    normalized = " ".join(
        answer.lower().strip().split()
    )

    no_answer_phrases = [
        # Generic inability to answer
        "i couldn't find the answer",
        "i could not find the answer",
        "i don't have enough information",
        "i do not have enough information",
        "i can't answer",
        "i cannot answer",
        "i cannot provide an answer",
        "i can't provide an answer",

        # Document does not contain information
        "the document does not contain enough information",
        "the document doesn't contain enough information",
        "the document does not contain any information",
        "the document doesn't contain any information",
        "the document does not contain information",
        "the document doesn't contain information",
        "the document does not contain any information about",
        "the document doesn't contain any information about",
        "the document does not contain information about",
        "the document doesn't contain information about",
        "the document contains no information about",

        # Document does not provide information
        "the document does not provide information",
        "the document doesn't provide information",
        "the document does not provide any information",
        "the document doesn't provide any information",
        "the document does not provide information about",
        "the document doesn't provide information about",

        # Sources do not support answer
        "the provided sources do not contain",
        "the provided sources don't contain",
        "the provided sources do not provide",
        "the provided sources don't provide",
        "the provided sources contain no information",

        "the sources do not contain",
        "the sources don't contain",
        "the sources do not provide",
        "the sources don't provide",
        "the sources contain no information",

        # Insufficient information
        "there is not enough information",
        "there isn't enough information",
        "there is insufficient information",
        "there isn't sufficient information",
        "not enough information to answer",
        "insufficient information to answer",

        # Cannot answer from sources
        "cannot answer based on the provided sources",
        "can't answer based on the provided sources",
        "cannot be answered from the provided sources",
        "can't be answered from the provided sources",

        "cannot answer based on the document",
        "can't answer based on the document",
        "cannot be answered from the document",
        "can't be answered from the document",

        # Answer absent
        "the answer is not present in the document",
        "the answer is not in the document",
        "the answer is not provided in the document",
        "the document does not contain the answer",
        "the document doesn't contain the answer",

        # Answer cannot be determined
        "the answer cannot be determined from the document",
        "the answer can't be determined from the document",
        "the answer cannot be determined",
        "the answer can't be determined",

        # Very common direct refusal pattern
        "so i cannot answer that question",
        "so i can't answer that question",
        "therefore i cannot answer",
        "therefore i can't answer",

        # Short generic phrase
        "no information about",
    ]

    return any(
        phrase in normalized
        for phrase in no_answer_phrases
    )


# ============================================================
# HELPER: NO-ANSWER RESPONSE
# ============================================================

def no_answer_response(
    question: str,
    answer: str | None = None,
):
    """
    Create a consistent no-answer response.

    Both no_answer and noAnswer are returned so the backend
    remains compatible with different frontend naming styles.
    """

    final_answer = (
        answer.strip()
        if isinstance(answer, str) and answer.strip()
        else "The document does not contain enough information to answer this question."
    )

    return {
        "question": question,
        "answer": final_answer,
        "sources": [],
        "no_answer": True,
        "noAnswer": True,
    }


# ============================================================
# HELPER: SUCCESS RESPONSE
# ============================================================

def success_response(
    question: str,
    answer: str,
    sources: list,
):
    """
    Create a consistent successful answer response.
    """

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "no_answer": False,
        "noAnswer": False,
    }


# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Upload a PDF, extract its pages and word coordinates,
    create searchable chunks, and store the document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_id = str(uuid.uuid4())

    safe_filename = Path(file.filename).name

    file_path = (
        UPLOAD_DIR /
        f"{document_id}_{safe_filename}"
    )

    try:
        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # ----------------------------------------------------
        # Process PDF
        # ----------------------------------------------------

        result = process_pdf(
            str(file_path)
        )

        if not result:
            raise ValueError(
                "PDF processing returned no data."
            )

        pages = result.get(
            "pages",
            [],
        )

        page_count = result.get(
            "page_count",
            len(pages),
        )

        # ----------------------------------------------------
        # Check text layer
        # ----------------------------------------------------

        total_words = sum(
            len(page.get("words", []))
            for page in pages
        )

        if total_words == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This PDF does not contain a readable text layer. "
                    "Scanned/image-only PDFs are not supported."
                ),
            )

        # ----------------------------------------------------
        # Create searchable chunks
        # ----------------------------------------------------

        chunks = create_chunks(
            {
                "pages": pages,
            }
        )

        if not chunks:
            raise ValueError(
                "No searchable text chunks were created."
            )

        # ----------------------------------------------------
        # Give every chunk a unique ID
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):
            chunk["id"] = (
                f"{document_id}_chunk_{index}"
            )

        # ----------------------------------------------------
        # Store complete document
        # ----------------------------------------------------

        document_data = {
            "document_id": document_id,
            "filename": safe_filename,
            "page_count": page_count,
            "pages": pages,
            "chunks": chunks,
        }

        save_document(
            document_id,
            document_data,
        )

        # ----------------------------------------------------
        # Return upload information
        # ----------------------------------------------------

        return {
            "document_id": document_id,
            "filename": safe_filename,
            "page_count": page_count,
            "message": (
                "PDF uploaded and processed successfully."
            ),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process PDF: {str(e)}",
        )

    finally:
        try:
            file.file.close()
        except Exception:
            pass


# ============================================================
# GET DOCUMENT
# ============================================================

@app.get("/documents/{document_id}")
def get_document(
    document_id: str
):
    document = load_document(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document


# ============================================================
# HIGHLIGHT TEXT
# ============================================================

@app.post("/highlight")
async def highlight_text(
    data: dict[str, Any]
):
    page_words = data.get(
        "page_words",
        [],
    )

    target_text = data.get(
        "text",
        "",
    )

    if not page_words:
        raise HTTPException(
            status_code=400,
            detail="page_words are required.",
        )

    if not target_text:
        raise HTTPException(
            status_code=400,
            detail="text is required.",
        )

    try:
        rectangles = find_highlight(
            page_words,
            target_text,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Highlight calculation failed: {str(e)}"
            ),
        )

    return {
        "found": bool(rectangles),
        "rectangles": rectangles or [],
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
async def ask_question(
    data: dict[str, Any]
):
    """
    Retrieve relevant document chunks and generate a
    grounded answer.

    Important behavior:

    1. If nothing is retrieved:
       -> no answer
       -> no sources
       -> no highlights

    2. If the LLM says the document does not contain
       the answer:
       -> no answer
       -> no sources
       -> no highlights

    3. If retrieved chunks cannot be mapped to actual
       PDF coordinates:
       -> no answer
       -> no sources
       -> no highlights

    4. Only successfully highlighted chunks become
       sources.
    """

    # --------------------------------------------------------
    # Read request
    # --------------------------------------------------------

    document_id = data.get(
        "document_id"
    )

    question = data.get(
        "question"
    )

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="document_id is required.",
        )

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question is required.",
        )

    question = str(question).strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question cannot be empty.",
        )

    # --------------------------------------------------------
    # Load document
    # --------------------------------------------------------

    document = load_document(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    try:
        results = search_document(
            document,
            question,
            top_k=5,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Document search failed: {str(e)}"
            ),
        )

    # --------------------------------------------------------
    # Nothing retrieved
    # --------------------------------------------------------

    if not results:
        return no_answer_response(
            question,
            "I couldn't find the answer in the document.",
        )

    # --------------------------------------------------------
    # Generate grounded answer
    # --------------------------------------------------------

    try:
        answer = generate_answer(
            question,
            results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Answer generation failed: {str(e)}"
            ),
        )

    answer = (
        str(answer).strip()
        if answer is not None
        else ""
    )

    # --------------------------------------------------------
    # CRITICAL NO-ANSWER CHECK
    # --------------------------------------------------------

    if answer_is_unsupported(answer):
        return no_answer_response(
            question,
            answer,
        )

    # --------------------------------------------------------
    # Build valid sources
    # --------------------------------------------------------

    sources = []

    for index, result in enumerate(results):

        words = result.get(
            "words",
            [],
        )

        if not words:
            continue

        # ----------------------------------------------------
        # Calculate actual PDF highlight coordinates
        # ----------------------------------------------------

        rectangles = []

        try:
            rectangles = find_highlight(
                words,
                result.get(
                    "text",
                    "",
                ),
            )

        except Exception:
            rectangles = []

        # ----------------------------------------------------
        # Only accept sources with real coordinates
        # ----------------------------------------------------

        if not rectangles:
            continue

        sources.append(
            {
                "id": index + 1,

                "source_id": index + 1,

                "page": result.get(
                    "page",
                    1,
                ),

                "chunkId": result.get(
                    "id"
                ),

                "text": result.get(
                    "text",
                    "",
                ),

                "score": float(
                    result.get(
                        "score",
                        0.0,
                    )
                ),

                "semantic_score": float(
                    result.get(
                        "semantic_score",
                        0.0,
                    )
                ),

                "keyword_score": float(
                    result.get(
                        "keyword_score",
                        0.0,
                    )
                ),

                "words": words,

                "rectangles": rectangles,
            }
        )

    # --------------------------------------------------------
    # CRITICAL: Answer without valid highlights
    # --------------------------------------------------------

    if not sources:
        return no_answer_response(
            question,
            "The document does not contain enough information to answer this question.",
        )

    # --------------------------------------------------------
    # Successful grounded answer
    # --------------------------------------------------------

    return success_response(
        question,
        answer,
        sources,
    )


# ============================================================
# RUN LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )