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

app = FastAPI(title="FindMe API")


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
        "message": "FindMe backend is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# HELPER: DETECT NO-ANSWER RESPONSES
# ============================================================

def answer_is_unsupported(answer: str) -> bool:
    """
    Detect when the LLM correctly says that the retrieved
    document content does not contain enough information.

    This prevents irrelevant retrieved chunks from being
    displayed as citations/highlights.
    """

    if not answer:
        return True

    normalized = " ".join(
        answer.lower().strip().split()
    )

    no_answer_phrases = [
        "i couldn't find the answer",
        "i could not find the answer",
        "the document does not contain enough information",
        "the document does not provide information",
        "the provided sources do not contain",
        "the provided sources do not provide",
        "the sources do not contain",
        "the sources do not provide",
        "there is not enough information",
        "there isn't enough information",
        "not enough information to answer",
        "cannot answer based on the provided sources",
        "can't answer based on the provided sources",
        "cannot be answered from the provided sources",
        "can't be answered from the provided sources",
        "the answer is not present in the document",
        "the document does not contain the answer",
        "the document doesn't contain the answer",
        "the answer cannot be determined from the document",
        "the answer can't be determined from the document",
    ]

    return any(
        phrase in normalized
        for phrase in no_answer_phrases
    )


# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its pages/word coordinates,
    create chunks and store the document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
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
                buffer
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
            []
        )

        page_count = result.get(
            "page_count",
            len(pages)
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
                )
            )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = create_chunks(
            {
                "pages": pages
            }
        )

        if not chunks:
            raise ValueError(
                "No searchable text chunks were created."
            )

        # ----------------------------------------------------
        # Give every chunk an ID
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
            document_data
        )

        # ----------------------------------------------------
        # Return upload information
        # ----------------------------------------------------

        return {
            "document_id": document_id,
            "filename": safe_filename,
            "page_count": page_count,
            "message": "PDF uploaded and processed successfully.",
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not process PDF: {str(e)}"
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
def get_document(document_id: str):

    document = load_document(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document


# ============================================================
# HIGHLIGHT TEXT
# ============================================================

@app.post("/highlight")
async def highlight_text(data: dict[str, Any]):

    page_words = data.get(
        "page_words",
        []
    )

    target_text = data.get(
        "text",
        ""
    )

    if not page_words:
        raise HTTPException(
            status_code=400,
            detail="page_words are required."
        )

    if not target_text:
        raise HTTPException(
            status_code=400,
            detail="text is required."
        )

    try:

        rectangles = find_highlight(
            page_words,
            target_text
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Highlight calculation failed: {str(e)}"
        )

    return {
        "found": bool(rectangles),
        "rectangles": rectangles or []
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
async def ask_question(data: dict[str, Any]):
    """
    Retrieve relevant chunks and generate an answer.

    Important:
    If the LLM determines that the retrieved document
    content does not answer the question, we return
    NO SOURCES. This prevents irrelevant page jumps
    and random highlights.
    """

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
            detail="document_id is required."
        )

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question is required."
        )

    question = str(question).strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question cannot be empty."
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
            detail="Document not found."
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    try:

        results = search_document(
            document,
            question,
            top_k=5
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Nothing retrieved
    # --------------------------------------------------------

    if not results:

        return {
            "question": question,
            "answer": (
                "I couldn't find the answer in the document."
            ),
            "sources": [],
            "no_answer": True,
        }

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            question,
            results
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )

    # --------------------------------------------------------
    # CRITICAL:
    # Check whether LLM says answer is unsupported
    # --------------------------------------------------------

    if answer_is_unsupported(answer):

        return {
            "question": question,
            "answer": (
                "The document does not contain enough information "
                "to answer this question."
            ),
            "sources": [],
            "no_answer": True,
        }

    # --------------------------------------------------------
    # Build valid sources
    # --------------------------------------------------------

    sources = []

    for index, result in enumerate(results):

        words = result.get(
            "words",
            []
        )

        # ----------------------------------------------------
        # Calculate highlight rectangles
        # ----------------------------------------------------

        rectangles = []

        if words:

            try:

                rectangles = find_highlight(
                    words,
                    result.get("text", "")
                )

            except Exception:
                rectangles = []

        # ----------------------------------------------------
        # Only return sources that have actual coordinates
        # ----------------------------------------------------

        if not rectangles:
            continue

        sources.append(
            {
                "id": index + 1,

                "source_id": index + 1,

                "page": result.get(
                    "page",
                    1
                ),

                "chunkId": result.get(
                    "id"
                ),

                "text": result.get(
                    "text",
                    ""
                ),

                "score": float(
                    result.get(
                        "score",
                        0.0
                    )
                ),

                "semantic_score": float(
                    result.get(
                        "semantic_score",
                        0.0
                    )
                ),

                "keyword_score": float(
                    result.get(
                        "keyword_score",
                        0.0
                    )
                ),

                "words": words,

                "rectangles": rectangles,
            }
        )

    # --------------------------------------------------------
    # If answer exists but no source can actually be
    # highlighted, do NOT show fake citations.
    # --------------------------------------------------------

    if not sources:

        return {
            "question": question,
            "answer": (
                "The document does not contain enough information "
                "to answer this question."
            ),
            "sources": [],
            "no_answer": True,
        }

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "no_answer": False,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )