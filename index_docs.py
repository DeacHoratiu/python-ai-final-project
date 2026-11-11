from pathlib import Path

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cool_db"
CURRENT_DIR = Path(__file__).resolve().parent
DATA_SOURCE_DIR = CURRENT_DIR / "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def index_file(collection, model, file_path):
    print(f"Processing: {file_path.name}")

    pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True, use_ocr=False)
    records = []

    for page in pages:
        page_number = page["metadata"]["page_number"]
        page_text = page["text"]

        chunks = chunk_text(page_text)
        chunk_number = 0

        for chunk in chunks:
            records.append((chunk, page_number, chunk_number))
            chunk_number += 1

    collection.delete(where={"source": file_path.name})
    ids = []
    documents = []
    metadatas = []

    for record in records:
        chunk = record[0]
        page_number = record[1]
        chunk_number = record[2]
        ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")
        documents.append(chunk)
        metadatas.append(
            {
                "source": file_path.name,
                "page": page_number,
                "chunk": chunk_number,
            }
        )

    if not documents:
        print(f"No documents found for {file_path.name}")
        return documents, metadatas

    collection.upsert(
        ids=ids,

        embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Successfully indexed {len(documents)} chunks.\n")

    return documents, metadatas


def main():

    model = SentenceTransformer(MODEL_NAME)

    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000,
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("\nStarting document indexing...\n")

    for file_path in sorted(DATA_SOURCE_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".txt", ".pdf"}:
            continue

        index_file(collection,model,file_path)

    print("Indexing completed successfully!\n")


if __name__ == "__main__":
    main()
