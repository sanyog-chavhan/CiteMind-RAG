from vector_store import get_collection
import pymupdf4llm
from config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PAPER_URL, PAPER_FILENAME
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import urllib.request

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def download_paper():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPERS_DIR / PAPER_FILENAME
    if target.exists():
        return target
    print(f"Downloading {PAPER_FILENAME} from {PAPER_URL}...")
    urllib.request.urlretrieve(PAPER_URL, target)
    with open(target, "rb") as f:
        if f.read(5) != b"%PDF-":
            target.unlink()  # Delete the file if it's not a valid PDF
            raise ValueError(f"Downloaded file {PAPER_FILENAME} is not a valid PDF.")
    return target

def load_pdfs():
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    all_pages = []
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the papers directory.")

    for pdf in pdf_files:
        page = pymupdf4llm.to_markdown(str(pdf), page_chunks=True)
        all_pages.extend(page)
    return all_pages

def normalize_text(each_page):
    normalized_pages = []
    for page in each_page:
        normalized_text = unicodedata.normalize("NFKC", page["text"])
        normalized_pages.append({"text": normalized_text, "metadata": page["metadata"]})
    return normalized_pages

def chunk_text(normalized_pages):
    all_chunks = []
    for page in normalized_pages:
        chunks = text_splitter.split_text(page["text"])
        for index, chunked_text in enumerate(chunks):
            all_chunks.append({"text": chunked_text, "metadata": {**page["metadata"], "chunk_index": index}})
    return all_chunks

def ingest_documents(all_chunks):
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    ids = []
    for chunk in all_chunks:
        ids.append(f"{Path(chunk['metadata']['file_path']).name}-p{chunk['metadata']['page_number']}-c{chunk['metadata']['chunk_index']}")

    collection = get_collection()
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return collection.count()


def build_index():
    download_paper()
    pages = load_pdfs()
    normalized_pages = normalize_text(pages)
    chunks = chunk_text(normalized_pages)
    return ingest_documents(chunks)

def ensure_index():
    if get_collection().count() == 0:
        print("No documents found in the collection. Building index...")
        build_index()

if __name__ == "__main__":
    result = build_index()
    print(f"Total documents in collection: {result}")