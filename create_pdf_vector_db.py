import os
import time
import base64
from glob import glob
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from google.api_core.exceptions import InternalServerError

# Load environment variables
load_dotenv()


def extract_clean_text_from_image(base64_image: str, max_retries: int = 5, delay: int = 3) -> str:
    """
    Extracts readable text from a base64 image using Gemini 2.5 Flash model with retry logic.
    """
    prompt = (
        "Extract all readable text from the image. "
        "Ignore logos. Remove excessive or unnecessary punctuation. "
        "Preserve meaningful content, structure, and line breaks."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

    for attempt in range(1, max_retries + 1):
        try:
            result = llm.invoke([message])
            return result.content
        except InternalServerError:
            print(f"[Attempt {attempt}] Internal server error. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    return "[ERROR] Failed to extract text after multiple retries."


def extract_pages_data(folder_path: str) -> List[Document]:
    """
    Extracts text from PDF files in the given folder using OCR,
    returning a list of LangChain Document objects with metadata.
    """
    pages_data = []
    pdf_paths = glob(os.path.join(folder_path, "*.pdf"))

    for pdf_path in pdf_paths:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = fitz.open(pdf_path)

        print(f"📄 Processing PDF: {pdf_name}")
        for page_number in tqdm(range(len(doc)), desc=f"{pdf_name}", unit="page"):
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

            try:
                text = extract_clean_text_from_image(img_base64)
                text = text.encode("utf-8", errors="ignore").decode("utf-8")
            except Exception as e:
                print(f"❌ Error on page {page_number + 1} of {pdf_name}: {e}")
                continue

            pages_data.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_name,
                    "page_number": page_number + 1
                }
            ))

    return pages_data


def create_db(pages_data: List[Document], chunk_size: int, chunk_overlap: int, db_dir: str) -> Chroma:
    """
    Splits documents into chunks and stores their embeddings and metadata in a ChromaDB.
    """
    try:
        # Validate input
        if not pages_data or not isinstance(pages_data, list):
            raise ValueError("pages_data must be a non-empty list of Document objects.")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if not isinstance(db_dir, str) or not db_dir.strip():
            raise ValueError("db_dir must be a valid directory path.")

        os.makedirs(db_dir, exist_ok=True)

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = splitter.split_documents(pages_data)

        if not chunked_docs:
            raise ValueError("Document splitting resulted in an empty list. Check chunk size and content.")

        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
            persist_directory=db_dir
        )

        print(f"✅ Chroma DB created with {len(chunked_docs)} chunks at: {db_dir}")
        return vectorstore

    except Exception as e:
        print(f"❌ Failed to create Chroma DB: {e}")
        raise


if __name__ == "__main__":
    # Customize these paths/values
    PDF_FOLDER = "assets"
    DB_DIR = "data/chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    print("🔍 Starting PDF OCR + Vector DB Creation...")
    pages_data = extract_pages_data(PDF_FOLDER)
    create_db(pages_data, CHUNK_SIZE, CHUNK_OVERLAP, DB_DIR)
    print("✅ All done!")
