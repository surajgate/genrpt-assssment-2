import os
import logging
from typing import List
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a text-based PDF file.

    This function reads the given PDF and extracts textual content from all its pages.
    It skips any OCR-based extraction and assumes the PDF contains embedded text.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The combined textual content of the PDF.
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def create_chunks(text: str) -> List[str]:
    """
    Split a large block of text into smaller overlapping chunks.

    Uses a recursive character splitter with specified chunk size and overlap
    to preserve context across chunks.
    
    Args:
        text (str): The input text to split.

    Returns:
        List[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def store_in_milvus(chunks: List[str]) -> None:
    """
    Embed and store text chunks in a Milvus vector database with BM25 support.

    Converts each chunk into a document, generates dense and sparse embeddings,
    and stores them in the specified Milvus collection.

    Args:
        chunks (List[str]): A list of text chunks to be embedded and stored.

    Returns:
        None
    """
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )

    documents = [Document(page_content=chunk) for chunk in chunks]

    _ = Milvus.from_documents(
        connection_args={"uri": MILVUS_CONNECTION_URI},
        documents=documents,
        embedding=embeddings,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        consistency_level="Strong",
        collection_name=COLLECTION_NAME,
    )


def pdf_processing(pdf_path: str):
    """
    Full pipeline for processing a PDF file and storing its content in Milvus.

    This function:
        1. Extracts text from the given PDF.
        2. Splits the text into overlapping chunks.
        3. Embeds and stores the chunks in Milvus with BM25 support.

    Args:
        pdf_path (str): The path to the PDF file to process.

    Returns:
        None
    """
    logger.info(f"Processing file: {pdf_path}")

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        logger.warning("No valid text content found in the PDF.")
        return

    chunks = create_chunks(raw_text)
    store_in_milvus(chunks)

    logger.info("Text processing and storage complete.")


if __name__ == "__main__":
    file_path = "data/IPL_Teams.pdf"
    pdf_processing(file_path)
