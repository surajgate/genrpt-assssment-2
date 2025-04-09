import os
import logging
import tempfile

from typing import List, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv

# PDF processing
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Vector storage components
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text_with_pypdf(pdf_path: str) -> str:
    """
    Extract text content from a text-based PDF file using PyPDF2.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The combined textual content of the PDF.
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)

            # Extract metadata if available
            metadata = reader.metadata
            if metadata:
                logger.info(f"Document info: {metadata}")

            # Extract text from each page
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"  # Double newline for better paragraph separation
                    else:
                        logger.warning(f"No text extracted from page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {e}")

        return text.strip()
    except Exception as e:
        logger.error(f"Failed to process PDF with PyPDF2 {pdf_path}: {e}")
        return ""


def extract_text_with_ocr(pdf_path: str, dpi: int = 300, language: str = 'eng') -> str:
    """
    Extract text content from a PDF file using OCR.

    This function:
    1. Converts each PDF page to an image
    2. Performs OCR on each image to extract text
    3. Combines all text into a single string

    Args:
        pdf_path (str): The path to the PDF file
        dpi (int): The DPI resolution for image conversion (higher = better quality but slower)
        language (str): The language for OCR (default: 'eng' for English)

    Returns:
        str: The combined OCR-extracted text from the PDF
    """
    logger.info(f"Starting OCR extraction on {pdf_path} with {dpi} DPI")

    try:
        # Create a temporary directory to store images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF to images
            logger.info("Converting PDF to images...")
            pdf_images = convert_from_path(
                pdf_path,
                dpi=dpi,
                output_folder=temp_dir,
                fmt="jpeg",
                thread_count=os.cpu_count() or 4
            )

            logger.info(
                f"PDF converted to {len(pdf_images)} images. Starting OCR...")

            # Extract text from each image using OCR
            extracted_text = []
            for i, img in enumerate(pdf_images):
                try:
                    page_text = pytesseract.image_to_string(img, lang=language)
                    extracted_text.append(page_text)
                    logger.debug(f"OCR completed for page {i+1}")
                except Exception as e:
                    logger.error(f"OCR failed for page {i+1}: {e}")

            # Combine all extracted text
            full_text = "\n\n".join(extracted_text)
            logger.info(
                f"OCR extraction complete. Extracted {len(full_text)} characters.")
            return full_text.strip()

    except Exception as e:
        logger.error(f"PDF OCR extraction failed: {e}")
        return ""


def extract_text_from_pdf(pdf_path: str, ocr_fallback: bool = True) -> str:
    """
    Extract text from a PDF with fallback to OCR if needed.

    This function attempts to extract text using PyPDF2 first, and if
    the result is insufficient, falls back to OCR extraction.

    Args:
        pdf_path (str): Path to the PDF file
        ocr_fallback (bool): Whether to use OCR as a fallback

    Returns:
        str: Extracted text content
    """
    # First try regular text extraction
    text = extract_text_with_pypdf(pdf_path)

    # Check if we need to fall back to OCR
    if ocr_fallback and (not text or len(text) < 100):
        logger.info(
            "Standard text extraction produced insufficient results. Falling back to OCR...")
        text = extract_text_with_ocr(pdf_path)

    return text


def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split a large block of text into smaller overlapping chunks.

    Uses a recursive character splitter with specified chunk size and overlap
    to preserve context across chunks.

    Args:
        text (str): The input text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between consecutive chunks

    Returns:
        List[str]: A list of text chunks
    """
    if not text:
        logger.warning("No text to chunk. Returning empty list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def store_in_milvus(chunks: List[str], metadata: Optional[dict] = None) -> bool:
    """
    Embed and store text chunks in a Milvus vector database with BM25 support.

    Args:
        chunks (List[str]): A list of text chunks to be embedded and stored
        metadata (dict, optional): Metadata to attach to all documents

    Returns:
        bool: Success or failure
    """
    if not chunks:
        logger.warning("No chunks to store. Skipping Milvus storage.")
        return False

    try:
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-large"
        )

        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {"chunk_id": i}
            if metadata:
                doc_metadata.update(metadata)
            documents.append(
                Document(page_content=chunk, metadata=doc_metadata))

        logger.info(f"Storing {len(documents)} documents in Milvus...")

        _ = Milvus.from_documents(
            connection_args={"uri": MILVUS_CONNECTION_URI},
            documents=documents,
            embedding=embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            collection_name=COLLECTION_NAME,
        )

        logger.info("Successfully stored documents in Milvus")
        return True

    except Exception as e:
        logger.error(f"Failed to store documents in Milvus: {e}")
        return False


def process_pdf(pdf_path: str, use_ocr: bool = True, extract_metadata: bool = True) -> Tuple[bool, dict]:
    """
    Full pipeline for processing a PDF file and storing its content in Milvus.

    Args:
        pdf_path (str): The path to the PDF file to process
        use_ocr (bool): Whether to use OCR as a fallback
        extract_metadata (bool): Whether to extract PDF metadata

    Returns:
        Tuple[bool, dict]: Success status and processing statistics
    """
    stats = {
        "filename": os.path.basename(pdf_path),
        "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 2),
        "characters_extracted": 0,
        "chunks_created": 0,
        "storage_successful": False
    }

    logger.info(f"Processing file: {pdf_path}")

    # Extract text from PDF
    raw_text = extract_text_from_pdf(pdf_path, ocr_fallback=use_ocr)
    if not raw_text:
        logger.warning("No valid text content extracted from the PDF.")
        return False, stats

    stats["characters_extracted"] = len(raw_text)

    # Extract metadata if requested
    pdf_metadata = {}
    if extract_metadata:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                if reader.metadata:
                    for key, value in reader.metadata.items():
                        # Clean up metadata key names
                        clean_key = key.strip('/').lower()
                        pdf_metadata[clean_key] = value
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")

    # Create chunks from the extracted text
    chunks = create_chunks(raw_text)
    stats["chunks_created"] = len(chunks)

    # Store chunks in Milvus
    storage_result = store_in_milvus(chunks, metadata=pdf_metadata)
    stats["storage_successful"] = storage_result

    logger.info(f"Processing complete for {pdf_path}. Stats: {stats}")
    return storage_result, stats


def batch_process_pdfs(directory_path: str) -> List[dict]:
    """
    Process all PDF files in a directory.

    Args:
        directory_path (str): Path to directory containing PDFs

    Returns:
        List[dict]: Processing statistics for each file
    """
    pdf_dir = Path(directory_path)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    results = []
    for pdf_file in pdf_files:
        success, stats = process_pdf(str(pdf_file))
        results.append(stats)

    successful = sum(1 for r in results if r["storage_successful"])
    logger.info(
        f"Batch processing complete. {successful}/{len(results)} files processed successfully.")

    return results


if __name__ == "__main__":
    file_path = "data/IPL_Teams.pdf"
    process_pdf(file_path, use_ocr=True)
