import logging as logger
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

def load_and_chunk_documents(
    data_path: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[Document]:
    """Load documents from directory, chunk them, and prepare for ChromaDB."""

    data_dir = Path(data_path)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Invalid directory: {data_path}")

    # Define loaders for each file type
    loaders = {
        '.txt': lambda p: TextLoader(str(p), encoding='utf-8'),
        '.pdf': lambda p: PyPDFLoader(str(p))
    }

    # Load all documents
    documents = []
    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue

        loader_fn = loaders.get(file_path.suffix.lower())
        if not loader_fn:
            continue

        try:
            documents.extend(loader_fn(file_path).load())
            logger.info(f"Loaded: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    if not documents:
        raise ValueError(f"No documents loaded from {data_path}")

    logger.info(f"Total documents loaded: {len(documents)}")

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    # Add ChromaDB-friendly metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': f"chunk_{idx}",
            'chunk_index': idx,
            'source_file': Path(chunk.metadata.get('source', 'unknown')).name
        })
        # Ensure all metadata values are strings, ints, floats, or bools
        chunk.metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                         for k, v in chunk.metadata.items()}

    logger.info(f"Created {len(chunks)} chunks")
    return chunks