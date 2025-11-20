from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import CharacterTextSplitter


def load_documents(file_path):
    """
    Load documents from a directory or single file.
    Supports: PDF, TXT, DOCX, XLSX, CSV, PPTX

    Args:
        file_path: Path to a directory or single file

    Returns:
        List of split document chunks
    """
    documents = []

    # Map file extensions to their loaders
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".csv": CSVLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
    }

    path = Path(file_path)

    # Handle single file
    if path.is_file():
        files_to_process = [path]
    # Handle directory
    elif path.is_dir():
        files_to_process = [f for f in path.rglob("*") if f.is_file()]
    else:
        raise ValueError(f"Path does not exist: {file_path}")

    # Load each supported file
    for file in files_to_process:
        file_extension = file.suffix.lower()

        if file_extension in loader_mapping:
            try:
                loader_class = loader_mapping[file_extension]
                loader = loader_class(str(file))
                documents.extend(loader.load())
                print(f"Loaded: {file.name}")
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
        else:
            print(f"Skipped (unsupported): {file.name}")

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    return text_splitter.split_documents(documents)
