import os
import asyncio
import subprocess
from typing import Callable

from langchain_community.document_loaders import PDFPlumberLoader
from docx import Document


def _extract_pdf_sync(file_path: str) -> str:
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    return "\n".join(
        d.page_content.strip()
        for d in docs
        if d.page_content and d.page_content.strip()
    ).strip()


def _extract_docx_sync(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(
        p.text.strip()
        for p in doc.paragraphs
        if p.text and p.text.strip()
    ).strip()


def _convert_doc_to_docx(file_path: str) -> str:
    output_dir = os.path.dirname(file_path)
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "docx", file_path, "--outdir", output_dir],
        check=True
    )
    return file_path.replace(".doc", ".docx")


def _extract_doc_sync(file_path: str) -> str:
    converted_path = _convert_doc_to_docx(file_path)
    return _extract_docx_sync(converted_path)


async def _run_in_thread(func: Callable, file_path: str) -> str:
    return await asyncio.to_thread(func, file_path)


ASYNC_EXTRACTORS = {
    ".pdf": lambda fp: _run_in_thread(_extract_pdf_sync, fp),
    ".docx": lambda fp: _run_in_thread(_extract_docx_sync, fp),
    ".doc": lambda fp: _run_in_thread(_extract_doc_sync, fp),
}


async def extract_text(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    suffix = os.path.splitext(file_name)[1].lower()

    if suffix not in ASYNC_EXTRACTORS:
        raise ValueError(f"Unsupported file type: {suffix}")

    try:
        return await ASYNC_EXTRACTORS[suffix](file_path)
    except Exception as e:
        raise ValueError(f"Extraction failed for {file_name}: {str(e)}") from e