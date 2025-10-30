from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .db import init_raw_docs_db, insert_many_raw_documents


TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSION = ".pdf"


def _read_text_file(path: Path) -> str:
	"""Read and return the content of a text file as a string."""
	return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
	"""Read and return the extracted text content of a PDF file as a string."""
	try:
		from pypdf import PdfReader
	except Exception as e:
		raise RuntimeError("pypdf is required to read PDFs. Install pypdf.") from e

	reader = PdfReader(str(path))
	texts: List[str] = []
	for page in reader.pages:
		try:
			texts.append(page.extract_text() or "")
		except Exception:
			texts.append("")
	return "\n".join(texts)


def _gather_files(source_dir: str) -> List[Path]:
	"""Recursively collect all supported text and PDF files from the source directory."""
	root = Path(source_dir)
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Source dir not found: {source_dir}")
	paths: List[Path] = []
	for p in root.rglob("*"):
		if p.is_file() and (p.suffix.lower() in TEXT_EXTENSIONS or p.suffix.lower() == PDF_EXTENSION):
			paths.append(p)
	return paths


def ingest_files_to_db(
	source_dir: str,
	db_path: str,
	default_metadata: Optional[Dict] = None,
) -> int:
	"""Read files under `source_dir` and insert into SQLite.

	Returns the number of files successfully ingested.
	"""
	# Step 1: Initialize the SQLite database
	init_raw_docs_db(db_path)

	# Step 2: Gather all supported files from the source directory
	files = _gather_files(source_dir)

	# Step 3: Read the content of each file and insert into a list of tuples
	rows: List[Tuple[str, str, Optional[dict]]] = []
	for f in files:
		if f.suffix.lower() in TEXT_EXTENSIONS:
			content = _read_text_file(f)
		elif f.suffix.lower() == PDF_EXTENSION:
			content = _read_pdf_file(f)
		else:
			continue
		# Create metadata for the file
		metadata = {"filename": f.name, **(default_metadata or {})}
		rows.append((str(f), content, metadata))

	# Step 4: Insert the rows into SQLite
	if rows:
		insert_many_raw_documents(db_path, rows)
	return len(rows)
