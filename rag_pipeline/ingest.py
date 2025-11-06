from __future__ import annotations

import os
from pathlib import Path
from io import BytesIO
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


def _is_s3_url(url: str) -> bool:
	return url.startswith("s3://")


def _parse_s3_url(url: str) -> tuple[str, str]:
	"""Return (bucket, prefix) from s3://bucket/prefix style URLs."""
	without_scheme = url[5:]
	parts = without_scheme.split("/", 1)
	bucket = parts[0]
	prefix = parts[1] if len(parts) > 1 else ""
	return bucket, prefix


def _iter_s3_keys_public(bucket: str, prefix: str) -> List[str]:
	"""List object keys for a public S3 bucket/prefix (no credentials)."""
	try:
		from botocore import UNSIGNED
		from botocore.config import Config
		import boto3
	except Exception as e:
		raise RuntimeError("boto3 is required for S3 ingestion. Install boto3.") from e

	s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
	keys: List[str] = []
	kwargs = {"Bucket": bucket, "Prefix": prefix}
	while True:
		resp = s3.list_objects_v2(**kwargs)
		for obj in resp.get("Contents", []) or []:
			key = obj["Key"]
			if key.lower().endswith(PDF_EXTENSION) or any(key.lower().endswith(ext) for ext in TEXT_EXTENSIONS):
				keys.append(key)
		if resp.get("IsTruncated"):
			kwargs["ContinuationToken"] = resp.get("NextContinuationToken")
		else:
			break
	return keys


def _read_s3_object_text_public(bucket: str, key: str) -> str:
	from botocore import UNSIGNED
	from botocore.config import Config
	import boto3
	s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
	obj = s3.get_object(Bucket=bucket, Key=key)
	data: bytes = obj["Body"].read()
	return data.decode("utf-8", errors="ignore")


def _read_s3_object_pdf_public(bucket: str, key: str) -> str:
	from botocore import UNSIGNED
	from botocore.config import Config
	import boto3
	try:
		from pypdf import PdfReader
	except Exception as e:
		raise RuntimeError("pypdf is required to read PDFs. Install pypdf.") from e

	s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
	obj = s3.get_object(Bucket=bucket, Key=key)
	data: bytes = obj["Body"].read()
	reader = PdfReader(BytesIO(data))
	texts: List[str] = []
	for page in reader.pages:
		try:
			texts.append(page.extract_text() or "")
		except Exception:
			texts.append("")
	return "\n".join(texts)


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

	rows: List[Tuple[str, str, Optional[dict]]] = []

	# Step 2: Gather and read content from local or S3
	if _is_s3_url(source_dir):
		bucket, prefix = _parse_s3_url(source_dir)
		keys = _iter_s3_keys_public(bucket, prefix)
		for key in keys:
			content: str
			if key.lower().endswith(PDF_EXTENSION):
				content = _read_s3_object_pdf_public(bucket, key)
			else:
				content = _read_s3_object_text_public(bucket, key)
			filename = os.path.basename(key)
			metadata = {"filename": filename, **(default_metadata or {})}
			source_uri = f"s3://{bucket}/{key}"
			rows.append((source_uri, content, metadata))
	else:
		files = _gather_files(source_dir)
		for f in files:
			if f.suffix.lower() in TEXT_EXTENSIONS:
				content = _read_text_file(f)
			elif f.suffix.lower() == PDF_EXTENSION:
				content = _read_pdf_file(f)
			else:
				continue
			metadata = {"filename": f.name, **(default_metadata or {})}
			rows.append((str(f), content, metadata))

	# Step 4: Insert the rows into SQLite
	if rows:
		insert_many_raw_documents(db_path, rows)
	return len(rows)
