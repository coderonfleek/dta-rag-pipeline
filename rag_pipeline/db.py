from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional

from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, Session

try:
	from langchain_core.documents import Document
except Exception:  # fallback for older langchain
	from langchain.schema import Document  # type: ignore


Base = declarative_base()


class RawDocument(Base):
	__tablename__ = "raw_documents"

	id = Column(Integer, primary_key=True, autoincrement=True)
	source = Column(String, nullable=False)
	content = Column(Text, nullable=False)
	metadata_json = Column(Text, nullable=True)


def _make_engine(db_path: str):
	uri = f"sqlite:///{os.path.abspath(db_path)}"
	return create_engine(uri, future=True)


def init_raw_docs_db(db_path: str) -> None:
	"""Create the SQLite DB and `raw_documents` table if missing."""
	engine = _make_engine(db_path)
	Base.metadata.create_all(engine)


def insert_raw_document(
	db_path: str,
	source: str,
	content: str,
	metadata: Optional[dict] = None,
) -> int:
	"""Insert a single raw document and return its ID."""
	engine = _make_engine(db_path)
	with Session(engine) as session:
		row = RawDocument(
			source=source,
			content=content,
			metadata_json=json.dumps(metadata or {}),
		)
		session.add(row)
		session.commit()
		session.refresh(row)
		return int(row.id)


def insert_many_raw_documents(
	db_path: str,
	docs: Iterable[tuple[str, str, Optional[dict]]],
) -> None:
	"""Bulk insert multiple (source, content, metadata) tuples."""
	engine = _make_engine(db_path)
	with Session(engine) as session:
		for source, content, metadata in docs:
			row = RawDocument(
				source=source,
				content=content,
				metadata_json=json.dumps(metadata or {}),
			)
			session.add(row)
		session.commit()


def fetch_all_documents(db_path: str) -> List[Document]:
	"""Fetch all raw documents from the SQLite DB as LangChain `Document`s."""
	engine = _make_engine(db_path)
	documents: List[Document] = []
	with Session(engine) as session:
		rows: list[RawDocument] = session.query(RawDocument).all()
		for r in rows:
			metadata = {}
			if r.metadata_json:
				try:
					metadata = json.loads(r.metadata_json)
				except Exception:
					metadata = {"_metadata_parse_error": True}
			documents.append(
				Document(page_content=r.content, metadata={"source": r.source, **metadata})
			)
	return documents
