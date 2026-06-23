from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class RetrievedDoc:
    title: str
    content: str
    score: float
    category: str = "general"
    source: str = "knowledge_base.md"


def load_knowledge_base(path: Path | None = None) -> list[RetrievedDoc]:
    if path is not None:
        return _load_markdown_docs(path, category="general")

    root = Path(__file__).resolve().parent
    sources = [
        (root / "knowledge_base.md", "general"),
        (root / "or_knowledge_base.md", "modeling"),
    ]
    docs: list[RetrievedDoc] = []
    for source, category in sources:
        if source.exists():
            docs.extend(_load_markdown_docs(source, category=category))
    return docs


def _load_markdown_docs(source: Path, category: str) -> list[RetrievedDoc]:
    text = source.read_text(encoding="utf-8")
    docs: list[RetrievedDoc] = []
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        title = lines[0].lstrip("#").strip()
        content = "\n".join(lines[1:]).strip()
        doc_category = _infer_category(title, content, category)
        docs.append(RetrievedDoc(title=title, content=content, score=0.0, category=doc_category, source=source.name))
    return docs


def retrieve(query: str, top_k: int = 3, docs: list[RetrievedDoc] | None = None) -> list[RetrievedDoc]:
    candidates = docs or load_knowledge_base()
    if not candidates:
        return []

    corpus = [f"{doc.title} {doc.content}" for doc in candidates]
    tokens = [_tokenize(query), *[_tokenize(item) for item in corpus]]
    vocab = sorted({token for doc_tokens in tokens for token in doc_tokens})
    if not vocab:
        return candidates[:top_k]

    matrix = np.array([_vectorize(doc_tokens, vocab) for doc_tokens in tokens], dtype=float)
    query_vec = matrix[0]
    doc_matrix = matrix[1:]
    idf = _idf(doc_matrix)
    query_vec = query_vec * idf
    doc_matrix = doc_matrix * idf

    query_norm = np.linalg.norm(query_vec)
    scores = []
    for doc_vec in doc_matrix:
        denom = query_norm * np.linalg.norm(doc_vec)
        scores.append(float(np.dot(query_vec, doc_vec) / denom) if denom else 0.0)

    ranked = sorted(
        (
            RetrievedDoc(title=doc.title, content=doc.content, score=score, category=doc.category, source=doc.source)
            for doc, score in zip(candidates, scores, strict=True)
        ),
        key=lambda doc: doc.score,
        reverse=True,
    )
    return ranked[:top_k]


def retrieve_by_category(query: str, category: str, top_k: int = 2) -> list[RetrievedDoc]:
    docs = [doc for doc in load_knowledge_base() if doc.category == category]
    return retrieve(query, top_k=top_k, docs=docs)


def rag_context_pack(query: str) -> dict[str, list[dict]]:
    categories = {
        "modeling": "建模知识",
        "schema": "数据要求",
        "template": "代码模板",
        "solver": "求解策略",
    }
    pack: dict[str, list[dict]] = {}
    for category, label in categories.items():
        docs = retrieve_by_category(query, category, top_k=2)
        pack[label] = [
            {
                "title": doc.title,
                "score": round(doc.score, 4),
                "source": doc.source,
                "content": doc.content,
            }
            for doc in docs
        ]
    return pack


def rag_summary(query: str, top_k: int = 3) -> tuple[list[str], list[RetrievedDoc]]:
    docs = retrieve(query, top_k=top_k)
    notes = []
    for doc in docs:
        first_sentence = re.split(r"[。！？\n]", doc.content.strip())[0]
        if first_sentence:
            notes.append(f"根据《{doc.title}》：{first_sentence}。")
    return notes, docs


def _infer_category(title: str, content: str, fallback: str) -> str:
    text = f"{title}\n{content}"
    if "类别：schema" in text or "Schema" in title or "数据" in title and "字段" in text:
        return "schema"
    if "类别：template" in text or "模板" in title or "代码" in title:
        return "template"
    if "类别：solver" in text or "求解器" in title or "Gurobi" in title or "OR-Tools" in title:
        return "solver"
    if "类别：modeling" in text or "建模" in title or "模型" in title or "问题" in title:
        return "modeling"
    return fallback


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-zA-Z0-9_]+", lowered)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    bigrams = []
    for token in chinese_tokens:
        bigrams.extend(token[i : i + 2] for i in range(max(len(token) - 1, 0)))
    return latin_tokens + chinese_tokens + bigrams


def _vectorize(tokens: list[str], vocab: list[str]) -> np.ndarray:
    counts = {token: tokens.count(token) for token in set(tokens)}
    return np.array([counts.get(token, 0) for token in vocab], dtype=float)


def _idf(doc_matrix: np.ndarray) -> np.ndarray:
    doc_count = doc_matrix.shape[0]
    df = (doc_matrix > 0).sum(axis=0)
    return np.log((1 + doc_count) / (1 + df)) + 1
