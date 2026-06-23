from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._in_link = False
        self._in_snippet = False
        self._current_url = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        class_name = attr.get("class", "")
        if tag == "a" and "result__a" in class_name:
            self._in_link = True
            self._current_url = _normalize_duckduckgo_url(attr.get("href", ""))
            self._title_parts = []
            self._snippet_parts = []
        if tag in {"a", "div"} and "result__snippet" in class_name:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            title = " ".join("".join(self._title_parts).split())
            if title and self._current_url:
                self.results.append(WebSearchResult(title=title, url=self._current_url, snippet=""))
        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join("".join(self._snippet_parts).split())
            if snippet and self.results:
                last = self.results[-1]
                self.results[-1] = WebSearchResult(title=last.title, url=last.url, snippet=snippet)

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title_parts.append(data)
        if self._in_snippet:
            self._snippet_parts.append(data)


def web_search(query: str, max_results: int = 5) -> list[WebSearchResult]:
    clean_query = " ".join((query or "").split())
    if not clean_query:
        return []
    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": clean_query},
        headers={"User-Agent": "Mozilla/5.0 OptiAgent/1.0"},
        timeout=12,
    )
    response.raise_for_status()
    parser = _DuckDuckGoHTMLParser()
    parser.feed(response.text)
    unique: list[WebSearchResult] = []
    seen = set()
    for item in parser.results:
        if not item.url or item.url in seen:
            continue
        seen.add(item.url)
        unique.append(item)
        if len(unique) >= max_results:
            break
    return unique


def _normalize_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url
