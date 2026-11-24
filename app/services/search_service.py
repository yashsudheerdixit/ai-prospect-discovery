import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache

from app.config import Settings
from app.schemas import RawSearchResult

logger = logging.getLogger(__name__)


class SearchBackendError(Exception):
    """Raised when an authorized search backend fails."""


class SearchService:
    """Unified async web search service supporting Serper, Brave, Tavily, and Google CSE."""

    def __init__(self, settings: Settings, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.settings = settings
        self._external_client = http_client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._cache: TTLCache[str, List[RawSearchResult]] = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=max(settings.cache_ttl_seconds, 1),
        )

    def available_backends(self) -> List[str]:
        """Return ordered list of configured authorized search backends."""
        backends: List[str] = []
        if self.settings.serper_api_key:
            backends.append("serper")
        if self.settings.brave_search_api_key:
            backends.append("brave")
        if self.settings.tavily_api_key:
            backends.append("tavily")
        if self.settings.google_cse_api_key and self.settings.google_cse_cx:
            backends.append("google_cse")
        return backends

    async def search(
        self,
        query: str,
        source_platform: str,
        num_results: Optional[int] = None,
    ) -> List[RawSearchResult]:
        """Execute a web search across configured authorized backends with caching & rate limiting."""
        limit = num_results or self.settings.max_results_per_platform
        cache_key = f"{source_platform}|{limit}|{query.strip().lower()}"

        if self.settings.cache_ttl_seconds > 0 and cache_key in self._cache:
            return self._cache[cache_key]

        backends = self.available_backends()
        if not backends:
            logger.warning("No search API keys configured; skipping web search for %s", source_platform)
            return []

        results: List[RawSearchResult] = []
        async with self._semaphore:
            for backend in backends:
                try:
                    results = await self._execute_backend(backend, query, source_platform, limit)
                    if results:
                        break
                except Exception as exc:
                    logger.warning(
                        "Search backend %s failed for platform %s (%s): %s",
                        backend,
                        source_platform,
                        query,
                        exc,
                    )

        if self.settings.cache_ttl_seconds > 0 and results:
            self._cache[cache_key] = results
        return results

    async def _execute_backend(
        self,
        backend: str,
        query: str,
        source_platform: str,
        limit: int,
    ) -> List[RawSearchResult]:
        if self._external_client is not None:
            return await self._dispatch(self._external_client, backend, query, source_platform, limit)

        async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
            return await self._dispatch(client, backend, query, source_platform, limit)

    async def _dispatch(
        self,
        client: httpx.AsyncClient,
        backend: str,
        query: str,
        source_platform: str,
        limit: int,
    ) -> List[RawSearchResult]:
        if backend == "serper":
            return await self._search_serper(client, query, source_platform, limit)
        if backend == "brave":
            return await self._search_brave(client, query, source_platform, limit)
        if backend == "tavily":
            return await self._search_tavily(client, query, source_platform, limit)
        if backend == "google_cse":
            return await self._search_google_cse(client, query, source_platform, limit)
        raise SearchBackendError(f"Unknown search backend: {backend}")

    async def _search_serper(
        self,
        client: httpx.AsyncClient,
        query: str,
        source_platform: str,
        limit: int,
    ) -> List[RawSearchResult]:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.settings.serper_api_key or "",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"q": query, "num": min(limit, 20)}
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("organic") or []
        return [
            RawSearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source_platform=source_platform,
                metadata={"position": item.get("position"), "backend": "serper"},
            )
            for item in items
            if item.get("link")
        ][:limit]

    async def _search_brave(
        self,
        client: httpx.AsyncClient,
        query: str,
        source_platform: str,
        limit: int,
    ) -> List[RawSearchResult]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.settings.brave_search_api_key or "",
            "Accept": "application/json",
        }
        params = {"q": query, "count": min(limit, 20)}
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("web", {}).get("results") or []
        return [
            RawSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source_platform=source_platform,
                metadata={"backend": "brave"},
            )
            for item in items
            if item.get("url")
        ][:limit]

    async def _search_tavily(
        self,
        client: httpx.AsyncClient,
        query: str,
        source_platform: str,
        limit: int,
    ) -> List[RawSearchResult]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": min(limit, 20),
            "include_answer": False,
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("results") or []
        return [
            RawSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source_platform=source_platform,
                metadata={"score": item.get("score"), "backend": "tavily"},
            )
            for item in items
            if item.get("url")
        ][:limit]

    async def _search_google_cse(
        self,
        client: httpx.AsyncClient,
        query: str,
        source_platform: str,
        limit: int,
    ) -> List[RawSearchResult]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.settings.google_cse_api_key,
            "cx": self.settings.google_cse_cx,
            "q": query,
            "num": min(limit, 10),
        }
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        return [
            RawSearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                display_link=item.get("displayLink"),
                source_platform=source_platform,
                metadata={"backend": "google_cse"},
            )
            for item in items
            if item.get("link")
        ][:limit]
