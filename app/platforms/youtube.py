import logging
from typing import List, Optional

import httpx

from app.platforms.base import BasePlatformDiscoverer, PlatformDiscoveryResult
from app.schemas import QueryIntent, RawSearchResult
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class YouTubePlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public YouTube channels using official YouTube Data API v3 with search fallback."""

    platform_name = "YouTube"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "YouTube")
        return f"site:youtube.com (inurl:/@ OR inurl:/channel/ OR inurl:/c/) {terms}"

    async def discover(
        self,
        intent: QueryIntent,
        search_service: SearchService,
    ) -> PlatformDiscoveryResult:
        if self.settings.youtube_api_key:
            try:
                api_results = await self._discover_via_youtube_api(intent)
                if api_results:
                    return PlatformDiscoveryResult(
                        platform=self.platform_name,
                        results=api_results,
                    )
            except Exception as exc:
                logger.warning("YouTube Data API v3 failed; falling back to web search: %s", exc)

        return await super().discover(intent, search_service)

    async def _discover_via_youtube_api(self, intent: QueryIntent) -> List[RawSearchResult]:
        query = self._base_terms(intent, "YouTube")
        limit = min(self.settings.max_results_per_platform, 10)
        async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
            search_resp = await client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "type": "channel",
                    "q": query,
                    "maxResults": limit,
                    "key": self.settings.youtube_api_key,
                },
            )
            search_resp.raise_for_status()
            items = search_resp.json().get("items") or []
            channel_ids = [
                item.get("snippet", {}).get("channelId") or item.get("id", {}).get("channelId")
                for item in items
            ]
            channel_ids = [cid for cid in channel_ids if cid]
            if not channel_ids:
                return []

            channels_resp = await client.get(
                "https://www.googleapis.com/youtube/v3/channels",
                params={
                    "part": "snippet,brandingSettings",
                    "id": ",".join(channel_ids),
                    "key": self.settings.youtube_api_key,
                },
            )
            channels_resp.raise_for_status()
            channels = channels_resp.json().get("items") or []

            results: List[RawSearchResult] = []
            for ch in channels:
                snippet = ch.get("snippet") or {}
                cid = ch.get("id", "")
                custom_url: Optional[str] = snippet.get("customUrl")
                if custom_url:
                    handle = custom_url if custom_url.startswith("@") else f"@{custom_url}"
                    url = f"https://www.youtube.com/{handle}"
                else:
                    url = f"https://www.youtube.com/channel/{cid}"

                country = snippet.get("country", "")
                desc = snippet.get("description", "")
                results.append(
                    RawSearchResult(
                        title=snippet.get("title", ""),
                        url=url,
                        snippet=desc,
                        source_platform=self.platform_name,
                        metadata={
                            "backend": "youtube_data_api_v3",
                            "country": country,
                            "custom_url": custom_url,
                        },
                    )
                )
            return results
