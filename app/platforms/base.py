from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

from app.config import Settings
from app.schemas import QueryIntent, RawSearchResult
from app.services.search_service import SearchService
from app.utils.normalizers import normalize_url
from app.utils.validators import is_valid_platform_profile_url


class PlatformDiscoveryResult(BaseModel):
    """Container for per-platform discovery results and error tracking."""

    platform: str
    results: List[RawSearchResult] = Field(default_factory=list)
    error: Optional[str] = None


class BasePlatformDiscoverer(ABC):
    """Abstract base class for independent, extensible platform discoverers."""

    platform_name: str = ""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def build_search_query(self, intent: QueryIntent) -> str:
        """Construct a platform-specific search query from structured intent."""

    async def discover(
        self,
        intent: QueryIntent,
        search_service: SearchService,
    ) -> PlatformDiscoveryResult:
        """Execute discovery for this platform and filter to valid profile URLs."""
        try:
            query = self.build_search_query(intent)
            raw_results = await search_service.search(
                query=query,
                source_platform=self.platform_name,
                num_results=self.settings.max_results_per_platform,
            )
            filtered: List[RawSearchResult] = []
            seen_urls: set[str] = set()
            for item in raw_results:
                canonical = self.normalize_candidate_url(item.url)
                if not canonical or canonical in seen_urls:
                    continue
                if self.is_valid_result(canonical, item):
                    seen_urls.add(canonical)
                    filtered.append(item.model_copy(update={"url": canonical}))
            return PlatformDiscoveryResult(platform=self.platform_name, results=filtered)
        except Exception as exc:
            return PlatformDiscoveryResult(
                platform=self.platform_name,
                results=[],
                error=str(exc),
            )

    def normalize_candidate_url(self, url: str) -> str:
        """Normalize URL prior to validation."""
        return normalize_url(url)

    def is_valid_result(self, canonical_url: str, item: RawSearchResult) -> bool:
        """Validate whether the URL is an authentic public profile/target for this platform."""
        return is_valid_platform_profile_url(self.platform_name, canonical_url)

    @staticmethod
    def _base_terms(intent: QueryIntent, platform_key: str) -> str:
        custom = intent.platform_queries.get(platform_key)
        if custom:
            return custom
        loc = " ".join(p for p in (intent.location.city, intent.location.country) if p)
        return f"{intent.audience} {loc}".strip()
