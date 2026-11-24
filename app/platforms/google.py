from urllib.parse import urlparse

from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent, RawSearchResult
from app.utils.validators import is_valid_platform_profile_url


GENERIC_LISTICLE_TITLES = (
    "top 10",
    "top 20",
    "best ",
    "wikipedia",
    "near me",
    "directory",
    "jobs in",
    "salary",
    "news",
    "roster",
    "schedule",
    "standings",
)

EXCLUDED_URL_SEGMENTS = (
    "/roster",
    "/schedule",
    "/stats",
    "/standings",
    "/scores",
    "/news/",
    "/article/",
    "/articles/",
    "/jobs/",
    "/careers/",
    "/wiki/",
)


class GooglePlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers publicly indexed individual/business websites and public profiles via web search."""

    platform_name = "Google"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "Google")
        if intent.target_type == "business":
            return f"{terms} official website contact"
        return f"{terms} community OR club OR creator OR blog contact"

    def is_valid_result(self, canonical_url: str, item: RawSearchResult) -> bool:
        if not is_valid_platform_profile_url("Website", canonical_url):
            host = urlparse(canonical_url).netloc.lower()
            if not ("sites.google.com" in host or host == "g.page" or "maps.google." in host):
                return False

        path_low = (urlparse(canonical_url).path or "").lower()
        if any(seg in path_low for seg in EXCLUDED_URL_SEGMENTS):
            return False

        title_low = (item.title or "").lower()
        if any(marker in title_low for marker in GENERIC_LISTICLE_TITLES):
            return False
        return True
