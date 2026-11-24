from urllib.parse import urlparse

from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent, RawSearchResult
from app.utils.normalizers import normalize_url
from app.utils.validators import is_valid_platform_profile_url


class ThreadsPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public Threads profiles via authorized search."""

    platform_name = "Threads"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "Threads")
        return f"(site:threads.net/@ OR site:threads.com/@) -inurl:/post/ {terms}"

    def normalize_candidate_url(self, url: str) -> str:
        return normalize_url(url)

    def is_valid_result(self, canonical_url: str, item: RawSearchResult) -> bool:
        # Reject individual Threads posts so only genuine indexed profile pages are kept
        segments = [s for s in urlparse(canonical_url).path.strip("/").split("/") if s]
        if len(segments) != 1 or not segments[0].startswith("@"):
            return False
        return is_valid_platform_profile_url(self.platform_name, canonical_url)
