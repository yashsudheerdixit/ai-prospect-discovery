from urllib.parse import urlparse

from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent, RawSearchResult
from app.utils.normalizers import normalize_url
from app.utils.validators import is_valid_platform_profile_url


class XPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public X (formerly Twitter) user profiles via authorized search."""

    platform_name = "X"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "X")
        return f"(site:x.com OR site:twitter.com) -inurl:status -inurl:hashtag -inurl:search {terms}"

    def normalize_candidate_url(self, url: str) -> str:
        return normalize_url(url)

    def is_valid_result(self, canonical_url: str, item: RawSearchResult) -> bool:
        # Reject individual tweet/status pages so only genuine indexed profiles are kept
        segments = [s for s in urlparse(canonical_url).path.strip("/").split("/") if s]
        if len(segments) != 1:
            return False
        return is_valid_platform_profile_url(self.platform_name, canonical_url)
