from urllib.parse import urlparse

from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent
from app.utils.normalizers import normalize_url


class XPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public X (formerly Twitter) user profiles via authorized search."""

    platform_name = "X"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "X")
        return f"(site:x.com OR site:twitter.com) -inurl:status -inurl:hashtag -inurl:search {terms}"

    def normalize_candidate_url(self, url: str) -> str:
        canonical = normalize_url(url)
        parsed = urlparse(canonical)
        segments = [s for s in parsed.path.strip("/").split("/") if s]
        # If search returned a specific tweet/post URL (x.com/handle/status/123),
        # collapse cleanly to the verified author profile URL x.com/handle.
        if len(segments) >= 3 and segments[1] == "status":
            return f"https://x.com/{segments[0]}"
        return canonical
