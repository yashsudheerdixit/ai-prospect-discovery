from urllib.parse import urlparse

from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent
from app.utils.normalizers import normalize_url


class ThreadsPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public Threads profiles via authorized search."""

    platform_name = "Threads"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "Threads")
        return f"(site:threads.net/@ OR site:threads.com/@) -inurl:/post/ {terms}"

    def normalize_candidate_url(self, url: str) -> str:
        canonical = normalize_url(url)
        parsed = urlparse(canonical)
        segments = [s for s in parsed.path.strip("/").split("/") if s]
        # Collapse post URLs like threads.net/@user/post/xyz to profile URL threads.net/@user
        if segments and segments[0].startswith("@") and len(segments) >= 2 and segments[1] == "post":
            return f"https://{parsed.netloc}/{segments[0]}"
        return canonical
