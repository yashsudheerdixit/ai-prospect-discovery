from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent


class FacebookPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public Facebook profiles and pages via authorized search."""

    platform_name = "Facebook"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "Facebook")
        return f"site:facebook.com -inurl:/posts/ -inurl:/watch/ -inurl:/groups/ {terms}"
