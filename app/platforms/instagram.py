from app.platforms.base import BasePlatformDiscoverer
from app.schemas import QueryIntent


class InstagramPlatformDiscoverer(BasePlatformDiscoverer):
    """Discovers public Instagram profiles via authorized search."""

    platform_name = "Instagram"

    def build_search_query(self, intent: QueryIntent) -> str:
        terms = self._base_terms(intent, "Instagram")
        return f"site:instagram.com -inurl:/p/ -inurl:/reel/ -inurl:/explore/ {terms}"
