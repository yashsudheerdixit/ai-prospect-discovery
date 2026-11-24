import asyncio
import logging
from typing import List, Optional

import httpx

from app.config import Settings
from app.platforms import BasePlatformDiscoverer, PlatformDiscoveryResult, get_registered_platforms
from app.schemas import Prospect, ProspectDiscoveryResponse, RawSearchResult
from app.services.deduplication_service import DeduplicationService
from app.services.llm_service import LLMService
from app.services.profile_extractor import ProfileExtractor
from app.services.query_service import QueryService
from app.services.relevance_service import RelevanceService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class ProspectService:
    """Orchestrates the end-to-end multi-platform prospect discovery pipeline."""

    def __init__(
        self,
        settings: Settings,
        llm_service: Optional[LLMService] = None,
        search_service: Optional[SearchService] = None,
        platforms: Optional[List[BasePlatformDiscoverer]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings
        self.llm_service = llm_service or LLMService(settings, http_client=http_client)
        self.search_service = search_service or SearchService(settings, http_client=http_client)
        self.query_service = QueryService(settings, self.llm_service)
        self.platforms = platforms if platforms is not None else get_registered_platforms(settings)
        self.extractor = ProfileExtractor(settings, http_client=http_client)
        self.deduplicator = DeduplicationService()
        self.relevance_service = RelevanceService(settings, self.llm_service)

    async def discover_prospects(self, query: str) -> ProspectDiscoveryResponse:
        """Run full discovery pipeline for a natural-language query."""
        # 1. Natural Language Query Understanding
        intent = await self.query_service.understand_query(query)
        logger.info(
            "Parsed query intent: interest=%r audience=%r location=%r target_type=%r",
            intent.interest,
            intent.audience,
            intent.location.formatted(),
            intent.target_type,
        )

        # 2. Multi-Platform Discovery (isolated per-platform failures)
        platform_results: List[PlatformDiscoveryResult] = await asyncio.gather(
            *(platform.discover(intent, self.search_service) for platform in self.platforms)
        )

        raw_results: List[RawSearchResult] = []
        for plat_res in platform_results:
            if plat_res.error:
                logger.warning("Platform %s failed: %s", plat_res.platform, plat_res.error)
            else:
                logger.info("Platform %s returned %d candidates", plat_res.platform, len(plat_res.results))
                raw_results.extend(plat_res.results)

        if not raw_results:
            return ProspectDiscoveryResponse(
                success=True,
                query=intent.original_query,
                total_prospects=0,
                prospects=[],
            )

        # 3. Profile Extraction & Contact Validation
        candidates = await self.extractor.extract_batch(raw_results, intent)

        # 4. Duplicate Detection & Social Profile Consolidation
        consolidated = self.deduplicator.consolidate(candidates)

        # 5. Relevance Evaluation & Filtering
        verified_prospects = await self.relevance_service.evaluate_and_filter(consolidated, intent)

        # 6. Optional URL reachability verification
        if self.settings.verify_profile_urls and verified_prospects:
            await self._verify_links(verified_prospects)

        prospects_out = [
            Prospect(
                name=cand.name,
                location=cand.location,
                social_link=cand.social_links,
            )
            for cand in verified_prospects
            if cand.social_links
        ]

        return ProspectDiscoveryResponse(
            success=True,
            query=intent.original_query,
            total_prospects=len(prospects_out),
            prospects=prospects_out,
        )

    async def _verify_links(self, candidates: list) -> None:
        """Drop links that return explicit 404/410; tolerate 403/999 anti-bot headers."""
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            for cand in candidates:
                valid_links = []
                for s_link in cand.social_links:
                    if s_link.platform in {"Email", "Phone"}:
                        valid_links.append(s_link)
                        continue
                    try:
                        resp = await client.head(s_link.link)
                        if resp.status_code not in {404, 410}:
                            valid_links.append(s_link)
                    except Exception:
                        valid_links.append(s_link)
                cand.social_links = valid_links
