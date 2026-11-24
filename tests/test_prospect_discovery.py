import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_prospect_service
from app.platforms.base import BasePlatformDiscoverer, PlatformDiscoveryResult
from app.schemas import CandidateProfile, QueryIntent, RawSearchResult, SocialLink
from app.services.deduplication_service import DeduplicationService
from app.services.llm_service import LLMService
from app.services.prospect_service import ProspectService
from app.services.query_service import QueryService
from app.services.search_service import SearchService
from app.utils.normalizers import normalize_url, normalize_whatsapp_link
from app.utils.validators import (
    is_valid_platform_profile_url,
    validate_and_normalize_email,
    validate_and_normalize_phone,
)


def make_settings(**overrides) -> Settings:
    defaults = {
        "enable_page_enrichment": False,
        "verify_profile_urls": False,
        "cache_ttl_seconds": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# --- 1. Validator & Normalizer Tests ---


def test_url_validation_and_normalization() -> None:
    assert normalize_url("https://twitter.com/dcfootballfan/?utm_source=x") == "https://x.com/dcfootballfan"
    assert is_valid_platform_profile_url("LinkedIn", "https://www.linkedin.com/in/john-smith-dc")
    assert not is_valid_platform_profile_url("LinkedIn", "https://www.linkedin.com/jobs/view/123")
    assert is_valid_platform_profile_url("Instagram", "https://www.instagram.com/dc_football_club")
    assert not is_valid_platform_profile_url("Instagram", "https://www.instagram.com/p/C8xYz123")
    assert is_valid_platform_profile_url("YouTube", "https://www.youtube.com/@DCFootballTV")
    assert is_valid_platform_profile_url("Threads", "https://www.threads.net/@dcfootball")
    assert not is_valid_platform_profile_url("Google", "https://www.google.com/search?q=football")
    assert normalize_whatsapp_link("https://api.whatsapp.com/send?phone=+1-202-555-0147") == "https://wa.me/12025550147"


def test_contact_validation() -> None:
    assert validate_and_normalize_email("mailto:John.Doe@dcfootball.org") == "john.doe@dcfootball.org"
    assert validate_and_normalize_email("noreply@linkedin.com") is None
    assert validate_and_normalize_email("avatar@2x.png") is None
    assert validate_and_normalize_email("user@example.com") is None

    assert validate_and_normalize_phone("+1 (202) 555-0199", default_region="US") == "+12025550199"
    assert validate_and_normalize_phone("12345", default_region="US") is None


# --- 2. Query Understanding Tests (All 5 Test Cases) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected_city,expected_country,expected_type",
    [
        ("Football lovers in Washington DC in USA", "Washington DC", "USA", "individual"),
        ("Digital marketing agency owners in New York USA", "New York", "USA", "business"),
        ("Startup founders in London UK", "London", "UK", "individual"),
        ("Cricket lovers in Ahmedabad India", "Ahmedabad", "India", "individual"),
        ("Restaurant owners in Dubai UAE", "Dubai", "UAE", "business"),
    ],
)
async def test_heuristic_query_understanding_test_cases(
    query: str,
    expected_city: str,
    expected_country: str,
    expected_type: str,
) -> None:
    settings = make_settings()
    service = QueryService(settings, LLMService(settings))
    intent = await service.understand_query(query)

    assert intent.location.city == expected_city
    assert intent.location.country == expected_country
    assert intent.target_type == expected_type
    assert intent.interest
    assert intent.platform_queries


# --- 3. Deduplication & Consolidation Tests ---


def test_deduplication_merges_same_person_and_separates_namesakes() -> None:
    dedup = DeduplicationService()

    c1 = CandidateProfile(
        name="John Smith",
        location="Washington DC, USA",
        bio="Football coach and Washington Commanders fan",
        social_links=[SocialLink(platform="LinkedIn", link="https://www.linkedin.com/in/johnsmithdc")],
        handles={"LinkedIn": "johnsmithdc"},
        emails=["john@dcfootball.org"],
    )
    c2 = CandidateProfile(
        name="John Smith",
        location="Washington DC, USA",
        bio="DC Football community & matchday vlogs",
        social_links=[SocialLink(platform="Instagram", link="https://www.instagram.com/johnsmithdc")],
        handles={"Instagram": "johnsmithdc"},
    )
    # Different person with the exact same name in London UK -> must NOT merge!
    c3 = CandidateProfile(
        name="John Smith",
        location="London, UK",
        bio="Premier League football blogger in London",
        social_links=[SocialLink(platform="X", link="https://x.com/johnsmith_london")],
        handles={"X": "johnsmith_london"},
    )

    consolidated = dedup.consolidate([c1, c2, c3])
    assert len(consolidated) == 2

    dc_john = next(p for p in consolidated if "Washington DC" in (p.location or ""))
    london_john = next(p for p in consolidated if "London" in (p.location or ""))

    assert {s.platform for s in dc_john.social_links} == {"LinkedIn", "Instagram"}
    assert [s.platform for s in london_john.social_links] == ["X"]


# --- 4. End-to-End API Tests with Stubbed Platforms ---


class StubSuccessPlatform(BasePlatformDiscoverer):
    def __init__(self, settings: Settings, name: str, items: list[RawSearchResult]) -> None:
        super().__init__(settings)
        self.platform_name = name
        self._items = items

    def build_search_query(self, intent: QueryIntent) -> str:
        return intent.original_query

    async def discover(self, intent: QueryIntent, search_service: SearchService) -> PlatformDiscoveryResult:
        return PlatformDiscoveryResult(platform=self.platform_name, results=self._items)


class StubFailingPlatform(BasePlatformDiscoverer):
    platform_name = "Instagram"

    def build_search_query(self, intent: QueryIntent) -> str:
        return intent.original_query

    async def discover(self, intent: QueryIntent, search_service: SearchService) -> PlatformDiscoveryResult:
        return PlatformDiscoveryResult(platform=self.platform_name, results=[], error="Simulated 429 rate limit")


def test_root_and_health_endpoints() -> None:
    client = TestClient(app)
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Prospect Discovery" in root_resp.text

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"


def test_post_prospect_discovery_validation_errors() -> None:
    client = TestClient(app)
    resp_empty = client.post("/prospect-discovery", json={"query": "   "})
    assert resp_empty.status_code == 422
    assert resp_empty.json()["success"] is False

    resp_punct = client.post("/prospect-discovery", json={"query": "???"})
    assert resp_punct.status_code == 422
    assert resp_punct.json()["success"] is False


def test_post_prospect_discovery_end_to_end_and_resilience() -> None:
    settings = make_settings()

    platforms = [
        StubSuccessPlatform(
            settings,
            "LinkedIn",
            [
                RawSearchResult(
                    title="Alex Rivera - Football Community Organizer | LinkedIn",
                    url="https://www.linkedin.com/in/alexriveradc",
                    snippet="Passionate football lover in Washington DC, USA. Contact alex@dcfootball.org",
                    source_platform="LinkedIn",
                ),
                RawSearchResult(
                    title="Top 10 Best Football Bars in Washington DC",
                    url="https://www.linkedin.com/in/top10bars",
                    snippet="Directory listicle of bars",
                    source_platform="LinkedIn",
                ),
            ],
        ),
        StubSuccessPlatform(
            settings,
            "X",
            [
                RawSearchResult(
                    title="Alex Rivera (@alexriveradc) / X",
                    url="https://x.com/alexriveradc",
                    snippet="Washington DC football enthusiast. Reach me on https://wa.me/12025550142",
                    source_platform="X",
                )
            ],
        ),
        StubFailingPlatform(settings),
    ]

    stub_service = ProspectService(settings=settings, platforms=platforms)
    app.dependency_overrides[get_prospect_service] = lambda: stub_service

    try:
        client = TestClient(app)
        response = client.post(
            "/prospect-discovery",
            json={"query": "Football lovers in Washington DC in USA"},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["success"] is True
        assert body["query"] == "Football lovers in Washington DC in USA"
        assert body["total_prospects"] == 1
        assert len(body["prospects"]) == 1

        prospect = body["prospects"][0]
        assert prospect["name"] == "Alex Rivera"
        assert "Washington DC" in prospect["location"]

        links_by_platform = {item["platform"]: item["link"] for item in prospect["social_link"]}
        assert links_by_platform["LinkedIn"] == "https://www.linkedin.com/in/alexriveradc"
        assert links_by_platform["X"] == "https://x.com/alexriveradc"
        assert links_by_platform["Email"] == "mailto:alex@dcfootball.org"
        assert links_by_platform["WhatsApp"] == "https://wa.me/12025550142"
    finally:
        app.dependency_overrides.clear()
