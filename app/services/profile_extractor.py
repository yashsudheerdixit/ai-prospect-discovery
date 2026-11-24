import asyncio
import logging
import re
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.schemas import CandidateProfile, QueryIntent, RawSearchResult, SocialLink
from app.utils.normalizers import (
    normalize_handle,
    normalize_name,
    normalize_url,
    normalize_whatsapp_link,
    normalize_whitespace,
)
from app.utils.validators import (
    infer_platform_from_url,
    is_valid_platform_profile_url,
    validate_and_normalize_email,
    validate_and_normalize_phone,
)

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}")
WHATSAPP_URL_PATTERN = re.compile(
    r"https?://(?:wa\.me/\d+|api\.whatsapp\.com/send\?[^\s\"'<>]+|chat\.whatsapp\.com/[A-Za-z0-9]+)",
    re.IGNORECASE,
)
HANDLE_IN_TEXT_PATTERN = re.compile(r"@([A-Za-z0-9._]{2,30})")

# Never fetch login-walled social media HTML directly; rely on search/official APIs for those.
AUTH_WALLED_HOSTS = {
    "linkedin.com",
    "www.linkedin.com",
    "instagram.com",
    "www.instagram.com",
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "x.com",
    "twitter.com",
    "threads.net",
    "www.threads.net",
    "threads.com",
    "www.threads.com",
    "youtube.com",
    "www.youtube.com",
}

COUNTRY_DEFAULT_REGION = {
    "usa": "US",
    "united states": "US",
    "uk": "GB",
    "united kingdom": "GB",
    "india": "IN",
    "uae": "AE",
    "united arab emirates": "AE",
}


class ProfileExtractor:
    """Extracts structured candidate profiles and public contact details from search results."""

    def __init__(self, settings: Settings, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.settings = settings
        self._external_client = http_client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def extract_batch(
        self,
        raw_results: List[RawSearchResult],
        intent: QueryIntent,
    ) -> List[CandidateProfile]:
        """Extract CandidateProfile objects concurrently from raw search results."""
        tasks = [self.extract_one(item, intent) for item in raw_results]
        profiles = await asyncio.gather(*tasks, return_exceptions=True)
        extracted: List[CandidateProfile] = []
        for res in profiles:
            if isinstance(res, CandidateProfile):
                extracted.append(res)
            elif isinstance(res, Exception):
                logger.debug("Profile extraction error skipped: %s", res)
        return extracted

    async def extract_one(
        self,
        item: RawSearchResult,
        intent: QueryIntent,
    ) -> Optional[CandidateProfile]:
        canonical_url = normalize_url(item.url)
        platform = infer_platform_from_url(canonical_url) or item.source_platform
        if platform == "Website" and item.source_platform == "Google":
            platform = "Website"

        combined_text = normalize_whitespace(f"{item.title} {item.snippet}")
        name = self._extract_name(item.title, canonical_url, platform)
        if not name:
            return None

        default_region = self._infer_phone_region(intent)
        emails: Set[str] = set()
        phones: Set[str] = set()
        whatsapp_links: Set[str] = set()
        social_links: List[SocialLink] = []
        handles: Dict[str, str] = {}

        # Primary profile link (do not add generic WhatsApp/Email search page URL unless it's wa.me)
        if platform == "WhatsApp":
            wa = normalize_whatsapp_link(canonical_url)
            if wa:
                whatsapp_links.add(wa)
                social_links.append(SocialLink(platform="WhatsApp", link=wa))
        elif platform != "Email":
            if is_valid_platform_profile_url(platform, canonical_url):
                social_links.append(SocialLink(platform=platform, link=canonical_url))
                handle = self._extract_handle_from_url(platform, canonical_url)
                if handle:
                    handles[platform] = handle

        # Extract contacts from search snippet/title
        self._extract_contacts_from_text(
            text=combined_text,
            default_region=default_region,
            emails=emails,
            phones=phones,
            whatsapp_links=whatsapp_links,
        )

        # Extract @handles from title/snippet
        for h_match in HANDLE_IN_TEXT_PATTERN.findall(combined_text):
            norm_h = normalize_handle(h_match)
            if norm_h and "." not in norm_h[-4:]:
                handles.setdefault("mentioned", norm_h)

        # Optionally enrich from safe public personal/business websites (non-auth-walled)
        extra_bio = item.snippet or ""
        if self.settings.enable_page_enrichment and self._should_fetch_public_page(canonical_url):
            page_data = await self._fetch_public_page_contacts(canonical_url, default_region)
            if page_data:
                if page_data["title"] and (len(name) < 3 or name.lower() in {"home", "contact", "about"}):
                    cand_name = self._extract_name(page_data["title"], canonical_url, platform)
                    if cand_name:
                        name = cand_name
                if page_data["description"]:
                    extra_bio = normalize_whitespace(f"{extra_bio} {page_data['description']}")
                emails.update(page_data["emails"])
                phones.update(page_data["phones"])
                whatsapp_links.update(page_data["whatsapp_links"])
                for link_obj in page_data["social_links"]:
                    social_links.append(link_obj)
                    h = self._extract_handle_from_url(link_obj.platform, link_obj.link)
                    if h:
                        handles[link_obj.platform] = h

        # Add verified contact channels to social_links
        for email in sorted(emails):
            social_links.append(SocialLink(platform="Email", link=f"mailto:{email}"))
        for phone in sorted(phones):
            social_links.append(SocialLink(platform="Phone", link=f"tel:{phone}"))
        for wa in sorted(whatsapp_links):
            social_links.append(SocialLink(platform="WhatsApp", link=wa))

        deduped_links = self._dedupe_links(social_links)
        if not deduped_links:
            return None

        inferred_location = self._extract_location_hint(f"{extra_bio} {item.title}", intent)

        return CandidateProfile(
            name=name,
            location=inferred_location,
            bio=extra_bio,
            social_links=deduped_links,
            emails=sorted(emails),
            phones=sorted(phones),
            whatsapp_links=sorted(whatsapp_links),
            handles=handles,
            source_urls=[canonical_url],
            evidence_snippets=[combined_text] if combined_text else [],
        )

    def _should_fetch_public_page(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return not any(host == blocked or host.endswith(f".{blocked}") for blocked in AUTH_WALLED_HOSTS)

    async def _fetch_public_page_contacts(
        self,
        url: str,
        default_region: str,
    ) -> Optional[dict]:
        """Fetch a safe public non-social web page and parse contact & outbound social links."""
        async with self._semaphore:
            try:
                if self._external_client is not None:
                    resp = await self._external_client.get(url, follow_redirects=True)
                else:
                    async with httpx.AsyncClient(
                        timeout=min(self.settings.http_timeout_seconds, 8.0),
                        follow_redirects=True,
                        headers={"User-Agent": "ProspectDiscoveryBot/1.0 (+public-metadata)"},
                    ) as client:
                        resp = await client.get(url)
                if resp.status_code >= 400:
                    return None
                content_type = resp.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    return None
                html = resp.text[:400_000]
            except Exception:
                return None

        soup = BeautifulSoup(html, "html.parser")
        title = normalize_whitespace(soup.title.string) if soup.title and soup.title.string else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = normalize_whitespace(str(meta_tag["content"]))

        emails: Set[str] = set()
        phones: Set[str] = set()
        whatsapp_links: Set[str] = set()
        social_links: List[SocialLink] = []

        for anchor in soup.find_all("a", href=True):
            href = str(anchor["href"]).strip()
            low = href.lower()
            if low.startswith("mailto:"):
                em = validate_and_normalize_email(href.split("?", 1)[0])
                if em:
                    emails.add(em)
            elif low.startswith("tel:"):
                ph = validate_and_normalize_phone(href, default_region=default_region)
                if ph:
                    phones.add(ph)
            elif "wa.me/" in low or "api.whatsapp.com" in low:
                wa = normalize_whatsapp_link(href)
                if wa:
                    whatsapp_links.add(wa)
            elif href.startswith(("http://", "https://", "/")):
                abs_url = normalize_url(urljoin(url, href))
                plat = infer_platform_from_url(abs_url)
                if plat in {"LinkedIn", "X", "YouTube", "Instagram", "Facebook", "Threads"}:
                    if is_valid_platform_profile_url(plat, abs_url):
                        social_links.append(SocialLink(platform=plat, link=abs_url))

        visible_text = normalize_whitespace(soup.get_text(" ", strip=True))[:8000]
        self._extract_contacts_from_text(
            text=visible_text,
            default_region=default_region,
            emails=emails,
            phones=phones,
            whatsapp_links=whatsapp_links,
        )

        # Filter out navbar/portal link dumps (e.g. university athletics pages linking 15 teams)
        links_per_platform: Dict[str, List[SocialLink]] = {}
        for s_link in social_links:
            links_per_platform.setdefault(s_link.platform, []).append(s_link)
        filtered_social_links: List[SocialLink] = []
        for plat_name, items in links_per_platform.items():
            unique_links = list({i.link: i for i in items}.values())
            if len(unique_links) <= 2:
                filtered_social_links.extend(unique_links)

        return {
            "title": title,
            "description": meta_desc or visible_text[:400],
            "emails": emails,
            "phones": phones,
            "whatsapp_links": whatsapp_links,
            "social_links": filtered_social_links,
        }

    @staticmethod
    def _extract_contacts_from_text(
        text: str,
        default_region: str,
        emails: Set[str],
        phones: Set[str],
        whatsapp_links: Set[str],
    ) -> None:
        for raw_email in EMAIL_PATTERN.findall(text):
            em = validate_and_normalize_email(raw_email)
            if em:
                emails.add(em)

        for raw_wa in WHATSAPP_URL_PATTERN.findall(text):
            wa = normalize_whatsapp_link(raw_wa)
            if wa:
                whatsapp_links.add(wa)

        for raw_phone in PHONE_PATTERN.findall(text):
            ph = validate_and_normalize_phone(raw_phone, default_region=default_region)
            if ph:
                phones.add(ph)

    @staticmethod
    def _extract_name(title: str, url: str, platform: str) -> str:
        cleaned = normalize_name(title)
        # Split common profile title separators like "John Smith - Football Coach | LinkedIn"
        if platform == "LinkedIn" and " - " in cleaned:
            cleaned = cleaned.split(" - ", 1)[0].strip()
        elif " | " in cleaned:
            cleaned = cleaned.split(" | ", 1)[0].strip()
        elif " — " in cleaned:
            cleaned = cleaned.split(" — ", 1)[0].strip()

        # Clean sentence-like or truncated titles (e.g. post text) by falling back to handle
        if len(cleaned.split()) > 6 or cleaned.endswith("..."):
            handle = ProfileExtractor._extract_handle_from_url(platform, url)
            if handle:
                return handle.replace("_", " ").replace(".", " ").title()

        if cleaned and len(cleaned) >= 2 and cleaned.lower() not in {"instagram", "facebook", "linkedin", "youtube", "x", "threads"}:
            return cleaned

        handle = ProfileExtractor._extract_handle_from_url(platform, url)
        if handle:
            return handle.replace("_", " ").replace(".", " ").title()

        host = urlparse(url).netloc.removeprefix("www.")
        return host.split(".")[0].replace("-", " ").title() if host else ""

    @staticmethod
    def _extract_handle_from_url(platform: str, url: str) -> Optional[str]:
        segments = [s for s in urlparse(url).path.strip("/").split("/") if s]
        if not segments:
            return None
        if platform in {"X", "Instagram", "Threads"}:
            return normalize_handle(segments[0])
        if platform == "YouTube" and segments[0].startswith("@"):
            return normalize_handle(segments[0])
        if platform == "LinkedIn" and len(segments) >= 2:
            return normalize_handle(segments[1])
        if platform == "Facebook":
            return normalize_handle(segments[0])
        return None

    @staticmethod
    def _infer_phone_region(intent: QueryIntent) -> str:
        country = (intent.location.country or "").strip().lower()
        return COUNTRY_DEFAULT_REGION.get(country, "US")

    @staticmethod
    def _extract_location_hint(text: str, intent: QueryIntent) -> Optional[str]:
        low = (text or "").lower()
        city = intent.location.city
        state = intent.location.state
        country = intent.location.country

        city_matched = bool(
            city
            and (
                city.lower() in low
                or ("washington" in city.lower() and ("washington dc" in low or "washington, dc" in low or " d.c." in low or " dc " in low))
            )
        )
        state_matched = bool(state and state.lower() in low and (not city or state.lower() != city.lower()))

        # If a specific city was requested, do NOT assign location from country alone
        if city and not (city_matched or state_matched):
            return None

        matched_parts: List[str] = []
        if city_matched and city:
            matched_parts.append(city)
        if state_matched and state:
            matched_parts.append(state)
        if country and (
            country.lower() in low
            or (country.upper() in {"USA", "US"} and ("united states" in low or " usa" in low))
            or (country.upper() == "UK" and ("united kingdom" in low or " uk" in low))
            or (country.upper() == "UAE" and ("united arab emirates" in low or " uae" in low))
        ):
            matched_parts.append(country)

        return ", ".join(matched_parts) if matched_parts else None

    @staticmethod
    def _dedupe_links(links: List[SocialLink]) -> List[SocialLink]:
        seen: Set[tuple[str, str]] = set()
        result: List[SocialLink] = []
        for link in links:
            key = (link.platform, link.link)
            if key not in seen:
                seen.add(key)
                result.append(link)
        return result
