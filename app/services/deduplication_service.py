from difflib import SequenceMatcher
from typing import List, Set

from app.schemas import CandidateProfile, SocialLink
from app.utils.normalizers import (
    canonical_name_key,
    normalize_handle,
    normalize_location_token,
    normalize_whitespace,
)

PLATFORM_ORDER = {
    "Facebook": 1,
    "Instagram": 2,
    "LinkedIn": 3,
    "X": 4,
    "YouTube": 5,
    "Threads": 6,
    "Google": 7,
    "Website": 8,
    "Email": 9,
    "Phone": 10,
    "WhatsApp": 11,
}


class DeduplicationService:
    """Consolidates duplicate profiles belonging to the same person or business."""

    def consolidate(self, candidates: List[CandidateProfile]) -> List[CandidateProfile]:
        """Group and merge candidate profiles using multi-signal entity resolution."""
        groups: List[CandidateProfile] = []
        for candidate in candidates:
            merged = False
            for idx, existing in enumerate(groups):
                if self._should_merge(existing, candidate):
                    groups[idx] = self._merge_profiles(existing, candidate)
                    merged = True
                    break
            if not merged:
                groups.append(candidate.model_copy(deep=True))
        return groups

    def _should_merge(self, a: CandidateProfile, b: CandidateProfile) -> bool:
        """Determine if two candidates represent the same real-world entity."""
        # 1. Hard conflict guard: incompatible explicit locations prevent accidental namesake merge
        #    unless linked by verified shared email/phone/URL.
        has_shared_email = bool(set(a.emails) & set(b.emails))
        has_shared_phone = bool(set(a.phones) & set(b.phones))
        has_shared_wa = bool(set(a.whatsapp_links) & set(b.whatsapp_links))
        urls_a = {link.link for link in a.social_links}
        urls_b = {link.link for link in b.social_links}
        has_shared_url = bool(urls_a & urls_b)

        if has_shared_email or has_shared_phone or has_shared_wa or has_shared_url:
            return True

        # Reject merge if both have explicit, conflicting locations
        if self._locations_conflict(a.location, b.location):
            return False

        # 2. Cross-platform handle match + compatible name
        handles_a = {normalize_handle(h) for h in a.handles.values() if h}
        handles_b = {normalize_handle(h) for h in b.handles.values() if h}
        shared_handles = {h for h in (handles_a & handles_b) if len(h) >= 4}

        name_sim = self._name_similarity(a.name, b.name)
        if shared_handles and name_sim >= 0.55:
            return True

        # 3. High name similarity + compatible non-empty location + token overlap in bio/domain
        if name_sim >= 0.90:
            loc_a = normalize_location_token(a.location or "").lower()
            loc_b = normalize_location_token(b.location or "").lower()
            same_location = bool(loc_a and loc_b and (loc_a in loc_b or loc_b in loc_a))
            bio_overlap = self._bio_token_overlap(a.bio, b.bio)
            if same_location and (bio_overlap >= 0.15 or self._handles_compatible(a, b)):
                return True

        return False

    @staticmethod
    def _name_similarity(name_a: str, name_b: str) -> float:
        key_a = canonical_name_key(name_a)
        key_b = canonical_name_key(name_b)
        if not key_a or not key_b:
            return 0.0
        if key_a == key_b:
            return 1.0
        return SequenceMatcher(None, key_a, key_b).ratio()

    @staticmethod
    def _locations_conflict(loc_a: str | None, loc_b: str | None) -> bool:
        if not loc_a or not loc_b:
            return False
        norm_a = normalize_location_token(loc_a).lower()
        norm_b = normalize_location_token(loc_b).lower()
        tokens_a = {t.strip() for t in norm_a.replace(",", " ").split() if len(t.strip()) > 2}
        tokens_b = {t.strip() for t in norm_b.replace(",", " ").split() if len(t.strip()) > 2}
        if not tokens_a or not tokens_b:
            return False
        return not bool(tokens_a & tokens_b)

    @staticmethod
    def _handles_compatible(a: CandidateProfile, b: CandidateProfile) -> bool:
        """If both have handles on the exact same platform, ensure they don't contradict."""
        for platform, handle_a in a.handles.items():
            if platform in b.handles and normalize_handle(handle_a) != normalize_handle(b.handles[platform]):
                return False
        return True

    @staticmethod
    def _bio_token_overlap(bio_a: str, bio_b: str) -> float:
        stopwords = {"the", "and", "for", "with", "from", "this", "that", "official", "profile"}
        tokens_a = {
            w for w in normalize_whitespace(bio_a).lower().split() if len(w) > 3 and w not in stopwords
        }
        tokens_b = {
            w for w in normalize_whitespace(bio_b).lower().split() if len(w) > 3 and w not in stopwords
        }
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))

    @staticmethod
    def _merge_profiles(primary: CandidateProfile, secondary: CandidateProfile) -> CandidateProfile:
        # Prefer the richer human/business display name
        chosen_name = primary.name if len(primary.name) >= len(secondary.name) else secondary.name
        chosen_location = primary.location or secondary.location
        if primary.location and secondary.location and len(secondary.location) > len(primary.location):
            chosen_location = secondary.location

        seen_links: Set[tuple[str, str]] = set()
        merged_links: List[SocialLink] = []
        for link in primary.social_links + secondary.social_links:
            key = (link.platform, link.link)
            if key not in seen_links:
                seen_links.add(key)
                merged_links.append(link)

        merged_links.sort(key=lambda item: (PLATFORM_ORDER.get(item.platform, 99), item.link))

        merged_handles = dict(primary.handles)
        merged_handles.update(secondary.handles)

        return CandidateProfile(
            name=chosen_name,
            location=chosen_location,
            bio=normalize_whitespace(f"{primary.bio} {secondary.bio}")[:1000],
            social_links=merged_links,
            emails=sorted(set(primary.emails) | set(secondary.emails)),
            phones=sorted(set(primary.phones) | set(secondary.phones)),
            whatsapp_links=sorted(set(primary.whatsapp_links) | set(secondary.whatsapp_links)),
            handles=merged_handles,
            source_urls=list(dict.fromkeys(primary.source_urls + secondary.source_urls)),
            evidence_snippets=list(dict.fromkeys(primary.evidence_snippets + secondary.evidence_snippets))[:5],
            entity_type=primary.entity_type if primary.entity_type != "unknown" else secondary.entity_type,
        )
