"""Extensible platform registry for multi-platform prospect discovery."""

from typing import List, Type

from app.config import Settings
from app.platforms.base import BasePlatformDiscoverer, PlatformDiscoveryResult
from app.platforms.email_platform import EmailPlatformDiscoverer
from app.platforms.facebook import FacebookPlatformDiscoverer
from app.platforms.google import GooglePlatformDiscoverer
from app.platforms.instagram import InstagramPlatformDiscoverer
from app.platforms.linkedin import LinkedInPlatformDiscoverer
from app.platforms.threads import ThreadsPlatformDiscoverer
from app.platforms.whatsapp import WhatsAppPlatformDiscoverer
from app.platforms.x_platform import XPlatformDiscoverer
from app.platforms.youtube import YouTubePlatformDiscoverer

PLATFORM_CLASSES: List[Type[BasePlatformDiscoverer]] = [
    GooglePlatformDiscoverer,
    LinkedInPlatformDiscoverer,
    XPlatformDiscoverer,
    YouTubePlatformDiscoverer,
    InstagramPlatformDiscoverer,
    FacebookPlatformDiscoverer,
    ThreadsPlatformDiscoverer,
    WhatsAppPlatformDiscoverer,
    EmailPlatformDiscoverer,
]


def get_registered_platforms(settings: Settings) -> List[BasePlatformDiscoverer]:
    """Instantiate all registered platform discoverers."""
    return [cls(settings) for cls in PLATFORM_CLASSES]


__all__ = [
    "BasePlatformDiscoverer",
    "PlatformDiscoveryResult",
    "PLATFORM_CLASSES",
    "get_registered_platforms",
]
