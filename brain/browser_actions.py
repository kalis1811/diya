"""Explicit, allow-listed browser actions for common Google/YouTube requests."""

import re
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode


@dataclass(frozen=True)
class BrowserCommand:
    label: str
    url: str


_OPEN_SITE = re.compile(
    r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:open|go\s+to|visit)\s+(google|youtube)(?:\.com)?[.!?]*$",
    re.IGNORECASE,
)
_SEARCH_FOR = re.compile(
    r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"search\s+(google|youtube)\s+for\s+(.+?)[.!?]*$",
    re.IGNORECASE,
)
_SEARCH_ON = re.compile(
    r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:search|find)\s+(.+?)\s+on\s+(google|youtube)[.!?]*$",
    re.IGNORECASE,
)


def parse_browser_command(text: str) -> BrowserCommand | None:
    """Recognize only explicit site-open and search requests."""
    text = text.strip()
    match = _OPEN_SITE.fullmatch(text)
    if match:
        site = match.group(1).lower()
        if site == "youtube":
            return BrowserCommand("YouTube", "https://www.youtube.com/")
        return BrowserCommand("Google", "https://www.google.com/")

    match = _SEARCH_FOR.fullmatch(text)
    if match:
        site, terms = match.groups()
    else:
        match = _SEARCH_ON.fullmatch(text)
        if not match:
            return None
        terms, site = match.groups()

    terms = terms.strip().strip("\"'")
    if not terms:
        return None
    site = site.lower()
    if site == "youtube":
        url = "https://www.youtube.com/results?" + urlencode({"search_query": terms})
        return BrowserCommand("YouTube search", url)
    url = "https://www.google.com/search?" + urlencode({"q": terms})
    return BrowserCommand("Google search", url)


def run_browser_command(
    text: str,
    opener: Callable[..., bool] | None = None,
) -> str | None:
    """Open an allow-listed destination in the user's default browser."""
    command = parse_browser_command(text)
    if command is None:
        return None
    opener = opener or webbrowser.open
    try:
        opened = opener(command.url, new=2)
    except Exception:
        return f"I couldn't open {command.label} in the default browser."
    if opened is False:
        return f"I couldn't open {command.label} in the default browser."
    return f"Opened {command.label} in your default browser."
