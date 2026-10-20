from urllib.parse import parse_qs, urlparse

from brain.browser_actions import parse_browser_command, run_browser_command


def test_open_youtube_is_allow_listed():
    command = parse_browser_command("Can you open YouTube?")
    assert command is not None
    assert command.url == "https://www.youtube.com/"


def test_google_search_encodes_query():
    command = parse_browser_command('search google for "best local RAG tools"')
    assert command is not None
    parsed = urlparse(command.url)
    assert parsed.netloc == "www.google.com"
    assert parse_qs(parsed.query) == {"q": ["best local RAG tools"]}


def test_youtube_search_encodes_query():
    command = parse_browser_command("find piano basics on youtube")
    assert command is not None
    parsed = urlparse(command.url)
    assert parsed.netloc == "www.youtube.com"
    assert parse_qs(parsed.query) == {"search_query": ["piano basics"]}


def test_arbitrary_urls_are_not_opened():
    opened = []
    answer = run_browser_command(
        "open https://not-google.example", opener=lambda url, **kwargs: opened.append(url) or True)
    assert answer is None
    assert opened == []


def test_browser_failure_is_reported():
    result = run_browser_command("open youtube", opener=lambda *a, **kw: False)
    assert result == "I couldn't open YouTube in the default browser."
