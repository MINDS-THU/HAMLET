from langchain.docstore.document import Document
import requests
import fitz  # PyMuPDF
from requests.exceptions import HTTPError
import re
from smolagents import Tool
import os
from typing import Optional, List
from bs4 import BeautifulSoup  # HTML text extraction fallback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional Selenium imports (not required if unavailable)
try:  # pragma: no cover - optional path
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    _SELENIUM_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
    _SELENIUM_AVAILABLE = False
    webdriver = None  # type: ignore[assignment]
    ChromeOptions = None  # type: ignore[assignment]
    ChromeService = None  # type: ignore[assignment]


def remove_irrelevant_sections(text: str) -> str:
    """Cut content at first occurrence of terminal sections (references, appendix, etc.)."""
    heading_max_chars = 120
    stop_headings = (
        "references",
        "reference",
        "bibliography",
        "acknowledgment",
        "acknowledgement",
        "appendix",
        "supplementary material",
        "supplementary materials",
        "supplementary",
    )
    pattern = re.compile(
        rf"^\s*(\d+\s*[\.\-–])?\s*(?:{'|'.join(stop_headings)})\b.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    cut_pos: Optional[int] = None
    for m in pattern.finditer(text):
        line = m.group(0)
        if len(line) <= heading_max_chars:
            cut_pos = m.start()
            break
    return text[:cut_pos].rstrip() if cut_pos is not None else text


# --- Abstract extraction helpers ---
_ABSTRACT_INLINE_RE = re.compile(r"^\s*abstract\s*[:\.]?\s*(.+)$", re.IGNORECASE)
_ABSTRACT_HEADING_ONLY_RE = re.compile(r"^\s*abstract\s*[:\.]?\s*$", re.IGNORECASE)

_SECTION_HEADINGS_RE = re.compile(
    r"^(?:\d+\s*[\.-–])?\s*(?:"
    r"introduction|background|related\s+work|methods?|materials|results|discussion|conclusion|conclusions|"
    r"acknowledg?ments?|references|bibliography|appendix|supplementary|keywords?|index\s+terms|contents|"
    r"authors?|affiliations?|citation|subjects?|comments?|submission\s+history)\b",
    re.IGNORECASE,
)


def _is_section_heading(line: str) -> bool:
    return bool(_SECTION_HEADINGS_RE.match(line.strip()))


def _clean_join(lines: List[str]) -> str:
    """Join lines into a paragraph, fixing common hyphenation and whitespace."""
    text = " ".join(l.strip() for l in lines if l is not None)
    text = re.sub(r"(\w)-(\s+)(\w)", r"\1\3", text)  # fix split words like "ap- plication"
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_abstract_from_text(text: str) -> Optional[str]:
    lines = text.splitlines()
    n = len(lines)
    # Pass 1: inline abstract
    for i, raw in enumerate(lines):
        m = _ABSTRACT_INLINE_RE.match(raw)
        if m:
            content = [m.group(1).strip()] if m.group(1).strip() else []
            for j in range(i + 1, n):
                l = lines[j].strip()
                if not l:
                    if content:
                        break
                    else:
                        continue
                if _is_section_heading(l):
                    break
                content.append(l)
            return _clean_join(content) if content else None
    # Pass 2: heading-only then following paragraph(s)
    for i, raw in enumerate(lines):
        if _ABSTRACT_HEADING_ONLY_RE.match(raw):
            content: List[str] = []
            for j in range(i + 1, n):
                l = lines[j].strip()
                if not l:
                    if content:
                        break
                    else:
                        continue
                if _is_section_heading(l):
                    break
                content.append(l)
            if content:
                return _clean_join(content)
    return None


# --- Title extraction helpers ---
_TITLE_NOISE_RE = re.compile(
    r"^(arXiv:|submitted\s+on|preprint|doi\b|copyright|university\b|figure\s+\d+|proceedings\b)",
    re.IGNORECASE,
)

_TRAILING_SITE_SUFFIXES = (
    " - arXiv",
    " | arXiv",
    " - arXiv.org",
    " - PMC",
    " | PMC",
    " - PubMed Central",
    " - ScienceDirect",
    " | ScienceDirect",
    " - SpringerLink",
    " | SpringerLink",
    " - ACM Digital Library",
    " | ACM Digital Library",
)


def _clean_title_suffix(title: str) -> str:
    t = title.strip()
    for suf in _TRAILING_SITE_SUFFIXES:
        if t.endswith(suf):
            t = t[: -len(suf)].rstrip()
    return re.sub(r"\s{2,}", " ", t)


def _alpha_ratio(s: str) -> float:
    letters = sum(c.isalpha() for c in s)
    return letters / max(1, len(s))


def extract_title_from_content(text: str) -> Optional[str]:
    """Heuristically pick a plausible title line before Abstract or within first ~60 lines."""
    lines = [ln.strip() for ln in text.splitlines()[:200]]
    abs_idx = None
    for i, ln in enumerate(lines):
        if _ABSTRACT_HEADING_ONLY_RE.match(ln) or _ABSTRACT_INLINE_RE.match(ln):
            abs_idx = i
            break
    search_until = abs_idx if abs_idx is not None else min(len(lines), 60)
    candidates: List[str] = []
    for ln in lines[:search_until]:
        if not ln:
            continue
        if _TITLE_NOISE_RE.match(ln):
            continue
        if 8 <= len(ln) <= 180 and _alpha_ratio(ln) >= 0.5:
            candidates.append(ln)
    if candidates:
        best = max(candidates, key=len)
        return _clean_title_suffix(best)
    return None


def extract_text_from_pdf_url(url: str, return_title: bool = False):
    try:
        session = _get_requests_session()
        response = session.get(url, timeout=(10, 60))  # connect, read timeouts
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text("text") for page in doc)  # type: ignore[attr-defined]
            if return_title:
                try:
                    meta_title = (doc.metadata or {}).get("title")
                except Exception:
                    meta_title = None
                return text, meta_title
            return text
    except HTTPError as http_err:
        status = getattr(getattr(http_err, 'response', None), 'status_code', 'unknown')
        print(f"HTTP error occurred: {http_err} (Status code: {status})")
        return ("", None) if return_title else ""
    except Exception as err:
        print(f"Other error occurred: {err}")
        return ("", None) if return_title else ""


def _get_requests_session() -> requests.Session:
    """Create a requests session with retry/backoff and proxy support via env vars.

    Requests uses HTTP(S)_PROXY env vars automatically; we mainly add retries and timeouts.
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Optional default headers (user agent helps with some CDNs)
    ua = os.environ.get(
        "HAMLET_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36",
    )
    session.headers.update({
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def _env_truthy(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _extract_text_from_html(html: str) -> str:
    """Extract visible text from HTML with basic cleanup."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\n\s*)+", "\n", text)
    return text.strip()


def _load_web_docs_via_requests(urls: List[str]) -> List[Document]:
    session = _get_requests_session()
    docs: List[Document] = []
    for url in urls:
        try:
            resp = session.get(url, timeout=(10, 60))
            if resp.status_code >= 400:
                continue
            html = resp.text
            text = _extract_text_from_html(html) if html else ""
            if text:
                docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception:
            # Silently skip; Selenium path may handle it
            continue
    return docs


def _load_web_docs_via_selenium(urls: List[str]) -> List[Document]:  # pragma: no cover - I/O heavy
    try:
        from selenium import webdriver as _webdriver
        from selenium.webdriver.chrome.options import Options as _ChromeOptions
        from selenium.webdriver.chrome.service import Service as _ChromeService
    except Exception:
        return []

    headless = _env_truthy("HAMLET_SELENIUM_HEADLESS", True)
    proxy = os.environ.get("HAMLET_SELENIUM_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    chromedriver_path = os.environ.get("HAMLET_CHROMEDRIVER_PATH")
    user_agent = os.environ.get(
        "HAMLET_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36",
    )
    quiet_logs = _env_truthy("HAMLET_SELENIUM_QUIET", True)

    options = _ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-agent={user_agent}")
    if quiet_logs:
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])  # hides many USB/device logs
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")

    # Build service; prefer user-specified driver
    try:
        if chromedriver_path:
            service = _ChromeService(executable_path=chromedriver_path)
        else:
            service = _ChromeService()
    except Exception:
        service = None  # type: ignore[assignment]

    driver = None
    try:
        driver = _webdriver.Chrome(options=options, service=service) if service else _webdriver.Chrome(options=options)
        docs: List[Document] = []
        for url in urls:
            try:
                driver.get(url)
                try:
                    driver.implicitly_wait(2)
                except Exception:
                    pass
                title = (driver.title or "").strip()
                page_source = driver.page_source or ""
                text = _extract_text_from_html(page_source) if page_source else ""
                meta = {"source": url}
                if title:
                    meta["title"] = title
                if text:
                    docs.append(Document(page_content=text, metadata=meta))
            except Exception:
                continue
        return docs
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def extract_docs_from_urls(urls: List[str]) -> List[Document]:
    """Extract documents from URLs and convert to Document objects.

    - PDFs: fetched via requests with retry/timeout, parsed by PyMuPDF.
    - HTML: try Selenium (if available) with headless, proxy, and quiet-logs options; otherwise fall back to requests+BS4.
    """
    pdf_links = [link for link in urls if 'pdf' in link.lower()]
    web_links = [link for link in urls if 'pdf' not in link.lower()]

    docs: List[Document] = []

    # Process web pages
    if web_links:
        web_docs: List[Document] = []
        use_selenium = _env_truthy("HAMLET_USE_SELENIUM", True)
        if use_selenium and _SELENIUM_AVAILABLE:
            web_docs = _load_web_docs_via_selenium(web_links)
        if not web_docs:
            # Fallback to requests-only path
            web_docs = _load_web_docs_via_requests(web_links)
        docs.extend(web_docs)

    # Process PDFs (capture metadata title when available)
    for link in pdf_links:
        text, meta_title = extract_text_from_pdf_url(link, return_title=True)
        text = remove_irrelevant_sections(text)
        if text:
            meta = {"source": link}
            if meta_title and meta_title.strip():
                meta["title"] = meta_title.strip()
            docs.append(Document(page_content=text, metadata=meta))

    return docs


class GetPaperFromURL(Tool):
    name = "get_paper_from_url"
    description = (
        "Fetch research papers from a list of URLs, extract their content, save each paper as a Markdown (.md) file, and return a summary string for each document including its title, abstract, and the saved filename."
    )
    inputs = {
        "urls": {
            "type": "any",
            "description": "List of paper URLs to fetch. Example: ['https://arxiv.org/abs/2401.12345','https://openreview.net/pdf?id=abc123','https://proceedings.mlr.press/v123/paper.pdf']",
        }
    }
    output_type = "string"

    def __init__(self, working_dir: str):
        super().__init__()
        self.working_dir = working_dir

    def forward(self, urls: list) -> str:  # type: ignore[override]
        if not urls:
            return "No URLs provided."
        try:
            docs = extract_docs_from_urls(urls)
            if not docs:
                return "No valid documents found at the provided URLs."

            summary_lines: List[str] = []
            for doc in docs:
                meta = doc.metadata or {}
                title = (meta.get('title') or '').strip()
                # Heuristic cleanup/guess when title is missing or noisy
                noisy = (not title) or title.lower().startswith('arxiv:') or title.lower() in {
                    'sciencedirect', 'researchgate - temporarily unavailable'
                }
                if noisy:
                    guessed = extract_title_from_content(doc.page_content)
                    if guessed:
                        title = guessed
                if not title:
                    # Final fallback: first non-empty line
                    for line in doc.page_content.splitlines():
                        if line.strip():
                            title = line.strip()
                            break
                    if not title:
                        title = 'Untitled Document'
                # Clean common site suffixes
                title = _clean_title_suffix(title)

                filename = re.sub(r"\W+", "_", title).lower() + ".md"
                safe_filename = self._safe_path(filename)
                # Compose simple Markdown: H1 title, source link (if any), then content
                md_lines = [f"# {title}"]
                source = meta.get('source') or meta.get('url')
                if source:
                    md_lines.append("")
                    md_lines.append(f"Source: {source}")
                md_lines.append("")
                md_lines.append(doc.page_content)
                with open(safe_filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(md_lines))

                # Extract abstract using robust parser
                abstract = extract_abstract_from_text(doc.page_content)
                if not abstract:
                    # Fallback: take the first ~250 words before 'Introduction'
                    pre_intro = re.split(r"\n\s*(?:\d+\s*[\.-–])?\s*Introduction\b", doc.page_content, flags=re.IGNORECASE)[0]
                    words = re.findall(r"\S+", pre_intro)
                    abstract = " ".join(words[:250]).strip() if words else '(No abstract found)'
                else:
                    # If the abstract is extremely short, append a bit more context after it
                    if len(abstract) < 120:
                        tail = re.split(r"\n\s*(?:\d+\s*[\.-–])?\s*Introduction\b", doc.page_content, flags=re.IGNORECASE)[0]
                        extra_words = re.findall(r"\S+", tail)
                        if extra_words:
                            abstract = (abstract + " " + " ".join(extra_words[:120])).strip()
                # Length sanity: trim overly long abstracts
                if len(abstract) > 3000:
                    abstract = abstract[:3000].rstrip() + " …"

                summary_lines.append(
                    f"Title: {title}\nAbstract: {abstract}\nSaved as: {os.path.basename(safe_filename)}\n"
                )

            return "\n---\n".join(summary_lines)
        except Exception as e:
            return f"Error occurred: {e}"

    def _safe_path(self, path: str) -> str:
        # Prevent absolute paths and directory traversal
        abs_working_dir = os.path.abspath(self.working_dir)
        abs_path = os.path.abspath(os.path.join(self.working_dir, path))
        if not abs_path.startswith(abs_working_dir):
            raise PermissionError("Access outside the working directory is not allowed.")
        return abs_path


if __name__ == "__main__":
    # Test GetPaperFromURL tool with example URLs
    urls = [
        "https://zenodo.org/records/15042478/files/PhD-Thesis-LukasJohannesBreitwieser.pdf?download=1",
    ]
    working_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(working_dir, exist_ok=True)
    tool = GetPaperFromURL(working_dir)
    result = tool.forward(urls)
    print("Summary of fetched documents:\n")
    print(result)
