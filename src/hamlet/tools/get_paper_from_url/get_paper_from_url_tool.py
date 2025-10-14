from langchain.docstore.document import Document
import requests
from urllib.parse import quote as _urlquote
import fitz  # PyMuPDF
from requests.exceptions import HTTPError
import re
from src.hamlet.core.tools import Tool
import os
import tempfile
import shutil
import hashlib
from typing import Optional, List
from bs4 import BeautifulSoup  # HTML text extraction fallback
try:  # For type checking safety with BeautifulSoup elements
    from bs4 import Tag  # type: ignore
except Exception:  # pragma: no cover - fallback if bs4 changes
    Tag = object  # type: ignore
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
    """Fetch a PDF from a URL and extract text.

    Adds robust per-page error handling so that a single corrupt / non-standard
    page (e.g. MuPDF color space parsing issue) does not abort extraction of
    the remaining pages. Optionally returns the metadata title.

    Env flags:
      HAMLET_PDF_DEBUG=1     -> verbose logging of skipped pages / fallbacks
      HAMLET_PDF_PAGE_MODE   -> comma list of extraction methods to try per page
                                 (default: "text,raw,blocks")
    """
    debug = _env_truthy("HAMLET_PDF_DEBUG", False)
    try:
        session = _get_requests_session()
        response = session.get(url, timeout=(10, 60))  # (connect, read) timeouts
        response.raise_for_status()
        try:
            doc = fitz.open(stream=response.content, filetype="pdf")
        except Exception as e:
            if debug:
                print(f"[PDFDebug] Failed to open PDF {url}: {e}")
            return ("", None) if return_title else ""
        try:
            page_modes_env = os.environ.get("HAMLET_PDF_PAGE_MODE", "text,raw,blocks")
            page_modes = [m.strip() for m in page_modes_env.split(',') if m.strip()]
            collected: List[str] = []
            skipped_pages: List[tuple[int, str]] = []
            for page_index in range(len(doc)):
                try:
                    page = doc.load_page(page_index)  # may itself fail
                except Exception as e:
                    skipped_pages.append((page_index, f"load_page error: {e}"))
                    continue
                page_text = ""
                last_err: Optional[str] = None
                for mode in page_modes:
                    try:
                        # Common modes: "text" (layout-aware), "raw", "blocks"
                        page_text = page.get_text(mode)  # type: ignore[arg-type]
                        if page_text:
                            break
                    except Exception as e_mode:
                        last_err = str(e_mode)
                        continue
                if not page_text:
                    # As an extreme fallback, try extracting individual spans (if accessible)
                    try:  # pragma: no cover - rare path
                        blocks = page.get_text("dict").get("blocks", [])  # type: ignore
                        spans_accum: List[str] = []
                        for b in blocks:
                            for l in b.get("lines", []):
                                for s in l.get("spans", []):
                                    txt = s.get("text")
                                    if txt:
                                        spans_accum.append(txt)
                        page_text = "\n".join(spans_accum)
                    except Exception as e_span:
                        last_err = last_err or str(e_span)
                if page_text:
                    collected.append(page_text)
                else:
                    skipped_pages.append((page_index, last_err or "unknown extraction failure"))
            if debug and skipped_pages:
                print(f"[PDFDebug] Skipped {len(skipped_pages)} page(s) in {url}:")
                for idx, reason in skipped_pages[:10]:  # cap log spam
                    print(f"  - page {idx}: {reason}")
                if len(skipped_pages) > 10:
                    print(f"  ... {len(skipped_pages)-10} more skipped page(s) ...")
            full_text = "\n".join(collected)
            if return_title:
                try:
                    meta_title = (doc.metadata or {}).get("title")
                except Exception:
                    meta_title = None
                finally:
                    try:
                        doc.close()
                    except Exception:
                        pass
                return full_text, meta_title
            else:
                try:
                    doc.close()
                except Exception:
                    pass
                return full_text
        finally:
            try:  # ensure doc is closed if not already
                if not doc.is_closed:  # type: ignore[attr-defined]
                    doc.close()
            except Exception:
                pass
    except HTTPError as http_err:
        status = getattr(getattr(http_err, 'response', None), 'status_code', 'unknown')
        if debug:
            print(f"[PDFDebug] HTTP error fetching {url}: {http_err} (status={status})")
        return ("", None) if return_title else ""
    except Exception as err:
        if debug:
            print(f"[PDFDebug] Other error for {url}: {err}")
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
    debug = _env_truthy("HAMLET_GET_PAPER_DEBUG", False)

    def _discover_pdf_and_extract(parent_url: str, html: str) -> Optional[Document]:
        """Attempt to locate a PDF link inside an HTML page (meta tags or anchors) and extract it.

        Strategies:
          1. <meta name="citation_pdf_url" content="..."> (common on ACM / Springer / etc.)
          2. Anchor tags where href endswith .pdf or contains '/pdf/' or 'download' with pdf inside.
        Returns a Document if successful, else None.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None
        # Meta tag
        meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        candidates: List[str] = []
        if isinstance(meta, Tag):
            content_val = meta.get("content")  # type: ignore[index]
            if isinstance(content_val, str) and content_val.strip():
                candidates.append(content_val.strip())
        # Anchor tags
        for a in soup.find_all("a"):  # type: ignore
            if not isinstance(a, Tag):
                continue
            # Use getattr to satisfy static analysis; BeautifulSoup Tag supports .get at runtime
            href_val = getattr(a, 'get', lambda *_: None)("href")  # type: ignore[call-arg]
            if not isinstance(href_val, str):
                continue
            href = href_val.strip()
            if not href:
                continue
            href_lower = href.lower()
            if any(x in href_lower for x in [".pdf", "/pdf/", "download"]) and (".pdf" in href_lower or href_lower.endswith("pdf")):
                candidates.append(href)
        # Normalize / dedupe
        normed: List[str] = []
        seen = set()
        for c in candidates:
            if c.startswith("//"):
                # protocol-relative
                c = ("https:" if parent_url.lower().startswith("https") else "http:") + c
            elif c.startswith("/"):
                # relative to domain
                try:
                    from urllib.parse import urlparse, urljoin
                    c = urljoin(parent_url, c)
                except Exception:
                    pass
            if c not in seen:
                seen.add(c)
                normed.append(c)
        for pdf_url in normed[:3]:  # Cap attempts to avoid long chains
            try:
                if debug:
                    print(f"[GetPaperFromURL][DiscoverPDF] Trying embedded PDF: {pdf_url}")
                text, meta_title = extract_text_from_pdf_url(pdf_url, return_title=True)
                if text.strip():
                    text = remove_irrelevant_sections(text)
                    meta = {"source": pdf_url}
                    if meta_title:
                        meta["title"] = meta_title
                    return Document(page_content=text, metadata=meta)
            except Exception as e:  # pragma: no cover - network / parse variability
                if debug:
                    print(f"[GetPaperFromURL][DiscoverPDF] Failed {pdf_url}: {e}")
                continue
        return None
    for url in urls:
        try:
            resp = session.get(url, timeout=(10, 60))
            if resp.status_code >= 400:
                if debug:
                    print(f"[GetPaperFromURL] HTTP {resp.status_code} for {url}; skipping")
                continue
            html = resp.text or ""
            text = _extract_text_from_html(html) if html else ""
            # If no text or extremely short (< 200 chars), attempt PDF discovery
            doc_obj: Optional[Document] = None
            if (not text) or len(text) < 200:
                if debug:
                    print(f"[GetPaperFromURL] Weak HTML extraction for {url} (len={len(text)}); attempting embedded PDF discovery")
                doc_obj = _discover_pdf_and_extract(url, html)
            if doc_obj is None and text:
                doc_obj = Document(page_content=text, metadata={"source": url})
            if doc_obj:
                docs.append(doc_obj)
            else:
                if debug:
                    print(f"[GetPaperFromURL] No usable content derived from {url}")
        except Exception as e:
            if debug:
                print(f"[GetPaperFromURL] Exception processing {url}: {e}")
            continue

    # Fallback: If no docs were captured for ResearchGate URLs, optionally try OpenAlex API
    if not docs:
        rg_candidates = [u for u in urls if 'researchgate.net/publication/' in u.lower()]
        if rg_candidates and _env_truthy("HAMLET_ENABLE_OPENALEX_FALLBACK", True):
            if debug:
                print(f"[GetPaperFromURL][OpenAlex] Attempting OpenAlex fallback for {len(rg_candidates)} ResearchGate URL(s)")
            for rg_url in rg_candidates:
                # Derive a search query from the tail slug
                try:
                    slug = rg_url.rstrip('/').split('/')[-1]
                    # Remove leading numeric id part if present (pattern: digits_)
                    parts = slug.split('_')
                    if parts and parts[0].isdigit():
                        parts = parts[1:]
                    query = ' '.join(p for p in parts if p).strip()
                    # Guard length
                    if len(query) < 8:
                        if debug:
                            print(f"[GetPaperFromURL][OpenAlex] Query '{query}' too short; skipping")
                        continue
                    oa_url = f"https://api.openalex.org/works?search={_urlquote(query)}&per-page=3"
                    if debug:
                        print(f"[GetPaperFromURL][OpenAlex] Fetching {oa_url}")
                    try:
                        r = session.get(oa_url, timeout=(10, 30))
                        if r.status_code != 200:
                            if debug:
                                print(f"[GetPaperFromURL][OpenAlex] HTTP {r.status_code} for query '{query}'")
                            continue
                        data = r.json()
                    except Exception as e_json:
                        if debug:
                            print(f"[GetPaperFromURL][OpenAlex] Failed querying OpenAlex: {e_json}")
                        continue
                    results = (data or {}).get('results') or []
                    for work in results:
                        try:
                            # Prefer open access PDF link
                            pdf_url = None
                            open_access = work.get('open_access') if isinstance(work, dict) else None
                            if isinstance(open_access, dict):
                                pdf_url = open_access.get('oa_url') or open_access.get('pdf_url')
                            # Fallback to primary location
                            if not pdf_url and isinstance(work, dict):
                                loc = work.get('primary_location') or {}
                                if isinstance(loc, dict):
                                    pdf_url = (loc.get('source') or {}).get('host_organization_page_url')  # type: ignore
                            if not pdf_url:
                                continue
                            if debug:
                                print(f"[GetPaperFromURL][OpenAlex] Attempt PDF {pdf_url}")
                            text, meta_title = extract_text_from_pdf_url(pdf_url, return_title=True)
                            if not text.strip():
                                continue
                            text = remove_irrelevant_sections(text)
                            meta = {"source": pdf_url}
                            if meta_title:
                                meta['title'] = meta_title
                            docs.append(Document(page_content=text, metadata=meta))
                            # Stop after first successful fallback per ResearchGate URL
                            break
                        except Exception as e_work:  # pragma: no cover - external service variability
                            if debug:
                                print(f"[GetPaperFromURL][OpenAlex] Work processing error: {e_work}")
                            continue
                except Exception as e_rg:
                    if debug:
                        print(f"[GetPaperFromURL][OpenAlex] Error handling slug for {rg_url}: {e_rg}")
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
    # Core hardening / container friendly flags
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")  # harmless in headless; avoids some env warnings
    options.add_argument("--disable-extensions")
    options.add_argument(f"--user-agent={user_agent}")
    if quiet_logs:
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])  # hides many USB/device logs
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")

    # Ensure each run gets a unique Chrome profile dir (can be disabled)
    supplied_profile = os.environ.get("HAMLET_CHROME_USER_DATA_DIR")
    disable_profile = _env_truthy("HAMLET_CHROME_NO_PROFILE", False)
    temp_profile_dir: Optional[str] = None
    try:
        if not disable_profile:
            if supplied_profile:
                os.makedirs(supplied_profile, exist_ok=True)
                options.add_argument(f"--user-data-dir={supplied_profile}")
            else:
                temp_profile_dir = tempfile.mkdtemp(prefix="hamlet_chrome_profile_")
                options.add_argument(f"--user-data-dir={temp_profile_dir}")
            if _env_truthy("HAMLET_SELENIUM_DEBUG", False):
                which = supplied_profile if supplied_profile else temp_profile_dir
                print(f"[Selenium] Using user data dir: {which}")
    except Exception as e:
        if _env_truthy("HAMLET_SELENIUM_DEBUG", False):
            print(f"[Selenium] Failed setting profile dir: {e}; continuing without user-data-dir")
        temp_profile_dir = None

    # Build service; prefer user-specified driver
    try:
        if chromedriver_path:
            service = _ChromeService(executable_path=chromedriver_path)
        else:
            service = _ChromeService()
    except Exception:
        service = None  # type: ignore[assignment]

    from selenium.common.exceptions import SessionNotCreatedException, TimeoutException  # type: ignore
    driver = None
    tried_without_profile = False
    try:
        try:
            driver = _webdriver.Chrome(options=options, service=service) if service else _webdriver.Chrome(options=options)
        except SessionNotCreatedException as e:
            if _env_truthy("HAMLET_SELENIUM_DEBUG", False):
                print(f"[Selenium] SessionNotCreatedException: {e}")
            # Retry once without profile dir if we haven't already and profiles enabled
            if not tried_without_profile and not disable_profile:
                tried_without_profile = True
                # Rebuild options without any user-data-dir
                new_options = _ChromeOptions()
                for arg in options.arguments:
                    if not str(arg).startswith('--user-data-dir'):
                        new_options.add_argument(arg)
                options = new_options
                temp_profile_dir = None
                supplied_profile = None
                try:
                    driver = _webdriver.Chrome(options=options, service=service) if service else _webdriver.Chrome(options=options)
                except Exception:
                    if _env_truthy("HAMLET_SELENIUM_DEBUG", False):
                        print("[Selenium] Retry without profile failed; abandoning Selenium path.")
                    return []
            else:
                return []
        # Set conservative timeouts so we don't hang indefinitely in CI / headless servers
        try:
            if driver is not None:
                driver.set_page_load_timeout(int(os.environ.get("HAMLET_SELENIUM_PAGELOAD_TIMEOUT", "25")))
        except Exception:
            pass

        docs: List[Document] = []
        for url in urls:
            try:
                # Guard each navigation with a timeout; if it times out we still attempt to collect partial DOM.
                try:
                    driver.get(url)
                except TimeoutException:
                    if _env_truthy("HAMLET_SELENIUM_DEBUG", False):
                        print(f"[Selenium] Timeout loading {url}; continuing with partial content")
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
        try:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass
        finally:
            # Remove temp profile dir only if we created it (not user supplied)
            if 'temp_profile_dir' in locals() and temp_profile_dir and not supplied_profile:
                try:
                    shutil.rmtree(temp_profile_dir, ignore_errors=True)
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
        # Track filenames this instance has produced to avoid collisions
        self._seen_filenames: set[str] = set()

    # ---------------- Filename / slug helpers -----------------
    def _make_filename(self, title: str, source: Optional[str]) -> str:
        """Create a filesystem-safe, reasonably short, collision-resistant markdown filename.

        Strategy:
        1. Normalize title -> slug (alnum + underscores) lowercased.
        2. Collapse multiple underscores; trim leading/trailing underscores.
        3. If empty, fall back to 'document'.
        4. Truncate to configurable max (HAMLET_MAX_FILENAME_LEN, default 120) BEFORE adding hash.
        5. Append short hash (first 8 hex of sha1(title + source)).
        6. Guarantee total filename length (bytes) <= 240 (leave margin for various filesystems) and < 255 hard limit.
        7. Deduplicate within the instance by adding incremental suffix if needed.
        8. Always end with .md
        """
        raw = title if title else (source or "document")
        # Basic slug
        slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).lower()
        slug = re.sub(r"_+", "_", slug).strip("_")
        if not slug:
            slug = "document"

        # Configurable max length (characters) before hash
        try:
            max_len_env = int(os.environ.get("HAMLET_MAX_FILENAME_LEN", "120"))
            # clamp to sane bounds
            if max_len_env < 20:
                max_len_env = 20
            if max_len_env > 180:
                max_len_env = 180
        except Exception:
            max_len_env = 120

        # Short hash for disambiguation
        h = hashlib.sha1((title + "|" + str(source)).encode("utf-8", errors="ignore")).hexdigest()[:8]
        base = slug[:max_len_env]
        candidate = f"{base}-{h}.md"

        # Enforce byte-length safety (<255). Use 240 as soft cap.
        def _shrink(name: str) -> str:
            b = name.encode("utf-8", errors="ignore")
            if len(b) <= 240:
                return name
            # shrink base further
            keep = max(15, max_len_env // 2)
            new_base = slug[:keep]
            return f"{new_base}-{h}.md"

        candidate = _shrink(candidate)

        # Deduplicate if already seen in this run
        if candidate in self._seen_filenames:
            idx = 1
            stem, ext = candidate.rsplit(".", 1)
            while True:
                alt = f"{stem}_{idx}.{ext}"
                if alt not in self._seen_filenames and len(alt.encode('utf-8')) < 250:
                    candidate = alt
                    break
                idx += 1
        self._seen_filenames.add(candidate)
        return candidate

    def forward(self, urls: list) -> str:  # type: ignore[override]
        if not urls:
            return "No URLs provided."
        try:
            # Optional truncation for debugging / batching large URL lists
            try:
                max_urls_env = os.environ.get("HAMLET_MAX_URLS")
                if max_urls_env:
                    max_urls = int(max_urls_env)
                    urls = urls[:max_urls]
            except Exception:
                pass

            progress = _env_truthy("HAMLET_GET_PAPER_PROGRESS", False)
            if progress:
                print(f"[GetPaperFromURL] Starting extraction for {len(urls)} URL(s)")
            docs = extract_docs_from_urls(urls)
            if progress:
                print(f"[GetPaperFromURL] Fetched {len(docs)} document(s); beginning summarization")
            if not docs:
                return "No valid documents found at the provided URLs."

            summary_lines: List[str] = []
            for idx, doc in enumerate(docs, 1):
                if progress:
                    print(f"[GetPaperFromURL] Processing doc {idx}/{len(docs)}")
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


                filename = self._make_filename(title, meta.get('source'))
                safe_filename = self._safe_path(filename)
                # ensure the parent folder exists
                os.makedirs(os.path.dirname(safe_filename), exist_ok=True)
                # print(safe_filename)
                # print("=============")
                
                # create 
                # Compose simple Markdown: H1 title, source link (if any), then content
                md_lines = [f"# {title}"]
                source = meta.get('source') or meta.get('url')
                if source:
                    md_lines.append("")
                    md_lines.append(f"Source: {source}")
                md_lines.append("")
                md_lines.append(doc.page_content)
                try:
                    with open(safe_filename, 'w', encoding='utf-8') as f:
                        f.write("\n".join(md_lines))
                except OSError as oe:
                    # Retry with ultra-short fallback on filename too long or similar issues
                    if getattr(oe, 'errno', None) == 36 or 'File name too long' in str(oe):
                        short_hash = hashlib.sha1(title.encode('utf-8', errors='ignore')).hexdigest()[:10]
                        fallback = self._safe_path(f"doc-{short_hash}.md")
                        try:
                            with open(fallback, 'w', encoding='utf-8') as f:
                                f.write("\n".join(md_lines))
                            filename = os.path.basename(fallback)
                            safe_filename = fallback
                        except Exception as oe2:
                            return f"Error occurred: failed to write fallback filename as well: {oe2}"
                    else:
                        return f"Error occurred: {oe}"

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
                # Length sanity: configurable trim
                try:
                    max_abs_env = int(os.environ.get("HAMLET_MAX_ABSTRACT_CHARS", "1500"))
                    if max_abs_env < 200:
                        max_abs_env = 200  # guard against extreme low values
                    if max_abs_env > 5000:
                        max_abs_env = 5000  # hard ceiling
                except Exception:
                    max_abs_env = 1500
                original_len = len(abstract)
                if len(abstract) > max_abs_env:
                    abstract = abstract[:max_abs_env].rstrip() + " …"

                # Optional second (tighter) limit just for the returned summary string
                try:
                    summary_abs_limit = int(os.environ.get("HAMLET_SUMMARY_ABSTRACT_CHARS", str(max_abs_env)))
                    if summary_abs_limit < 100:
                        summary_abs_limit = 100
                    if summary_abs_limit > max_abs_env:
                        # Don't silently allow it to exceed the main extracted limit
                        summary_abs_limit = max_abs_env
                except Exception:
                    summary_abs_limit = max_abs_env
                if len(abstract) > summary_abs_limit:
                    abstract = abstract[:summary_abs_limit].rstrip() + " …"

                if _env_truthy("HAMLET_ABSTRACT_DEBUG", False):
                    print(f"[AbstractDebug] title_hash={hash(title)} original_len={original_len} final_len={len(abstract)} max_abs_env={max_abs_env} summary_abs_limit={summary_abs_limit}")

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
