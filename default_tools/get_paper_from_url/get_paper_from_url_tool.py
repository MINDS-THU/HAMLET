from langchain.docstore.document import Document
from langchain_community.document_loaders import SeleniumURLLoader
import requests
import fitz  # PyMuPDF
from requests.exceptions import HTTPError
import re
from smolagents import Tool
import os
from typing import Optional, List

def remove_irrelevant_sections(text):
    # Lowercased and normalized version of common section headings to remove
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
        "supplementary")
    pattern = re.compile(
        rf"^\s*(\d+\s*[\.\-–])?\s*"
        rf"(?:{'|'.join(stop_headings)})"
        rf"\b.*$",               # heading text to end‑of‑line
        re.IGNORECASE | re.MULTILINE
        )

    # Find the first heading line satisfying length ≤ heading_max_chars
    match_iter = pattern.finditer(text)
    cut_pos = None
    for m in match_iter:
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
    """Join lines into a paragraph, fixing common hyphenation splits like 'ap- plication'."""
    text = " ".join(l.strip() for l in lines if l is not None)
    # Fix hyphenation across line breaks that became '- ' patterns
    text = re.sub(r"(\w)-(\s+)(\w)", r"\1\3", text)
    # Collapse excessive whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_abstract_from_text(text: str) -> Optional[str]:
    lines = text.splitlines()
    n = len(lines)
    # Pass 1: look for inline abstract (same line)
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

    # Pass 2: heading-only 'Abstract' line, then collect following paragraph(s)
    for i, raw in enumerate(lines):
        if _ABSTRACT_HEADING_ONLY_RE.match(raw):
            content: list[str] = []
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
            # continue searching in case this 'Abstract' was decorative
    return None

def extract_docs_from_urls(urls):
    """Extract documents from URLs and convert to Document objects"""
    pdf_links = [link for link in urls if 'pdf' in link]
    web_links = [link for link in urls if 'pdf' not in link]
    
    docs = []
    
    # Process web pages
    if web_links:
        html_docs = SeleniumURLLoader(urls=web_links).load()
        docs.extend(html_docs)
        
    # Process PDFs
    for link in pdf_links:
        text = remove_irrelevant_sections(extract_text_from_pdf_url(link))
        if text:
            docs.append(Document(page_content=text, metadata={"source": link}))
            
    return docs

def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)  # type: ignore[attr-defined]
    except HTTPError as http_err:
        status = getattr(getattr(http_err, 'response', None), 'status_code', 'unknown')
        print(f"HTTP error occurred: {http_err} (Status code: {status})")
        return ""
    except Exception as err:
        print(f"Other error occurred: {err}")
        return ""

class GetPaperFromURL(Tool):
    name = "get_paper_from_url"
    description = (
        "Fetch research papers from a list of URLs, extract their content, save each paper as a text file, and return a summary string for each document including its title, abstract, and the filename used."
    )
    inputs = {
        "urls": {"type": "any", "description": "List of URLs to fetch papers from."}
    }
    output_type = "string"

    def __init__(self, working_dir):
        super().__init__()
        self.working_dir = working_dir

    def forward(self, urls: list) -> str:  # type: ignore[override]
        # use extract_docs_from_urls to get Document objects
        # then save each to a text file in working_dir
        # return a summary string for each situation
        if not urls:
            return "No URLs provided."
        try:
            docs = extract_docs_from_urls(urls)
            if not docs:
                return "No valid documents found at the provided URLs."

            # save each document to a text file
            # in the end, return a summary string of the fetched documents,
            # including their titles, abstract, and safe_filenames
            summary_lines = []
            for doc in docs:
                title = doc.metadata.get('title')
                # Fallback logic for missing title in doc metadata
                if not title or not title.strip():
                    # Try to use first non-empty line of content
                    for line in doc.page_content.splitlines():
                        if line.strip():
                            title = line.strip()
                            break
                    else:
                        title = 'Untitled Document'
                filename = re.sub(r'\W+', '_', title).lower() + '.md'
                safe_filename = self._safe_path(filename)
                # Compose simple Markdown: H1 title, source link (if any), then content
                md_lines = [f"# {title}"]
                source = doc.metadata.get('source') or doc.metadata.get('url')
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
                    # Fallback: take the first 120–200 words before 'Introduction'
                    pre_intro = re.split(r"\n\s*(?:\d+\s*[\.-–])?\s*Introduction\b", doc.page_content, flags=re.IGNORECASE)[0]
                    words = re.findall(r"\S+", pre_intro)
                    abstract = " ".join(words[:180]).strip() if words else '(No abstract found)'
                # Length sanity: trim overly long abstracts
                if len(abstract) > 3000:
                    abstract = abstract[:3000].rstrip() + " …"

                summary_lines.append(f"Title: {title}\nAbstract: {abstract}\nSaved as: {os.path.basename(safe_filename)}\n")

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
        "https://arxiv.org/html/2412.10400v1",
        "https://openreview.net/forum?id=inpkC8UrDu",
        "https://raw.githubusercontent.com/mlresearch/v258/main/assets/zhang25d/zhang25d.pdf",
        "https://arxiv.org/html/2306.04891v2",
    ]
    working_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(working_dir, exist_ok=True)
    tool = GetPaperFromURL(working_dir)
    result = tool.forward(urls)
    print("Summary of fetched documents:\n")
    print(result)
