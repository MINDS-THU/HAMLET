from glob import glob
from anyio import Path
from literature_survey_app.paper_search_agent import create_paper_search_agent
from literature_survey_app.survey_writing_agent import create_survey_writing_agent
import os
from pathlib import Path
import tempfile
import re
import chardet
from typing import List
from smolagents import LogLevel
from src.models import LiteLLMModel
# Import utility functions from the new modules
from literature_survey_app.utils import (
    try_compile_with_fallback,
)

from dotenv import load_dotenv
load_dotenv(override=True)


LIST_MARKER_RE = re.compile(
    r""" ^\s{0,3}               # optional indent (≤ 3 spaces)
         (?:                    # list marker alternatives
             [-+*]              #   • unordered: -, +, *
           | \d+[.)]            #   • ordered:   1.   1)
           | \[\d+\]            #   • ordered:   [1]  [23]
         )
         \s+                    # at least one space after the marker
    """,
    re.VERBOSE,
)

def extract_markdown_paths(report_text: str, dedupe: bool = True) -> List[str]:
    """
    Parse a block of text (e.g., a boxed report of curated papers) and return
    the list of relative Markdown file paths that appear after 'Saved as:' lines.

    Handles these cases:
      1. Path on the same line:
         │     Saved as: ./training_language_models_to_follow_instructions.md │
      2. Path on the following line (blank placeholder after 'Saved as:'):
         │     Saved as:                                                     │
         │ ./remax_a_simple_effective_and_efficient_reinforcement...md       │
      3. Path followed by extra commentary before the right border:
         │     Saved as: ./file_name.md  (See attachments…)                  │
      4. Mixed box characters (│) or plain text.
      5. Ignores malformed or missing entries gracefully.

    Parameters
    ----------
    report_text : str
        The full text block to parse.
    dedupe : bool
        If True, preserves first occurrence order and removes duplicates.

    Returns
    -------
    List[str]
        List of extracted markdown paths (exact as they appeared, stripped of
        trailing spaces / box artifacts).
    """
    lines = report_text.splitlines()
    results: List[str] = []
    seen = set()

    # Regex to detect a path on the SAME line as 'Saved as:'
    same_line_pattern = re.compile(
        r'Saved as:\s*(\./[A-Za-z0-9][^\s|]*?\\.md)(?=\s|\\|$)', re.IGNORECASE
    )

    # Regex to detect a standalone path line (used for lookahead case)
    path_line_pattern = re.compile(
        r'^\s*[│|]?\s*(\./[A-Za-z0-9][^\s|]*?\\.md)(?=\s|\\|$)', re.IGNORECASE
    )

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # Normalize by stripping the right-side box char if present
        # (We only remove the final │ if it's detached by spaces)
        raw_line = line.rstrip()
        if raw_line.endswith('│'):
            raw_line = raw_line[:-1].rstrip()

        # Case 1: path present on same line
        m = same_line_pattern.search(raw_line)
        if m:
            path = m.group(1).strip()
            if not dedupe or path not in seen:
                results.append(path)
                seen.add(path)
            i += 1
            continue

        # Case 2: line contains 'Saved as:' but no path -> look ahead
        if 'Saved as:' in raw_line and not m:
            j = i + 1
            while j < n:
                look = lines[j].rstrip()
                if look.endswith('│'):
                    look = look[:-1].rstrip()
                # Skip blank / decorative lines
                if look.strip() == '' or look.strip(' │') == '':
                    j += 1
                    continue
                pm = path_line_pattern.match(look)
                if pm:
                    path = pm.group(1).strip()
                    if not dedupe or path not in seen:
                        results.append(path)
                        seen.add(path)
                    break
                # If the next non-empty line isn't a path, abandon this 'Saved as:' block
                break
            i = j
            continue

        i += 1

    return results

def _read_file_utf8_safe(path: str) -> str:
    """Read any text file as UTF-8, falling back to detected encoding if needed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try to detect encoding
        with open(path, "rb") as f:
            raw = f.read()
        import chardet
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(enc, errors="replace")

def blank_lines_around_list_items(md: str) -> str:
    """
    Ensure every list item has *exactly one* blank line before it,
    between it and the next item, and after the last item in the list.
    """
    lines = md.splitlines()
    out   = []
    i     = 0

    while i < len(lines):
        line = lines[i]

        if LIST_MARKER_RE.match(line):
            # ---- blank line BEFORE the item ---------------------------
            if out and out[-1].strip():          # prev line exists & not blank
                out.append("")

            # ---- the item itself -------------------------------------
            out.append(line)

            # ---- blank line AFTER the item ----------------------------
            next_is_blank = (i + 1 < len(lines) and not lines[i + 1].strip())
            if not next_is_blank:
                out.append("")

            i += 1
            continue

        # not a list item → copy verbatim
        out.append(line)
        i += 1

    return "\n".join(out)

def write_report(query: str, working_directory: str, search_agent_model: LiteLLMModel, writing_agent_model: LiteLLMModel, max_retries: int = 3, min_papers: int = 5):
    print("===== First stage: paper search =====")
    paper_search_agent = create_paper_search_agent(
        working_directory=os.path.join(working_directory, "relevant_papers"),
        model=search_agent_model,
        max_steps=5,
        verbosity_level=LogLevel.DEBUG,
    )
    paper_summary = paper_search_agent.run(query).to_string()

    paper_md_list = [str(p) for p in Path(os.path.join(working_directory, "relevant_papers")).rglob("*.md")]

    retries = 0
    while len(paper_md_list) < min_papers and retries < max_retries:
        print("Not enough markdown files found in paper search agent report. Minimum required to write a comprehensive report:", min_papers)
        print("Ask paper search agent to retrieve paper content based on history.")
        paper_summary = paper_search_agent.run(f"Not enough markdown files were found in the working directory. Most likely reason was that you did not call get_paper_from_url, the calls failed, or the selected number of URLs were not enough (minimum: {min_papers}). Please try to retrieve enough number of papers from the URLs you have found so far without any further search.", reset=False).to_string()
        paper_md_list = [str(p) for p in Path(os.path.join(working_directory, "relevant_papers")).rglob("*.md")]
        retries += 1

    print(f"Extracted {len(paper_md_list)} markdown files from paper search agent report.")
    print(paper_md_list)
    print("===== Second stage: literature survey writing =====")
    writing_agent, survey_writing_agent = create_survey_writing_agent(model=writing_agent_model, output_dir=working_directory)

    for i,full_path in enumerate(paper_md_list):
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        writing_agent(f"""Read the following paper and write a concise but comprehensive summary in markdown format (MODE=PER_PAPER). Name of the resulting markdown file should start with index {i+1}. \n{content}
                          """, reset=True)
    # survey_writing_agent.run(f"Write a literature survey on {query}. Your audience will be students and researchers in the relevant fields. \nThe writing_agent has already generated summary for each paper under the working directory. What you need to do now is just reading these summaries and writing an overview section based on them (save the overview section with name 00_overview.md). You should also include a reference list at the end of the overview.") 
    survey_writing_agent.run(f"Write a literature survey on {query}. Your audience will be students and researchers in the relevant fields. \nThe following list of papers under folder relevant_papers has been provided as context:\n{paper_summary}. \nThe writing_agent has already generated summary for each of these papers under the working directory. What you need to do now is reading these summaries and writing an overview section based on them (save the overview section with name 00_overview.md). You should also include a reference list at the end of the overview.")        
    # # survey_writing_agent.run(f"Write a literature survey on current research on {query}. Your audience will be students and researchers in the relevant fields. Be sure to include technical details and references to the literature.\nThe following list of papers under folder relevant_papers has been provided as context:\n{paper_summary}.\nYour should write a comprehensive literature survey based on these papers. You must include detailed descriptions and analyses of each paper in your literature survey.")

    print("===== Third stage: compile the report =====")
    # combine all .md files under working_directory into a single .md file
    md_files = sorted([f for f in glob(os.path.join(working_directory, "*.md"))])
    if not md_files:
        raise FileNotFoundError("No main content *.md files found in the working directory.")

    # 2) concatenate their contents (UTF‑8‑safe)
    full_markdown = "\n\n".join(_read_file_utf8_safe(p) for p in md_files)
    full_markdown = blank_lines_around_list_items(full_markdown)
    # write markdown to a single file
    output_file = os.path.join(working_directory, "final_report.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    # if not try_compile_with_fallback(working_directory, 'full_report.html'):
    #     print("\nAll compilation attempts failed. Please check the markdown files manually.")
    #     print("Common issues to check:")
    #     print("1. YAML metadata blocks")
    #     print("2. Malformed headers")
    #     print("3. Inconsistent list formatting")
    #     print("4. Unclosed code blocks")
    #     print("5. Malformed math expressions")
    #     print("6. Special characters in headers")
    #     print("7. Inconsistent line breaks")
    
    return {
        "status": "success",
        "directory": working_directory,
        "filename": "final_report.md"
    }

if __name__ == "__main__":
    base_temp_dir = "literature_survey_app/temp_files"
    Path(base_temp_dir).mkdir(parents=True, exist_ok=True)
    working_directory = None
    if working_directory is None:
        working_directory = tempfile.mkdtemp(dir=base_temp_dir, prefix="working_directory_")
    print(f"Using working directory: {working_directory}")
    search_agent_model = LiteLLMModel(model_id="gpt-5", api_base=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    writing_agent_model = LiteLLMModel(model_id="gpt-4.1", api_base=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    write_report(
        query="current research on meta-heuristic methods enhanced by large language models",
        working_directory=working_directory,
        search_agent_model=search_agent_model,
        writing_agent_model=writing_agent_model
    )