import os
import glob
import pypandoc
import chardet
import re
from smolagents import Tool
from pathlib import Path
import subprocess

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

class CompileMarkdownToHTML(Tool):
    """
    Reads all zero‑padded Markdown files in `working_dir`, concatenates them
    in filename order, and produces both a combined Markdown file and a well‑formatted HTML document.
    """
    name = "compile_markdown_to_html"
    description = (
        "Combine Markdown files (01_…, 02_…, …) into one Markdown file and HTML with equations, "
        "clickable links, a TOC, and correct section numbering."
    )
    inputs = {
        "output_filename": {
            "type": "string",
            "description": "Name of the HTML to create (e.g. 'report.html'). The combined markdown will be saved as 'survey_combined.md'."
        }
    }
    output_type = "string"

    def __init__(self, working_dir: str):
        super().__init__()
        self.working_dir = working_dir
        self.css_content = """
        body {
            font-family: "Times New Roman", Times, serif;
            font-size: 11pt;
            line-height: 1.5;
            margin: 1in;
            max-width: 8.5in;
        }
        h1, h2, h3 {
            color: #333;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        h1 { font-size: 24pt; }
        h2 { font-size: 18pt; }
        h3 { font-size: 14pt; }
        p { margin-bottom: 1em; }
        a { color: blue; }
        code {
            font-family: "Courier New", Courier, monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #ddd;
            padding-left: 1em;
            color: #666;
            margin-left: 0;
        }
        .math {
            margin: 1em 0;
        }
        """

    @staticmethod
    def _read_file_utf8_safe(path: str) -> str:
        """Read any text file as UTF-8, auto-detecting original encoding."""
        with open(path, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(enc)

    def forward(self, output_filename: str) -> str:
        # 1) find main content Markdown files (excluding reference files, review feedback, and any files containing "review" or "feedback") and keep their natural (zero‑padded) order
        md_files = sorted([f for f in glob.glob(os.path.join(self.working_dir, "*.md"))])
        if not md_files:
            raise FileNotFoundError("No main content *.md files found in the working directory.")

        # 2) concatenate their contents (UTF‑8‑safe)
        full_markdown = "\n\n".join(self._read_file_utf8_safe(p) for p in md_files)
        full_markdown = blank_lines_around_list_items(full_markdown)

        # 3) Append bibliography if it exists
        bibliography_path = os.path.join(self.working_dir, "bibliography.md")
        if os.path.exists(bibliography_path):
            full_markdown += "\n\n" + self._read_file_utf8_safe(bibliography_path)

        # 4) build output paths
        output_path = os.path.join(self.working_dir, output_filename)
        css_path = os.path.join(self.working_dir, "style.css")

        # 5) Save CSS file
        with open(css_path, 'w') as f:
            f.write(self.css_content)

        # 6) Convert to HTML using pandoc
        try:
            pypandoc.convert_text(
                full_markdown,
                to="html",
                format="markdown+tex_math_dollars+tex_math_single_backslash",
                outputfile=output_path,
                extra_args=[
                    "--standalone",
                    "--mathjax",
                    "--toc",
                    f"--css={css_path}"
                ],
            )
        except RuntimeError as e:
            raise RuntimeError(f"HTML compilation failed: {e}") from e

        return f"HTML generated at {output_path}"

def compile_markdown_to_html(output_filename="bayesian_icl_literature_survey.html"):
    """Compile markdown to HTML with proper formatting and MathJax support."""
    try:
        # Read the combined markdown file
        with open("survey_combined.md", "r") as f:
            content = f.read()
            
        # Remove line numbers and clean up formatting
        content = re.sub(r'^\d+:', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\s*:', '', content, flags=re.MULTILINE)
        
        # Create HTML template with proper styling
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bayesian In-Context Learning Literature Survey</title>
    <link rel="stylesheet" href="style.css">
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            },
            svg: {
                fontCache: 'global'
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 2em;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        h1 { font-size: 2.5em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }
        h2 { font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.2em; }
        h3 { font-size: 1.5em; }
        h4 { font-size: 1.2em; }
        p { margin: 1em 0; }
        code {
            background: #f5f5f5;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }
        pre {
            background: #f5f5f5;
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
        }
        pre code {
            background: none;
            padding: 0;
        }
        blockquote {
            border-left: 4px solid #ddd;
            margin: 1em 0;
            padding-left: 1em;
            color: #666;
        }
        a {
            color: #0366d6;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .math {
            margin: 1em 0;
        }
        .math.display {
            display: block;
            text-align: center;
            margin: 1em 0;
        }
        .toc {
            background: #f8f9fa;
            padding: 1em;
            border-radius: 5px;
            margin: 1em 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 1em;
        }
        .toc li {
            margin: 0.5em 0;
        }
        .references {
            margin-top: 2em;
            padding-top: 1em;
            border-top: 1px solid #eee;
        }
    </style>
</head>
<body>
    <div class="toc">
        <h2>Table of Contents</h2>
        {toc}
    </div>
    <div class="content">
        {content}
    </div>
    <div class="references">
        <h2>References</h2>
        {references}
    </div>
</body>
</html>"""

        # Convert markdown to HTML using pandoc
        html_content = subprocess.check_output(
            ["pandoc", "--from", "markdown", "--to", "html", "--standalone"],
            input=content.encode(),
            stderr=subprocess.PIPE
        ).decode()

        # Extract TOC and references
        toc_match = re.search(r'<nav id="TOC".*?</nav>', html_content, re.DOTALL)
        toc = toc_match.group(0) if toc_match else ""
        
        references_match = re.search(r'<h2 id="references".*?</div>', html_content, re.DOTALL)
        references = references_match.group(0) if references_match else ""

        # Clean up the content
        content = re.sub(r'<nav id="TOC".*?</nav>', '', html_content, flags=re.DOTALL)
        content = re.sub(r'<h2 id="references".*?</div>', '', content, flags=re.DOTALL)
        content = re.sub(r'<div class="line-block">.*?</div>', '', content, flags=re.DOTALL)
        
        # Format the final HTML
        final_html = html_template.format(
            toc=toc,
            content=content,
            references=references
        )

        # Write the HTML file
        with open(output_filename, "w") as f:
            f.write(final_html)

        print(f"Successfully compiled {output_filename}")
        return True

    except Exception as e:
        print(f"Error compiling HTML: {e}")
        return False

# ------------------------------------------------------------------ CLI test
if __name__ == "__main__":
    tool = CompileMarkdownToHTML("apps/literature_survey/report_74")
    print(tool.forward("final_report.html"))