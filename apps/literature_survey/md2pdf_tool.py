import os
import glob
import pypandoc
import chardet
import re
from smolagents import Tool

import re

LIST_MARKER_RE = re.compile(
    r""" ^\s{0,3}               # optional indent (≤ 3 spaces)
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

class CompileMarkdownToPDF(Tool):
    """
    Reads all zero‑padded Markdown files in `working_dir`, concatenates them
    in filename order, and produces a single, well‑formatted PDF.
    """
    name = "compile_markdown_to_pdf"
    description = (
        "Combine Markdown files (01_…, 02_…, …) into one PDF with equations, "
        "clickable links, a TOC, and correct section numbering."
    )
    inputs = {
        "output_filename": {
            "type": "string",
            "description": "Name of the PDF to create (e.g. 'report.pdf')."
        }
    }
    output_type = "string"

    # ------------------------------------------------------------------ utils
    def __init__(self, working_dir: str):
        super().__init__()
        self.working_dir = working_dir

    @staticmethod
    def _read_file_utf8_safe(path: str) -> str:
        """Read any text file as UTF-8, auto-detecting original encoding."""
        with open(path, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(enc)

    # ---------------------------------------------------------------- forward
    def forward(self, output_filename: str) -> str:
        # 1) find Markdown files and keep their natural (zero‑padded) order
        md_files = sorted(glob.glob(os.path.join(self.working_dir, "*.md")))
        if not md_files:
            raise FileNotFoundError("No *.md files found in the working directory.")

        # 2) concatenate their contents (UTF‑8‑safe)
        full_markdown = "\n\n".join(self._read_file_utf8_safe(p) for p in md_files)
        full_markdown = blank_lines_around_list_items(full_markdown)

        # 3) build output path
        output_path = os.path.join(self.working_dir, output_filename)

        # 4) let Pandoc do the heavy lifting
        try:
            pypandoc.convert_text(
                full_markdown,
                to="pdf",
                format="markdown+tex_math_dollars+tex_math_single_backslash",
                outputfile=output_path,
                extra_args=[
                    "--pdf-engine=xelatex",
                    "--toc",
                    # "--number-sections",
                ],
            )
        except RuntimeError as e:
            raise RuntimeError(f"PDF compilation failed: {e}") from e

        return f"PDF generated at {output_path}"


# ------------------------------------------------------------------ CLI test
if __name__ == "__main__":
    tool = CompileMarkdownToPDF("apps/literature_survey/example_report")
    print(tool.forward("final_report_fixed.pdf"))
