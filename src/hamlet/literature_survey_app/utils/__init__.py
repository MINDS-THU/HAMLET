"""
Literature Survey Utilities Package

This package contains utility modules for processing literature surveys:
- markdown_processor: Functions for cleaning and processing markdown content
- bibliography_manager: Functions for managing citations and bibliographies
- html_enhancer: Functions for HTML compilation and enhancement
"""

from .markdown_processor import clean_markdown_content, fix_markdown_files
from .bibliography_manager import create_bibliography, update_in_text_citations
from .html_enhancer import try_compile_with_fallback, add_internal_hyperlinks

__all__ = [
    'clean_markdown_content',
    'fix_markdown_files',
    'create_bibliography',
    'update_in_text_citations',
    'try_compile_with_fallback',
    'add_internal_hyperlinks'
] 