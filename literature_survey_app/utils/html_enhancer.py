"""
HTML Compilation and Enhancement Utilities

This module provides functions for compiling markdown to HTML and enhancing
the generated HTML with additional features like internal hyperlinks.
"""

import os
import re
from ..combine_md_to_html import CompileMarkdownToHTML
from .markdown_processor import fix_markdown_files


def try_compile_with_fallback(directory: str, output_file: str) -> bool:
    """Try to compile markdown to HTML with multiple fallback options.
    
    Args:
        directory: Directory containing markdown files
        output_file: Name of the output HTML file
        
    Returns:
        True if compilation successful, False otherwise
        
    This function:
    1. Attempts direct compilation using CompileMarkdownToHTML
    2. If that fails, cleans markdown files and retries
    3. Returns success/failure status for error handling
    """
    try:
        # Try direct compilation first
        print("\nAttempting direct compilation...")
        compile_tool = CompileMarkdownToHTML(directory)
        compile_tool.forward(output_file)
        print("HTML compilation successful!")
        return True
        
    except Exception as e:
        print(f"\nDirect compilation failed: {str(e)}")
        
        # Try cleaning and compiling
        print("\nCleaning markdown files and trying again...")
        fix_markdown_files(directory)
        
        try:
            compile_tool = CompileMarkdownToHTML(directory)
            compile_tool.forward(output_file)
            print("HTML compilation successful after cleaning!")
            return True
        except Exception as e2:
            print(f"\nCompilation still failed after cleaning: {str(e2)}")
            return False


def add_internal_hyperlinks(directory: str) -> None:
    """Add internal hyperlinks to the HTML file.
    
    Args:
        directory: Directory containing the HTML file
        
    This function enhances the generated HTML by:
    1. Adding IDs to all headers for section navigation
    2. Making citations clickable links to references
    3. Adding back-links from references to citations
    4. Injecting CSS for better styling
    """
    html_path = os.path.join(directory, 'output.html')
    if not os.path.exists(html_path):
        print(f"HTML file not found at {html_path}")
        return
    
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First, add IDs to all headers
    content = re.sub(r'<h1>(.*?)</h1>', lambda m: f'<h1 id="section-{m.group(1).lower().replace(" ", "-")}">{m.group(1)}</h1>', content)
    content = re.sub(r'<h2>(.*?)</h2>', lambda m: f'<h2 id="section-{m.group(1).lower().replace(" ", "-")}">{m.group(1)}</h2>', content)
    content = re.sub(r'<h3>(.*?)</h3>', lambda m: f'<h3 id="section-{m.group(1).lower().replace(" ", "-")}">{m.group(1)}</h3>', content)
    
    # Add IDs to citations and create reference links
    # First, find all citations in the format [n]
    citation_pattern = r'\[(\d+)\]'
    
    # Add IDs to citations and make them clickable
    content = re.sub(citation_pattern, lambda m: f'<a href="#ref-{m.group(1)}" class="citation">[{m.group(1)}]</a>', content)
    
    # Find the References section
    ref_section_pattern = r'(<h1>References</h1>)(.*?)(</body>)'
    ref_section = re.search(ref_section_pattern, content, re.DOTALL)
    
    if ref_section:
        ref_header, ref_content, body_end = ref_section.groups()
        
        # Process each reference entry
        ref_entry_pattern = r'<p>\[(\d+)\](.*?)</p>'
        ref_content = re.sub(ref_entry_pattern, 
                           lambda m: f'<p id="ref-{m.group(1)}" class="reference">[{m.group(1)}]{m.group(2)}</p>', 
                           ref_content)
        
        # Add back link to each reference
        ref_content = re.sub(r'<p id="ref-(\d+)"', 
                           lambda m: f'<p id="ref-{m.group(1)}" class="reference"><a href="#section-references" class="back-link">↑</a>', 
                           ref_content)
        
        # Reconstruct the content with processed references
        content = ref_header + ref_content + body_end
    
    # Add CSS for better styling
    style_tag = """
    <style>
        .citation {
            color: #0366d6;
            text-decoration: none;
            font-weight: bold;
        }
        .citation:hover {
            text-decoration: underline;
        }
        .reference {
            margin: 1em 0;
            padding-left: 2em;
            text-indent: -2em;
        }
        .back-link {
            color: #666;
            text-decoration: none;
            margin-right: 0.5em;
        }
        .back-link:hover {
            color: #0366d6;
        }
        #section-references {
            scroll-margin-top: 2em;
        }
    </style>
    """
    
    # Insert style tag in the head section
    content = content.replace('</head>', f'{style_tag}</head>')
    
    # Write updated content back
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Added internal hyperlinks to HTML file with improved styling") 