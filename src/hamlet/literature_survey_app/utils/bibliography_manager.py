"""
Bibliography and Citation Management Utilities

This module provides functions for managing citations and bibliographies
in literature survey documents.
"""

import json
import os
import re


def create_bibliography(directory: str) -> None:
    """Create both bibliography.json and bibliography.md files from all reference markdown files in the directory.
    
    Args:
        directory: Directory containing reference markdown files
        
    This function:
    1. Scans all files starting with 'references_' and ending with '.md'
    2. Extracts reference entries from these files
    3. Creates a unified bibliography with unique numeric indices
    4. Saves both JSON and Markdown versions of the bibliography
    """
    # Dictionary to store unique references
    bibliography = {}
    current_index = 1
    
    # Regular expression to match reference entries
    ref_pattern = r'\[(.*?)\](.*?)(?=\n\[|$)'
    
    # Find all reference markdown files
    for filename in os.listdir(directory):
        if filename.startswith('references_') and filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            
            # Read the reference file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all references in the file
            matches = re.finditer(ref_pattern, content, re.DOTALL)
            
            for match in matches:
                ref_name = match.group(1).strip()
                ref_content = match.group(2).strip()
                
                # Only add if not already in bibliography
                if ref_name not in bibliography:
                    bibliography[ref_name] = (current_index, ref_content)
                    current_index += 1
    
    # Write bibliography to JSON file
    output_path = os.path.join(directory, 'bibliography.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bibliography, f, indent=2, ensure_ascii=False)
    
    # Write bibliography to Markdown file
    md_output_path = os.path.join(directory, 'bibliography.md')
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write("# References\n\n")
        # Sort by index
        sorted_refs = sorted(bibliography.items(), key=lambda x: x[1][0])
        for ref_name, (index, content) in sorted_refs:
            f.write(f"[{index}] {content}\n\n")
    
    print(f"Created bibliography.json and bibliography.md with {len(bibliography)} unique references")


def update_in_text_citations(directory: str) -> None:
    """Update in-text citations in markdown files using bibliography.json.
    
    Args:
        directory: Directory containing markdown files and bibliography.json
        
    This function:
    1. Loads the bibliography mapping from bibliography.json
    2. Converts in-text citations from [paper_title] format to [n] format
    3. Updates all main content markdown files (excluding reference files)
    """
    # Load bibliography
    bib_path = os.path.join(directory, 'bibliography.json')
    with open(bib_path, 'r', encoding='utf-8') as f:
        bibliography = json.load(f)
    
    # Create citation mapping
    citation_map = {}
    for ref_name, (index, _) in bibliography.items():
        citation_map[ref_name] = str(index)
    
    # Find all main content markdown files (excluding reference files)
    for filename in os.listdir(directory):
        if filename.endswith('.md') and not filename.startswith('references_'):
            filepath = os.path.join(directory, filename)
            
            # Read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace citations
            for ref_name, index in citation_map.items():
                # Use regex to match citations in square brackets
                # This ensures we only match complete citation markers
                pattern = r'\[(' + re.escape(ref_name) + r')\]'
                content = re.sub(pattern, f'[{index}]', content)
            
            # Write updated content back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated citations in {filename}") 