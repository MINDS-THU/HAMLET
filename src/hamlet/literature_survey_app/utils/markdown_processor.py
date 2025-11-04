"""
Markdown Content Processing Utilities

This module provides functions for cleaning and processing markdown content
to ensure proper formatting and compilation.
"""

import re
import os


def clean_markdown_content(content: str) -> str:
    """Clean markdown content to ensure it's valid and readable.
    
    Args:
        content: Raw markdown content string
        
    Returns:
        Cleaned markdown content string
        
    This function fixes common markdown issues including:
    - YAML metadata blocks
    - Header formatting
    - List formatting
    - Code block formatting
    - Math block formatting
    - Link formatting
    - Emphasis formatting
    - Trailing spaces
    - Line break normalization
    """
    # Remove YAML metadata blocks
    if content.startswith('---'):
        end_yaml = content.find('---', 3)
        if end_yaml != -1:
            content = content[end_yaml + 3:].lstrip()
    
    # Fix common markdown issues
    # 1. Fix headers
    content = re.sub(r'^#\s+', '# ', content, flags=re.MULTILINE)
    content = re.sub(r'^##\s+', '## ', content, flags=re.MULTILINE)
    content = re.sub(r'^###\s+', '### ', content, flags=re.MULTILINE)
    
    # 2. Fix lists
    content = re.sub(r'^\s*[-*]\s+', '- ', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+', '1. ', content, flags=re.MULTILINE)
    
    # 3. Fix code blocks
    content = re.sub(r'```\s*\n', '```\n', content)
    content = re.sub(r'\n\s*```', '\n```', content)
    
    # 4. Fix math blocks
    content = re.sub(r'\$\$\s*\n', '$$\n', content)
    content = re.sub(r'\n\s*\$\$', '\n$$', content)
    
    # 5. Fix links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[\1](\2)', content)
    
    # 6. Fix emphasis
    content = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', content)
    content = re.sub(r'\*([^*]+)\*', r'*\1*', content)
    
    # 7. Remove trailing spaces
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # 8. Fix line breaks
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 9. Remove empty lines at start/end
    content = content.strip()
    
    return content


def fix_markdown_files(directory: str) -> None:
    """Fix markdown files in a directory to ensure they compile properly.
    
    Args:
        directory: Path to directory containing markdown files
        
    This function applies clean_markdown_content to all .md files in the
    specified directory, fixing common formatting issues that could prevent
    proper compilation.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            print(f"Fixing {filename}...")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean the content
            content = clean_markdown_content(content)
            
            # Write cleaned content back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Fixed {filename}") 