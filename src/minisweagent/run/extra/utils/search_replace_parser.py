"""Parser for SEARCH/REPLACE edits and converter to unified diff format."""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchReplaceEdit:
    """Represents a single SEARCH/REPLACE edit."""
    file_path: str
    search_block: str
    replace_block: str


def parse_search_replace_edits(content: str) -> list[SearchReplaceEdit]:
    """Parse SEARCH/REPLACE edits from model output.
    
    Expected format:
    ### file/path.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE
    
    Args:
        content: Model output containing SEARCH/REPLACE edits
        
    Returns:
        List of SearchReplaceEdit objects
    """
    edits = []
    
    # Pattern to match the entire SEARCH/REPLACE block
    pattern = r'###\s+([^\n]+)\s*\n<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        file_path = match.group(1).strip()
        search_block = match.group(2)
        replace_block = match.group(3)
        
        edits.append(SearchReplaceEdit(
            file_path=file_path,
            search_block=search_block,
            replace_block=replace_block,
        ))
    
    return edits


def apply_search_replace_to_content(file_content: str, search_block: str, replace_block: str) -> str:
    """Apply a single SEARCH/REPLACE edit to file content.
    
    Args:
        file_content: Original file content
        search_block: Code to search for
        replace_block: Code to replace with
        
    Returns:
        Modified file content
        
    Raises:
        ValueError: If search block is not found or found multiple times
    """
    count = file_content.count(search_block)
    
    if count == 0:
        raise ValueError(f"Search block not found in file")
    if count > 1:
        raise ValueError(f"Search block found {count} times in file (must be unique)")
    
    return file_content.replace(search_block, replace_block, 1)


def search_replace_to_unified_diff(edits: list[SearchReplaceEdit], base_path: Path | None = None) -> str:
    """Convert SEARCH/REPLACE edits to unified diff format.
    
    Args:
        edits: List of SearchReplaceEdit objects
        base_path: Base directory path to read original files from (if None, creates synthetic diff)
        
    Returns:
        Unified diff string suitable for `git apply`
    """
    diff_parts = []
    
    for edit in edits:
        file_path = edit.file_path
        
        if base_path:
            # Read actual file content
            full_path = base_path / file_path
            if not full_path.exists():
                # File doesn't exist, treat as new file
                original_lines = []
                modified_lines = edit.replace_block.splitlines(keepends=True)
            else:
                original_content = full_path.read_text()
                try:
                    modified_content = apply_search_replace_to_content(
                        original_content,
                        edit.search_block,
                        edit.replace_block
                    )
                    original_lines = original_content.splitlines(keepends=True)
                    modified_lines = modified_content.splitlines(keepends=True)
                except ValueError as e:
                    # If we can't apply the edit, create a synthetic diff anyway
                    print(f"Warning: Could not apply edit to {file_path}: {e}")
                    original_lines = edit.search_block.splitlines(keepends=True)
                    modified_lines = edit.replace_block.splitlines(keepends=True)
        else:
            # Create synthetic diff from search/replace blocks
            original_lines = edit.search_block.splitlines(keepends=True)
            modified_lines = edit.replace_block.splitlines(keepends=True)
        
        # Generate unified diff
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        )
        
        diff_text = '\n'.join(diff)
        if diff_text:
            diff_parts.append(diff_text)
    
    return '\n'.join(diff_parts)


def extract_and_convert_to_diff(content: str, base_path: Path | None = None) -> str:
    """Extract SEARCH/REPLACE edits from content and convert to unified diff.
    
    This is a convenience function that combines parsing and conversion.
    
    Args:
        content: Model output containing SEARCH/REPLACE edits
        base_path: Base directory path to read original files from
        
    Returns:
        Unified diff string
    """
    edits = parse_search_replace_edits(content)
    
    if not edits:
        # No SEARCH/REPLACE format found, return content as-is
        return content
    
    return search_replace_to_unified_diff(edits, base_path)

