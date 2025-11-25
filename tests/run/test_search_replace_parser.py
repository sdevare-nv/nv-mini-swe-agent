"""Tests for SEARCH/REPLACE parser."""

import pytest
from pathlib import Path
import tempfile

from minisweagent.run.extra.utils.search_replace_parser import (
    parse_search_replace_edits,
    apply_search_replace_to_content,
    search_replace_to_unified_diff,
    extract_and_convert_to_diff,
    SearchReplaceEdit,
)


def test_parse_single_edit():
    """Test parsing a single SEARCH/REPLACE edit."""
    content = """
### test.py
<<<<<<< SEARCH
def old_function():
    pass
=======
def new_function():
    return True
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].file_path == "test.py"
    assert "old_function" in edits[0].search_block
    assert "new_function" in edits[0].replace_block


def test_parse_multiple_edits():
    """Test parsing multiple SEARCH/REPLACE edits."""
    content = """
### file1.py
<<<<<<< SEARCH
old code 1
=======
new code 1
>>>>>>> REPLACE

### file2.py
<<<<<<< SEARCH
old code 2
=======
new code 2
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 2
    assert edits[0].file_path == "file1.py"
    assert edits[1].file_path == "file2.py"


def test_parse_with_extra_text():
    """Test parsing with extra text around the edits."""
    content = """
<think>
This is my reasoning.
</think>

<solution>
### test.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
</solution>
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].file_path == "test.py"


def test_parse_no_edits():
    """Test parsing content with no SEARCH/REPLACE edits."""
    content = "This is just plain text with no edits."
    edits = parse_search_replace_edits(content)
    assert len(edits) == 0


def test_apply_search_replace_success():
    """Test successfully applying a SEARCH/REPLACE edit."""
    file_content = """def hello():
    print("hello")
    return True
"""
    
    search_block = """def hello():
    print("hello")"""
    
    replace_block = """def hello():
    print("hello world")"""
    
    result = apply_search_replace_to_content(file_content, search_block, replace_block)
    assert "hello world" in result
    assert result.count("hello world") == 1


def test_apply_search_replace_not_found():
    """Test applying a SEARCH/REPLACE edit when search block is not found."""
    file_content = "def foo(): pass"
    search_block = "def bar(): pass"
    replace_block = "def baz(): pass"
    
    with pytest.raises(ValueError, match="Search block not found"):
        apply_search_replace_to_content(file_content, search_block, replace_block)


def test_apply_search_replace_multiple_matches():
    """Test applying a SEARCH/REPLACE edit when search block appears multiple times."""
    file_content = "foo\nfoo\nfoo"
    search_block = "foo"
    replace_block = "bar"
    
    with pytest.raises(ValueError, match="found 3 times"):
        apply_search_replace_to_content(file_content, search_block, replace_block)


def test_search_replace_to_unified_diff_synthetic():
    """Test converting SEARCH/REPLACE to unified diff without base path."""
    edits = [
        SearchReplaceEdit(
            file_path="test.py",
            search_block="old line",
            replace_block="new line",
        )
    ]
    
    diff = search_replace_to_unified_diff(edits, base_path=None)
    assert "--- a/test.py" in diff
    assert "+++ b/test.py" in diff
    assert "-old line" in diff
    assert "+new line" in diff


def test_search_replace_to_unified_diff_with_file():
    """Test converting SEARCH/REPLACE to unified diff with actual file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        test_file = base_path / "test.py"
        test_file.write_text("def foo():\n    pass\n")
        
        edits = [
            SearchReplaceEdit(
                file_path="test.py",
                search_block="def foo():\n    pass",
                replace_block="def bar():\n    return True",
            )
        ]
        
        diff = search_replace_to_unified_diff(edits, base_path=base_path)
        assert "--- a/test.py" in diff
        assert "+++ b/test.py" in diff
        assert "-def foo():" in diff or "-    pass" in diff
        assert "+def bar():" in diff or "+    return True" in diff


def test_extract_and_convert_to_diff():
    """Test the convenience function that extracts and converts."""
    content = """
### test.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
    
    diff = extract_and_convert_to_diff(content)
    assert "--- a/test.py" in diff
    assert "-old" in diff
    assert "+new" in diff


def test_extract_and_convert_no_edits():
    """Test extract_and_convert_to_diff when no SEARCH/REPLACE format is found."""
    content = "Just plain text"
    result = extract_and_convert_to_diff(content)
    assert result == content


def test_parse_multiline_blocks():
    """Test parsing SEARCH/REPLACE with multiline code blocks."""
    content = """
### example.py
<<<<<<< SEARCH
def calculate(a, b):
    result = a - b
    return result
=======
def calculate(a, b):
    result = a + b
    return result
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert "a - b" in edits[0].search_block
    assert "a + b" in edits[0].replace_block
    assert edits[0].search_block.count("\n") == 2


def test_parse_with_indentation():
    """Test parsing maintains proper indentation."""
    content = """
### test.py
<<<<<<< SEARCH
class MyClass:
    def method(self):
        return False
=======
class MyClass:
    def method(self):
        return True
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert "    def method" in edits[0].search_block
    assert "        return False" in edits[0].search_block
    assert "        return True" in edits[0].replace_block


def test_parse_file_path_with_directories():
    """Test parsing file paths with directory structure."""
    content = """
### src/myapp/models.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].file_path == "src/myapp/models.py"


def test_empty_search_block():
    """Test parsing with empty search block (new file creation) - requires newline."""
    content = """
### newfile.py
<<<<<<< SEARCH

=======
def new_function():
    pass
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].search_block.strip() == ""
    assert "new_function" in edits[0].replace_block


def test_empty_replace_block():
    """Test parsing with empty replace block (deletion) - requires newline."""
    content = """
### test.py
<<<<<<< SEARCH
def old_function():
    pass
=======

>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].replace_block.strip() == ""
    assert "old_function" in edits[0].search_block


def test_truly_empty_blocks_dont_parse():
    """Test that blocks without newlines don't parse (strict format)."""
    content = """
### test.py
<<<<<<< SEARCH
=======
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    # Parser requires newlines, so this won't match
    assert len(edits) == 0


def test_whitespace_preservation():
    """Test that whitespace is preserved in blocks."""
    content = """
### test.py
<<<<<<< SEARCH
def foo():
    return 1    
=======
def foo():
    return 2    
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].search_block.endswith("    ")
    assert edits[0].replace_block.endswith("    ")


def test_trailing_newlines_in_blocks():
    """Test handling of trailing newlines in blocks."""
    content = """
### test.py
<<<<<<< SEARCH
line1
line2

=======
line1
line2
modified

>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].search_block.endswith("\n")
    assert "modified" in edits[0].replace_block


def test_multiple_edits_same_file():
    """Test multiple edits to the same file."""
    content = """
### test.py
<<<<<<< SEARCH
old1
=======
new1
>>>>>>> REPLACE

### test.py
<<<<<<< SEARCH
old2
=======
new2
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 2
    assert edits[0].file_path == "test.py"
    assert edits[1].file_path == "test.py"
    assert "old1" in edits[0].search_block
    assert "old2" in edits[1].search_block


def test_parse_inside_code_fence():
    """Test parsing SEARCH/REPLACE inside markdown code fence."""
    content = """
```python
### test.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
```
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].file_path == "test.py"


def test_special_characters_in_code():
    """Test parsing with special regex characters in code."""
    content = """
### test.py
<<<<<<< SEARCH
pattern = r"^[a-z]+$"
result = re.match(pattern, text)
=======
pattern = r"^[a-zA-Z]+$"
result = re.match(pattern, text)
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert r"^[a-z]+$" in edits[0].search_block
    assert r"^[a-zA-Z]+$" in edits[0].replace_block


def test_unicode_characters():
    """Test parsing with unicode characters."""
    content = """
### test.py
<<<<<<< SEARCH
message = "Hello"
=======
message = "Hello 世界 🌍"
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert "世界" in edits[0].replace_block
    assert "🌍" in edits[0].replace_block


def test_file_path_special_chars():
    """Test file paths with special characters."""
    content = """
### src/my-app/file_v2.0.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].file_path == "src/my-app/file_v2.0.py"


def test_malformed_missing_separator():
    """Test malformed edit missing separator."""
    content = """
### test.py
<<<<<<< SEARCH
old
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 0


def test_malformed_missing_end_marker():
    """Test malformed edit missing end marker."""
    content = """
### test.py
<<<<<<< SEARCH
old
=======
new
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 0


def test_malformed_wrong_order():
    """Test malformed edit with markers in wrong order."""
    content = """
### test.py
=======
new
<<<<<<< SEARCH
old
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 0


def test_nested_markers_in_code():
    """Test when code itself contains marker-like strings."""
    content = """
### test.py
<<<<<<< SEARCH
# This is not a <<<<<<< SEARCH marker
code = "<<<<<<< SEARCH"
=======
# This is not a <<<<<<< SEARCH marker
code = "updated"
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert "<<<<<<< SEARCH" in edits[0].search_block


def test_apply_with_surrounding_context():
    """Test applying edit with surrounding unchanged code."""
    file_content = """def foo():
    pass

def bar():
    return 1

def baz():
    pass
"""
    
    search_block = """def bar():
    return 1"""
    
    replace_block = """def bar():
    return 2"""
    
    result = apply_search_replace_to_content(file_content, search_block, replace_block)
    assert "return 2" in result
    assert result.count("def foo") == 1
    assert result.count("def baz") == 1


def test_apply_whitespace_sensitive():
    """Test that apply is whitespace-sensitive."""
    file_content = "def foo():\n    pass"
    search_block = "def foo():\n  pass"  # Wrong indentation
    replace_block = "def bar():\n  pass"
    
    with pytest.raises(ValueError, match="Search block not found"):
        apply_search_replace_to_content(file_content, search_block, replace_block)


def test_apply_empty_content():
    """Test applying to empty file content."""
    file_content = ""
    search_block = ""
    replace_block = "new content"
    
    result = apply_search_replace_to_content(file_content, search_block, replace_block)
    assert result == "new content"


def test_unified_diff_empty_edit():
    """Test unified diff with empty edit list."""
    edits = []
    diff = search_replace_to_unified_diff(edits)
    assert diff == ""


def test_unified_diff_no_changes():
    """Test unified diff when search equals replace."""
    edits = [
        SearchReplaceEdit(
            file_path="test.py",
            search_block="same",
            replace_block="same",
        )
    ]
    
    diff = search_replace_to_unified_diff(edits)
    # Diff should be empty or minimal when content is identical
    assert "test.py" in diff or diff == ""


def test_unified_diff_addition():
    """Test unified diff for pure addition."""
    edits = [
        SearchReplaceEdit(
            file_path="test.py",
            search_block="line1\nline2",
            replace_block="line1\nline2\nline3",
        )
    ]
    
    diff = search_replace_to_unified_diff(edits)
    assert "+line3" in diff


def test_unified_diff_deletion():
    """Test unified diff for pure deletion."""
    edits = [
        SearchReplaceEdit(
            file_path="test.py",
            search_block="line1\nline2\nline3",
            replace_block="line1\nline3",
        )
    ]
    
    diff = search_replace_to_unified_diff(edits)
    assert "-line2" in diff


def test_unified_diff_multiple_files():
    """Test unified diff with multiple files."""
    edits = [
        SearchReplaceEdit(file_path="file1.py", search_block="old1", replace_block="new1"),
        SearchReplaceEdit(file_path="file2.py", search_block="old2", replace_block="new2"),
    ]
    
    diff = search_replace_to_unified_diff(edits)
    assert "file1.py" in diff
    assert "file2.py" in diff
    assert "-old1" in diff
    assert "+new1" in diff
    assert "-old2" in diff
    assert "+new2" in diff


def test_unified_diff_with_base_path_nonexistent():
    """Test unified diff with base_path for non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        edits = [
            SearchReplaceEdit(
                file_path="newfile.py",
                search_block="",
                replace_block="new content",
            )
        ]
        
        diff = search_replace_to_unified_diff(edits, base_path=base_path)
        assert "newfile.py" in diff
        assert "+new content" in diff


def test_extract_convert_multiple_formats():
    """Test extract_and_convert with different content formats."""
    # SEARCH/REPLACE format
    content1 = """
### test.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
    diff1 = extract_and_convert_to_diff(content1)
    assert "test.py" in diff1
    
    # Plain text (no SEARCH/REPLACE)
    content2 = "Just plain text"
    diff2 = extract_and_convert_to_diff(content2)
    assert diff2 == content2


def test_large_blocks():
    """Test parsing with very large code blocks."""
    large_block = "\n".join([f"line_{i}" for i in range(1000)])
    content = f"""
### test.py
<<<<<<< SEARCH
{large_block}
=======
{large_block}_modified
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert "line_999" in edits[0].search_block
    assert "_modified" in edits[0].replace_block


def test_mixed_line_endings():
    """Test handling mixed line endings."""
    content = "### test.py\r\n<<<<<<< SEARCH\r\nold\r\n=======\r\nnew\r\n>>>>>>> REPLACE"
    
    edits = parse_search_replace_edits(content)
    assert len(edits) == 1
    assert edits[0].file_path == "test.py"


def test_extra_whitespace_in_markers():
    """Test that markers must be exact (no extra spaces in markers themselves)."""
    content = """
###   test.py   
<<<<<<<   SEARCH  
old
=======  
new
>>>>>>>   REPLACE  
"""
    
    edits = parse_search_replace_edits(content)
    # Parser requires exact marker format, extra spaces in markers themselves don't work
    assert len(edits) == 0


def test_whitespace_before_file_path():
    """Test that whitespace before file path is OK."""
    content = """
###   test.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
"""
    
    edits = parse_search_replace_edits(content)
    # Extra spaces after ### are OK, stripped from filename
    assert len(edits) == 1
    assert edits[0].file_path == "test.py"


def test_case_sensitive_search():
    """Test that search is case-sensitive."""
    file_content = "Hello World"
    search_block = "hello world"
    replace_block = "goodbye world"
    
    with pytest.raises(ValueError, match="Search block not found"):
        apply_search_replace_to_content(file_content, search_block, replace_block)


def test_partial_line_match():
    """Test that substring matches DO work (search is text-based, not line-based)."""
    file_content = "def function():\n    return True"
    search_block = "return True"  # This IS found as substring
    replace_block = "return False"
    
    # This should work because "return True" exists as a substring
    result = apply_search_replace_to_content(file_content, search_block, replace_block)
    assert "return False" in result


def test_must_match_with_indentation():
    """Test that you need to match indentation to avoid ambiguity."""
    file_content = "def function():\n    return True\n\ndef other():\n    return True"
    search_block = "return True"  # Ambiguous - appears twice
    replace_block = "return False"
    
    # This should fail because the search block appears multiple times
    with pytest.raises(ValueError, match="found 2 times"):
        apply_search_replace_to_content(file_content, search_block, replace_block)

