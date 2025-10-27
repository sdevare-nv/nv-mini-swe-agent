import re


def get_changed_files_from_diff(diff_text: str) -> list[str]:
    pattern = r"^diff --git a/(.*?)\s"
    return re.findall(pattern, diff_text, re.MULTILINE)

