#!/usr/bin/env python3
"""Main entry point for pipx run nano-swe-agent commands."""

import sys

from nanoswea.run_github_issue import run_from_cli as run_github_from_cli
from nanoswea.run_local import run_from_cli as run_local_from_cli


def main() -> None:
    """Main entry point for pipx run commands."""
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "gh":
        run_github_from_cli(args[1:])
    else:
        run_local_from_cli(args)


if __name__ == "__main__":
    main()
