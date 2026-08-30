#!/usr/bin/env python3
"""Patch comfy_kitchen for torch<=2.5 custom_op schema (list[int] -> typing.List[int])."""
from __future__ import annotations

import logging
import re
import site
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wan.patch")


def ensure_typing_import(text: str) -> str:
    if re.search(r"(?m)^import typing\b", text) or re.search(r"(?m)^from typing import", text):
        return text
    lines = text.splitlines(keepends=True)
    insert_at = 0
    # Keep module docstring / coding / future imports first
    i = 0
    if i < len(lines) and (lines[i].startswith("#!") or "coding" in lines[i]):
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith('"""'):
        # skip docstring
        if lines[i].count('"""') >= 2:
            i += 1
        else:
            i += 1
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
            i += 1
    while i < len(lines) and (
        lines[i].startswith("from __future__")
        or lines[i].startswith("import __future__")
        or lines[i].strip() == ""
    ):
        i += 1
        insert_at = i
    insert_at = max(insert_at, i)
    lines.insert(insert_at, "import typing\n")
    return "".join(lines)


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # Undo a previously broken prepend of "import typing" before __future__
    text = re.sub(
        r"(?m)^import typing\n(?=from __future__ import annotations\n)",
        "",
        text,
    )
    original = text
    replacements = (
        ("list[int]", "typing.List[int]"),
        ("list[bool]", "typing.List[bool]"),
        ("list[float]", "typing.List[float]"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    if "typing." in text:
        text = ensure_typing_import(text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        logger.info("patched %s", path)
        return True
    return False


def main() -> None:
    roots = [Path(p) for p in site.getsitepackages()]
    roots.append(Path("/opt/conda/lib"))
    seen = set()
    changed = 0
    for root in roots:
        for path in root.rglob("comfy_kitchen/**/*.py"):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            if patch_file(path):
                changed += 1
    logger.info("comfy_kitchen patch done, files changed=%s", changed)


if __name__ == "__main__":
    main()
