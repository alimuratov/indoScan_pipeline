from pathlib import Path
from typing import Set, Dict
from collections import Counter 
from typing import List
import logging
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str       # e.g., "MISSING_IMAGE", "STEM_MISMATCH", "MULTIPLE_PCDS"
    severity: str   # "error" | "warning"
    message: str

def index_by_stem(dir: Path, exts: Set[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if dir.is_dir():
        for file in dir.iterdir():
            if file.is_file() and file.suffix.lower() in exts:
                out[file.stem] = file
    return out

def count_stems(dir: Path, exts: Set[str]) -> Counter:
    stems = []
    for p in dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            stems.append(p.stem)
    return Counter(stems)

def get_stems(dir: Path, exts: Set[str]) -> Set[str]:
    stems = []
    for p in dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            stems.append(p.stem)
    return set(stems)

def log_issues(issues: List[ValidationIssue], severity: str) -> bool:
    for issue in issues:
        (logging.error if issue.severity == severity else logging.warning)("❌ %s: %s", issue.code, issue.message)
    return any(i.severity == severity for i in issues)

def exit_on_issues(issues: List[ValidationIssue], severity: str) -> None:
    if issues:
        for issue in issues:
            (logging.error if issue.severity == "error" else logging.warning)("%s: %s", issue.code, issue.message)
        sys.exit(1)