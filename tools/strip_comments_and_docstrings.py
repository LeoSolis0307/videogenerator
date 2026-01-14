import io
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


CODING_RE = re.compile(r"^#\s*[-_*]?\s*coding\s*[:=]\s*([-\w.]+)")


@dataclass
class SuiteState:
    at_suite_start: bool = True


def _is_shebang(comment_text: str, line_no: int) -> bool:
    return line_no == 1 and comment_text.startswith("#!")


def _is_coding_cookie(comment_text: str, line_no: int) -> bool:
    if line_no not in (1, 2):
        return False
    return CODING_RE.match(comment_text) is not None


def strip_comments_and_docstrings(source: str) -> str:
    tokens: list[tokenize.TokenInfo] = []

    suite_stack: list[SuiteState] = [SuiteState(at_suite_start=True)]          
    pending_skip_newline = False

    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        tok_type = tok.type

        if pending_skip_newline:
            if tok_type in (tokenize.NL, tokenize.NEWLINE):
                pending_skip_newline = False
                continue
            pending_skip_newline = False

        if tok_type == tokenize.COMMENT:
            if _is_shebang(tok.string, tok.start[0]) or _is_coding_cookie(tok.string, tok.start[0]):
                tokens.append(tok)
            continue

        if tok_type == tokenize.INDENT:
            suite_stack.append(SuiteState(at_suite_start=True))
            tokens.append(tok)
            continue

        if tok_type == tokenize.DEDENT:
            if len(suite_stack) > 1:
                suite_stack.pop()
            tokens.append(tok)
            continue

        if tok_type in (tokenize.NL, tokenize.NEWLINE):
            tokens.append(tok)
            continue

                                                             
        if tok_type == tokenize.STRING and suite_stack[-1].at_suite_start:
                                                                               
            pending_skip_newline = True
            continue

                                                          
        suite_stack[-1].at_suite_start = False
        tokens.append(tok)

    return tokenize.untokenize(tokens)


def iter_project_py_files(root: Path) -> list[Path]:
    targets: list[Path] = []

    explicit = [
        root / "main.py",
        root / "ver_voces.py",
        root / "habilitar_jorge_sapi5.py",
    ]
    for p in explicit:
        if p.exists() and p.suffix == ".py":
            targets.append(p)

    for folder in (root / "core", root / "utils", root / "tools"):
        if not folder.exists():
            continue
        for p in folder.rglob("*.py"):
            targets.append(p)

                          
    uniq = sorted({p.resolve() for p in targets})
    return uniq


def process_file(path: Path, dry_run: bool) -> bool:
    raw = path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except Exception:
        encoding = "utf-8"

    text = raw.decode(encoding, errors="replace")
    stripped = strip_comments_and_docstrings(text)

    if stripped == text:
        return False

    if not dry_run:
        path.write_text(stripped, encoding=encoding, newline="")

    return True


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    dry_run = "--dry-run" in argv

    changed = 0
    files = iter_project_py_files(root)
    for p in files:
                                                                       
        parts_lower = {part.lower() for part in p.parts}
        if ".venv" in parts_lower or ".git" in parts_lower:
            continue
        if "output" in parts_lower:
            continue

        if process_file(p, dry_run=dry_run):
            changed += 1

    print(f"Processed {len(files)} files, changed {changed}.")
    if dry_run:
        print("(dry-run: no files written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
