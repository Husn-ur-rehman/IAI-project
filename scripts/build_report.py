"""Check the LaTeX report and package it as a zip for Overleaf.

    python scripts/build_report.py      -> dist/cs2-expert-system-report.zip

The zip holds main.tex, references.bib, figures/*.pdf (copied from reports/figures, so the
evaluation figures are always the latest) and screenshots/*.png. Upload it on Overleaf with
New Project > Upload Project and compile with pdfLaTeX.

LaTeX is not needed locally. The script checks the structure before packaging instead:
balanced braces and environments, every \\includegraphics file present, every \\cite key in
the bibliography, every \\ref label defined.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
SOURCES = {"figures": ROOT / "reports" / "figures", "screenshots": REPORT / "screenshots"}
OUT = ROOT / "dist" / "cs2-expert-system-report.zip"


def strip(tex: str) -> str:
    """Remove comments and verbatim blocks, which do not follow normal LaTeX syntax."""
    tex = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", tex, flags=re.S)
    return re.sub(r"(?<!\\)%.*", "", tex)


def check(tex: str, bib: str) -> list[str]:
    errors = []
    body = strip(tex)

    depth = 0
    for i, line in enumerate(body.splitlines(), 1):
        clean = re.sub(r"\\[{}]", "", line)
        depth += clean.count("{") - clean.count("}")
        if depth < 0:
            errors.append(f"line {i}: closing brace without an opening one")
            depth = 0
    if depth:
        errors.append(f"{depth} unclosed brace(s)")

    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        kind, env = m.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            errors.append(f"\\end{{{env}}} does not match the open environment")
    errors += [f"environment {e} is never closed" for e in stack]

    for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        folder, name = path.split("/", 1)
        if folder not in SOURCES or not (SOURCES[folder] / name).exists():
            errors.append(f"missing image {path}")

    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        errors += [f"citation {k.strip()} not in references.bib" for k in group.split(",") if k.strip() not in keys]

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    errors += [f"\\ref{{{r}}} has no matching \\label" for r in re.findall(r"\\ref\{([^}]+)\}", body) if r not in labels]
    return errors


def main():
    tex = (REPORT / "main.tex").read_text(encoding="utf-8")
    bib = (REPORT / "references.bib").read_text(encoding="utf-8")
    errors = check(tex, bib)
    if errors:
        print("Report check failed:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    images = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip(tex))
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", tex)
        z.writestr("references.bib", bib)
        for path in images:
            folder, name = path.split("/", 1)
            z.write(SOURCES[folder] / name, path)
    n_refs = len(re.findall(r"@\w+\{", bib))
    print(f"Report check passed: {len(images)} images, {n_refs} references.")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
