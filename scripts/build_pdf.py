"""markdown -> HTML -> PDF via chrome headless."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "report.md"
CSS_PATH = ROOT / "docs" / "persian.css"
HTML_PATH = ROOT / "docs" / "_report.html"
PDF_PATH = ROOT / "docs" / "report.pdf"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<title>گزارش فنی</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<style>
{css}
</style>
</head>
<body>
{body}
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ],
            throwOnError: false
        }});"></script>
</body>
</html>
"""


def find_browser() -> str:
    for c in [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]:
        if Path(c).exists():
            return c
    raise RuntimeError("no chromium browser found")


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
    )
    css_text = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    HTML_PATH.write_text(HTML_TEMPLATE.format(css=css_text, body=html_body), encoding="utf-8")

    browser = find_browser()
    out = str(PDF_PATH.resolve())
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={out}",
        "--print-to-pdf-no-header",
        HTML_PATH.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, timeout=120)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
