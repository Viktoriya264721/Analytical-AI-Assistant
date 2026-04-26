"""Markdown to HTML converter for report download.

Converts a Markdown report to a self-contained HTML file with inline CSS.
The HTML can be opened in a browser and printed to PDF via Ctrl+P →
"Save as PDF". Works reliably with Cyrillic text without external dependencies.
"""

import re


CSS = """\
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #1a1a1a;
    max-width: 800px;
    margin: 40px auto;
    padding: 0 20px;
}
h1 {
    font-size: 22px;
    text-align: center;
    margin-bottom: 24px;
    color: #111;
    border-bottom: 2px solid #333;
    padding-bottom: 12px;
}
h2 {
    font-size: 18px;
    margin-top: 28px;
    margin-bottom: 10px;
    color: #222;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
}
h3 {
    font-size: 15px;
    margin-top: 16px;
    margin-bottom: 8px;
    color: #333;
}
ul {
    margin-left: 24px;
    margin-top: 6px;
    margin-bottom: 6px;
    padding-left: 0;
}
li {
    margin-bottom: 4px;
}
em {
    color: #2c5282;
}
hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 20px 0;
}
@media print {
    body { margin: 0; max-width: 100%; }
    h2 { page-break-before: auto; }
}
"""


def _md_line_to_html(line: str) -> str:
    """Convert a single Markdown line to its HTML equivalent.

    Handles headings (h1–h3), horizontal rules, list items, and plain
    paragraphs. Inline formatting is applied via :func:`_inline_format`.

    Args:
        line: A single line of Markdown text.

    Returns:
        Corresponding HTML string, or an empty string for blank lines.
    """
    stripped = line.strip()

    if not stripped:
        return ""

    if stripped.startswith("### "):
        return f"<h3>{_inline_format(stripped[4:])}</h3>"
    if stripped.startswith("## "):
        return f"<h2>{_inline_format(stripped[3:])}</h2>"
    if stripped.startswith("# "):
        return f"<h1>{_inline_format(stripped[2:])}</h1>"

    if stripped.startswith("---"):
        return "<hr>"

    if stripped.startswith("- "):
        return f"<li>{_inline_format(stripped[2:])}</li>"

    return f"<p>{_inline_format(stripped)}</p>"


def _inline_format(text: str) -> str:
    """Convert Markdown inline markers to HTML tags.

    Handles ``**bold**`` → ``<strong>`` and ``*italic*`` → ``<em>``.

    Args:
        text: Inline Markdown text.

    Returns:
        Text with bold and italic markers replaced by HTML tags.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def markdown_to_html_bytes(md_text: str) -> bytes:
    """Convert a Markdown report to a UTF-8 encoded HTML document.

    Wraps the converted content in a full HTML page with inline CSS
    ready for browser printing. List items are automatically grouped
    inside ``<ul>`` tags.

    Args:
        md_text: Full Markdown report text.

    Returns:
        UTF-8 encoded HTML bytes suitable for a Streamlit download button.
    """
    lines = md_text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        html = _md_line_to_html(line)

        if html.startswith("<li>"):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(html)
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if html:
                html_lines.append(html)

    if in_list:
        html_lines.append("</ul>")

    body = "\n".join(html_lines)

    full_html = f"""\
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="utf-8">
    <title>Фінансовий звіт</title>
    <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    return full_html.encode("utf-8")
