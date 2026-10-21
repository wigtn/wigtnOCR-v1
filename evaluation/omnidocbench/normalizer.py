"""Text normalization for OmniDocBench evaluation.

Normalizes text, formulas, and tables before metric computation.
Aligned with OmniDocBench's data_preprocess.py for identical evaluation results.

Reference: OmniDocBench (Ouyang et al., CVPR 2025)
"""

import html as html_lib
import re
import unicodedata

from pylatexenc.latex2text import LatexNodes2Text


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Inline formula regex (same as OmniDocBench's inline_reg)
_INLINE_RE = re.compile(
    r"\$(.*?)\$|"
    r"\\\((.*?)\\\)",
)


def textblock2unicode(text: str) -> str:
    """Convert inline LaTeX in text blocks to Unicode characters.

    Matches OmniDocBench's textblock2unicode() exactly:
    - Finds inline formulas ($...$ or \\(...\\))
    - If they contain \\, ^, or _, converts via pylatexenc
    - Replaces the formula span with the Unicode result

    Args:
        text: Text block potentially containing inline LaTeX.

    Returns:
        Text with inline LaTeX converted to Unicode.
    """
    inline_matches = _INLINE_RE.finditer(text)
    removal_positions = []

    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)

        # Remove escape characters
        clean_content = re.sub(r"\\([\\_&%^])", "", content)

        try:
            if any(char in clean_content for char in r"\^_"):
                if clean_content.endswith("\\"):
                    clean_content += " "
                unicode_content = LatexNodes2Text().latex_to_text(clean_content)
                removal_positions.append((position[0], position[1], unicode_content))
        except Exception:
            continue

    # Replace inline formulas in reverse order to preserve positions
    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]

    return text


def clean_string(text: str) -> str:
    """Normalize text to alphanumeric + CJK characters only.

    Identical to OmniDocBench's clean_string():
    - Removes escape sequences (\\t, \\n, /t, /n)
    - Removes whitespace and actual tab/newline
    - Keeps only word characters and CJK range
    """
    text = (text
            .replace("\\t", "").replace("\\n", "")
            .replace("\t", "").replace("\n", "")
            .replace("/t", "").replace("/n", ""))
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text


def normalize_unicode(text: str) -> str:
    """NFKC normalize for consistent comparison."""
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# Formula normalization
# ---------------------------------------------------------------------------

# Identical to OmniDocBench's normalized_formula() filter_list
_FORMULA_STRIP = [
    r"\mathbf", r"\mathrm", r"\mathnormal", r"\mathit", r"\mathbb",
    r"\mathcal", r"\mathscr", r"\mathfrak", r"\mathsf", r"\mathtt",
    r"\textbf", r"\text", r"\boldmath", r"\boldsymbol", r"\operatorname", r"\bm",
    r"\symbfit", r"\mathbfcal", r"\symbf", r"\scriptscriptstyle", r"\notag",
    r"\setlength", r"\coloneqq", r"\space", r"\thickspace", r"\thinspace",
    r"\medspace", r"\nobreakspace", r"\negmedspace",
    r"\quad", r"\qquad", r"\enspace", r"\substackw",
    " ", "$$", r"\left", r"\right", r"\displaystyle", r"\text",
]

_FORMULA_REMOVE_PATTERNS = [
    (r"\\tag\{.*?\}", ""),
    (r"\\hspace\{.*?\}", ""),
    (r"\\begin\{.*?\}", ""),
    (r"\\end\{.*?\}", ""),
    (r"\\arraycolsep.*?\}", ""),
]


def normalize_formula(text: str) -> str:
    """Normalize LaTeX formula for comparison.

    Identical to OmniDocBench's normalized_formula():
    - Strip $ delimiters and \\[...\\] wrappers
    - Remove \\tag{}, \\hspace{}, \\begin{}, \\end{}, \\arraycolsep
    - Strip trailing dot
    - Remove all styling commands from filter_list
    - Lowercase
    """
    # Delimiter filter (matches OmniDocBench exactly)
    text = text.strip().strip("$").strip("\n")

    # Extract content from \[...\] wrapper
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    match = pattern.search(text)
    if match:
        text = match.group(1).strip()

    # Remove patterns
    for pat, repl in _FORMULA_REMOVE_PATTERNS:
        text = re.sub(pat, repl, text)

    # Strip trailing dot
    text = text.strip(".")

    # Remove styling commands
    for cmd in _FORMULA_STRIP:
        text = text.replace(cmd, "")

    text = text.lower()
    return text


# ---------------------------------------------------------------------------
# Table normalization
# ---------------------------------------------------------------------------

def normalize_html_table(html_str: str) -> str:
    """Normalize HTML table for TEDS comparison.

    Aligned with OmniDocBench's normalized_html_table():
    - Converts <th> to <td>, unwraps <thead>
    - Replaces <math> tags with alttext
    - Unwraps <span> tags
    - html.unescape + NFKC normalization
    - Removes styling attributes
    - Removes <tbody>, <sup>, <sub>, <div>, <p>, <colgroup> tags
    - Wraps in standard HTML structure with border="1"
    """
    from bs4 import BeautifulSoup

    # --- Phase 1: OmniDocBench's process_table_html() ---
    soup = BeautifulSoup(html_str, "html.parser")

    # Convert th → td
    for th in soup.find_all("th"):
        th.name = "td"

    # Unwrap thead
    for thead in soup.find_all("thead"):
        thead.unwrap()

    # Replace <math> tags with alttext (OmniDocBench specific)
    for math_tag in soup.find_all("math"):
        alttext = math_tag.get("alttext", "")
        if alttext:
            alttext = f"${alttext}$"
        math_tag.replace_with(alttext)

    # Unwrap span
    for span in soup.find_all("span"):
        span.unwrap()

    processed = str(soup)

    # --- Phase 2: OmniDocBench's main normalization ---
    # html.unescape + NFKC
    table_res = html_lib.unescape(processed).replace("\n", "")
    table_res = unicodedata.normalize("NFKC", table_res).strip()

    # Extract inner table content
    inner = _inner_table(table_res)

    # Remove styling attributes
    inner = re.sub(r'( style=".*?")', "", inner)
    inner = re.sub(r'( height=".*?")', "", inner)
    inner = re.sub(r'( width=".*?")', "", inner)
    inner = re.sub(r'( align=".*?")', "", inner)
    inner = re.sub(r'( class=".*?")', "", inner)
    inner = re.sub(r"</?tbody>", "", inner)

    # Normalize whitespace
    inner = re.sub(r"\s+", " ", inner)

    # --- Phase 3: OmniDocBench's clean_table() ---
    inner = (inner
             .replace("<sup>", "").replace("</sup>", "")
             .replace("<sub>", "").replace("</sub>", "")
             .replace("<span>", "").replace("</span>", "")
             .replace("<div>", "").replace("</div>", "")
             .replace("<p>", "").replace("</p>", "")
             .replace('<spandata-span-identity="">', ""))
    inner = re.sub(r"<colgroup>.*?</colgroup>", "", inner)

    return f'<html><body><table border="1" >{inner}</table></body></html>'


def _inner_table(html_str: str) -> str:
    """Extract inner content of <table> tags."""
    match = re.search(r"<table\b[^>]*>(.*)</table>", html_str, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else html_str


# ---------------------------------------------------------------------------
# Markdown table → HTML conversion
# ---------------------------------------------------------------------------

def markdown_table_to_html(md_table: str) -> str:
    """Convert a markdown table string to HTML table.

    Args:
        md_table: Markdown table with | delimiters and |---| separator.

    Returns:
        HTML table string.
    """
    lines = [l.strip() for l in md_table.strip().split("\n") if l.strip()]
    if not lines:
        return ""

    rows = []
    for line in lines:
        # Skip separator row
        if re.match(r"^\|[\s\-:]+\|$", line):
            continue
        # Parse cells
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last from leading/trailing |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    html_parts = ["<table>"]
    for row in rows:
        html_parts.append("<tr>")
        for cell in row:
            html_parts.append(f"<td>{cell}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")

    return "".join(html_parts)
