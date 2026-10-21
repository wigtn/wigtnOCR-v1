"""Element extractor for markdown predictions.

Parses VLM-generated markdown into structured elements (text, tables, formulas)
for comparison against OmniDocBench GT annotations.

Based on OmniDocBench's extract.py (md_tex_filter).
"""

import re
from dataclasses import dataclass, field


@dataclass
class PredElement:
    """A structural element extracted from prediction markdown."""
    category: str          # text, table, formula, code
    content: str           # raw content
    position: tuple[int, int] = (0, 0)  # (start, end) in original text
    order: int = 0         # sequential order in document


def extract_elements_from_markdown(content: str) -> dict[str, list[PredElement]]:
    """Extract structured elements from prediction markdown.

    Splits markdown into text blocks, tables (HTML/markdown), formulas,
    and code blocks. Follows OmniDocBench's md_tex_filter() protocol.

    Args:
        content: Full page markdown content.

    Returns:
        Dict mapping category to list of PredElement.
    """
    # Pre-processing
    content = _remove_images(content)
    content = _remove_fences(content)
    content = _limit_repeated_chars(content, max_repeat=4)

    elements: dict[str, list[PredElement]] = {
        "text": [],
        "html_table": [],
        "formula": [],
        "code": [],
    }

    # Track positions that have been consumed
    consumed = set()
    order = 0

    # 1. Extract HTML tables
    for m in re.finditer(r"<table[\s\S]*?</table>", content, re.IGNORECASE):
        elements["html_table"].append(PredElement(
            category="html_table",
            content=m.group(),
            position=(m.start(), m.end()),
            order=order,
        ))
        consumed.update(range(m.start(), m.end()))
        order += 1

    # 2. Extract display formulas ($$...$$, \[...\])
    formula_patterns = [
        r"\$\$([\s\S]*?)\$\$",
        r"\\\[([\s\S]*?)\\\]",
    ]
    for pattern in formula_patterns:
        for m in re.finditer(pattern, content):
            if m.start() not in consumed:
                elements["formula"].append(PredElement(
                    category="formula",
                    content=m.group(1).strip() if m.group(1) else m.group().strip(),
                    position=(m.start(), m.end()),
                    order=order,
                ))
                consumed.update(range(m.start(), m.end()))
                order += 1

    # 3. Extract markdown tables (|...|)
    md_table_blocks = _extract_markdown_tables(content)
    for start, end, table_content in md_table_blocks:
        if start not in consumed:
            # Convert to HTML for TEDS comparison
            html = _markdown_table_to_html(table_content)
            if html:
                elements["html_table"].append(PredElement(
                    category="html_table",
                    content=html,
                    position=(start, end),
                    order=order,
                ))
                consumed.update(range(start, end))
                order += 1

    # 4. Extract code blocks
    for m in re.finditer(r"```(\w*)\n([\s\S]*?)```", content):
        if m.start() not in consumed:
            elements["code"].append(PredElement(
                category="code",
                content=m.group(2).strip(),
                position=(m.start(), m.end()),
                order=order,
            ))
            consumed.update(range(m.start(), m.end()))
            order += 1

    # 5. Remaining text blocks (split by double newline)
    text_content = list(content)
    for idx in consumed:
        if idx < len(text_content):
            text_content[idx] = " "
    remaining = "".join(text_content)

    for block in re.split(r"\n\n+", remaining):
        block = block.strip()
        if block and len(block) > 1:
            # Find approximate position
            pos = content.find(block[:20]) if len(block) >= 20 else content.find(block)
            pos = max(pos, 0)
            elements["text"].append(PredElement(
                category="text",
                content=block,
                position=(pos, pos + len(block)),
                order=order,
            ))
            order += 1

    return elements


# ---------------------------------------------------------------------------
# GT element extraction from OmniDocBench JSON
# ---------------------------------------------------------------------------

@dataclass
class GTElement:
    """A ground truth element from OmniDocBench JSON."""
    category: str          # text_block, title, table, equation_isolated, etc.
    content: str           # text content
    html: str = ""         # HTML content (for tables)
    latex: str = ""        # LaTeX content (for formulas)
    order: int = 0         # reading order
    anno_id: int = 0
    attributes: dict = field(default_factory=dict)


def extract_gt_elements(page_data: dict) -> dict[str, list[GTElement]]:
    """Extract GT elements from OmniDocBench page JSON.

    Handles truncated text block merging.

    Args:
        page_data: Single page from OmniDocBench.json

    Returns:
        Dict mapping category group to list of GTElement.
    """
    layout_dets = page_data.get("layout_dets", [])
    extra = page_data.get("extra", {})

    # Handle truncated text merging
    truncated_ids = set()
    merge_groups = []
    for rel in extra.get("relation", []):
        if rel.get("relation_type") == "truncated":
            src, tgt = rel["source_anno_id"], rel["target_anno_id"]
            truncated_ids.add(src)
            truncated_ids.add(tgt)
            # Merge into existing group or create new
            merged = False
            for group in merge_groups:
                if src in group or tgt in group:
                    group.add(src)
                    group.add(tgt)
                    merged = True
                    break
            if not merged:
                merge_groups.append({src, tgt})

    # Build element dict, excluding truncated
    elements_by_id = {}
    normal_elements = []
    for item in layout_dets:
        aid = item.get("anno_id", -1)
        elements_by_id[aid] = item
        if aid not in truncated_ids:
            normal_elements.append(item)

    # Merge truncated groups
    merged_elements = []
    for group in merge_groups:
        blocks = [elements_by_id[aid] for aid in group if aid in elements_by_id]
        blocks.sort(key=lambda x: x.get("order", 0))
        merged_text = "".join(b.get("text", "") for b in blocks)
        merged_el = {
            "category_type": blocks[0].get("category_type", "text_block"),
            "text": merged_text,
            "order": blocks[0].get("order", 0),
            "anno_id": blocks[0].get("anno_id", 0),
            "attribute": blocks[0].get("attribute", {}),
            "html": blocks[0].get("html", ""),
            "latex": blocks[0].get("latex", ""),
        }
        merged_elements.append(merged_el)

    all_items = normal_elements + merged_elements

    # Group by category
    result: dict[str, list[GTElement]] = {
        "text": [],
        "table": [],
        "formula": [],
        "reading_order_items": [],  # all matchable items for reading order
    }

    text_categories = {
        "text_block", "title", "code_txt", "code_txt_caption", "reference",
    }
    ignore_categories = {
        "figure_caption", "figure_footnote", "table_caption", "table_footnote",
        "header", "footer", "page_footnote", "page_number",
        "equation_caption", "code_algorithm", "code_algorithm_caption",
    }
    formula_categories = {"equation_isolated"}

    for item in sorted(all_items, key=lambda x: x.get("order") or 0):
        cat = item.get("category_type", "")
        gt_el = GTElement(
            category=cat,
            content=item.get("text", ""),
            html=item.get("html", ""),
            latex=item.get("latex", ""),
            order=item.get("order", 0),
            anno_id=item.get("anno_id", 0),
            attributes=item.get("attribute", {}),
        )

        if cat == "table":
            result["table"].append(gt_el)
        elif cat in formula_categories:
            result["formula"].append(gt_el)
            result["reading_order_items"].append(gt_el)
        elif cat in text_categories:
            result["text"].append(gt_el)
            result["reading_order_items"].append(gt_el)
        elif cat not in ignore_categories:
            # Unknown category → treat as text
            result["text"].append(gt_el)
            result["reading_order_items"].append(gt_el)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMG_PATTERN = re.compile(r"!\[.*?\]\(.*?\)|<img[^>]*>", re.IGNORECASE)


def _remove_images(content: str) -> str:
    return _IMG_PATTERN.sub("", content)


def _remove_fences(content: str) -> str:
    """Remove outermost markdown fence if the entire content is wrapped."""
    content = content.strip()
    if content.startswith("```markdown"):
        content = content[len("```markdown"):]
        if content.endswith("```"):
            content = content[:-3]
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3]
    return content.strip()


def _limit_repeated_chars(content: str, max_repeat: int = 4) -> str:
    """Limit consecutive repeated characters."""
    return re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, content)


def _extract_markdown_tables(content: str) -> list[tuple[int, int, str]]:
    """Find markdown table blocks (consecutive lines starting with |)."""
    tables = []
    lines = content.split("\n")
    i = 0
    pos = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines = []
            start_pos = pos
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                pos += len(lines[i]) + 1
                i += 1
            if len(table_lines) >= 2:  # Need at least header + separator
                tables.append((start_pos, pos, "\n".join(table_lines)))
        else:
            pos += len(line) + 1
            i += 1

    return tables


def _markdown_table_to_html(md_table: str) -> str:
    """Convert markdown table to HTML."""
    lines = [l.strip() for l in md_table.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return ""

    rows = []
    for line in lines:
        if re.match(r"^\|[\s\-:]+\|$", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    parts = ["<table>"]
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)
