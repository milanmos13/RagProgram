"""Extract prose from an Article's Parsoid HTML, grouped by section."""

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

# Sections that hold boilerplate rather than Article content.
BOILERPLATE_SECTIONS = {
    "See also",
    "Notes",
    "References",
    "Citations",
    "Sources",
    "Bibliography",
    "Further reading",
    "External links",
}

# Marks where an infobox value has line breaks or list items; newlines in the source do not.
_BREAK = "\x1f"

HEADINGS = ("h2", "h3", "h4", "h5", "h6")


@dataclass(frozen=True)
class Paragraph:
    section_path: tuple[str, ...]
    text: str


def extract_prose(html: str) -> list[Paragraph]:
    """Return the Article's prose paragraphs in reading order.

    Only paragraphs that sit directly in a section count as prose, so text inside
    infoboxes, tables and navigation boxes is skipped. Reference markers are removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    for ref in soup.select("sup.mw-ref, sup.reference, style"):
        ref.decompose()

    paragraphs: list[Paragraph] = []
    for section in soup.find_all("section"):
        path = _section_path(section)
        if any(title in BOILERPLATE_SECTIONS for title in path):
            continue
        for p in section.find_all("p", recursive=False):
            text = " ".join(p.get_text(" ").split())
            text = _tidy_punctuation(text)
            if text:
                paragraphs.append(Paragraph(section_path=path, text=text))
    return paragraphs


def extract_infobox(html: str) -> list[str]:
    """Return the Article's infobox as ``Field: value`` lines, in reading order.

    Rows without both a label and a value (section headers, images, captions) are skipped.
    Reference markers are removed; list items and line breaks inside a value become ``; ``.
    """
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.select_one("table.infobox")
    if infobox is None:
        return []
    for ref in infobox.select("sup.mw-ref, sup.reference, style"):
        ref.decompose()

    lines: list[str] = []
    for row in infobox.find_all("tr"):
        if row.find_parent("table") is not infobox:
            continue  # a nested table's rows are already part of its cell's text
        label = row.find("th", class_="infobox-label")
        data = row.find("td", class_="infobox-data")
        if label is None or data is None:
            continue
        for br in data.find_all("br"):
            br.replace_with(_BREAK)
        for item in data.find_all("li"):
            item.insert_before(_BREAK)
        field = " ".join(label.get_text(" ").split())
        parts = (" ".join(part.split()) for part in data.get_text(" ").split(_BREAK))
        value = "; ".join(_tidy_punctuation(part) for part in parts if part)
        if field and value:
            lines.append(f"{field}: {value}")
    return lines


@dataclass(frozen=True)
class Table:
    section_path: tuple[str, ...]
    caption: str
    lines: list[str]


def extract_tables(html: str) -> list[Table]:
    """Return the Article's data tables, each row as one ``Header: value | ...`` line.

    Only ``wikitable`` tables with a header row count; layout tables have nothing to
    repeat. Header rows are the leading rows made only of ``th`` cells, and a column with
    several header rows gets them joined (``Score Cavaliers``). Merged cells (``rowspan``
    and ``colspan``) are repeated in every column and row they cover. Empty cells are
    skipped and reference markers removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    for ref in soup.select("sup.mw-ref, sup.reference, style"):
        ref.decompose()

    tables: list[Table] = []
    for table in soup.select("table.wikitable"):
        if table.find_parent("table") is not None:
            continue  # a nested table's text is already part of its cell
        grid, header_count = _table_grid(table)
        if header_count == 0:
            continue
        headers = [
            " ".join(_dedupe_runs(cells)) for cells in zip(*grid[:header_count], strict=False)
        ]
        rows = []
        for row in grid[header_count:]:
            pairs = [
                f"{header}: {value}" if header else value
                for header, value in zip(headers, row, strict=False)
                if value
            ]
            if pairs:
                rows.append(" | ".join(pairs))
        if not rows:
            continue
        section = table.find_parent("section")
        caption = table.find("caption")
        tables.append(
            Table(
                section_path=_section_path(section) if section else (),
                caption=_tidy_punctuation(" ".join(caption.get_text(" ").split()))
                if caption
                else "",
                lines=rows,
            )
        )
    return tables


def _table_grid(table: Tag) -> tuple[list[list[str]], int]:
    """Expand a table into a rectangular grid of cell texts, and count its header rows."""
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}  # (row, column) covered by an earlier rowspan
    header_count = 0
    in_header = True
    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    for r, tr in enumerate(rows):
        cells = tr.find_all(["th", "td"], recursive=False)
        if in_header and cells and all(cell.name == "th" for cell in cells):
            header_count += 1
        else:
            in_header = False
        row: list[str] = []
        for cell in cells:
            _take_covered(row, pending, r)
            text = _tidy_punctuation(" ".join(cell.get_text(" ").split()))
            colspan = _span(cell, "colspan")
            rowspan = _span(cell, "rowspan")
            for _ in range(colspan):
                for extra in range(1, rowspan):
                    pending[(r + extra, len(row))] = text
                row.append(text)
        _take_covered(row, pending, r)
        grid.append(row)
    return grid, header_count


def _take_covered(row: list[str], pending: dict[tuple[int, int], str], r: int) -> None:
    while (r, len(row)) in pending:
        row.append(pending.pop((r, len(row))))


def _span(cell: Tag, attribute: str) -> int:
    value = str(cell.get(attribute, "1"))
    return max(1, int(value)) if value.isdigit() else 1


def _dedupe_runs(values: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and (not result or result[-1] != value):
            result.append(value)
    return result


def _section_path(section: Tag) -> tuple[str, ...]:
    titles = []
    for node in [section, *section.find_parents("section")]:
        heading = node.find(HEADINGS, recursive=False)
        if heading is not None:
            titles.append(" ".join(heading.get_text(" ").split()))
    return tuple(reversed(titles))


def _tidy_punctuation(text: str) -> str:
    # get_text(" ") puts a space between inline elements, which strands punctuation.
    for mark in (".", ",", ";", ":", ")", "!", "?"):
        text = text.replace(f" {mark}", mark)
    return text.replace("( ", "(")
