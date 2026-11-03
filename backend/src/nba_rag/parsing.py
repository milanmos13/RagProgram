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
