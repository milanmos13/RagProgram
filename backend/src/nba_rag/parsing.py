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
