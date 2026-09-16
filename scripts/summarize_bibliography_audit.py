from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for record in data["records"]:
        metadata = record.get("metadata", {})
        source = record.get("metadata_source", "")
        if source.startswith("Crossref"):
            meta_title = (metadata.get("title") or [""])[0]
            parts = (metadata.get("published") or {}).get("date-parts") or [[""]]
            meta_year = str(parts[0][0])
            meta_doi = metadata.get("DOI", "")
            meta_journal = (metadata.get("container-title") or [""])[0]
            meta_volume = str(metadata.get("volume", ""))
            meta_issue = str(metadata.get("issue", ""))
            meta_pages = str(metadata.get("page", "") or metadata.get("article-number", ""))
            meta_authors = "; ".join(
                " ".join(filter(None, (person.get("given", ""), person.get("family", ""))))
                for person in metadata.get("author", [])
            )
        elif source == "DataCite":
            meta_title = (metadata.get("titles") or [{}])[0].get("title", "")
            meta_year = str(metadata.get("publicationYear", ""))
            meta_doi = metadata.get("doi", "")
            meta_journal = metadata.get("publisher", "")
            meta_volume = meta_issue = meta_pages = ""
            meta_authors = "; ".join(
                creator.get("name", "") for creator in metadata.get("creators", [])
            )
        else:
            meta_title = meta_year = meta_doi = meta_journal = ""
            meta_volume = meta_issue = meta_pages = meta_authors = ""
        bib = record["bib"]
        print(
            "|".join(
                (
                    record["key"],
                    source,
                    f"title={norm(bib['title']) == norm(meta_title)}",
                    f"year={bib['year']}->{meta_year}",
                    f"doi={bib['doi']}->{meta_doi}",
                    f"journal={bib['journal'] or bib['publisher']}->{meta_journal}",
                    f"volume={bib['volume']}->{meta_volume}",
                    f"issue={bib['number']}->{meta_issue}",
                    f"pages={bib['pages']}->{meta_pages}",
                    f"authors={bib['author']}->{meta_authors}",
                    f"metadata_title={meta_title}",
                )
            )
        )


if __name__ == "__main__":
    main()
