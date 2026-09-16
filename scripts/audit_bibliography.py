from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path


ENTRY_START = re.compile(r"(?m)^\s*@([A-Za-z]+)\s*\{\s*([^,]+)\s*,")
FIELD_START = re.compile(r"(?m)^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*")
CITE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")


def balanced_block(text: str, start: int) -> tuple[str, int]:
    depth = 0
    quote = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quote = not quote
            continue
        if quote:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
    raise ValueError(f"Unbalanced BibTeX entry beginning at offset {start}")


def parse_entries(text: str) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for match in ENTRY_START.finditer(text):
        opening = text.find("{", match.start())
        raw, end = balanced_block(text, opening)
        entry_text = text[match.start() : end]
        fields: dict[str, str] = {}
        body_offset = entry_text.find(",") + 1
        body = entry_text[body_offset:-1]
        starts = list(FIELD_START.finditer(body))
        for i, field_match in enumerate(starts):
            value_start = field_match.end()
            value_end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
            value = body[value_start:value_end].strip().rstrip(",").strip()
            while len(value) >= 2 and (
                (value[0] == "{" and value[-1] == "}")
                or (value[0] == '"' and value[-1] == '"')
            ):
                value = value[1:-1].strip()
            fields[field_match.group(1).lower()] = value
        entries[match.group(2).strip()] = {
            "type": match.group(1).lower(),
            "fields": fields,
            "raw": entry_text.strip() + "\n",
        }
    return entries


def latex_plain(value: str) -> str:
    replacements = {
        "--": "-",
        "~": " ",
        "\\&": "&",
        "\\aa": "a",
        "\\AA": "A",
        "\\o": "o",
        "\\O": "O",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\\['\"`^~=Hckruv]\s*\{?([A-Za-z])\}?", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", latex_plain(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def request_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "JOG-bibliography-audit/1.0 (mailto:literature-audit@example.invalid)",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def crossref_candidate(fields: dict[str, str]) -> dict | None:
    title = latex_plain(fields.get("title", ""))
    if not title:
        return None
    params = {"query.title": title, "rows": 3, "select": "DOI,title,author,published,container-title,volume,issue,page,type"}
    author = latex_plain(fields.get("author", "")).split(" and ")[0]
    if author:
        params["query.author"] = author
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    items = request_json(url).get("message", {}).get("items", [])
    if not items:
        return None
    wanted = normalize(title)
    ranked = sorted(
        items,
        key=lambda item: len(set(wanted) ^ set(normalize((item.get("title") or [""])[0]))),
    )
    return ranked[0]


def doi_metadata(doi: str) -> tuple[str, dict] | None:
    encoded = urllib.parse.quote(doi, safe="")
    try:
        data = request_json(f"https://api.crossref.org/works/{encoded}")
        return "Crossref", data["message"]
    except Exception:
        pass
    try:
        data = request_json(f"https://api.datacite.org/dois/{encoded}")
        attributes = data["data"]["attributes"]
        return "DataCite", attributes
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    tex = args.tex.read_text(encoding="utf-8")
    bib_text = args.bib.read_text(encoding="utf-8")
    entries = parse_entries(bib_text)
    cited = sorted(
        {
            key.strip()
            for group in CITE.findall(tex)
            for key in group.split(",")
            if key.strip()
        }
    )
    missing = sorted(set(cited) - set(entries))
    unused = sorted(set(entries) - set(cited))
    records = []
    for index, key in enumerate(cited, start=1):
        entry = entries.get(key)
        if entry is None:
            records.append({"key": key, "status": "missing_from_bib"})
            continue
        fields = entry["fields"]
        record: dict[str, object] = {
            "key": key,
            "entry_type": entry["type"],
            "bib": {name: latex_plain(str(fields.get(name, ""))) for name in ("author", "title", "journal", "booktitle", "publisher", "year", "volume", "number", "pages", "doi", "url")},
        }
        if args.fetch:
            doi = latex_plain(str(fields.get("doi", ""))).strip()
            try:
                result = doi_metadata(doi) if doi else None
                if result is None:
                    candidate = crossref_candidate(fields)
                    if candidate:
                        record["metadata_source"] = "Crossref title query"
                        record["metadata"] = candidate
                    else:
                        record["metadata_source"] = "unresolved"
                else:
                    record["metadata_source"], record["metadata"] = result
            except Exception as exc:
                record["metadata_source"] = "error"
                record["metadata_error"] = repr(exc)
            print(f"Checked {index}/{len(cited)}: {key}", flush=True)
        records.append(record)

    report = {
        "tex": str(args.tex),
        "bib": str(args.bib),
        "cited_count": len(cited),
        "bib_entry_count": len(entries),
        "missing_cited_keys": missing,
        "unused_keys": unused,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
