from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request


query = " ".join(sys.argv[1:])
url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
    {"query.bibliographic": query, "rows": 20}
)
request = urllib.request.Request(
    url,
    headers={"User-Agent": "JOG-bibliography-audit/1.0"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    items = json.load(response)["message"]["items"]
for item in items:
    title = (item.get("title") or [""])[0]
    container = (item.get("container-title") or [""])[0]
    date = (item.get("published") or {}).get("date-parts", [[""]])[0]
    print(
        json.dumps(
            {
                "score": item.get("score"),
                "title": title,
                "author": item.get("author"),
                "container": container,
                "published": date,
                "volume": item.get("volume"),
                "issue": item.get("issue"),
                "page": item.get("page"),
                "article_number": item.get("article-number"),
                "DOI": item.get("DOI"),
                "type": item.get("type"),
            },
            ensure_ascii=False,
        )
    )
