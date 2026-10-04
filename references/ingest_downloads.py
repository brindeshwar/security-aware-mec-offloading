"""Add PDFs you downloaded by hand to the reference collection.

    python ingest_downloads.py <folder with PDFs>

Each PDF is matched to a cited paper by comparing the title words with its first two pages, then
  - filed under open_access/  if the publisher-hosted version carries a Creative Commons licence (OpenAlex),
  - filed under local_only/   otherwise (free to read, not redistributed, gitignored),
and renamed  <number>_<Title>.pdf.  index.json is updated; run build_index.py afterwards.
All PDFs already in the collection are renamed to the same scheme.
"""
import json
import os
import re
import shutil
import sys
import unicodedata
import urllib.parse

import pymupdf

import fetch_references as F

HERE = F.HERE
IDX = os.path.join(HERE, "index.json")


def norm(s):
    """Lower-case letters and digits only, with ligatures expanded (fi, ffl, ...)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def page_text(path):
    d = pymupdf.open(path)
    return " ".join(" ".join(d[i].get_text().split()) for i in range(min(2, len(d)))), len(d)


def licence_of_published_version(b):
    try:
        base = "https://api.openalex.org/works"
        work = F.get(f"{base}/doi:{urllib.parse.quote(b['doi'])}") if b["doi"] else None
    except Exception:
        return ""
    for loc in (work or {}).get("locations") or []:
        if loc and loc.get("version") == "publishedVersion" and (loc.get("license") or "").lower() in F.REDISTRIBUTABLE:
            return loc["license"].lower()
    return ""


def main(src):
    rows = json.load(open(IDX, encoding="utf-8"))
    by_key = {r["key"]: r for r in rows}
    unmatched = []
    for name in sorted(os.listdir(src)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(src, name)
        txt, pages = page_text(path)
        tk = F.tokens(txt)
        norm_txt = norm(txt)

        def score_of(r):
            if r["doi"] and norm(r["doi"]) in norm_txt:
                return 3.0                                   # the paper's DOI is printed in the PDF
            if norm(r["title"]) in norm_txt:
                return 2.0                                   # the exact title is on the first pages
            return len(F.tokens(r["title"]) & tk) / max(len(F.tokens(r["title"])), 1)
        scored = sorted(((score_of(r), r) for r in rows), key=lambda x: -x[0])
        score, r = scored[0]
        runner = scored[1][0]
        if score < 0.6 or (score < 2 and score - runner < 0.05):
            unmatched.append((name, round(score, 2), r["title"][:60]))
            continue
        lic = licence_of_published_version(r)
        folder = "open_access" if lic else "local_only"
        old_dst = os.path.join(HERE, r["folder"], r["stem"] + ".pdf") if r["status"] == "ok" else None
        dst = os.path.join(HERE, folder, r["stem"] + ".pdf")
        if old_dst and os.path.exists(old_dst) and old_dst != dst:
            os.remove(old_dst)      # the downloaded publisher PDF replaces a preprint/other copy
        shutil.move(path, dst)
        if lic in F.RECOMPRESS_OK:
            F.shrink(dst)
        r.update(status="ok", folder=folder, license=lic or "none stated", source="downloaded by the author",
                 size_kb=round(os.path.getsize(dst) / 1024), pages=pages, note="")
        print(f"{name}  ->  {folder}/{r['stem']}.pdf  (match {score:.2f}, licence {lic or 'none'})")
    # rename everything already collected to the title-based scheme
    for r in rows:
        new = F.file_stem(r["n"], r["title"])
        if r["status"] == "ok" and r["stem"] != new:
            old = os.path.join(HERE, r["folder"], r["stem"] + ".pdf")
            if os.path.exists(old):
                os.replace(old, os.path.join(HERE, r["folder"], new + ".pdf"))
        r["stem"] = new
    json.dump(rows, open(IDX, "w", encoding="utf-8"), indent=1)
    for u in unmatched:
        print("NOT MATCHED:", u)
    print("collected:", sum(r["status"] == "ok" for r in rows), "of", len(rows))


if __name__ == "__main__":
    main(sys.argv[1])
