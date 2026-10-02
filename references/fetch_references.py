"""Collect the PDFs of the papers cited in ../paper/paper.tex from legal open-access locations.

    python fetch_references.py

For every cited key in refs.bib the script tries, in order:
  1. a PDF the author already owns (LOCAL_FILES),
  2. open-access PDF locations listed by OpenAlex (api.openalex.org),
  3. the OpenAlex open-access landing URL, Semantic Scholar's openAccessPdf, an arXiv preprint found by title,
     and the publisher's direct PDF link for Nature/Scientific Reports.
Each download must be a PDF whose first pages contain the title words. The licence is taken from the copy that was
actually downloaded: an explicit Creative Commons licence -> open_access/ (redistribution allowed, tracked in git);
anything else -> local_only/ (free to read, NOT redistributed, gitignored). Papers that are paywalled everywhere are
reported in MISSING.md with their DOI. No shadow libraries are used.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "application/pdf,application/json;q=0.9,*/*;q=0.8"}
LOCAL_FILES = {   # key -> path of a PDF the author already has
    "lin2019": r"D:\Major\IEEE paper on computation offloading.pdf",
    "kiran2020": r"D:\Major\Joint_resource_allocation_and_computation_offloadi.pdf",
    "kovachev2012": r"D:\Major\Dialnet-FrameworkForComputationOffloadingInMobileCloudComp-4114048.pdf",
    "barbarossa2013": r"D:\Major\Funems13.pdf",
    "akherfi2018": r"D:\Major\1488793.pdf",
}
REDISTRIBUTABLE = ("cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd", "cc0", "public-domain")
RECOMPRESS_OK = ("cc-by", "cc0", "public-domain")      # never alter ND/SA/NC copies


def get(url, binary=False, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=timeout).read()
    return data if binary else json.loads(data)


def _field(body, name):
    """Value of `name = {...}` with balanced braces."""
    m = re.search(name + r"\s*=\s*\{", body)
    if not m:
        return ""
    i, depth = m.end(), 1
    while i < len(body) and depth:
        depth += {"{": 1, "}": -1}.get(body[i], 0)
        i += 1
    return body[m.end():i - 1].strip()


def parse_bib():
    txt = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()
    out = {}
    for chunk in re.split(r"\n(?=@)", txt):
        m = re.match(r"@(\w+)\{(\w+),", chunk.strip())
        if not m:
            continue
        f = lambda k: _field(chunk, k)
        out[m.group(2)] = {"title": re.sub(r"[{}\\]", "", f("title")), "author": f("author"), "year": f("year"),
                           "doi": f("doi"), "venue": f("journal") or f("booktitle") or f("howpublished")}
    return out


def citation_order():
    tex = open(os.path.join(PAPER, "paper.tex"), encoding="utf-8").read()
    order = []
    for m in re.finditer(re.escape(chr(92) + "cite{") + r"([^}]*)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    return order


def tokens(s):
    return set(w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3)


def matches(pdf_path, title):
    try:
        d = pymupdf.open(pdf_path)
        txt = " ".join(d[0].get_text().split() + (d[1].get_text().split() if len(d) > 1 else []))
        want = tokens(title)
        return len(want & tokens(txt)) / max(len(want), 1) >= 0.6, len(d)
    except Exception:
        return False, 0


def shrink(path):
    try:
        d = pymupdf.open(path)
        tmp = path + ".tmp"
        d.save(tmp, garbage=4, deflate=True, clean=True)
        d.close()
        if os.path.getsize(tmp) < 0.95 * os.path.getsize(path):
            os.replace(tmp, path)
        else:
            os.remove(tmp)
    except Exception:
        if os.path.exists(path + ".tmp"):
            os.remove(path + ".tmp")


def host(url):
    return urllib.parse.urlparse(url).netloc.replace("www.", "")


def find_candidates(key, b):
    """Return [(url, licence, source_label)], best first, and the OpenAlex work."""
    cands, work = [], None
    if key in LOCAL_FILES and os.path.exists(LOCAL_FILES[key]):
        cands.append(("file:" + LOCAL_FILES[key], "?", "author's copy"))
    try:
        base = "https://api.openalex.org/works"
        work = get(f"{base}/doi:{urllib.parse.quote(b['doi'])}") if b["doi"] else (
            get(f"{base}?search={urllib.parse.quote(b['title'])}&per-page=1")["results"] or [None])[0]
    except Exception:
        pass
    locs = [l for l in ((work or {}).get("locations") or []) if l]
    lic_by_host = {}
    for l in locs:
        if (l.get("license") or "").lower() and l.get("landing_page_url"):
            lic_by_host.setdefault(host(l["landing_page_url"]), l["license"].lower())
    for l in sorted(locs, key=lambda l: not (l.get("license") or "").lower().startswith("cc")):
        if l.get("pdf_url"):
            cands.append((l["pdf_url"], (l.get("license") or "").lower(), "OpenAlex location"))
    oa = (work or {}).get("open_access") or {}
    if oa.get("oa_url"):
        cands.append((oa["oa_url"], lic_by_host.get(host(oa["oa_url"]), ""), "OpenAlex OA link"))
    if b["doi"]:
        try:   # Europe PMC mirrors many open-access journals and states the licence of each article
            q = urllib.parse.quote('DOI:"' + b["doi"] + '"')
            res = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?resultType=core&format=json&query="
                      + q)["resultList"]["result"]
            if res and res[0].get("pmcid") and res[0].get("isOpenAccess") == "Y":
                lic = (res[0].get("license") or "").lower().replace(" ", "-")
                cands.append(("https://europepmc.org/backend/ptpmcrender.fcgi?accid=" + res[0]["pmcid"]
                              + "&blobtype=pdf", lic, "Europe PMC"))
        except Exception:
            pass
        try:
            s2 = get("https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(b["doi"])
                     + "?fields=openAccessPdf")
            if (s2.get("openAccessPdf") or {}).get("url"):
                u = s2["openAccessPdf"]["url"]
                cands.append((u, lic_by_host.get(host(u), ""), "Semantic Scholar"))
        except Exception:
            pass
        if b["doi"].startswith("10.1038/"):
            u = "https://www.nature.com/articles/" + b["doi"].split("/")[1] + ".pdf"
            cands.append((u, lic_by_host.get("nature.com", ""), "Nature direct PDF"))
    if key == "bae2020":
        cands.append(("https://arxiv.org/pdf/2012.07279", "", "arXiv"))
    else:
        try:
            q = urllib.parse.quote('ti:"' + b["title"].replace('"', "") + '"')
            xml = get("http://export.arxiv.org/api/query?max_results=1&search_query=" + q, binary=True).decode()
            m = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>.*?<title>(.*?)</title>", xml, re.S)
            if m and len(tokens(m.group(2)) & tokens(b["title"])) / max(len(tokens(b["title"])), 1) > 0.8:
                cands.append(("https://arxiv.org/pdf/" + m.group(1), "", "arXiv preprint"))
        except Exception:
            pass
    seen, uniq = set(), []
    for c in cands:
        if c[0] not in seen:
            seen.add(c[0])
            uniq.append(c)
    return uniq, work


def main():
    bib, order = parse_bib(), citation_order()
    for d in ("open_access", "local_only"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    index = []
    tmp = os.path.join(HERE, "_tmp.pdf")
    for n, key in enumerate(order, 1):
        b = bib[key]
        first = re.split(r"\s+and\s+|,", b["author"])[0].split()[-1] if b["author"] else key
        stem = f"{n:02d}_{first}_{b['year']}"
        rec = dict(n=n, key=key, stem=stem, **b, status="missing", folder=None, license="", source="",
                   size_kb=None, pages=None, note="")
        print(f"[{n:02d}] {key}: ", end="", flush=True)
        cands, work = find_candidates(key, b)
        for url, lic, label in cands:
            try:
                data = open(url[5:], "rb").read() if url.startswith("file:") else get(url, binary=True)
                if not data.startswith(b"%PDF"):
                    rec["note"] += f" {label}: not a PDF;"
                    continue
                open(tmp, "wb").write(data)
                ok, pages = matches(tmp, b["title"])
                if not ok:
                    rec["note"] += f" {label}: first page does not match;"
                    continue
                lic = lic if lic in REDISTRIBUTABLE else ""
                if url.startswith("file:"):      # author's own copy: licence from the publisher's CC location, if any
                    for l in ((work or {}).get("locations") or []):
                        if l and (l.get("license") or "").lower() in REDISTRIBUTABLE:
                            lic = l["license"].lower()
                            break
                folder = "open_access" if lic else "local_only"
                dst = os.path.join(HERE, folder, stem + ".pdf")
                os.replace(tmp, dst)
                if lic in RECOMPRESS_OK:
                    shrink(dst)
                rec.update(status="ok", folder=folder, license=lic or "none stated", source=label,
                           size_kb=round(os.path.getsize(dst) / 1024), pages=pages)
                break
            except Exception as e:
                rec["note"] += f" {label}: {type(e).__name__};"
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        if rec["status"] != "ok":
            oa = (work or {}).get("open_access") or {}
            rec["is_oa"], rec["oa_url"] = bool(oa.get("is_oa")), oa.get("oa_url")
        print(rec["status"], rec["folder"] or "", rec["license"], rec["source"], rec["size_kb"] or "", flush=True)
        index.append(rec)
        time.sleep(0.5)
    json.dump(index, open(os.path.join(HERE, "index.json"), "w", encoding="utf-8"), indent=1)
    print("done:", sum(r["status"] == "ok" for r in index), "of", len(index), "obtained")


if __name__ == "__main__":
    sys.exit(main())
