"""Generate INDEX.md and MISSING.md from index.json (written by fetch_references.py).

    python build_index.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "index.json"), encoding="utf-8"))


def cite(r):
    names = [a.strip() for a in r["author"].replace(" and ", ", ").split(",") if a.strip()]
    etal = "others" in names or len(names) > 3
    names = [a for a in names if a != "others"][:3]
    au = ", ".join(names) + (" et al." if etal else "")
    return f'{au}, "{r["title"]}", {r["venue"]}, {r["year"]}.'


def doi_link(r):
    return f"[{r['doi']}](https://doi.org/{r['doi']})" if r["doi"] else "n/a"


def folder_size(folder):
    p = os.path.join(HERE, folder)
    return sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p) if f.endswith(".pdf")) / 1e6


ok = [r for r in rows if r["status"] == "ok"]
oa = [r for r in ok if r["folder"] == "open_access"]
lo = [r for r in ok if r["folder"] == "local_only"]
miss = [r for r in rows if r["status"] != "ok"]

with open(os.path.join(HERE, "INDEX.md"), "w", encoding="utf-8") as f:
    f.write("# Reference papers\n\n")
    f.write("Papers cited in [`paper/paper.tex`](../paper/paper.tex), numbered in order of first citation "
            "(the numbers match the reference list of the paper).\n\n")
    f.write(f"- **{len(oa)} open-access PDFs** are stored in [`open_access/`](open_access/) "
            f"({folder_size('open_access'):.1f} MB). Their licence permits redistribution (Creative Commons).\n")
    f.write(f"- **{len(lo)} further PDFs** are free to read but carry no redistribution licence. They are kept only "
            "on the author's machine in `local_only/` (excluded from git); use the DOI links below to read them.\n")
    f.write(f"- **{len(miss)} papers** could not be downloaded automatically; see [MISSING.md](MISSING.md).\n\n")
    f.write("| # | Paper | DOI | Copy | Licence |\n|---|---|---|---|---|\n")
    for r in rows:
        if r["status"] == "ok" and r["folder"] == "open_access":
            copy = f"[PDF](open_access/{r['stem']}.pdf) ({r['size_kb'] / 1024:.1f} MB)"
        elif r["status"] == "ok":
            copy = f"local only ({r['size_kb'] / 1024:.1f} MB)"
        else:
            copy = "not obtained"
        f.write(f"| {r['n']} | {cite(r)} | {doi_link(r)} | {copy} | {r['license'] or '-'} |\n")
    f.write("\nPDFs were obtained from open-access locations only (author copies, arXiv preprints, publisher-hosted "
            "open access, Semantic Scholar/OpenAlex open-access links). No shadow libraries were used. "
            "`python fetch_references.py` repeats the search.\n")

with open(os.path.join(HERE, "MISSING.md"), "w", encoding="utf-8") as f:
    f.write("# Papers not obtained automatically\n\n")
    f.write("These could not be downloaded by script. Most publisher sites block automated downloads, so the ones "
            "marked **open access** can simply be saved from a normal browser. For the others, use an institutional "
            "login (for example the Manipal University Jaipur library: IEEE Xplore, ACM Digital Library, "
            "ScienceDirect, SpringerLink) or ask the authors for a copy. Save a PDF as "
            "`references/local_only/<number>_<Author>_<Year>.pdf` (not committed) unless its licence allows "
            "redistribution.\n\n")
    f.write("| # | Paper | DOI | Access | Where to get it |\n|---|---|---|---|---|\n")
    for r in miss:
        if r.get("is_oa"):
            acc = "**open access**"
            where = (f"[publisher page]({r['oa_url']})" if r.get("oa_url") else "publisher page via DOI")
        else:
            acc = "paywalled"
            where = "institutional access / author copy"
        f.write(f"| {r['n']} | {cite(r)} | {doi_link(r)} | {acc} | {where} |\n")
print("INDEX.md and MISSING.md written;", len(ok), "obtained,", len(miss), "missing")
print("open_access %.1f MB, local_only %.1f MB" % (folder_size("open_access"), folder_size("local_only")))
