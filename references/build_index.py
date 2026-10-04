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

# Hand-checked details for papers that could not be fetched (re-search done 2026-10-04)
HINTS = {
    "liu2024dep": ("**open access** (SpringerOpen)",
                   "Free to download in a browser (the site blocks scripts): "
                   "[article page](https://link.springer.com/article/10.1186/s13677-024-00701-0) "
                   "(click *Download PDF*), "
                   "[direct PDF](https://link.springer.com/content/pdf/10.1186/s13677-024-00701-0.pdf), "
                   "[DOAJ record](https://doaj.org/article/46e89a34390a423182b234220e3797c9), "
                   "[ResearchGate](https://www.researchgate.net/publication/383791431_Dependency-aware_online_task_offloading_based_on_deep_reinforcement_learning_for_IoV)."),
    "zhang2025": ("paywalled; **no free legal copy found**",
                  "OpenAlex lists it as closed and neither Semantic Scholar, CORE nor arXiv has a copy. Options: "
                  "(1) read it through institutional access (Manipal University Jaipur library -> SpringerLink): "
                  "[article page](https://link.springer.com/article/10.1007/s12083-025-02101-w), "
                  "[PDF](https://link.springer.com/content/pdf/10.1007/s12083-025-02101-w.pdf); "
                  "(2) email the corresponding author (address on the article page) and ask for the accepted "
                  "manuscript, which authors are generally allowed to share; "
                  "(3) the abstract and metadata are public, so the paper can be cited without the PDF. "
                  "Authors: X. Zhang, C. Fang, Z. Bai, L. Zhang, P. Wang, Z. Cao; "
                  "Peer-to-Peer Netw. Appl. 18(6), article 288, 26 Sep 2025."),
}

with open(os.path.join(HERE, "MISSING.md"), "w", encoding="utf-8") as f:
    f.write("# Papers not in the collection\n\n")
    if not miss:
        f.write("All cited papers are in the collection.\n")
    f.write("These could not be downloaded by script (most publisher sites block automated downloads). After saving "
            "a PDF from a browser, put it in any folder and run `python ingest_downloads.py <folder>`: it matches "
            "the file to the paper, files it under `open_access/` or `local_only/` by licence, renames it with the "
            "paper's title and updates the index (then run `python build_index.py`).\n\n")
    f.write("| # | Paper | DOI | Access | Where to get it |\n|---|---|---|---|---|\n")
    for r in miss:
        if r["key"] in HINTS:
            acc, where = HINTS[r["key"]]
        elif r.get("is_oa"):
            acc = "**open access**"
            where = (f"[publisher page]({r['oa_url']})" if r.get("oa_url") else "publisher page via DOI")
        else:
            acc, where = "paywalled", "institutional access / author copy"
        f.write(f"| {r['n']} | {cite(r)} | {doi_link(r)} | {acc} | {where} |\n")
# ---- list of all papers inside the repository README (between the REFS markers) ----------------
EXTRA_LINKS = {   # papers without a DOI
    "bae2020": "https://arxiv.org/abs/2012.07279",
    "barbarossa2013": "https://www.researchgate.net/publication/261053174_Computation_offloading_for_mobile_cloud_computing_based_on_wide_cross-layer_optimization",
}
lines = ["| # | Paper | Where to read it |", "|---|---|---|"]
for r in rows:
    pub = f"https://doi.org/{r['doi']}" if r["doi"] else EXTRA_LINKS.get(r["key"], "")
    if r["status"] == "ok" and r["folder"] == "open_access":
        where = f"[PDF in this repository](references/open_access/{r['stem']}.pdf) ({r['license'].upper()})"
    else:
        label = "arXiv" if "arxiv.org" in pub else ("ResearchGate" if "researchgate" in pub else "Publisher page")
        where = f"[{label}]({pub})" if pub else "-"
    lines.append(f"| {r['n']} | {cite(r)} | {where} |")
readme = os.path.join(HERE, "..", "README.md")
txt = open(readme, encoding="utf-8").read()
a, b = txt.index("<!-- REFS:START -->"), txt.index("<!-- REFS:END -->")
txt = txt[:a] + "<!-- REFS:START -->\n" + "\n".join(lines) + "\n" + txt[b:]
open(readme, "w", encoding="utf-8", newline="\n").write(txt)

print("INDEX.md and MISSING.md written;", len(ok), "obtained,", len(miss), "missing")
print("open_access %.1f MB, local_only %.1f MB" % (folder_size("open_access"), folder_size("local_only")))
