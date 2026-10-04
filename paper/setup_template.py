"""Download the official IEEE Access LaTeX template and prepare it for this paper.

    python setup_template.py [destination folder, default ./ieee]

The template (class, fonts, bibliography style) is IEEE's and is not stored in this repository. The script fetches
the zip from ieeeaccess.ieee.org and applies three small edits to ieeeaccess.cls so that a preprint does not look
like an IEEE-issued article:
  1. the page footer reads "PREPRINT -- NOT PEER REVIEWED" instead of "VOLUME 11, 2023",
  2. the empty "Digital Object Identifier" line is dropped,
  3. the IEEE Access logo in the page header is replaced by a plain blue "PREPRINT" label (same position).
The page layout is otherwise unchanged.
"""
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip"
KEEP_SUFFIX = (".cls", ".bst", ".sty", ".pfb", ".tfm", ".map", ".fd")
BS = chr(92)


def patch_cls(text):
    t = text.replace(BS + "footervolfont VOLUME" + BS + " " + BS + "thevol, " + BS + "theyear",
                     BS + "footervolfont PREPRINT -- NOT PEER REVIEWED")
    t = t.replace(BS + "doifont Digital Object Identifier" + BS + "space" + BS + "@doi", BS + "doifont" + BS + "@doi")
    label = BS + "raisebox{-2pt}{" + BS + "headerfont" + BS + "color{accessblue}" + BS + "textbf{PREPRINT}}"
    for name, logo in (("headerlogo", "logo.png"), ("headerlogoall", "notaglinelogo.png")):
        old = BS + "def" + BS + name + "{" + BS + "raisebox{-2pt}{" + BS + "includegraphics[width=7.61pc]{" + logo + "}}}"
        t = t.replace(old, BS + "def" + BS + name + "{" + label + "}")
    return t


def main(dest):
    os.makedirs(dest, exist_ok=True)
    data = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "curl/8.5.0", "Accept": "*/*"}), timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.lower().endswith(KEEP_SUFFIX):
                continue
            blob = z.read(info)
            if name == "ieeeaccess.cls":
                blob = patch_cls(blob.decode("latin-1")).encode("latin-1")
            open(os.path.join(dest, name), "wb").write(blob)
    print("template installed in", os.path.abspath(dest))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ieee"))
