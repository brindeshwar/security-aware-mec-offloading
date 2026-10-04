"""Build paper.pdf:  python build.py

Uses the LaTeX distribution at ../.local/tools/TinyTeX if present, otherwise the pdflatex/bibtex on PATH.
The IEEE Access class and its fonts live in ./ieee and are found through the TEX* search paths below.
Figures and tables are read from ../results (run the simulation scripts first to regenerate them).
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
tiny = os.path.join(HERE, "..", ".local", "tools", "TinyTeX", "bin", "windows")
env = dict(os.environ)
if os.path.isdir(tiny):
    env["PATH"] = os.path.abspath(tiny) + os.pathsep + env["PATH"]
if not shutil.which("pdflatex", path=env["PATH"]):
    sys.exit("pdflatex not found: install TeX Live / MiKTeX, or unpack TinyTeX into ../.local/tools/")
ieee = os.path.join(HERE, "ieee")
sep = os.pathsep
for var in ("TEXINPUTS", "TFMFONTS", "T1FONTS", "TEXFONTMAPS", "BSTINPUTS"):
    env[var] = ieee + "//" + sep + env.get(var, "")


def run(*cmd):
    exe = shutil.which(cmd[0], path=env["PATH"])
    return subprocess.run((exe,) + tuple(cmd[1:]), cwd=HERE, env=env, capture_output=True, text=True)


for step in (("pdflatex", "-interaction=nonstopmode", "paper.tex"), ("bibtex", "paper"),
             ("pdflatex", "-interaction=nonstopmode", "paper.tex"),
             ("pdflatex", "-interaction=nonstopmode", "paper.tex")):
    r = run(*step)
log = open(os.path.join(HERE, "paper.log"), encoding="latin-1").read()
ok = os.path.exists(os.path.join(HERE, "paper.pdf")) and "Output written" in log
undefined = [l for l in log.splitlines() if "undefined" in l.lower()]
print("built paper.pdf" if ok else "BUILD FAILED (see paper.log)")
for l in undefined[:10]:
    print("  warning:", l)
for ext in (".aux", ".out", ".blg", ".bbl", ".toc"):
    p = os.path.join(HERE, "paper" + ext)
    if os.path.exists(p):
        os.remove(p)
sys.exit(0 if ok else 1)
