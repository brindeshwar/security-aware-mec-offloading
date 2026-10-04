# References

- [`INDEX.md`](INDEX.md): every paper cited in the manuscript, with DOI, licence and a link to the PDF where one is stored here.
- [`MISSING.md`](MISSING.md): papers that could not be downloaded automatically, with where to get them.
- `open_access/`: PDFs whose licence allows redistribution (Creative Commons). These are the only PDFs committed to git.
- `local_only/`: further PDFs that are free to read but have no redistribution licence. They exist only on the author's machine and are excluded by `.gitignore`.

`python ingest_downloads.py <folder>` adds PDFs you saved by hand (matches each to its paper, files it by licence, renames it with the title). `python fetch_references.py` repeats the automatic search (open-access sources only: author copies, arXiv, publisher open access, OpenAlex and Semantic Scholar open-access links), and `python build_index.py` regenerates the two index files. Requires `pymupdf` (`pip install pymupdf`).
