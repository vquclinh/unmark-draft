# UNMARK — Proposal build

Files:
- `unmark-proposal.tex`   — the proposal source
- `vietnamese-t1.tex`     — Unicode → T1 mappings so pdfLaTeX renders Vietnamese
- `unmark-proposal.pdf`   — compiled output (14 pages)

## Build

```bash
pdflatex unmark-proposal.tex
pdflatex unmark-proposal.tex     # run twice for the table of contents
```

Requires only a standard TeX Live install (`graphicx`, `booktabs`, `tabularx`,
`titlesec`, `fancyhdr`, `listings`, `hyperref`). No `vntex` / T5 encoding needed:
`vietnamese-t1.tex` declares every Vietnamese precomposed character in terms of
T1 accent commands, plus two hand-built macros (`\vhook`, `\vhorn`) for the
hook-above and horn marks that T1 lacks.

If you later move to XeLaTeX/LuaLaTeX, drop `vietnamese-t1.tex` and the two
macros and use `fontspec` with a font that covers Vietnamese.
