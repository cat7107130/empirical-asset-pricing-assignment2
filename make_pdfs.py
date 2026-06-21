"""
make_pdfs.py
============
Renders the two required submission PDFs using only matplotlib (no extra deps):

  architecture.pdf : ARCHITECTURE.md laid out as paginated text.
  results.pdf      : for Set1 and Set2 -> Table 1 (numeric table + bar figure),
                     Figure 4, Figure 9.

Usage:
    python make_pdfs.py --out_dir outputs            # full-run results
    python make_pdfs.py --out_dir outputs_pilot      # pilot results
"""

from __future__ import annotations
import argparse
import os
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)                          # portrait A4 in inches


# --------------------------------------------------------------------------- #
# Architecture PDF                                                            #
# --------------------------------------------------------------------------- #
def architecture_pdf(md_path: str, pdf_path: str,
                     lines_per_page: int = 58, wrap: int = 98):
    """Render a markdown file as monospaced, paginated text pages.

    Light touch only: headings are bolded, everything else is wrapped at
    ``wrap`` characters. Markdown tables render as their raw pipe text (still
    readable). This keeps the toolchain to matplotlib alone."""
    with open(md_path, encoding="utf-8") as f:
        raw = f.read().splitlines()

    # Wrap long lines while preserving blank lines and code/table rows verbatim.
    flat = []
    for ln in raw:
        if ln.strip() == "" or ln.lstrip().startswith(("|", "```")) or "  " in ln[:4]:
            flat.append(ln)
        else:
            wrapped = textwrap.wrap(ln, wrap) or [""]
            flat.extend(wrapped)

    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(flat), lines_per_page):
            chunk = flat[i:i + lines_per_page]
            fig = plt.figure(figsize=A4)
            fig.text(0.5, 0.97, "Architecture Description — GKX (2020) Replication",
                     ha="center", va="top", fontsize=10, weight="bold")
            y = 0.93
            for ln in chunk:
                weight = "bold" if ln.startswith("#") else "normal"
                txt = ln.lstrip("#").strip() if ln.startswith("#") else ln
                size = 9.5 if ln.startswith("#") else 7.6
                fig.text(0.06, y, txt, ha="left", va="top",
                         fontsize=size, family="monospace", weight=weight)
                y -= 0.0155
            pdf.savefig(fig)
            plt.close(fig)
    print("wrote", pdf_path)


# --------------------------------------------------------------------------- #
# Results PDF                                                                  #
# --------------------------------------------------------------------------- #
def _image_page(pdf, png_path, title):
    """Embed a PNG as a full page with a title."""
    if not os.path.exists(png_path):
        return
    fig = plt.figure(figsize=A4)
    fig.suptitle(title, fontsize=12, weight="bold", y=0.97)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.86])
    ax.axis("off")
    ax.imshow(plt.imread(png_path))
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf, csv_path, title):
    """Render the Table-1 CSV in GKX layout: models as columns, an ``All`` row,
    monthly out-of-sample R^2 in percent (matching Table 1 of GKX 2020)."""
    if not os.path.exists(csv_path):
        return
    # CSV already holds percent values with models as columns and an "All" row.
    df = pd.read_csv(csv_path, index_col=0).round(3)

    fig = plt.figure(figsize=A4)
    fig.suptitle(title, fontsize=12, weight="bold", y=0.95)
    fig.text(0.5, 0.88, "Monthly out-of-sample stock-level prediction "
             r"performance (percentage $R^2_{oos}$)",
             ha="center", fontsize=10)
    ax = fig.add_axes([0.05, 0.6, 0.9, 0.22])
    ax.axis("off")
    tbl = ax.table(cellText=df.reset_index().values,
                   colLabels=[""] + list(df.columns),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    # Bold the header row and the row label, as in the paper.
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def results_pdf(out_dir: str, pdf_path: str):
    with PdfPages(pdf_path) as pdf:
        # Cover page.
        fig = plt.figure(figsize=A4)
        fig.text(0.5, 0.6, "Results — GKX (2020) Replication",
                 ha="center", fontsize=18, weight="bold")
        fig.text(0.5, 0.52, "Set 1: test 2021-2023   |   Set 2: test 2021-2025",
                 ha="center", fontsize=12)
        fig.text(0.5, 0.47,
                 "Compute-limited window (assignment-permitted): sample starts 2006,\n"
                 "OOS test period restricted to 2021-2025.",
                 ha="center", fontsize=9)
        fig.text(0.5, 0.42, f"(artifacts: {out_dir})", ha="center", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        for tag in ["Set1", "Set2"]:
            _table_page(pdf, f"{out_dir}/table1_{tag}.csv",
                        f"Table 1 — Monthly OOS R^2 ({tag})")
            _image_page(pdf, f"{out_dir}/table1_{tag}.png",
                        f"Table 1 (figure) — Monthly OOS R^2 ({tag})")
            _image_page(pdf, f"{out_dir}/figure4_{tag}.png",
                        f"Figure 4 — Variable importance ({tag})")
            _image_page(pdf, f"{out_dir}/figure9_{tag}.png",
                        f"Figure 9 — Cumulative ML portfolio return ({tag})")
    print("wrote", pdf_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--arch_md", default="ARCHITECTURE.md")
    args = ap.parse_args()
    suffix = "" if args.out_dir == "outputs" else "_" + args.out_dir.split("_")[-1]
    architecture_pdf(args.arch_md, f"architecture{suffix}.pdf")
    results_pdf(args.out_dir, f"results{suffix}.pdf")


if __name__ == "__main__":
    main()
