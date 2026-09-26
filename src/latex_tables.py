"""
Generate the LaTeX tables of the paper (Tables 1, 3-7, B.1 and H.1) from the stored results in output/.

The layout follows the paper's LaTeX source: stars for significance at 10/5/1 percent, p-values (or t-statistics)
underneath the numbers. Each file contains only the tabular environment, to replace the one in the paper,
except table_H1.tex, which contains the complete appendix table (three table environments).

    python src/latex_tables.py          # writes output/tables_tex/table_*.tex
"""

from pathlib import Path

import pandas as pd

import tables

MATS = [24, 36, 48, 60, 84, 120]
OUT = tables.OUTPUT / "tables_tex"


def stars(p):
    return tables._stars(p)


def cell(value, decimals, p=None, hspace=True):
    """Number with the p-value underneath (if available) and significance stars."""
    v = f"{value:.{decimals}f}"
    if p is None or pd.isna(p):
        return f"${v}$"
    s = stars(p)
    return ("\\hspace{0.2cm} " if hspace else "") + f"$\\underset{{({p:.3f})}}{{{v}}}^{{{s}}}$"


def star_only(value, decimals, p=None):
    """Number with significance stars only (Table 7 style)."""
    s = stars(p) if p is not None and pd.notna(p) else ""
    return f"${value:.{decimals}f}^{{{s}}}$" if s else f"{value:.{decimals}f}"


def row(cells):
    return " & ".join(cells) + " \\\\"


# ----------------------------------------------------------------------------------------------------------
# Accessors


def r2_cells(run, method, decimals=3):
    outp = tables.load(run, method)
    r2, p = outp["R2"].loc[method].astype(float), outp["R2_pval"].loc[method].astype(float)
    return [cell(r2[m], decimals, p[m], hspace=False) for m in MATS]


def cer_values(run, method):
    u = tables.load(run, method)["Utilities"]
    stat = u.loc[("CER", method, "stat")].astype(float) * 100
    pv = u.loc[("CER", method, "p-val")].astype(float) if ("CER", method, "p-val") in u.index else None
    return [(stat[m], None if pv is None else pv[m]) for m in MATS]


def sharpe_values(run, method):
    u = tables.load(run, method)["Utilities"]
    return [float(u.loc[("Sharpe Ratio", method, "stat")][m]) for m in MATS]


def sharpe_cells(run, method):
    """Sharpe ratios with significance stars for gains over the benchmark (Ledoit-Wolf test, src/sharpe_test.py)."""
    p = tables.sharpe_pvalues()
    return [star_only(v, 3, p.get((method, m))) for v, m in zip(sharpe_values(run, method), MATS)]


# ----------------------------------------------------------------------------------------------------------
# Tables


def table_1():
    outp = tables.load("baseline_main")
    ys, dys = (outp[f"{m}: full sample regression"].astype(float) for m in ["Yields", "Yield changes"])
    lines = [r"\begin{tabular}{cccccclccccc}", r"	\hline",
             r"	\multicolumn{12}{c}{Full-sample bond risk premia regressions}\\\hline",
             r"	& \multicolumn{5}{c}{Panel A: Using $Y$} && \multicolumn{5}{c}{Panel B: Using $\Delta Y$}\\\cline{2-6}\cline{8-12}",
             r"$n$ & const  & PC1      & PC2       & PC3       & $R^{2}$ && const   & PC1     & PC2     & PC3    & $R^{2}$\\",
             r"\hline"]
    for m in MATS:
        coef, tstat = [str(m // 12)], [""]
        for k, cof in enumerate([ys, dys]):
            for c in ["const", 1, 2, 3]:
                p = cof.loc[(m, "p-value"), c]
                s = stars(p)
                coef.append(f"${cof.loc[(m, 'coef'), c]:.2f}" + (f"^{{{s}}}$" if s else "$"))
                tstat.append(f"({cof.loc[(m, 't-stat'), c]:.2f})")
            coef.append(f"{cof.loc[(m, 'coef'), 'R2']:.3f}")
            tstat.append("")
            if k == 0:
                coef.append("")
                tstat.append("")
        lines += [row(coef), row(tstat)]
    lines += [r"\hline", r"    \end{tabular}"]
    return "\n".join(lines)


def table_3():
    rows = [("Benchmark (EH)", "baseline_main", "Benchmark"), ("$Y$", "baseline_main", "Yields"),
            ("$\\Delta Y$", "baseline_main", "Yield changes"), ("$F$", "baseline_main", "Forwards"),
            ("$\\Delta F$", "baseline_main", "Forward changes"),
            ("Cochrane and Piazzesi", "baseline_main", "Cochrane Piazessi"),
            ("Diebold and Li", "baseline_diebold_li", "NN-3-1-Diebold Li direct")]
    lines = [r"\begin{tabular}{llcccccc}", r"    \hline",
             r"	\multicolumn{8}{c}{Mean-variance utility investor -- Linear models}\\\hline",
             r"	&& \multicolumn{6}{c}{$n$}\\\cline{3-8}",
             r"&Factors   & 2  & 3      & 4       & 5       & 7 & 10 \\", r"\hline"]
    for i, (label, run, method) in enumerate(rows):
        lines.append(row(["CER (\\%)" if i == 0 else "", label] + [cell(v, 2, p) for v, p in cer_values(run, method)]))
    lines.append(r"    \hline")
    for i, (label, run, method) in enumerate(rows):
        lines.append(row(["Sharpe Ratio" if i == 0 else "", label] + sharpe_cells(run, method)))
    lines += [r"\hline", r"    \end{tabular}"]
    return "\n".join(lines)


NN_TABLE_4 = [("$y$", "NN-3-1-Yields"), ("$\\Delta y$", "NN-3-1-Yield changes"), ("$f$", "NN-3-1-Forwards"),
              ("$\\Delta f$", "NN-3-1-Forward changes"), None,
              ("$Y$", "NN-3-1-Yields PC"), ("$\\Delta Y$", "NN-3-1-Yield changes PC"),
              ("Cochrane and Piazzesi", "NN-3-1-Cochrane Piazessi"), ("Diebold and Li", "NN-3-1-Diebold Li")]


def table_4():
    lines = [r"\begin{tabular}{llccccccc}", r"\hline",
             r"	\multicolumn{8}{c}{Out-of-sample $R^2$ -- Neural networks}\\\hline",
             r"	&& \multicolumn{6}{c}{$n$}\\\cline{3-8}",
             r"Model & Input  & 2  & 3      & 4       & 5       & 7 & 10 \\", r"\hline"]
    first = True
    for entry in NN_TABLE_4:
        if entry is None:
            lines.append(r"\hline")
            first = True
            continue
        label, method = entry
        lines.append(row(["\\ Neural Network" if first else "", label] + r2_cells("nn", method)))
        first = False
    lines += [r"\hline", r"    \end{tabular}"]
    return "\n".join(lines)


def table_5():
    lines = [r"\begin{tabular}{lllcccccc}", r"    \hline",
             r"	\multicolumn{9}{c}{Mean-variance utility investor -- Neural networks}\\\hline",
             r"	&& \multicolumn{6}{c}{$n$}\\\cline{4-9}",
             r"&Models&Input   & 2  & 3      & 4       & 5       & 7 & 10 \\", r"\hline"]
    for metric in ["CER", "SR"]:
        bench = cer_values("baseline_main", "Benchmark") if metric == "CER" else sharpe_values("baseline_main", "Benchmark")
        fmt = (lambda x: f"{x[0]:.2f}") if metric == "CER" else (lambda x: f"{x:.3f}")
        lines.append(row([("CER (\\%)" if metric == "CER" else "SR"), "Benchmark (EH)", ""] + [fmt(x) for x in bench]))
        lines.append(r"    \cline{2-9}")
        first = True
        for entry in NN_TABLE_4:
            if entry is None:
                lines.append(r"    \cline{2-9}")
                first = True
                continue
            label, method = entry
            vals = ([cell(v, 2, p) for v, p in cer_values("nn", method)] if metric == "CER"
                    else sharpe_cells("nn", method))
            lines.append(row(["", "Neural Net" if first else "", label] + vals))
            first = False
        lines.append(r"    \hline")
    lines += [r"    \end{tabular}"]
    return "\n".join(lines)


MACRO_REGRESSION = [("$Y$ and $Z$", "Yields + macro (revised)"), ("$F$ and $Z$", "Forwards + macro (revised)"),
                    ("$\\Delta Y$ and $Z$", "Yield changes + macro (revised)"),
                    ("$\\Delta F$ and $Z$", "Forward changes + macro (revised)"),
                    ("Cieslak and Povala", "Cieslak Povala (param 0.987)")]
MACRO_NN = [("$y$ and $Z$", "NN-32-1-Yields + macro (revised)"), ("$f$ and $Z$", "NN-32-1-Forwards + macro (revised)"),
            ("$\\Delta y$ and $Z$", "NN-32-1-Yield changes + macro (revised)"),
            ("$\\Delta f$ and $Z$", "NN-32-1-Forward changes + macro (revised)"),
            ("Cieslak and Povala", "NN-3-1-Cieslak Povala (param 0.987)")]


def table_6():
    lines = [r"\begin{tabular}{llccccccc}", r"\hline",
             r"	\multicolumn{8}{c}{Out-of-sample $R^2$ -- With macro data}\\\hline",
             r"	&& \multicolumn{6}{c}{$n$}\\\cline{3-8}",
             r"Model & Factors  & 2  & 3      & 4       & 5       & 7 & 10 \\", r"\hline"]
    for i, (label, method) in enumerate(MACRO_REGRESSION):
        lines.append(row(["\\ Regression" if i == 0 else "", label] + r2_cells("baseline_macro", method)))
    lines.append(r"\hline")
    for i, (label, method) in enumerate(MACRO_NN):
        lines.append(row(["\\ Neural Network" if i == 0 else "", label] + r2_cells("nn", method)))
    lines += [r"\hline", r"    \end{tabular}"]
    return "\n".join(lines)


def table_7():
    lines = [r"\begin{tabular}{lllcccccc}", r"    \hline",
             r"	\multicolumn{9}{c}{Mean-variance utility investor - Including macro data}\\\hline",
             r"	&& \multicolumn{6}{c}{$n$}\\\cline{4-9}",
             r"&Models&Factors   & 2  & 3      & 4       & 5       & 7 & 10 \\", r"\hline"]
    for metric in ["CER", "SR"]:
        if metric == "CER":
            bench = [f"{v:.2f}" for v, _ in cer_values("baseline_main", "Benchmark")]
        else:
            bench = [f"{v:.3f}" for v in sharpe_values("baseline_main", "Benchmark")]
        lines.append(row([("CER (\\%)" if metric == "CER" else "SR"), "Benchmark", ""] + bench))
        for group, entries, run in [("Regression", MACRO_REGRESSION, "baseline_macro"), ("Neural Net", MACRO_NN, "nn")]:
            lines.append(r"\cline{2-9}")
            for i, (label, method) in enumerate(entries):
                vals = ([star_only(v, 2, p) for v, p in cer_values(run, method)] if metric == "CER"
                        else sharpe_cells(run, method))
                lines.append(row(["", group if i == 0 else "", label] + vals))
        lines.append(r"    \hline")
    lines += [r"    \end{tabular}"]
    return "\n".join(lines)


def table_h1():
    """Table H.1: complete replacement (three table environments), with variable descriptions per data column
    from data/fredmd_series.csv, grouped in panels by blocks of consecutive columns."""
    series = pd.read_csv(tables.ROOT / "data" / "fredmd_series.csv", encoding="utf-8")
    loadings, _ = tables.macro_loadings()
    header = r"&Variable&	PC1	&	PC2	&	PC3	&	PC4	&	PC5	&	PC6	&	PC7	&	PC8	\\ \hline"
    pages = [["Output and income", "Labor market", "Housing"],
             ["Orders and inventories", "Money and credit", "Stock market", "Interest and exchange rates"],
             ["Prices", "Earnings, sentiment, money and credit, volatility"]]
    caption = (r"    \caption{ The factor loadings of the 128 macro variables on the first 8 components. The sample "
               r"period is 1971:08-2018:12. Variables are numbered and ordered as in the January 2019 vintage of "
               r"FRED-MD; variable descriptions follow \cite{mccracken2016fred}.}")
    letters = iter("ABCDEFGHI")
    out = [r"\setlength{\tabcolsep}{3.8pt}"]
    for p, blocks in enumerate(pages):
        if p == 2:
            out += [r"\newpage", r"\vspace*{250px}"]
        out += [r"\begin{table}[!h]", r"\scriptsize", r"    \centering", r"    \begin{tabular}{llcccccccc}",
                r"    \hline"]
        if p == 0:
            out.append(r"\multicolumn{10}{c}{Macro factors: Principal Component Loadings} \\ \hline")
        for b, name in enumerate(blocks):
            if b > 0:
                out.append(r"\\ \hline")
            out += [f" \\multicolumn{{10}}{{c}}{{Panel {next(letters)}: {name.replace('and', chr(92) + '&', 1) if name in ('Orders and inventories', 'Money and credit') else name}}} \\\\ ", header]
            for _, s in series[series.block == name].iterrows():
                vals = "	&	".join(f"{loadings.loc[s.column, f'PC{i}']:.3f}" for i in range(1, 9))
                out.append(f"{s.column}&{s.description_tex}	&	{vals}	\\\\")
        out.append(r" \hline")
        if p < 2:
            out.append(r"\multicolumn{10}{l}{\it Continued on next page}")
        out.append(r"    \end{tabular}")
        out.append(caption if p == 2 else "    %" + caption.strip())
        out.append(f"    \\label{{macro_loadings{p + 1}}}" if p > 0 else "")
        out += [r"\end{table}", ""]
    return "\n".join(line for line in out if line is not None)


def table_b1():
    """Table B.1: loadings of the principal components of forward-rate changes and eigenvalues, components in
    descending order of eigenvalue. Signs of principal components are arbitrary; each component is signed as in
    the printed version of the paper."""
    ours, paper, _ = tables.forward_change_loadings()
    loadings = ours.iloc[:10].astype(float)
    for c in loadings.columns:
        if (loadings[c] * paper[c].iloc[:10]).sum() < 0:
            loadings[c] = -loadings[c]
    eig = ours.loc["λ"].astype(float)
    lines = [r"\begin{tabular}{ccccccccccc}",
             r"\hline &\multicolumn{10}{c}{Change in forward rates: Principal Component loadings} \\ \cline{2-11}",
             r" n	&	" + "	&	".join(f"PC {i}" for i in range(1, 11)) + r"\\", r" 		\hline"]
    for i, (idx, row_) in enumerate(loadings.iterrows()):
        lines.append(row([str(12 * (i + 1))] + [f"{v:.2f}" for v in row_]))
    lines += [r"		\hline", row(["$\\lambda$"] + [f"{v:.2f}" for v in eig]), r"\hline", r"    \end{tabular}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("table_1", table_1), ("table_3", table_3), ("table_4", table_4), ("table_5", table_5),
                     ("table_6", table_6), ("table_7", table_7), ("table_B1", table_b1), ("table_H1", table_h1)]:
        (OUT / f"{name}.tex").write_text(fn() + "\n", encoding="utf-8")
        print(f"written {OUT / name}.tex")


if __name__ == "__main__":
    main()
