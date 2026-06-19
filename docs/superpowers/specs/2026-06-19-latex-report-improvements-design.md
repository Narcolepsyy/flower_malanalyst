# Design Document: LaTeX Report Enhancements & Consistency

**Date:** 2026-06-19
**Status:** Approved

## 1. Objectives & Context
The user requested an enhancement and review of the LaTeX report under `IE105_Q11/`. Following a compilation check and a detailed content audit, we found several layout warnings (small headheight, float specifiers fallback) and content/mathematical inconsistencies (chapter title casing, $n$ vs $K$ client counts, model count mismatch). 

The objective is to resolve all warnings, prevent margins overflow, standardize casing and variables, and align numbers with actual experimental data text-only.

## 2. Target Files
- `/home/khaitran/project/flmal/IE105_Q11/main.tex`
- `/home/khaitran/project/flmal/IE105_Q11/chapters/chapter2_theory.tex`
- `/home/khaitran/project/flmal/IE105_Q11/chapters/chapter3_methodology.tex`
- `/home/khaitran/project/flmal/IE105_Q11/chapters/chapter4_experiments.tex`
- `/home/khaitran/project/flmal/IE105_Q11/chapters/appendix.tex`

## 3. Specifications

### 3.1. Layout & Compilation Fixes
1. **Preamble Setup (`main.tex`)**: 
   Add `headheight=15pt` to the `\geometry` block to fix the `fancyhdr` warning.
2. **Table Placements (`chapters/chapter4_experiments.tex`)**:
   Adjust float specifiers for all top tables from `[h!]` to `[ht!]` to eliminate the fallback warning.
3. **Margin Overflows (`chapters/chapter4_experiments.tex`)**:
   - Use `\path{...}` instead of `\texttt{...}` for file paths and variable names containing underscores/slashes (e.g. `state/explanations.json`, `handles.nthread`, `ldrmodules.not_in_load`, `not_in_mem`).
   - Extract the long command string from line 193 into a centered display environment `\begin{center}\texttt{...}\end{center}`.

### 3.2. Casing & Notation Consistency
1. **Chapter 3 Title (`chapters/chapter3_methodology.tex`)**:
   Change `\section{Phương pháp thực hiện}` to `\section{PHƯƠNG PHÁP THỰC HIỆN}` for consistency.
2. **Client Count Notation (`chapters/chapter2_theory.tex`)**:
   Change the Krum distance search limit `n - f - 2` on line 49 to `K - f - 2` to align with the definition of total client count $K$.
3. **Typo Fix (`chapters/chapter2_theory.tex`)**:
   On line 53, change `thay vị chỉ một` to `thay vì chỉ một`.

### 3.3. Model Count Standardization
1. **Chapter 4 (`chapters/chapter4_experiments.tex`)**:
   Change model counts from `53` to `50` on lines 24 and 27 to match the table entries.
2. **Appendix (`chapters/appendix.tex`)**:
   Change model count from `53` to `50` on line 6 to align with the table entries.

## 4. Verification Plan
- Run `pdflatex -interaction=nonstopmode main.tex` twice to check that referencing is complete.
- Verify that `fancyhdr` warnings are gone.
- Verify that float specifier warnings are gone.
- Verify that there are no new overfull `\hbox`es on page margins.
