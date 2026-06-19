# LaTeX Report Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the LaTeX report by resolving all layout and compilation warnings (fancyhdr, float specifiers), preventing margins overflow (overfull hboxes), standardizing titles/notations (Chapter 3 casing, Krum variables), and aligning model count references.

**Architecture:** Modifying LaTeX source files and compiling via pdflatex to verify changes.

**Tech Stack:** LaTeX, pdflatex.

## Global Constraints
- Target compilation directory is `/home/khaitran/project/flmal/IE105_Q11`.
- Clean compilation requires 0 errors and zero fancyhdr/float layout warnings.

---

### Task 1: Preamble and Table Placements (Layout Fixes)

**Files:**
- Modify: `main.tex:28-34`
- Modify: `chapters/chapter4_experiments.tex:29,88,115,144,173`

**Interfaces:**
- Consumes: Existing files and document structure.
- Produces: Corrected page geometries and table positioning.

- [ ] **Step 1: Edit geometry configuration in main.tex**
  Change the geometry configuration to include `headheight=15pt`:
  ```latex
  \geometry{
    a4paper,
    left=2.5cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm,
    headheight=15pt
  }
  ```

- [ ] **Step 2: Update table float specifiers in chapters/chapter4_experiments.tex**
  Update the tables on lines 29, 88, 115, 144, and 173 to use `[ht!]` instead of `[h!]` to allow top placement fallback.
  * Line 29:
    ```latex
    \begin{table}[ht!]
    ```
  * Line 88:
    ```latex
    \begin{table}[ht!]
    ```
  * Line 115:
    ```latex
    \begin{table}[ht!]
    ```
  * Line 144:
    ```latex
    \begin{table}[ht!]
    ```
  * Line 173:
    ```latex
    \begin{table}[ht!]
    ```

- [ ] **Step 3: Run compilation check**
  Run: `pdflatex -interaction=nonstopmode main.tex` in `/home/khaitran/project/flmal/IE105_Q11`
  Expected: Compiled successfully. Inspect the output log and ensure that all "fancyhdr headheight too small" warnings and "`!h' float specifier changed to `!ht'" warnings are resolved.

- [ ] **Step 4: Commit changes**
  ```bash
  git add main.tex chapters/chapter4_experiments.tex
  git commit -m "style: fix headheight and float specifier warnings"
  ```

---

### Task 2: Margin Overflows & Variable Formatting

**Files:**
- Modify: `chapters/chapter4_experiments.tex:193,202,205,208,211,224,231`

**Interfaces:**
- Consumes: Task 1 updates.
- Produces: Overflow-free page margins for technical paths/commands.

- [ ] **Step 1: Format long command paragraph on line 193**
  Extract the long `python explain.py --model logreg --top-k 10` command out of inline text and place it in a centered `center` block. Wrap the path in `\path` instead of `\texttt`.
  Target content:
  ```latex
  \textit{\textbf{Lưu ý về tái lập:} Bảng trên được trích xuất từ kết quả giải thích mô hình Logistic Regression trong một phiên thực nghiệm trước đó. Artifact hiện tại trong \texttt{state/explanations.json} chứa kết quả của mô hình MLP (round 3) với các score bị triệt tiêu ($\approx 0$) do hiện tượng bão hòa gradient. Để tái lập bảng này, cần chạy lại module XAI với mô hình logreg đã hội tụ: \texttt{python explain.py --model logreg --top-k 10}.}
  ```
  Replacement content:
  ```latex
  \textit{\textbf{Lưu ý về tái lập:} Bảng trên được trích xuất từ kết quả giải thích mô hình Logistic Regression trong một phiên thực nghiệm trước đó. Artifact hiện tại trong \path{state/explanations.json} chứa kết quả của mô hình MLP (round 3) với các score bị triệt tiêu ($\approx 0$) do hiện tượng bão hòa gradient. Để tái lập bảng này, cần chạy lại module XAI với mô hình logreg đã hội tụ bằng lệnh:}
  \begin{center}
  \texttt{python explain.py --model logreg --top-k 10}
  \end{center}
  ```

- [ ] **Step 2: Update remaining paths and variables to use `\path`**
  Convert the inline file/variable names from `\texttt{...}` with escaped underscores `\_` to `\path{...}` with literal underscores `_`.
  * Line 202:
    Change `\texttt{pslist.avg\_threads}` to `\path{pslist.avg_threads}`
  * Line 205:
    Change `\texttt{process\_services}` to `\path{process_services}`
    Change `\texttt{shared\_process\_services}` to `\path{shared_process_services}`
  * Line 208:
    Change `\texttt{handles.nthread}` to `\path{handles.nthread}`
    Change `\texttt{ldrmodules.not\_in\_load}` to `\path{ldrmodules.not_in_load}`
    Change `\texttt{not\_in\_mem}` to `\path{not_in_mem}`
  * Line 211:
    Change `\texttt{handles.nmutant}` to `\path{handles.nmutant}`
  * Line 224:
    Change `\texttt{state/metrics.json}` to `\path{state/metrics.json}`
    Change `\texttt{state/explanations.json}` to `\path{state/explanations.json}`
  * Line 231:
    Change `\texttt{dashboard\_interactive.py}` to `\path{dashboard_interactive.py}`
    Change `\texttt{flmal-dashboard}` to `\path{flmal-dashboard}`
    Change `\texttt{dashboard.py}` to `\path{dashboard.py}`
    Change `\texttt{dashboard\_flask.py}` to `\path{dashboard_flask.py}`

- [ ] **Step 3: Run compilation check**
  Run: `pdflatex -interaction=nonstopmode main.tex` in `/home/khaitran/project/flmal/IE105_Q11`
  Expected: Verify that the overfull hbox warnings at lines 193, 207-209, and 231-232 are resolved.

- [ ] **Step 4: Commit changes**
  ```bash
  git add chapters/chapter4_experiments.tex
  git commit -m "style: fix margin overflows by formatting variables with \path"
  ```

---

### Task 3: Casing & Notation Consistency

**Files:**
- Modify: `chapters/chapter3_methodology.tex:1`
- Modify: `chapters/chapter2_theory.tex:49,53`

**Interfaces:**
- Consumes: Task 2 updates.
- Produces: Correct chapter title capitalization, math variables, and spelling.

- [ ] **Step 1: Standardize Chapter 3 Title casing**
  In `chapters/chapter3_methodology.tex` line 1:
  Change `\section{Phương pháp thực hiện}` to `\section{PHƯƠNG PHÁP THỰC HIỆN}`

- [ ] **Step 2: Correct Krum notation and Vietnamese typo**
  In `chapters/chapter2_theory.tex`:
  * Line 49:
    Change `n - f - 2` to `K - f - 2`
  * Line 53:
    Change `thay vị chỉ một` to `thay vì chỉ một`

- [ ] **Step 3: Run compilation check**
  Run: `pdflatex -interaction=nonstopmode main.tex` in `/home/khaitran/project/flmal/IE105_Q11`
  Expected: Document compiles without errors. Verify the updated section title in the Table of Contents matches other chapters' uppercase casing.

- [ ] **Step 4: Commit changes**
  ```bash
  git add chapters/chapter3_methodology.tex chapters/chapter2_theory.tex
  git commit -m "docs: standardize chapter title casing and correct math variables"
  ```

---

### Task 4: Model Count Alignment

**Files:**
- Modify: `chapters/chapter4_experiments.tex:24,27`
- Modify: `chapters/appendix.tex:6`

**Interfaces:**
- Consumes: Task 3 updates.
- Produces: Consistent model counts (50 instead of 53).

- [ ] **Step 1: Update model count references in Chapter 4**
  In `chapters/chapter4_experiments.tex`:
  * Line 24: Change `\textbf{53 cấu hình mô hình}` to `\textbf{50 cấu hình mô hình}`
  * Line 27: Change `toàn bộ 53 mô hình` to `toàn bộ 50 mô hình`

- [ ] **Step 2: Update model count reference in Appendix**
  In `chapters/appendix.tex`:
  * Line 6: Change `toàn bộ 53 mô hình` to `toàn bộ 50 mô hình`

- [ ] **Step 3: Run final compilation checks**
  Run:
  ```bash
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  ```
  Expected: Zero warnings from fancyhdr/float placements. No overfull hboxes from the modified files. Table of contents and citations display correctly.

- [ ] **Step 4: Commit changes**
  ```bash
  git add chapters/chapter4_experiments.tex chapters/appendix.tex
  git commit -m "docs: align model count references to match 50 benchmarked models"
  ```
