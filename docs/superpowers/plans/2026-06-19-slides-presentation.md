# Slides Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a professional LaTeX Beamer presentation slides file matching the final research report.

**Architecture:** A standalone Beamer document `IE105_Q11/slides.tex` utilizing the `Boadilla` theme customized with UIT brand colors (deep navy and amber accents) and proper Vietnamese character typesetting.

**Tech Stack:** LaTeX, Beamer, BibTeX.

## Global Constraints

- Must compile cleanly with `pdflatex` and `bibtex` in the `IE105_Q11/` directory.
- Must use Vietnamese language with proper T5 font encoding.
- All technical terms and figures must align with the core report and code execution configurations.

---

### Task 1: Initialize Beamer Document and Structural Setup

**Files:**
- Create: `IE105_Q11/slides.tex`

**Interfaces:**
- Produces: The document skeleton, title slide, and agenda slide.

- [ ] **Step 1: Write the Beamer document preamble and title setup**

Write code:
```latex
\documentclass[10pt]{beamer}
\usepackage[utf8]{inputenc}
\usepackage[vietnam]{babel}
\usepackage{t5enc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}

% Custom UIT Academic light theme colors
\definecolor{UITnavy}{HTML}{0F2B5C}
\definecolor{UITblue}{HTML}{2E6BB8}
\definecolor{UITgold}{HTML}{D9A70D}

\usetheme{Boadilla}
\usecolortheme[named=UITnavy]{structure}

\setbeamercolor{palette primary}{bg=UITnavy,fg=white}
\setbeamercolor{palette secondary}{bg=UITblue,fg=white}
\setbeamercolor{palette tertiary}{bg=UITnavy,fg=white}
\setbeamercolor{titlelike}{parent=structure,fg=UITnavy}
\setbeamercolor{block title}{bg=UITblue,fg=white}
\setbeamercolor{block body}{bg=black!5,fg=black}

\title[Phát hiện Mã độc Bộ nhớ với FL \& QML]{Hệ thống Phát hiện Mã độc Phân tán trên Bộ nhớ với Học liên kết và Học máy Lượng tử}
\subtitle{Báo cáo Đồ án môn học IE105}
\author[Nhóm Q11]{Nguyễn Văn A \and Trần Văn B \and Lê Văn C}
\institute[UIT]{Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM}
\date{\today}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Nội dung Báo cáo}
    \tableofcontents
\end{frame}

\end{document}
```

- [ ] **Step 2: Run test to verify it compiles**

Run: `pdflatex -interaction=nonstopmode slides.tex` in the `IE105_Q11/` directory.
Expected: PASS with 0 compilation errors and a basic 2-page PDF.

- [ ] **Step 3: Commit**

```bash
git add IE105_Q11/slides.tex
git commit -m "feat: initialize slides presentation and basic setup"
```

---

### Task 2: Implement Slide Content for Background and Theoretical Foundations

**Files:**
- Modify: `IE105_Q11/slides.tex`

**Interfaces:**
- Consumes: The basic setup from Task 1.
- Produces: Slides covering Motivation, System Architecture, FL, Aggregation Strategies, and QML.

- [ ] **Step 1: Insert Background and FL/QML Theory slides before \end{document}**

Modify code in `IE105_Q11/slides.tex`:
```latex
% Replace "\end{document}" in the existing file with the following code blocks:

\section{Bối cảnh \& Lý thuyết}

\begin{frame}{Bối cảnh \& Động lực Đề tài}
    \begin{itemize}
        \item \textbf{Mối đe dọa an ninh mạng:} Mã độc đa hình và kỹ thuật che giấu hành vi trong bộ nhớ RAM (\textit{Obfuscated-MalMem2022}).
        \item \textbf{Hạn chế của mô hình tập trung (Centralized Learning):}
        \begin{itemize}
            \item Nguy cơ rò rỉ dữ liệu nhạy cảm của người dùng (rào cản GDPR, CCPA).
            \item Chi phí băng thông lớn khi truyền tải dữ liệu thô về máy chủ trung tâm.
        \end{itemize}
        \item \textbf{Giải pháp Học liên kết (Federated Learning - FL):} Di chuyển tính toán đến nơi có dữ liệu, bảo mật thông tin tại biên.
    \end{itemize}
\end{frame}

\begin{frame}{Kiến trúc Tổng quan Hệ thống}
    \begin{figure}
        \centering
        \includegraphics[width=0.85\textwidth]{asset/download.jpeg}
        \caption{Sơ đồ phân tán gRPC FL kết hợp Web Dashboard}
    \end{figure}
\end{frame}

\begin{frame}{Học Liên kết \& Tối ưu hóa FedAvg}
    \begin{block}{Bài toán Tối ưu hóa Toàn cục}
        $$\min_{w \in \mathbb{R}^d} f(w) = \sum_{k=1}^{K} p_k F_k(w)$$
        Trong đó $p_k = \frac{n_k}{n}$, $F_k(w)$ là hàm mất mát cục bộ của client $k$.
    \end{block}
    \begin{itemize}
        \item \textbf{FedAvg (Federated Averaging):}
        \begin{enumerate}
            \item Server phát tham số toàn cục $w_t$ đến các client.
            \item Client huấn luyện cục bộ qua $E$ epochs và gửi $w_{t+1}^k$ về Server.
            \item Server tính trung bình trọng số cập nhật: $w_{t+1} = \sum p_k w_{t+1}^k$.
        \end{enumerate}
    \end{itemize}
\end{frame}

\begin{frame}{Chiến lược Tổng hợp Bền vững (Byzantine-Robust)}
    Đối mặt với nguy cơ bị đầu độc mô hình từ các client độc hại:
    \begin{itemize}
        \item \textbf{Median:} Lấy trung vị từng tọa độ của các gradient gửi về.
        \item \textbf{Trimmed Mean:} Loại bỏ tỷ lệ $\beta$ các cập nhật biên lớn/nhỏ trước khi tính trung bình.
        \item \textbf{Krum:} Chọn ra một cập nhật mô hình thực tế có khoảng cách Euclidean nhỏ nhất đến số đông lân cận.
        \item \textbf{Bulyan \& Median-of-Means (MoM):} Loại bỏ các cập nhật phá hoại quy mô lớn, gia tăng tính bền vững hệ thống.
    \end{itemize}
\end{frame}

\begin{frame}{Cơ sở Máy học Lượng tử (QML)}
    \begin{itemize}
        \item \textbf{Qubit \& Nguyên lý Lượng tử:} Trạng thái chồng chập $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ kết hợp sự vướng víu giúp mở rộng không gian biểu diễn phi tuyến.
        \item \textbf{Khảo sát lý thuyết ba kiến trúc tiêu biểu:}
        \begin{itemize}
            \item Mạng nơ-ron tích chập lượng tử (QCNN).
            \item Phân tích thành phần chính lượng tử (QPCA).
            \item Mạng Perceptron đa lớp lượng tử (QMLP).
        \end{itemize}
        \item \textbf{Ứng dụng thực tế:} Hiện thực hóa một mô hình mạng nơ-ron lượng tử lai (Hybrid QNN) 4-qubit.
    \end{itemize}
\end{frame}

\end{document}
```

- [ ] **Step 2: Run test to verify it compiles**

Run: `pdflatex -interaction=nonstopmode slides.tex`
Expected: PASS with 0 compilation errors.

- [ ] **Step 3: Commit**

```bash
git add IE105_Q11/slides.tex
git commit -m "feat: implement slides for background and theory"
```

---

### Task 3: Implement Slide Content for Methodology

**Files:**
- Modify: `IE105_Q11/slides.tex`

**Interfaces:**
- Consumes: The background slides from Task 2.
- Produces: Slides covering Preprocessing, Hybrid Architecture, Quantum Gradients, Differential Privacy, Secure channels, and Advanced features.

- [ ] **Step 1: Insert Methodology slides before \end{document}**

Modify code in `IE105_Q11/slides.tex`:
```latex
% Replace "\end{document}" in the existing file with the following code blocks:

\section{Phương pháp Thực hiện}

\begin{frame}{Xử lý dữ liệu \& Tiền xử lý tại Biên}
    \begin{itemize}
        \item \textbf{Bộ dữ liệu bộ nhớ RAM Obfuscated-MalMem2022:}
        \begin{itemize}
            \item 58,596 bản ghi, cân bằng giữa lớp Lành tính (Benign) và Mã độc (Malicious).
            \item 55 đặc trưng số học (sau khi loại bỏ \texttt{Class} và \texttt{Category}).
        \end{itemize}
        \item \textbf{Quy trình tiền xử lý:}
        \begin{enumerate}
            \item \textbf{Data Cleaning}: Loại bỏ các nhãn định danh họ mã độc.
            \item \textbf{Normalization}: Chuẩn hóa Z-score scaling đưa dữ liệu về $\mu=0, \sigma=1$.
            \item \textbf{Partitioning}: Phân chia dữ liệu cho các client sử dụng phân phối Dirichlet với hệ số $\alpha = 0.5$ (thiết lập môi trường Non-IID).
        \end{enumerate}
    \end{itemize}
\end{frame}

\begin{frame}{Kiến trúc Mô hình Lai (Hybrid QNN)}
    \begin{block}{Kiến trúc "Bánh kẹp" (Sandwich)}
        \begin{enumerate}
            \item \textbf{Classical Encoder:} Mạng nơ-ron tuyến tính giảm chiều:
            $$55 \xrightarrow{\text{Linear}} 64 \xrightarrow{\text{ReLU}} 16 \xrightarrow{\text{ReLU}} 4 \xrightarrow{\text{Tanh}} \xrightarrow{\times \pi} \text{Rotation Angles}$$
            \item \textbf{Quantum Layer (VQC):} Mạch 4-qubit:
            \begin{itemize}
                \item Nhúng góc \textit{AngleEmbedding} (cổng xoay $R_x$).
                \item Biến phân vướng víu \textit{BasicEntanglerLayers} ($2$ lớp biến phân).
                \item Đo lường giá trị kỳ vọng Pauli-Z ($\langle PauliZ \rangle$) trên qubit 0.
            \end{itemize}
            \item \textbf{Head Classifier:} Lớp tuyến tính ($1 \rightarrow 1$) kết hợp hàm mất mát \textit{BCEWithLogitsLoss}.
        \end{enumerate}
    \end{block}
\end{frame}

\begin{frame}{Huấn luyện \& Tính toán Gradient Lượng tử}
    \begin{itemize}
        \item Giao diện PyTorch tích hợp qua PennyLane \texttt{qnn.TorchLayer} giúp truyền tải tham số trong Flower trong suốt.
        \item \textbf{Tính toán Gradient trong Huấn luyện:}
        \begin{itemize}
            \item \textbf{Giả lập (Simulator):} Sử dụng bộ giả lập \texttt{default.qubit} với lan truyền ngược trực tiếp (\texttt{diff\_method="backprop"}) để tối ưu tốc độ CPU.
            \item \textbf{Vật lý (QPU):} Định hình sử dụng phương pháp \textit{Parameter-Shift Rule} dịch chuyển góc quay $\pm s$ để tính đạo hàm:
            $$\frac{\partial f}{\partial \theta} \approx \frac{f(\theta + s) - f(\theta - s)}{2 \sin(s)}$$
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Cơ chế bảo mật Differential Privacy (DP)}
    \begin{itemize}
        \item Tích hợp thư viện \textbf{Opacus} chuyển đổi optimizer thành \texttt{DPOptimizer}.
        \item \textbf{Quy trình bảo mật gradient:}
        \begin{enumerate}
            \item Cắt chuẩn gradient cục bộ của từng mẫu (Gradient Clipping).
            \item Cộng nhiễu Gaussian được điều khiển bởi hệ số \texttt{--dp-noise-multiplier}.
        \end{enumerate}
        \item \textbf{Tính toán Ngân sách Riêng tư ($\epsilon, \delta$):}
        Epsilon không ràng buộc đầu vào mà được đo lường hậu nghiệm (post-facto) thông qua cơ chế Privacy Accountant của Opacus sau khi huấn luyện.
    \end{itemize}
\end{frame}

\begin{frame}{Bảo mật Kênh truyền \& Các Cơ chế Nâng cao}
    \begin{itemize}
        \item \textbf{Secure Channel (TLS Phía Máy chủ):}
        Sử dụng gRPC bảo mật qua Root CA nội bộ cấp phát chứng chỉ SSL/TLS cho máy chủ (\texttt{server.crt}), tích hợp SAN tránh lỗi xác thực domain.
        \item \textbf{Các Cơ chế Nâng cao Đã Hiện thực:}
        \begin{itemize}
            \item \textbf{Giảm băng thông:} Lượng tử hóa tham số (4/8-bit), thưa thớt hóa Top-k.
            \item \textbf{Chống độc độc:} Giới hạn biên độ gradient, bộ lọc thống kê ngoại lai Z-score (FLANDERS).
            \item \textbf{Kiểm toán}: Chuỗi băm audit bảo mật SHA-256 trên log hiệu năng qua từng vòng.
            \item \textbf{Tối ưu hóa & Phân tích}: Thuật toán FedProx xử lý dữ liệu Non-IID; đo lường ROC-AUC, PR-AUC và Confusion Matrix.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Giao diện Giám sát Dashboard thời gian thực}
    \begin{itemize}
        \item Backend được xây dựng bằng Flask (\path{dashboard_interactive.py}).
        \item Kiến trúc tách biệt dựa trên tệp JSON chia sẻ (\path{state/metrics.json} và \path{state/explanations.json}).
        \item \textbf{API chính}: `/api/status` và `/api/run`.
        \item \textbf{Cơ chế Polling AJAX}: Cập nhật trạng thái định kỳ 1 giây, nạp kết quả XAI định kỳ 5 giây.
        \item Trực quan hóa qua Plotly.js (đồ thị hiệu năng) và Chart.js (đồ thị cột XAI).
    \end{itemize}
\end{frame}

\end{document}
```

- [ ] **Step 2: Run test to verify it compiles**

Run: `pdflatex -interaction=nonstopmode slides.tex`
Expected: PASS with 0 compilation errors.

- [ ] **Step 3: Commit**

```bash
git add IE105_Q11/slides.tex
git commit -m "feat: implement slides for methodology, security, and dashboard"
```

---

### Task 4: Implement Slide Content for Experiments, Results, and Conclusions

**Files:**
- Modify: `IE105_Q11/slides.tex`

**Interfaces:**
- Consumes: The methodology slides from Task 3.
- Produces: Slides covering Aggregation benchmark results, QNN comparison, XAI cybersecurity analysis, and Conclusions/Future work.

- [ ] **Step 1: Insert Results and Conclusion slides before \end{document}**

Modify code in `IE105_Q11/slides.tex`:
```latex
% Replace "\end{document}" in the existing file with the following code blocks:

\section{Thực nghiệm \& Kết quả}

\begin{frame}{Thực nghiệm: So sánh Chiến lược Tổng hợp}
    Đánh giá hiệu năng huấn luyện phân tán (Logistic Regression, IID, 2 clients, 5 rounds):
    \begin{table}
        \centering
        \begin{tabular}{|l|c|c|c|c|c|}
            \hline
            \textbf{Strategy} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Precision} & \textbf{Recall} & \textbf{Loss} \\ \hline
            FEDAVG & 0.9983 & 0.9983 & 0.9981 & 0.9985 & 0.0090 \\ \hline
            MEDIAN & 0.9983 & 0.9983 & 0.9981 & 0.9985 & 0.0090 \\ \hline
            KRUM & 0.9983 & 0.9983 & 0.9977 & 0.9989 & 0.0090 \\ \hline
        \end{tabular}
        \caption{So sánh hiệu năng giữa các chiến lược tổng hợp}
    \end{table}
    \begin{itemize}
        \item Trong điều kiện dữ liệu IID sạch không có tấn công, cả ba chiến lược đều hội tụ hiệu quả về cùng kết quả tối ưu toàn cục ($\approx 99.83\%$), đúng như kỳ vọng lý thuyết.
    \end{itemize}
\end{frame}

\begin{frame}{Thực nghiệm: Centralized vs. FL cho Hybrid QNN}
    So sánh mô hình Hybrid QNN trong hai kịch bản huấn luyện:
    \begin{table}
        \centering
        \begin{tabular}{|l|c|c|}
            \hline
            \textbf{Chỉ số (Metric)} & \textbf{Centralized (Notebook)} & \textbf{Federated Learning} \\ \hline
            Số vòng lặp & 30 Epochs & 5 Rounds \\ \hline
            \textbf{Độ chính xác (Accuracy)} & $\approx 93.75\%$ & \textbf{99.92\%} \\ \hline
            Thời gian tổng cộng & $\approx 120$s & $\approx 65$s \\ \hline
            Bảo mật dữ liệu & Không & \textbf{Có (Dữ liệu tại chỗ)} \\ \hline
        \end{tabular}
        \caption{So sánh Centralized vs. Federated Learning cho Hybrid QNN}
    \end{table}
    \begin{itemize}
        \item \textbf{Giải thích độ chính xác lượng tử:} Độ chính xác cao ($\approx 99.92\%$) thực chất phần lớn phụ thuộc vào năng lực trích xuất phi tuyến mạnh mẽ của bộ mã hóa cổ điển ($55 \rightarrow 64 \rightarrow 16 \rightarrow 4$) trước khi đi qua mạch lượng tử (bottleneck đo 1 qubit).
    \end{itemize}
\end{frame}

\begin{frame}{Thực nghiệm: Giải thích Đặc trưng (XAI)}
    Sử dụng XAI trích xuất trọng số của mô hình toàn cục ở vòng 5:
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{table}
                \centering
                \scalebox{0.75}{
                    \begin{tabular}{|l|l|c|}
                        \hline
                        \textbf{Hạng} & \textbf{Tên đặc trưng} & \textbf{Điểm số} \\ \hline
                        1 & \texttt{pslist.avg\_threads} & 1.3751 \\ \hline
                        2 & \texttt{modules.nmodules} & 1.1942 \\ \hline
                        3 & \texttt{svcscan.process\_services} & 1.1857 \\ \hline
                        4 & \texttt{svcscan.shared\_process\_services} & 1.0359 \\ \hline
                        5 & \texttt{handles.nmutant} & 0.9672 \\ \hline
                    \end{tabular}
                }
                \caption{Top đặc trưng quan trọng}
            \end{table}
        \end{column}
        \begin{column}{0.5\textwidth}
            \textbf{Phân tích an ninh mạng:}
            \begin{itemize}
                \item \texttt{pslist.avg\_threads}: Mã độc Ransomware, Spyware thường tạo nhiều luồng song song để phá hoại hệ thống.
                \item \texttt{svcscan.*}: Dấu hiệu đăng ký dịch vụ để tự khởi chạy cùng hệ thống (Persistence).
                \item \texttt{handles.nmutant}: Sử dụng Mutex đồng bộ hóa và duy trì độc quyền chạy.
            \end{itemize}
        \end{column}
    \end{columns}
\end{frame}

\section{Kết luận}

\begin{frame}{Kết luận \& Hướng phát triển}
    \begin{block}{Kết quả Đạt được}
        \begin{itemize}
            \item Triển khai thành công hệ thống Học liên kết phát hiện mã độc bộ nhớ.
            \item Xây dựng mô hình Hybrid QNN có khả năng hội tụ nhanh, giải quyết giới hạn phần cứng lượng tử thông qua khối mã hóa cổ điển.
            \item Tích hợp Dashboard Flask trực quan hóa real-time và XAI.
        \end{itemize}
    \end{block}
    \begin{block}{Hướng Phát triển Tiếp theo}
        \begin{itemize}
            \item Kết nối các bộ xử lý lượng tử thực tế (QPU) qua Amazon Braket, IBM Quantum.
            \item Thử nghiệm mở rộng số lượng client lớn qua Docker/Kubernetes.
            \item Đóng gói triển khai trên các thiết bị biên vật lý thực tế (Raspberry Pi).
        \end{itemize}
    \\end{block}
\end{frame}

\begin{frame}
    \centering
    \Huge \color{UITnavy} \textbf{CẢM ƠN THẦY VÀ CÁC BẠN LẮNG NGHE!}\\
    \vspace{1cm}
    \normalsize Nhóm Q11 - Khoa Mạng máy tính và Truyền thông - UIT
\end{frame}

\end{document}
```

- [ ] **Step 2: Run test to verify it compiles**

Run: `pdflatex -interaction=nonstopmode slides.tex`
Expected: PASS with 0 compilation errors.

- [ ] **Step 3: Commit**

```bash
git add IE105_Q11/slides.tex
git commit -m "feat: complete slide content for results and conclusion"
```

---

### Task 5: Compile Final Slides with Bibliography and Clean Up

**Files:**
- Modify: `IE105_Q11/slides.tex`

**Interfaces:**
- Consumes: The complete slides from Task 4.
- Produces: The final `slides.pdf` with properly formatted references and no layout overflow warnings.

- [ ] **Step 1: Check bibliography compilation for slides**

Add references block to the slides before the final thank you slide and run compiling commands:
```latex
\begin{frame}[allowframebreaks]{Tài liệu Tham khảo}
    \bibliographystyle{ieeetr}
    \bibliography{ref}
\end{frame}
```
Run compiling pipeline:
`pdflatex -interaction=nonstopmode slides.tex && bibtex slides && pdflatex -interaction=nonstopmode slides.tex && pdflatex -interaction=nonstopmode slides.tex`
Expected: PASS with 0 errors, 0 BibTeX warnings, and output `slides.pdf` generated.

- [ ] **Step 2: Commit**

```bash
git add IE105_Q11/slides.tex
git commit -m "docs: compile final slides and integrate bibliography"
```
