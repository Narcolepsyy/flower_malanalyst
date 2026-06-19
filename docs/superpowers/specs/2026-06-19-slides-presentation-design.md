# Design Spec: LaTeX Beamer Presentation for IE105_Q11 Report

**Date**: 2026-06-19
**Topic**: Slides Presentation of the Federated Learning & Hybrid QNN Memory Malware Detection Report
**Target File**: `IE105_Q11/slides.tex`
**Output**: `IE105_Q11/slides.pdf`

---

## 1. Goal and Purpose
The goal is to create a professional, academically styled presentation (slides) in LaTeX Beamer corresponding to the final research report. The slides will present the background, methodology (Federated Learning, Byzantine robustness, Hybrid QNN, Differential Privacy, Secure channels, and Web dashboard), experimental results, and conclusion of the project.

---

## 2. Design System and Styling Tokens

- **Document Class**: `beamer`
- **Theme**: `Boadilla` or `Madrid` with clean layouts, customized using specific colors.
- **Color Palette (UIT Themed)**:
  - **Primary Color (Navy)**: `#0F2B5C` (RGB: 15, 43, 92)
  - **Secondary Color (Medium Blue)**: `#2E6BB8` (RGB: 46, 107, 184)
  - **Accent Color (Amber/Gold)**: `#D9A70D` (RGB: 217, 167, 13)
- **Typesetting**:
  - T5 encoding with the `vietnam` package for proper Vietnamese character rendering.
  - Standard Beamer font themes (sans-serif) for high legibility on projectors.
  - Standard shapes for structures, headers, and footer lines.

---

## 3. Slide-by-Slide Outline (18 Slides)

1. **Slide 1: Title Slide (Trang tiêu đề)**
   - Title: "Hệ thống Phát hiện Mã độc Phân tán trên Bộ nhớ với Học liên kết và Học máy Lượng tử"
   - Authors, Institution (UIT), and Date.
2. **Slide 2: Mục tiêu và Nội dung (Agenda)**
   - Overview roadmap of the presentation.
3. **Slide 3: Bối cảnh và Động lực (Motivation)**
   - Cyber threats, obfuscated memory malware (using `Obfuscated-MalMem2022`).
   - Limits of centralized learning: privacy concerns, bandwidth costs $\rightarrow$ transition to FL.
4. **Slide 4: Kiến trúc Hệ thống Tổng quan (System Architecture)**
   - Block scheme of Client-Server FL setup (Flower, gRPC over TLS, Flask Dashboard).
5. **Slide 5: Học Liên kết và Tối ưu hóa (Federated Learning Theory)**
   - FedAvg optimization formulation: minimizing global loss $f(w) = \sum p_k F_k(w)$.
6. **Slide 6: Chiến lược Tổng hợp Byzantine-robust (Robust Aggregation)**
   - Robust aggregators: Median, Trimmed Mean, Krum, Bulyan, and Median-of-Means.
7. **Slide 7: Học máy Lượng tử và Khảo sát Lý thuyết (QML Theory)**
   - Qubits, superposition, entanglement.
   - Summary of theoretical models surveyed: QCNN, QPCA, and QMLP.
8. **Slide 8: Kiến trúc Mô hình Lai (Hybrid QNN Architecture)**
   - Classical Encoder: $55 \rightarrow 64 \rightarrow 16 \rightarrow 4 \rightarrow \text{Tanh} \rightarrow \times \pi$.
   - Quantum Layer: 4-qubit VQC with AngleEmbedding and BasicEntanglerLayers.
9. **Slide 9: Huấn luyện và Gradient Lượng tử (QML Gradients)**
   - Backpropagation for simulation (`default.qubit` with `diff_method="backprop"`) vs. Parameter-Shift rule for physical QPUs.
10. **Slide 10: Cơ chế Bảo vệ Riêng tư (Differential Privacy)**
    - Gradient clipping, Gaussian noise, post-facto Privacy Accounting ($\epsilon, \delta$ calculation).
11. **Slide 11: Bảo mật Kênh truyền (gRPC Secure Channel)**
    - Server-side TLS, CA root chain, SAN configuration (`generate_certs.sh`).
12. **Slide 12: Các Cơ chế Nâng cao (Băng thông & Kiểm toán)**
    - Parameter Quantization (4/8-bit), Top-k sparsification, Audit Hash Chain (SHA-256), and FedProx.
13. **Slide 13: Giao diện Giám sát (Web Dashboard)**
    - Decoupled Flask backend (`/api/status`, `/api/run`), Plotly/Chart.js CDN, status (1s) & XAI (5s) polling.
14. **Slide 14: Thực nghiệm: So sánh Chiến lược Tổng hợp**
    - Performance summary table under IID conditions: FedAvg vs Median vs Krum ($99.83\%$ accuracy).
15. **Slide 15: Thực nghiệm: Centralized vs. Federated cho Hybrid QNN**
    - $93.75\%$ centralized (from `notebooks/quantum_experiments.ipynb`) vs. $99.92\%$ FL (and bottleneck explanation).
16. **Slide 16: Thực nghiệm: Giải thích Đặc trưng (XAI)**
    - Top 10 features importance list and cybersecurity analysis (`pslist.avg_threads` at $1.3751$, `modules.nmodules` at $1.1942$).
17. **Slide 17: Kết luận và Hướng phát triển (Conclusion & Future Work)**
    - Summary of achievements, scale tests (Docker/K8s), real QPU integration, physical edge devices (Raspberry Pi/Jetson Nano).
18. **Slide 18: Trang kết thúc (Q&A)**
    - Closing message and contact info.

---

## 4. Implementation Steps

1. Create a standalone `slides.tex` file referencing the same `ref.bib` bibliography file.
2. Configure slide colors using Beamer color command specifications.
3. Write clean, bulleted slide pages with standard block environments for equations and key findings.
4. Compile using `pdflatex` and `bibtex` to verify warning-free execution.
