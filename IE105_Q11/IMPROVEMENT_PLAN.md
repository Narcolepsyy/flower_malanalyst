# Kế hoạch Cải thiện Báo cáo IE105_Q11 — Đối chiếu với Mã nguồn

> **Mục đích:** Tài liệu này đối chiếu từng tuyên bố trong báo cáo LaTeX (`IE105_Q11/`) với **mã nguồn và dữ liệu thực tế** trong repo `flmal`, chỉ ra các điểm sai lệch (discrepancy), và đề xuất cách sửa chi tiết, minh bạch. Mỗi mục có trích dẫn `file:line` để kiểm chứng.
>
> Ngày lập: 2026-06-19 · Đối chiếu trên commit `d259707`.

---

## 0. Tóm tắt mức độ nghiêm trọng

| # | Vấn đề | Mức độ | Loại |
|---|--------|--------|------|
| D1 | "mTLS / xác thực 2 chiều" — **code chỉ làm TLS 1 chiều** (client không gửi cert) | 🔴 Nghiêm trọng | Sai sự thật kỹ thuật |
| D2 | Bảng so sánh FedAvg/Median/Krum gán cho **MLP Deep + Non-IID**, nhưng metric thực tế chạy bằng **logreg + IID** | 🔴 Nghiêm trọng | Sai gán thí nghiệm |
| D3 | Bảng XAI (Top-10, score 1.37...) gán cho **Logistic Regression**, nhưng `explanations.json` thực tế là **MLP**, **toàn bộ score = 0.0** | 🔴 Nghiêm trọng | Số liệu không tái lập |
| D4 | QML "Centralized 93.75% vs FL 99.92%" và bảng 5 vòng — **không có file metric quantum nào trong `state/`** | 🔴 Nghiêm trọng | Số liệu không truy vết được |
| D5 | Encoder lượng tử mô tả kết thúc ở `Latent(4)` + RX; code có thêm `Tanh()` và scale ×π; embedding là `AngleEmbedding` (mặc định RX) | 🟡 Trung bình | Thiếu chính xác |
| D6 | DP: báo cáo nhấn mạnh `--dp-epsilon` điều khiển nhiễu; code thực tế dùng `noise_multiplier` cố định, epsilon chỉ được *báo cáo lại* | 🟡 Trung bình | Hiểu sai cơ chế |
| D7 | Tên lớp & file: "metrics.json", "LoggedFedAvg/RobustLoggedFedAvg" — phần lớn đúng, nhưng dashboard và vài chi tiết lệch | 🟡 Trung bình | Lệch nhỏ |
| D8 | Báo cáo bỏ sót nhiều tính năng **đã có trong code** (Bulyan, MoM, FedNova-arg, quantization, top-k, server-DP, audit-hash, ROC/PR-AUC) | 🟢 Cơ hội | Thiếu sót (under-claiming) |
| D9 | Phần 1 (Giới thiệu) hứa QCNN/QPCA/QMLP + dữ liệu PE/Android; code chỉ có 1 mô hình Hybrid QNN + dữ liệu MalMem (memory dump) | 🟡 Trung bình | Phóng đại phạm vi |

---

## 1. Bảo mật kênh truyền — mTLS (D1) 🔴

**Báo cáo nói** (ch3 dòng 34; ch2 dòng 100–116; ch5 dòng 296–306):
> "hệ thống triển khai giao thức mTLS (Mutual TLS). Cả Server và Client đều phải xác thực danh tính… Phía Server: cần cả 3 thành phần để xác thực ngược lại client."

**Code thực tế:**
- `server.py:127-136` — server **có** truyền tuple `(server_cert, server_key, ca_cert)` cho `start_server`.
- `client.py:295-305` — client **chỉ** đọc `--ssl-ca-certfile` và truyền `root_certificates=...`. **Client không bao giờ gửi `--ssl-certfile/--ssl-keyfile` của mình** (các tham số này thậm chí không tồn tại theo hướng dùng — `client.py:159-161` có khai báo `--ssl-certfile/--ssl-keyfile` nhưng `main()` **không dùng** chúng).
- `start_client(...)` chỉ nhận `root_certificates` → đây là **TLS phía server (one-way)**, client xác thực server, **server KHÔNG xác thực client bằng cert**.
- `certs/generate_certs.sh:50-72` có sinh `client_i.crt/key` với `extendedKeyUsage = clientAuth`, nhưng **không có đường dẫn nào nạp chúng vào kết nối**.

**→ Sai lệch:** Hệ thống hiện là **server-side TLS**, *không phải* mTLS. Tuyên bố "chống Sybil Attack nhờ mTLS" (ch2:104) là **không đúng** với code hiện tại.

**Cách sửa (chọn 1):**
- **(A) Sửa báo cáo cho khớp code (nhanh, trung thực):** Đổi mọi chỗ "mTLS / xác thực hai chiều" → "TLS phía máy chủ (server-authenticated TLS), client xác thực server qua Root CA". Bỏ tuyên bố chống Sybil bằng cert. Ghi chú rằng cert client đã được sinh sẵn và *mTLS là hướng mở rộng*.
- **(B) Sửa code cho khớp báo cáo (đúng kỹ thuật hơn):** Trong `client.py:main()`, đọc `--ssl-certfile/--ssl-keyfile` và truyền vào `start_client` (Flower hỗ trợ qua tham số mTLS của phiên bản tương ứng); ở server kiểm tra phiên bản `flwr` có bật xác thực client. Sau đó giữ nguyên báo cáo.
- **Khuyến nghị:** Làm (A) trước để báo cáo trung thực; nêu (B) trong "Hướng phát triển".

---

## 2. Bảng so sánh chiến lược tổng hợp gán sai thí nghiệm (D2) 🔴

**Báo cáo nói** (ch5 dòng 69, 137–166): Hệ thống Flower triển khai với **`MLP Deep` trên phân phối Non-IID**; bảng so sánh FedAvg/Median/Krum (Acc 0.9983 / 0.9983 / 0.9979).

**Dữ liệu thực tế** (`state/metrics_fedavg.json` metadata, `state/results_summary.csv`):
```
model_name: logreg        ← KHÔNG phải MLP Deep
partition_method: iid     ← KHÔNG phải Non-IID
num_rounds: 5, num_clients: 2, epochs: 2, batch_size: 32, lr: 0.05
preset: dev, malicious_clients: 0
```
- `results_summary.csv`: FEDAVG/MEDIAN/KRUM đều = **0.9983 acc, 0.9983 F1**. KRUM trong CSV có Precision 0.9977 / Recall 0.9989. Báo cáo ghi KRUM Acc=0.9979 — **không khớp** CSV (0.9983).
- Vì chạy **IID + 2 client + không có client độc hại**, ba chiến lược cho kết quả gần như **trùng nhau** — đây là điều kiện *không* làm nổi bật ưu thế robust aggregation. Lập luận ch5:161 ("Median loại outlier trong môi trường Non-IID") **không được dữ liệu hậu thuẫn** vì thí nghiệm chạy IID, không có outlier/tấn công.

**→ Sai lệch:** (1) Sai mô hình (logreg ↔ MLP Deep); (2) sai phân phối (IID ↔ Non-IID); (3) số Krum không khớp file; (4) diễn giải robust aggregation không có cơ sở thực nghiệm.

**Cách sửa:**
1. Sửa câu mô tả: bảng so sánh aggregation được chạy với **mô hình `logreg`, phân phối IID, 2 client, 5 vòng** (trích đúng metadata). 
2. Sửa số Krum theo `results_summary.csv` (0.9983) hoặc nói rõ nguồn nếu lấy từ file khác.
3. Viết lại phần phân tích: *"Trong điều kiện IID không tấn công, ba chiến lược hội tụ về cùng kết quả — đúng như kỳ vọng lý thuyết."* Để **thực sự** chứng minh ưu thế robust aggregation, chạy lại với `--partition-method noniid` và inject client độc hại (xem mục 8 — code đã có `malicious_clients` trong metadata và FLANDERS filter).
4. Nếu muốn giữ tuyên bố "MLP Deep": phải chạy lại bộ FedAvg/Median/Krum với `--model mlp` và cập nhật toàn bộ số liệu + ảnh hội tụ.

---

## 3. Bảng XAI không tái lập được (D3) 🔴

**Báo cáo nói** (ch5 dòng 168–193): Áp XAI lên **Logistic Regression** (round 5), Top-10 đặc trưng với score như `pslist.avg_threads = 1.3751`, `modules.nmodules = 1.1942`…

**Dữ liệu thực tế** (`state/explanations.json`):
```
model_type: mlp          ← KHÔNG phải logreg
round: 3                 ← KHÔNG phải round 5
method: auto
TẤT CẢ score = 0.0       ← gradient saliency bị triệt tiêu (vanishing), rơi về 0
```
Top features hiện tại (thứ tự & tên khác báo cáo): `svcscan.process_services, svcscan.nservices, pslist.nppid, handles.nmutant, pslist.avg_threads, …` — **tất cả 0.0**.

- Code `explain.py:94-100`: MLP dùng *gradient-based saliency* trung bình trên `background_size` (mặc định 256 — khớp công thức ch3:193, đúng N=256).
- Code `explain.py:90-92`: logreg importance = `|weights|` (khớp công thức ch3:184 `I_j=|w_j|`).
- **Nhưng** file đầu ra hiện tại là MLP với score=0.0 → con số 1.3751… trong báo cáo **không đến từ artifact hiện có**. Có thể lấy từ một lần chạy logreg cũ không còn lưu, hoặc nhập tay.
- `explain.py:208-211` có cơ chế fallback: nếu gradient ≈ 0 thì chuyển sang `_mlp_weight_importance`. Việc file vẫn toàn 0.0 cho thấy fallback **chưa được kích hoạt đúng** hoặc weight cũng suy biến.

**→ Sai lệch:** Bảng XAI gán cho logreg/round-5 nhưng artifact là mlp/round-3/score=0; con số không tái lập.

**Cách sửa:**
1. **Tái sinh artifact đúng:** chạy 1 mô hình logreg đủ vòng, lưu `state/latest_model.npz`, rồi `python explain.py --model logreg --top-k 10`. Lấy đúng Top-10 + score từ `state/explanations.json` mới và dán vào bảng. Nêu rõ "round = N" theo `round` trong JSON.
2. Nếu muốn giữ phần MLP: phải xử lý bug score=0 (xem mục 10) rồi mới trích số.
3. Thêm 1 câu minh bạch về phương pháp đã dùng để sinh bảng (model nào, round nào, lệnh nào) — đảm bảo *reproducibility*.
4. Phần "phân tích an ninh mạng" (ch5:197-214) hiện diễn giải các feature như `pslist.avg_threads`, `svcscan.*`, `handles.nmutant` — các tên này **có thật** trong dataset (khớp `load_feature_names`), nên phần định tính giữ được; chỉ cần khớp lại thứ hạng/score với artifact thật.

---

## 4. Kết quả Quantum không truy vết được (D4) 🔴

**Báo cáo nói** (ch5 dòng 82–135): Bảng Hybrid QNN qua 5 vòng (99.58%→99.92%, loss 0.274→0.021); bảng "Centralized 93.75% vs FL 99.92%"; ~22s/vòng.

**Dữ liệu thực tế:**
- `state/` **không có** file `metrics_quantum*` / `model_quantum*` nào (đã liệt kê toàn bộ: chỉ có fedavg, median, trimmed, krum, bulyan, mom, catboost). → **Không có artifact nào hậu thuẫn** bảng quantum.
- Code `HybridQuantumModel` (`models/quantum.py`) **có thật và chạy được** (PennyLane `default.qubit`, 4 qubit, `qnn.TorchLayer`), nhưng số liệu trong báo cáo không đến từ pipeline FL đã lưu state.
- Con số "Centralized 93.75%" được mô tả là từ "Notebook" (ch5:120) — nằm ngoài repo code chính, không kiểm chứng được ở đây.

**→ Sai lệch:** Các bảng quantum (5 vòng FL; centralized-vs-FL) là **claim mạnh** ("FL vượt centralized cho QML") nhưng **không có dữ liệu lưu trữ** trong repo để tái lập.

**Cách sửa:**
1. **Chạy thật và lưu state:** `python server.py --agg-method fedavg ...` với client `--model hybrid-quantum`, hoặc dùng `run_experiments.py` nếu hỗ trợ; lưu `state/metrics_quantum.json`. Trích bảng 5 vòng từ file thật.
2. Nếu số "93.75% centralized" lấy từ notebook: **đưa notebook vào repo** (`notebooks/`) và trích dẫn đường dẫn, hoặc hạ cấp tuyên bố thành "thử nghiệm sơ bộ ngoài pipeline chính".
3. Câu "FL vượt trội centralized cho QML" (ch5:112,135) là một **khẳng định khoa học mạnh** — cần: cùng seed, cùng dữ liệu, nhiều lần chạy (mean±std). Hiện chỉ 1 con số → nên hạ thành "trong thử nghiệm của chúng tôi, … ; cần thêm thí nghiệm để khẳng định".
4. Bổ sung đoạn giải thích **vì sao chỉ đo PauliZ trên qubit 0** và head `Linear(1,1)` — đây là bottleneck biểu diễn rất hẹp (xem mục 5), nên độ chính xác ~99% chủ yếu đến từ **encoder cổ điển** chứ không phải mạch lượng tử. Tính minh bạch này quan trọng.

---

## 5. Mô tả kiến trúc Quantum chưa khớp code (D5) 🟡

**Báo cáo** (ch3 dòng 128–145, 163-164):
- Encoder: `Input(57) → 64 → 16 → Latent(4)` (kết thúc ở Linear 4).
- Embedding qua cổng `R_x` (Angle Encoding); `BasicEntanglerLayers`; đo `PauliZ` qubit 0; head `Linear(1→1)` + Sigmoid.

**Code** (`models/quantum.py:63-73, 84-97`):
- Encoder thực tế: `Linear(57,64) → ReLU → Linear(64,16) → ReLU → Linear(16,4) → **Tanh()**` (báo cáo **thiếu Tanh** ở cuối).
- Sau encoder còn nhân `× 3.14159` để đưa về ~[-π,π] (`quantum.py:88`) — báo cáo nói Tanh "scale to [-π,π]" trong comment nhưng thân bài không nêu bước ×π.
- Embedding: `qml.AngleEmbedding` (mặc định trục X ⇒ tương đương RX — *đúng tinh thần* nhưng nên ghi rõ "AngleEmbedding" thay vì chỉ "R_x").
- `n_layers = 2` (mặc định) — báo cáo không nêu số lớp biến phân.
- Head: `Linear(1,1)`; loss = `BCEWithLogitsLoss` → **Sigmoid nằm trong loss**, không phải lớp Sigmoid tường minh ở forward (`quantum.py:82,96`). Báo cáo nói "Sigmoid activation" — đúng về xác suất đầu ra nhưng nên nói rõ dùng `BCEWithLogitsLoss`.
- Công thức Parameter-Shift (ch3:170-171): code dùng `diff_method="backprop"` trên `default.qubit` (`quantum.py:51`) ⇒ **KHÔNG dùng parameter-shift** mà dùng backprop của simulator. → **Sai cơ chế gradient được trình bày.**

**Cách sửa:**
1. Cập nhật sơ đồ encoder: thêm `→ Tanh → (×π)`.
2. Ghi "AngleEmbedding (RX)" + "BasicEntanglerLayers, `n_layers=2`".
3. Sửa phần gradient: nêu rõ **dùng backpropagation của bộ giả lập `default.qubit`** (diff_method="backprop"); có thể giữ đoạn parameter-shift như "lý thuyết tổng quát" nhưng phải nói rõ implementation hiện tại dùng backprop.
4. Nói rõ head rất hẹp (1→1) và chỉ đo 1 qubit → mạch lượng tử đóng góp hạn chế; phần lớn năng lực biểu diễn nằm ở encoder cổ điển (minh bạch khoa học).

---

## 6. Differential Privacy: cơ chế epsilon (D6) 🟡

**Báo cáo** (ch3 dòng 257-266): `--dp-epsilon` (mặc định 1.0) là "ngân sách riêng tư", "ε càng nhỏ bảo vệ càng cao"; Opacus biến optimizer thành `DPOptimizer`.

**Code** (`models/dp_mlp.py:137-145, 174-175`):
- `PrivacyEngine.make_private(...)` được gọi với **`noise_multiplier`** và **`max_grad_norm`** — **KHÔNG dùng `make_private_with_epsilon`**. Nghĩa là **`target_epsilon` KHÔNG điều khiển lượng nhiễu**; nhiễu do `noise_multiplier` (mặc định 1.0) quyết định.
- `target_epsilon` chỉ được lưu (`self.target_epsilon`) và **không ảnh hưởng huấn luyện**; epsilon thực tế được *đo lại* sau train qua `get_epsilon(delta)` (`dp_mlp.py:175`).
- Gradient clipping per-sample + Gaussian noise: **đúng** (Opacus làm điều này), khớp công thức ch3:248-252.

**→ Sai lệch:** Báo cáo ngụ ý đặt `--dp-epsilon=1.0` sẽ ép mô hình đạt ε=1.0. Thực tế ε là **đầu ra đo được**, không phải ràng buộc đầu vào.

**Cách sửa:**
- **(A) Sửa báo cáo:** Nói rõ cơ chế hiện tại điều khiển riêng tư qua `noise_multiplier` + `max_grad_norm`; `epsilon`/`delta` dùng để **kế toán (accounting) và báo cáo** mức riêng tư đạt được sau huấn luyện (`get_epsilon`). 
- **(B) Sửa code:** đổi sang `make_private_with_epsilon(target_epsilon=…, target_delta=…, epochs=…)` để ε thực sự là ràng buộc. Nếu làm (B), báo cáo giữ nguyên ý.
- Khuyến nghị (A) + ghi (B) là cải tiến.

---

## 7. Dashboard: tên file, polling, SSL, thư viện (D7) 🟡

**Báo cáo** (ch3 218–239; ch5 216–256): `dashboard_flask.py` là backend Flask chính; route `/` và `/update_metrics`; polling **2000ms**; Chart.js (hoặc Plotly); SSL qua `--cert`.

**Code thực tế:**
- `dashboard_flask.py` và `dashboard.py` chỉ là **shim deprecated** (`dashboard_flask.py:1-22`) — in `DeprecationWarning` rồi gọi `dashboard_interactive.main()`. **Backend thật là `dashboard_interactive.py`** (CLI `flmal-dashboard --view live|comparison|explain`).
- Route thật: dùng `/api/status`, `/api/run`, … (`dashboard_interactive.py:179, 922-928`), **không có** `/update_metrics`. App chạy bằng `app.run(host, port=8503)`.
- Polling: có `setInterval(...,1000)` cho trạng thái chạy (`:624`) và `setInterval(loadExplanations,5000)` (`:901`) — **không phải đúng 2000ms** như báo cáo.
- Thư viện: **cả Plotly (2.35.2) lẫn Chart.js (4.4.1)** đều nạp từ CDN (`:199-200`); Chart.js dùng cho bar chart XAI (`:807`).
- SSL `--cert`: `parse_args` của dashboard **không có** `--cert`; `app.run` không bật `ssl_context`. → Tuyên bố "Dashboard bảo vệ HTTPS qua `--cert`" (ch5:255-256) **không đúng**.

**Cách sửa:**
1. Đổi tên file backend → `dashboard_interactive.py` (hoặc lệnh `flmal-dashboard --view live`). Nêu rõ `dashboard.py/dashboard_flask.py` là wrapper tương thích.
2. Sửa route: mô tả `/`, `/api/status`, `/api/run`, `/api/...` đúng như code; bỏ `/update_metrics` (hoặc đổi tên).
3. Sửa polling: trạng thái 1s, giải thích 5s (không phải 2s).
4. Nêu cả Plotly + Chart.js.
5. **Bỏ** tuyên bố HTTPS `--cert` cho dashboard (chưa có), hoặc thêm `ssl_context` vào `app.run` rồi giữ.

---

## 8. Tính năng đã có nhưng báo cáo bỏ sót (D8) 🟢 (cơ hội nâng điểm)

Code mạnh hơn báo cáo. Nên **bổ sung** (đều có thật, kèm `file:line`):

- **Bulyan** (`aggregators.py:74-97`) và **Median-of-Means / `mom`** (`aggregators.py:100-108`) — server hỗ trợ `--agg-method bulyan|mom` (`server.py:45`). Đã có `state/metrics_bulyan.json`, `metrics_mom.json`. Báo cáo (ch2) chỉ nói Median/Trimmed/Krum.
- **CatBoost aggregation strategy** riêng (`CatBoostLoggedFedAvg`, `strategy_factory.py:25-26`) + `state/metrics_catboost.json` (acc 0.9996, 1 round). Báo cáo nói "CatBoost không tương thích FedAvg nên loại" (ch3:110) — đúng cho FedAvg, nhưng **code vẫn có chiến lược CatBoost riêng**; nên nhắc tới.
- **Server-side DP noise** (`apply_server_dp`, `aggregators.py:154-166`; `--server-dp-noise`).
- **Quantization 4/8-bit** (`apply_quantization`, `:135-151`; `--quantization-bits`) và **Top-k sparsification** (`apply_topk`, `:123-132`; `--topk-ratio`) — kỹ thuật giảm băng thông truyền thông, rất hợp với động lực FL nêu ở ch1.
- **Clip update norm** (`clip_update`, `:111-120`; `--max-update-norm`).
- **FLANDERS-like z-score outlier filter** (`is_outlier`, `:169-188`; `--flanders-z`) — phòng thủ tấn công, hợp với ch2.
- **Audit hash chain (SHA-256)** trên metrics mỗi vòng (`base.py:120-130`) — tính toàn vẹn log, rất phù hợp môn "An ninh thông tin", nhưng báo cáo **không nhắc**.
- **ROC-AUC, PR-AUC, confusion matrix (tn/fp/fn/tp)** được log (`base.py:54-59, 114-119`) — báo cáo chỉ nói Acc/Prec/Rec/F1.
- **FedProx** (`--fedprox-mu`): code có proximal term thật (`quantum.py:160-165`, `dp_mlp.py:164-169`, và MLP). Báo cáo không đề cập dù `server.py`/`client.py` hỗ trợ.
- **`--agg-method fednova`** xuất hiện trong choices (`server.py:45`) nhưng **`strategy_factory` KHÔNG xử lý "fednova" riêng** → rơi vào nhánh `RobustLoggedFedAvg` và **mặc định về FedAvg** (`robust.py:133`). ⚠️ Đây là *bug nhẹ*: nên (a) bỏ "fednova" khỏi choices, hoặc (b) hiện thực thật. Báo cáo **không nên** liệt kê FedNova như tính năng hoạt động.

**Cách dùng:** Thêm 1 mục "Các cơ chế nâng cao đã hiện thực" trong ch3, liệt kê các tính năng trên. Điều này vừa tăng độ đầy đủ vừa **trung thực** (đang under-claim).

---

## 9. Phạm vi Giới thiệu phóng đại (D9) 🟡

**Báo cáo** (ch1 dòng 8, 10): nói xử lý dữ liệu "Windows PE và Android tại biên"; mở rộng "ba kiến trúc: QCNN, QPCA, QMLP".

**Code thực tế:**
- Dữ liệu: chỉ **Obfuscated-MalMem2022** (memory dump features), không có pipeline PE/Android (`dataset_utils.py:36-63`). Dataset có cột `Class` (benign/malware) + `Category`; **không** drop `hash/timestamp` (báo cáo ch3:56 nói drop hash/timestamp — dataset này không có các cột đó; code chỉ drop `Class`, `Category`).
- Mô hình lượng tử: chỉ **1** mô hình `HybridQuantumModel` (encoder cổ điển + VQC). **Không có** QCNN, QPCA, QMLP riêng biệt được hiện thực. ch2 mô tả lý thuyết QCNN/QPCA/QMLP nhưng đó là **nền lý thuyết**, không phải thứ được code.

**Cách sửa:**
1. ch1: đổi "Windows PE và Android" → "đặc trưng trích từ ảnh chụp bộ nhớ (memory forensics) của bộ MalMem2022". Bỏ/điều chỉnh nếu không có PE/Android.
2. ch1: làm rõ QCNN/QPCA/QMLP là **khảo sát lý thuyết** (ch2), còn **hiện thực thực tế là một mô hình Hybrid QNN 4-qubit** (ch3). Tránh để người đọc tưởng đã build cả ba.
3. ch3:56 (Data Cleaning): sửa lại — thực tế chỉ loại cột nhãn `Class` và `Category` (`dataset_utils.py:58`); xác nhận lại số đặc trưng (xem mục 11).

---

## 10. Bug cần xử lý trước khi trích số liệu (kỹ thuật)

1. **XAI score = 0.0 (MLP):** `state/explanations.json` toàn 0. Nguyên nhân khả dĩ: gradient saliency triệt tiêu trên mô hình đã bão hòa; fallback weight-based (`explain.py:209-211`) chỉ chạy khi `method="auto"` và `np.allclose(scores,0)` — cần kiểm tra vì sao vẫn ra 0 (có thể weight-importance cũng ra giá trị rất nhỏ nhưng *không* đúng 0, hoặc round/model mismatch). → Chạy lại explain với logreg để có bảng dùng được, hoặc debug nhánh MLP.
2. **`fednova` no-op** (mục 8): sửa choices hoặc hiện thực.
3. **`--ssl-certfile/--ssl-keyfile` ở client không được dùng** (mục 1): hoặc xóa khỏi `parse_args` (tránh gây hiểu nhầm), hoặc nối dây vào `start_client`.

---

## 11. Việc cần xác nhận bằng số (chưa kiểm bằng dataset thật)

> Dataset CSV (`Obfuscated-MalMem2022.csv`) hiện **đã bị xóa khỏi git** (git status: `D Obfuscated-MalMem2022.csv`) nhưng còn bản local. Cần chạy để chốt số chính xác:

- [ ] **Số dòng & số đặc trưng:** báo cáo nói **58,596 dòng, 57 đặc trưng**. Xác nhận bằng:
  `python -c "import pandas as pd; d=pd.read_csv('Obfuscated-MalMem2022.csv'); print(d.shape); print(d.drop(columns=['Class','Category'],errors='ignore').shape[1],'features')"`
  (MalMem2022 thường có 58,596 dòng và **55** đặc trưng sau khi bỏ `Class`+`Category`; nếu là 55 thì sửa "57" → số thật.)
- [ ] Tái sinh `state/explanations.json` bằng logreg, lấy Top-10 thật.
- [ ] Chạy & lưu `state/metrics_quantum.json` cho bảng QML.
- [ ] Nếu giữ tuyên bố MLP Deep/Non-IID: chạy lại bộ aggregation với `--model mlp --partition-method noniid` và cập nhật bảng + hình hội tụ.

---

## 12. Thứ tự thực hiện đề xuất (ưu tiên giảm dần)

1. **🔴 D1 (mTLS):** sửa câu chữ thành "server-side TLS" (15 phút) — hoặc nối dây client cert.
2. **🔴 D3 (XAI):** tái sinh `explanations.json` (logreg), thay bảng + nêu rõ model/round/lệnh.
3. **🔴 D2 (aggregation):** sửa metadata thí nghiệm (logreg/IID) + số Krum + diễn giải; lý tưởng là chạy lại Non-IID có tấn công.
4. **🔴 D4 (QML):** chạy & lưu state quantum; hoặc hạ cấp tuyên bố "FL > centralized" + đưa notebook vào repo.
5. **🟡 D5/D6/D9 (quantum arch / DP / phạm vi):** chỉnh cho khớp code (Tanh, ×π, backprop≠param-shift; noise_multiplier; bỏ PE/Android & "3 kiến trúc đã build").
6. **🟡 D7 (dashboard):** đổi tên file/route/polling/SSL.
7. **🟢 D8 (bổ sung tính năng):** thêm mục Bulyan/MoM/quantization/top-k/FLANDERS/audit-hash/ROC-PR/FedProx — tăng độ đầy đủ.
8. **Kỹ thuật:** vá `fednova` no-op, xóa client SSL args thừa, xác nhận shape dataset (mục 11).

---

## Phụ lục A — Bảng đối chiếu nhanh "Báo cáo nói ↔ Code làm"

| Chủ đề | Báo cáo | Code (file:line) | Khớp? |
|--------|---------|------------------|-------|
| Bảo mật kênh | mTLS 2 chiều | server: TLS+CA `server.py:127-136`; client chỉ root CA `client.py:295-305` | ❌ chỉ 1 chiều |
| Lớp strategy | LoggedFedAvg, RobustLoggedFedAvg | `strategy/base.py:22`, `strategy/robust.py:33` | ✅ |
| File metrics | `state/metrics.json` | `base.py:26,71-74` | ✅ |
| File model | `state/latest_model.npz` | `base.py:27,155-159` | ✅ |
| Aggregation | FedAvg/Median/Trimmed/Krum | + Bulyan/MoM/CatBoost `server.py:45` | ⚠️ thiếu liệt kê |
| Krum | chọn 1 client, n−f−2 hàng xóm | `aggregators.py:47-71` (single Krum) | ✅ (single, không multi) |
| Median/Trimmed | coordinate-wise | `aggregators.py:25-44` | ✅ |
| Partition Non-IID | Dirichlet α=0.5 | `dataset_utils.py:203-316`, mặc định 0.5 | ✅ (nhưng thí nghiệm chạy IID) |
| Encoder QML | 57→64→16→4, RX | + Tanh + ×π, AngleEmbedding `quantum.py:63-88` | ⚠️ thiếu chi tiết |
| Gradient QML | Parameter-Shift | backprop (`diff_method="backprop"`) `quantum.py:51` | ❌ sai cơ chế |
| DP | epsilon điều khiển nhiễu | noise_multiplier điều khiển; epsilon đo lại `dp_mlp.py:137-175` | ❌ hiểu sai |
| Dashboard backend | `dashboard_flask.py` | shim → `dashboard_interactive.py` | ⚠️ lệch tên |
| Polling | 2000ms, `/update_metrics` | 1000ms/5000ms, `/api/status` `dashboard_interactive.py:624,901` | ❌ |
| Dashboard HTTPS | `--cert` | không có ssl_context | ❌ |
| Certs | CA 4096, server 2048, SAN, N client loop | `generate_certs.sh:23,29,42-72` | ✅ (SAN có DNS.1/2,IP.1; **không có IP.2=0.0.0.0**) |
| XAI logreg | `I_j=|w_j|` | `explain.py:90-92` | ✅ công thức; ❌ artifact |
| XAI MLP | grad saliency, N=256 | `explain.py:94-100`, default 256 | ✅ công thức; ❌ score=0 |
| Metrics đo | Acc/Prec/Rec/F1 | + ROC-AUC/PR-AUC/confusion/audit-hash | ⚠️ thiếu liệt kê |

> Lưu ý phụ lục: báo cáo ch5:280 ghi `IP.2 = 0.0.0.0` trong SAN, nhưng `generate_certs.sh:42-45` **chỉ có** `DNS.1=localhost, DNS.2=server, IP.1=127.0.0.1` — **không có** `IP.2=0.0.0.0`. Sửa cho khớp.
