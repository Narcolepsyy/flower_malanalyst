
Triển khai Toàn diện Framework Flower cho Hệ thống Phát hiện Mã độc Phân tán với Giám sát Thời gian thực qua Giao diện Web


1. Giới thiệu: Sự Chuyển dịch Kiến trúc trong An ninh Mạng và AI

Trong kỷ nguyên số hóa hiện nay, cuộc chạy đua vũ trang giữa các chuyên gia bảo mật và tội phạm mạng đang diễn ra với tốc độ chưa từng có. Các hệ thống phát hiện xâm nhập (Intrusion Detection Systems - IDS) và phần mềm diệt virus truyền thống, vốn dựa trên chữ ký (signature-based), đang ngày càng trở nên lỗi thời trước sự gia tăng của mã độc đa hình (polymorphic malware) và các cuộc tấn công zero-day. Sự ra đời của Trí tuệ Nhân tạo (AI) và Học máy (Machine Learning - ML) đã mang lại một bước tiến lớn, cho phép phát hiện mã độc dựa trên hành vi và các đặc trưng tĩnh/động phức tạp thay vì chỉ so sánh chuỗi byte đơn thuần. Tuy nhiên, mô hình huấn luyện tập trung (Centralized Learning) - nơi dữ liệu từ hàng triệu thiết bị đầu cuối được gửi về một trung tâm dữ liệu khổng lồ để xử lý - đang vấp phải những rào cản kỹ thuật và pháp lý nghiêm trọng.1
Thứ nhất, vấn đề quyền riêng tư dữ liệu đang trở thành tâm điểm với sự ra đời của các quy định như GDPR (Châu Âu) hay CCPA (California). Dữ liệu nhật ký mạng, file thực thi, hoặc hành vi người dùng trên thiết bị di động chứa đựng thông tin nhạy cảm mà các tổ chức không muốn hoặc không được phép chia sẻ ra bên ngoài. Thứ hai, chi phí băng thông và độ trễ mạng khi truyền tải hàng terabyte dữ liệu thô về máy chủ trung tâm là một gánh nặng khổng lồ đối với cơ sở hạ tầng mạng.1
Trong bối cảnh đó, Học liên kết (Federated Learning - FL) nổi lên như một giải pháp kiến trúc mang tính cách mạng. Bằng cách đảo ngược quy trình học tập - di chuyển tính toán đến nơi có dữ liệu thay vì di chuyển dữ liệu đến nơi tính toán - FL giải quyết đồng thời bài toán bảo mật và hiệu năng.1 Báo cáo này sẽ trình bày một nghiên cứu toàn diện và chi tiết về việc thiết kế, triển khai và giám sát một hệ thống phát hiện mã độc phân tán sử dụng Framework Flower (flwr), kết hợp với giao diện giám sát thời gian thực dựa trên web. Chúng ta sẽ đi sâu vào từng lớp của kiến trúc, từ xử lý dữ liệu Windows PE và Android tại biên, xây dựng mô hình Neural Networks, tùy chỉnh chiến lược tổng hợp (Aggregation Strategy) trong Flower, đến việc trực quan hóa luồng huấn luyện thông qua Streamlit Dashboard.

2. Cơ sở Lý thuyết và Kiến trúc Hệ thống Phân tán

Để xây dựng một hệ thống phát hiện mã độc hiệu quả trên nền tảng Flower, việc thấu hiểu cơ chế toán học của FL và kiến trúc phần mềm của Flower là điều kiện tiên quyết.

2.1. Toán học của Học Liên kết (Federated Learning)

Mục tiêu của FL là tối ưu hóa một mô hình toàn cục (global model) với tham số $w$ mà không cần truy cập trực tiếp vào dữ liệu cục bộ $D_k$ của $K$ khách hàng (clients). Bài toán tối ưu hóa tổng quát có thể được biểu diễn như sau:

$$\min_{w} F(w) = \sum_{k=1}^{K} p_k F_k(w)$$
Trong đó:
$K$ là tổng số lượng client tham gia.
$p_k$ là trọng số của client thứ $k$, thường được xác định bởi tỷ lệ dữ liệu sở hữu: $p_k = \frac{n_k}{n}$, với $n_k = |D_k|$ và $n = \sum n_k$.
$F_k(w)$ là hàm mất mát cục bộ (local loss function) của client $k$, thường được định nghĩa là trung bình lỗi dự đoán trên tập dữ liệu $D_k$.
Thuật toán nền tảng được sử dụng rộng rãi nhất trong FL và cũng là chiến lược mặc định của Flower là Federated Averaging (FedAvg).1 Quy trình hoạt động của FedAvg bao gồm 4 bước lặp lại trong mỗi vòng huấn luyện (communication round):
Selection & Distribution: Máy chủ trung tâm (Server) chọn một tập con các client $S_t$ (với tỷ lệ $C$) và gửi tham số mô hình toàn cục hiện tại $w_t$ cho họ.3
Local Training: Mỗi client $k \in S_t$ thực hiện cập nhật mô hình cục bộ bằng phương pháp Gradient Descent (SGD) trên dữ liệu riêng của mình:

$$w_{t+1}^k \leftarrow w_t - \eta \nabla F_k(w_t)$$

Quá trình này có thể diễn ra trong nhiều epoch cục bộ (local epochs) $E$.2
Upload: Các client gửi tham số cập nhật $w_{t+1}^k$ (hoặc gradient $\Delta w$) trở lại máy chủ.
Aggregation: Máy chủ tổng hợp các cập nhật để tạo ra mô hình toàn cục mới:

$$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{n} w_{t+1}^k$$
Sự ưu việt của Flower nằm ở chỗ nó trừu tượng hóa các bước toán học này thành các thành phần phần mềm có thể tùy chỉnh cao, cho phép các nhà nghiên cứu can thiệp vào bất kỳ bước nào, từ việc chọn client (sampling) đến công thức tổng hợp (aggregation strategy).4

2.2. Kiến trúc Kỹ thuật của Flower Framework

Flower (hay flwr) được thiết kế với triết lý "client-agnostic" (không phụ thuộc client) và "framework-agnostic" (không phụ thuộc framework ML). Điều này có nghĩa là Flower có thể chạy trên các thiết bị biên yếu như Raspberry Pi, điện thoại di động, cho đến các cụm máy chủ lớn, và hỗ trợ PyTorch, TensorFlow, JAX hay thậm chí scikit-learn.3

2.2.1. Các Thành phần Cốt lõi

Hệ thống Flower hoạt động dựa trên mô hình Client-Server thông qua giao thức RPC (Remote Procedure Call).
SuperLink (Bộ liên kết trung tâm): Trong các phiên bản Flower hiện đại, SuperLink đóng vai trò trung gian quản lý kết nối. Nó duy trì trạng thái của hệ thống và điều phối các thông điệp giữa ServerApp và ClientApp.
ServerApp (Logic Máy chủ): Đây là nơi chứa "trí tuệ" của quá trình FL. ServerApp thực thi Strategy, quyết định khi nào bắt đầu vòng mới, cấu hình tham số cho client, và xử lý kết quả trả về. ServerApp không trực tiếp thực hiện tính toán nặng (như lan truyền ngược) mà chỉ thực hiện các phép toán đại số tuyến tính trên các trọng số mô hình.3
ClientApp (Logic Máy khách): Chạy trên các nút biên. ClientApp đóng gói logic huấn luyện cục bộ. Nó nhận Config và Parameters từ Server, khởi tạo mô hình ML cục bộ (ví dụ: PyTorch Module), nạp dữ liệu từ ổ cứng (như file PE hoặc dataset Android), thực hiện huấn luyện (fit), và trả về kết quả.3
Communication Stack (gRPC): Flower sử dụng gRPC làm giao thức nền tảng để truyền tải các payload lớn (trọng số mô hình). gRPC sử dụng Protocol Buffers (Protobuf) để tuần tự hóa dữ liệu, giúp giảm kích thước gói tin và tăng tốc độ truyền tải so với REST/JSON truyền thống.9 Đặc biệt, gRPC hỗ trợ mTLS (mutual TLS), cho phép xác thực hai chiều bảo mật cao, một yêu cầu bắt buộc trong triển khai hệ thống an ninh mạng.10

2.2.2. Vòng đời của một Message trong Flower

Hiểu rõ vòng đời message giúp chúng ta tùy chỉnh việc ghi log cho Dashboard Web:
Server gọi configure_fit: Strategy tạo ra các cấu hình (FitIns) cho các client được chọn.
gRPC truyền FitIns (chứa weights toàn cục) đến Client.
Client gọi fit: Thực hiện training cục bộ.
Client trả về FitRes: Chứa weights mới và các metrics cục bộ (ví dụ: training loss, accuracy).
Server gọi aggregate_fit: Strategy nhận danh sách FitRes, tổng hợp trọng số mới, và quan trọng nhất cho dự án này, là tổng hợp metrics để ghi vào cơ sở dữ liệu giám sát.12

3. Kỹ thuật Dữ liệu cho Phát hiện Mã độc (Data Engineering)

Một hệ thống AI chỉ tốt ngang với dữ liệu mà nó được học. Trong bài toán phát hiện mã độc, đặc trưng dữ liệu (features) phức tạp hơn nhiều so với hình ảnh hay văn bản thông thường. Chúng ta sẽ phân tích hai loại dữ liệu chính: Windows PE và Android, tương ứng với hai tập dữ liệu chuẩn là EMBER và DREBIN.

3.1. Phân tích Mã độc Windows: Tập dữ liệu EMBER

EMBER (Endgame Malware BEnchmark for Research) là tập dữ liệu mã nguồn mở tiêu chuẩn cho việc phát hiện mã độc dạng tĩnh trên Windows.14

3.1.1. Cấu trúc Đặc trưng (Feature Architecture)

EMBER không chứa file thực thi gốc (để đảm bảo an toàn và giảm dung lượng) mà chứa các vector đặc trưng đã trích xuất. Phiên bản EMBER 2018 sử dụng vector 2.381 chiều, trong khi phiên bản EMBER 2024 mở rộng lên 2.568 chiều.16 Các nhóm đặc trưng chính bao gồm:
Byte Histogram (256 đặc trưng): Đếm tần suất xuất hiện của các giá trị byte (0-255). Mã độc bị nén hoặc mã hóa thường có phân phối byte entropy cao, khác biệt so với phần mềm sạch.
Byte Entropy Histogram (256 đặc trưng): Đo lường độ hỗn loạn thông tin trên cửa sổ trượt của file, giúp phát hiện các vùng mã hóa hoặc packer.
String Information (104 đặc trưng): Thống kê về các chuỗi in được (printable strings), bao gồm số lượng, độ dài trung bình, và phân phối ký tự.
General File Information: Kích thước file, timestamp, linker version.
Header Information: Các thông tin từ COFF header và Optional header của định dạng PE (Portable Executable). Điều này rất quan trọng vì mã độc thường giả mạo hoặc làm hỏng các header này để đánh lừa các công cụ phân tích.15
Section Information: Đặc điểm của các section như .text, .data, .rsrc. Mã độc thường chèn mã thực thi vào các section không chuẩn hoặc thay đổi quyền truy cập bộ nhớ của section.
Import/Export Tables: Danh sách các hàm API mà file gọi từ hệ thống (Imported Functions) và các hàm nó cung cấp (Exported Functions). Ví dụ, mã độc thường import các hàm liên quan đến mạng (socket, connect) hoặc thao tác file (CreateFile, WriteFile).16

3.1.2. Quy trình Xử lý Dữ liệu trên Flower Client

Trên thực tế, ClientApp sẽ không nhận vector sẵn mà phải xử lý file thô. Quy trình tích hợp vào Flower như sau:
Bước 1: Parsing: Sử dụng thư viện LIEF (Library to Instrument Executable Formats) để parse cấu trúc file PE.16
Bước 2: Feature Extraction: Áp dụng logic của EMBER để chuyển đổi cấu trúc LIEF thành vector số học numpy.array.
Bước 3: Normalization: Chuẩn hóa dữ liệu (ví dụ: Log scaling cho kích thước file) để giúp mạng Neural hội tụ nhanh hơn.

3.2. Phân tích Mã độc Android: Tập dữ liệu DREBIN

Đối với nền tảng di động, DREBIN là tập dữ liệu kinh điển, chứa các đặc trưng tĩnh trích xuất từ AndroidManifest.xml và mã bytecode đã dịch ngược (disassembled code).20

3.2.1. Không gian Đặc trưng (Feature Space)

DREBIN sử dụng biểu diễn vector nhị phân thưa (sparse binary vectors) cho các đặc trưng chuỗi. Các nhóm đặc trưng bao gồm:
S1: Hardware Components: Các yêu cầu phần cứng (GPS, Camera, Touchscreen). Mã độc thường yêu cầu quyền truy cập phần cứng bất thường.
S2: Requested Permissions: Các quyền truy cập hệ thống (SEND_SMS, INTERNET, READ_CONTACTS). Đây là chỉ dấu quan trọng nhất. Ví dụ, một ứng dụng đèn pin (Flashlight) mà yêu cầu quyền gửi tin nhắn (SEND_SMS) là rất đáng ngờ.21
S3: App Components: Tên của Activities, Services, Content Providers.
S4: Filtered Intents: Các tín hiệu (Intents) mà ứng dụng lắng nghe (ví dụ: BOOT_COMPLETED để tự khởi động cùng hệ thống, một hành vi điển hình của malware).23
S5 - S8: Code Features: Bao gồm Restricted API calls, Suspicious API calls, và Network Addresses (URL, IP) được tìm thấy trong code.21

3.2.2. Chiến lược Mã hóa (Encoding Strategy)

Do số lượng quyền và API là rất lớn (hàng nghìn), vector đặc trưng của DREBIN thường có số chiều rất cao (ví dụ: 10.000+).
Multi-hot Encoding: Mỗi ứng dụng được biểu diễn bằng một vector mà giá trị 1 tại vị trí $i$ nghĩa là đặc trưng thứ $i$ xuất hiện.
Dimensionality Reduction: Trong môi trường FL, gửi các gradient của vector quá lớn sẽ tốn băng thông. Client có thể áp dụng các kỹ thuật chọn lọc đặc trưng (Feature Selection) như PCA hoặc Autoencoders để nén vector trước khi đưa vào mô hình phân lớp.14

Bảng 1: So sánh Đặc điểm Kỹ thuật giữa EMBER và DREBIN trong Ngữ cảnh FL


Đặc điểm
EMBER (Windows)
DREBIN (Android)
Nguồn dữ liệu
PE Header, Sections, Byte Histogram
AndroidManifest.xml, Dex code
Loại đặc trưng
Dense Vector (Số thực & Số nguyên)
Sparse Binary Vector (0/1)
Số chiều (Gốc)
2,381 (v2018) / 2,568 (v2024) 16
~545,000 (Toàn bộ) -> ~10,000 (Chọn lọc) 20
Mô hình phù hợp
Multi-Layer Perceptron (MLP), LightGBM
CNN 1D, MLP, Deep Belief Networks
Thách thức FL
Kích thước file lớn, trích xuất chậm (LIEF)
Vector thưa, mất cân bằng dữ liệu cao
Thư viện xử lý
lief, pefile
androguard, apktool


4. Thiết kế Mô hình Học máy cho Phát hiện Phân tán

Trong môi trường Federated Learning, mô hình cần cân bằng giữa độ chính xác và kích thước (để giảm chi phí truyền tải mạng). Chúng ta sẽ xem xét hai kiến trúc chính: MLP cho dữ liệu dạng bảng (EMBER) và CNN cho dữ liệu cấu trúc/chuỗi (DREBIN).

4.1. Kiến trúc MLP cho EMBER (Deep Feedforward Network)

Mô hình Multi-Layer Perceptron (MLP) là lựa chọn tiêu chuẩn cho dữ liệu EMBER vì tính chất dạng bảng (tabular) của các đặc trưng trích xuất.19
Input Layer: 2381 nodes (tương ứng với số chiều EMBER 2018).
Hidden Layers: Kiến trúc hình tháp ngược để nén dần thông tin đặc trưng:
Layer 1: 1024 neurons + ReLU + Batch Normalization + Dropout (0.2).
Layer 2: 512 neurons + ReLU + Batch Normalization + Dropout (0.2).
Layer 3: 256 neurons + ReLU + Batch Normalization.
Output Layer: 1 neuron + Sigmoid Activation (cho bài toán phân loại nhị phân Malware/Benign).25
Sự hiện diện của Batch Normalization là cực kỳ quan trọng trong FL. Vì dữ liệu ở các client là Non-IID (không phân phối độc lập và đồng nhất), việc chuẩn hóa batch giúp giảm sự thay đổi hiệp biến nội bộ (Internal Covariate Shift), giúp mô hình toàn cục hội tụ ổn định hơn khi tổng hợp từ nhiều nguồn dữ liệu khác nhau.26

4.2. Kiến trúc CNN cho DREBIN

Mặc dù CNN thường dùng cho ảnh, nhưng với dữ liệu Android Permissions (dạng vector thưa hoặc ma trận kề), CNN 1D hoặc 2D cho thấy hiệu quả vượt trội trong việc phát hiện các mẫu tương quan cục bộ (ví dụ: sự kết hợp của một nhóm quyền cụ thể thường đi kèm với malware).27
Embedding Layer: Chuyển đổi vector sparse đầu vào thành dense vector kích thước nhỏ hơn.
Convolutional Layers:
Conv1D (Filters: 64, Kernel Size: 3) + ReLU.
MaxPooling1D.
Conv1D (Filters: 128, Kernel Size: 3) + ReLU.
Fully Connected Layers: Dense layer (128) -> Output (1).
Các nghiên cứu gần đây 30 thậm chí còn đề xuất sử dụng Quantum Machine Learning (QML) với các kiến trúc lai như QMLP hoặc QCNN để cải thiện khả năng phát hiện các mẫu phức tạp trong không gian đặc trưng cao chiều của mã độc, tuy nhiên trong phạm vi triển khai thực tế hiện tại, CNN cổ điển vẫn là lựa chọn tối ưu về hiệu năng/tài nguyên.

5. Triển khai Hệ thống Flower (Implementation Strategy)

Phần này đi sâu vào mã nguồn và logic triển khai, tập trung vào việc tùy chỉnh Flower để hỗ trợ giám sát thời gian thực.

5.1. Xây dựng Custom Strategy để Ghi Log (Logging Strategy)

Đây là thành phần quan trọng nhất để kết nối Flower với Web Dashboard. Mặc định, Flower chỉ in log ra màn hình console. Chúng ta cần tạo một lớp kế thừa từ FedAvg (hoặc FedAdagrad, FedOpt) để "bắt" các sự kiện tổng hợp và ghi dữ liệu ra file JSON.12
Logic của chiến lược này như sau:
Ghi đè phương thức aggregate_evaluate (dùng cho đánh giá tập trung hoặc phân tán).
Sau khi nhận kết quả từ super().aggregate_evaluate(...), trích xuất loss và metrics (accuracy, precision, recall).
Lưu các giá trị này vào một cấu trúc dữ liệu bền vững (JSON file hoặc SQLite database) mà Web App có thể đọc được.12
Mã nguồn triển khai (Conceptual Implementation):

Python


import json
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Parameters, Scalar

class LoggedFedAvg(FedAvg):
    def __init__(self, log_file="state/metrics.json", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        # Khởi tạo file log với cấu trúc rỗng
        self._initialize_log_file()

    def _initialize_log_file(self):
        initial_data = {"rounds":, "accuracy":, "loss":, "val_loss":}
        with open(self.log_file, "w") as f:
            json.dump(initial_data, f)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List],
        failures: List, BaseException]],
    ) -> Tuple[Optional[float], Dict]:
        
        # Gọi hàm gốc để thực hiện thuật toán FedAvg
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if aggregated_loss is not None:
            # Logic ghi log tùy chỉnh
            self._save_metrics(server_round, aggregated_loss, aggregated_metrics)
        
        return aggregated_loss, aggregated_metrics

    def _save_metrics(self, round_num, loss, metrics):
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)
            
            data["rounds"].append(round_num)
            data["loss"].append(loss)
            
            # Trích xuất accuracy từ dictionary metrics
            # Lưu ý: Client phải trả về key "accuracy" trong hàm evaluate()
            acc = metrics.get("accuracy", 0.0)
            data["accuracy"].append(acc)
            
            with open(self.log_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Lỗi khi ghi log metrics: {e}")



5.2. Triển khai Flower Client (ClientApp)

Client cần thực hiện các tác vụ: tải dữ liệu, định nghĩa mô hình, và giao tiếp với Server.

Python


class MalwareClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        # Trả về trọng số mô hình dưới dạng list của numpy arrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        # Cập nhật trọng số mô hình từ server
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Thực hiện local training (1 hoặc nhiều epoch)
        train(self.net, self.trainloader, epochs=1) 
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        # Trả về accuracy để Server tổng hợp
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


Điểm mấu chốt ở đây là hàm evaluate phải trả về dictionary chứa key accuracy khớp với key mà LoggedFedAvg mong đợi.8

5.3. Cấu hình ServerApp và Khởi chạy

ServerApp kết nối Strategy với cấu hình vòng lặp.

Python


def server_fn(context: Context):
    # Định nghĩa hàm đánh giá tập trung (nếu có server-side dataset)
    # Hoặc để Strategy tự tổng hợp từ client
    strategy = LoggedFedAvg(
        fraction_fit=1.0,  # Chọn 100% client sẵn sàng để train
        fraction_evaluate=1.0,
        min_fit_clients=2, # Cần tối thiểu 2 client để bắt đầu
        min_evaluate_clients=2,
        log_file="state/metrics.json"
    )
    config = ServerConfig(num_rounds=50) # Chạy 50 vòng
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)



6. Giao diện Giám sát Web Thời gian thực (Web Dashboard)

Yêu cầu của dự án là một giao diện web để giám sát quá trình phát hiện mã độc. Chúng ta sẽ sử dụng Streamlit vì tính đơn giản và khả năng tương tác cao với dữ liệu Python.33 Tuy nhiên, Streamlit hoạt động theo cơ chế chạy lại script mỗi khi có tương tác, nên việc cập nhật thời gian thực (real-time) cần kỹ thuật xử lý khéo léo.

6.1. Kiến trúc Đọc Dữ liệu Bất đồng bộ (Asynchronous Data Reading)

Do ServerApp và Streamlit App là hai tiến trình độc lập (thậm chí chạy trên hai container khác nhau), chúng giao tiếp thông qua "Shared State" (Trạng thái chia sẻ). Trong thiết kế này, file state/metrics.json đóng vai trò là hàng đợi thông điệp đơn giản hóa (simplified message queue).31
ServerApp (Writer): Ghi thêm dữ liệu vào cuối file JSON sau mỗi vòng.
Streamlit (Reader): Đọc file JSON định kỳ và vẽ lại biểu đồ.

6.2. Triển khai Streamlit Dashboard

Mã nguồn Dashboard cần xử lý việc tự động làm mới (auto-refresh) mà không cần người dùng tải lại trang.

Python


import streamlit as st
import json
import pandas as pd
import time
import os
import plotly.express as px

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Giám sát Mã độc FL",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Federated Malware Detection Dashboard")
st.markdown("Giám sát quá trình huấn luyện mô hình phát hiện mã độc phân tán theo thời gian thực.")

METRICS_FILE = "state/metrics.json"

# Hàm load dữ liệu với Caching để tối ưu hiệu năng
# Tuy nhiên với realtime, ta cần clear cache hoặc đọc trực tiếp
def load_data():
    if not os.path.exists(METRICS_FILE):
        return None
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

# Layout Dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.info("Trạng thái: Đang hoạt động")
with col2:
    round_display = st.metric(label="Vòng huấn luyện (Round)", value="0")
with col3:
    acc_display = st.metric(label="Độ chính xác hiện tại", value="0%")

# Placeholder cho biểu đồ
st.subheader("Diễn biến Huấn luyện")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    loss_chart = st.empty()
with chart_col2:
    accuracy_chart = st.empty()

st.subheader("Dữ liệu chi tiết")
data_table = st.empty()

# Vòng lặp cập nhật (Polling Loop)
# Sử dụng st.empty() để thay thế nội dung cũ mà không duplicate
while True:
    data = load_data()
    
    if data and len(data["rounds"]) > 0:
        df = pd.DataFrame({
            "Round": data["rounds"],
            "Accuracy": data["accuracy"],
            "Loss": data["loss"]
        })
        
        # Cập nhật Metrics số
        current_round = df.iloc[-1]
        current_acc = df["Accuracy"].iloc[-1]
        round_display.metric(label="Vòng huấn luyện (Round)", value=f"{current_round}")
        acc_display.metric(label="Độ chính xác hiện tại", value=f"{current_acc:.4f}")

        # Vẽ biểu đồ Plotly
        fig_loss = px.line(df, x="Round", y="Loss", title="Hàm mất mát (Loss) theo thời gian", markers=True)
        fig_acc = px.line(df, x="Round", y="Accuracy", title="Độ chính xác (Accuracy) theo thời gian", markers=True)
        
        # Cập nhật vào placeholder
        loss_chart.plotly_chart(fig_loss, use_container_width=True)
        accuracy_chart.plotly_chart(fig_acc, use_container_width=True)
        
        # Cập nhật bảng
        data_table.dataframe(df.sort_values(by="Round", ascending=False).head(10))
    
    # Nghỉ 2 giây trước khi poll lại
    time.sleep(2)


Lưu ý kỹ thuật: Cách tiếp cận while True với time.sleep trong Streamlit là cách đơn giản nhất để mô phỏng realtime dashboard.35 Trong các phiên bản Streamlit mới, decorator @st.fragment(run_every=2) có thể được sử dụng để chỉ làm mới một phần giao diện, giúp trải nghiệm mượt mà hơn.37

6.3. Giải pháp Thay thế: Grafana và Prometheus

Đối với các hệ thống sản xuất (Production) yêu cầu độ tin cậy cao hơn, giải pháp chuyên nghiệp là sử dụng Prometheus để scrape metrics và Grafana để hiển thị.38
Flower Server có thể tích hợp thư viện prometheus_client để expose metrics tại một endpoint HTTP (ví dụ: /metrics).
Prometheus sẽ định kỳ "kéo" (scrape) dữ liệu từ endpoint này.
Grafana kết nối với Prometheus để vẽ biểu đồ.
Giải pháp này phức tạp hơn trong cấu hình nhưng mạnh mẽ hơn Streamlit về khả năng lưu trữ lịch sử dài hạn và cảnh báo (alerting).38

7. Bảo mật và Tính Bền vững của Hệ thống (Security & Robustness)

Trong lĩnh vực an ninh mạng, việc bảo vệ chính hệ thống phát hiện là tối quan trọng. Federated Learning mở ra các bề mặt tấn công mới mà hệ thống tập trung không có.

7.1. Bảo mật Kênh Truyền thông (Communication Security)

Flower sử dụng gRPC. Để ngăn chặn tấn công Man-in-the-Middle (MitM) và giả mạo client, hệ thống bắt buộc phải triển khai mTLS (Mutual Transport Layer Security).10
Quy trình triển khai mTLS:
Certificate Authority (CA): Tạo một CA tự ký (self-signed) đóng vai trò gốc tin cậy.
Server Certificates: Tạo private key và Certificate Signing Request (CSR) cho Server, sau đó dùng CA để ký chứng chỉ cho Server.
Client Certificates: Tương tự, tạo cặp key/cert riêng biệt cho từng Client (Client 1, Client 2,...).
Cấu hình Flower:
Server khởi chạy với root_certificates (CA cert), private_key, và certificate_chain (Server cert).10
Client kết nối với root_certificates (để verify server) và cặp key/cert của chính nó (để server verify client).

7.2. Tấn công Đầu độc và Phòng thủ (Poisoning Attacks & Defenses)

Kẻ tấn công có thể kiểm soát một số client và gửi các cập nhật mô hình độc hại (poisoned updates) nhằm làm sai lệch mô hình toàn cục (Untargeted Attack) hoặc cài cắm cửa hậu (Backdoor Attack).42
Các cơ chế phòng thủ tích hợp:
Robust Aggregation: Thay thế thuật toán FedAvg (trung bình cộng dễ bị ảnh hưởng bởi giá trị ngoại lai) bằng các thuật toán bền vững hơn:
FedMedian: Lấy trung vị của các tham số thay vì trung bình.
Trimmed Mean: Loại bỏ $k$ phần trăm giá trị lớn nhất và nhỏ nhất trước khi tính trung bình.
Krum / Multi-Krum: Chọn bản cập nhật giống với đa số các bản cập nhật khác nhất.43
FLANDERS (Pre-aggregation Filter): Một kỹ thuật tiên tiến sử dụng phân tích chuỗi thời gian ma trận. FLANDERS coi các bản cập nhật từ một client qua các vòng là một chuỗi thời gian. Nếu bản cập nhật tại vòng $t$ lệch quá xa so với dự đoán dựa trên lịch sử của client đó (sử dụng mô hình ARIMA hoặc tương tự), nó sẽ bị đánh dấu là độc hại và bị loại bỏ trước khi tổng hợp.43

7.3. Bảo vệ Quyền riêng tư (Differential Privacy)

Mặc dù dữ liệu không rời khỏi client, kẻ tấn công vẫn có thể khôi phục dữ liệu gốc từ gradient (Inference Attack). Để phòng chống, kỹ thuật Differential Privacy (DP) (Vi phân riêng tư) cần được áp dụng.
Cơ chế: Thêm nhiễu Gaussian (Gaussian Noise) vào các gradient tại phía client trước khi gửi đi (Local DP) hoặc tại phía server sau khi tổng hợp (Central DP).
Flower Support: Flower cung cấp các wrapper để tích hợp Opacus (thư viện DP của PyTorch) vào ClientApp một cách dễ dàng, cho phép kiểm soát ngân sách riêng tư (privacy budget $\epsilon$).46

8. Triển khai và Vận hành (Deployment & Operations)

Để chuyển từ mã nguồn nghiên cứu sang một hệ thống chạy được, công nghệ Containerization là chìa khóa.

8.1. Docker Compose cho Môi trường Phân tán

Sử dụng Docker Compose để định nghĩa và khởi chạy toàn bộ hệ sinh thái. Điều này đảm bảo tính nhất quán của môi trường (phiên bản Python, thư viện C++ cho LIEF, v.v.) giữa các node.38
Chiến lược Mount Volume:
Vấn đề khó khăn nhất khi kết hợp Flower và Streamlit trong Docker là chia sẻ dữ liệu. Chúng ta giải quyết bằng cách mount một volume vật lý từ máy chủ (host) vào cả hai container server và web-ui.
Container flower-server ghi log vào /app/state/metrics.json (thực chất là lưu trên host).
Container streamlit-ui đọc log từ /app/state/metrics.json (cũng từ host).
Điều này tạo ra một kênh giao tiếp qua file đơn giản nhưng hiệu quả cho việc giám sát.38

8.2. Xử lý Tính Không đồng nhất (Heterogeneity)

Trong thực tế, các client phát hiện mã độc có năng lực phần cứng rất khác nhau (Server mạnh vs Laptop yếu vs IoT). Flower hỗ trợ xử lý vấn đề này thông qua:
Timeout: Thiết lập thời gian chờ tối đa cho mỗi vòng huấn luyện. Nếu client yếu không phản hồi kịp, nó sẽ bị loại khỏi vòng đó (drop_client).
Resource Limits: Sử dụng Docker để giới hạn CPU/RAM cho từng client mô phỏng, giúp kiểm thử tính ổn định của hệ thống trước các client chậm.38

9. Kết luận và Hướng phát triển

Báo cáo này đã trình bày một kiến trúc tham chiếu toàn diện cho việc triển khai hệ thống phát hiện mã độc phân tán sử dụng Flower Framework và Streamlit. Giải pháp này giải quyết triệt để các vấn đề của mô hình tập trung: bảo vệ quyền riêng tư dữ liệu nhạy cảm, giảm tải băng thông mạng, và cung cấp khả năng giám sát trực quan thời gian thực.
Việc tích hợp sâu các kỹ thuật như trích xuất đặc trưng chuyên biệt cho mã độc (LIEF cho PE, Androguard cho Android), các chiến lược tổng hợp tùy chỉnh (LoggedFedAvg), và các cơ chế bảo mật nâng cao (mTLS, FLANDERS) tạo nên một hệ thống phòng thủ mạnh mẽ và linh hoạt.
Hướng phát triển tương lai:
Học bán giám sát (Semi-supervised FL): Tận dụng lượng dữ liệu khổng lồ không được gán nhãn tại các máy client.
Cá nhân hóa (Personalization): Cho phép mỗi client giữ lại một phần mô hình cục bộ (Fine-tuning) để thích nghi tốt hơn với đặc thù mã độc tại môi trường của họ (ví dụ: mã độc tài chính tấn công ngân hàng vs mã độc ransomware tấn công bệnh viện).
Tích hợp Quantum AI: Thử nghiệm các kiến trúc QCNN trên các thiết bị lượng tử mô phỏng để đón đầu xu hướng công nghệ tương lai.30

Phụ lục: Bảng Tóm tắt Cấu hình Hệ thống Đề xuất


Hạng mục
Chi tiết Cấu hình
Ghi chú
Framework FL
Flower (flwr) v1.13+
Hỗ trợ ClientApp/ServerApp API mới 3
Giao thức
gRPC + mTLS
Mã hóa SSL/TLS bắt buộc 10
Dataset (Windows)
EMBER 2024
2,568 features, yêu cầu LIEF parser 16
Dataset (Android)
DREBIN
Permissions & Intent filters, Sparse vector 20
Mô hình AI
MLP (4 layers)
Tối ưu cho dữ liệu dạng bảng, ReLU, BatchNorm 25
Web UI
Streamlit
Polling interval 2s, đọc JSON shared state 34
Tổng hợp
LoggedFedAvg
Kế thừa FedAvg, thêm logic ghi file JSON 13
Triển khai
Docker Compose
3 Services: Server, Client(xN), WebUI 38

Nguồn trích dẫn
What is Federated Learning? - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
Introduction - 2025 Tutorial: Federated AI Simulations with Flower - YouTube, truy cập vào tháng 11 29, 2025, https://www.youtube.com/watch?v=XK_dRVcSZqg
Get started with Flower - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
Exploring Flower: A Federated Learning Framework | by Salem Alqahtani - Medium, truy cập vào tháng 11 29, 2025, https://salemal.medium.com/exploring-flower-a-federated-learning-framework-29111892b389
Use strategies - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/how-to-use-strategies.html
Example projects - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/ref-example-projects.html
Flower Framework Documentation, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/index.html
Communicate custom Messages - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html
Configure logging - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/how-to-configure-logging.html
start_client - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/ref-api/flwr.client.start_client.html
Secure gRPC Client/Server over mTLS - Handra | Welcome to My Site, truy cập vào tháng 11 29, 2025, https://www.handracs.info/blog/grpcmtlsgo/
Customize a Flower Strategy, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html
Aggregate evaluation results - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/how-to-aggregate-evaluation-results.html
Identifying Useful Features for Malware Detection in the Ember Dataset - ResearchGate, truy cập vào tháng 11 29, 2025, https://www.researchgate.net/publication/338510121_Identifying_Useful_Features_for_Malware_Detection_in_the_Ember_Dataset
EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models - arXiv, truy cập vào tháng 11 29, 2025, https://arxiv.org/pdf/1804.04637
EMBER2024: A New Benchmark for Holistic Malware Classification - Chris Zhang - Medium, truy cập vào tháng 11 29, 2025, https://zhanghaolin66.medium.com/ember2024-a-new-benchmark-for-holistic-malware-classification-62dcb260b47a
An Efficient Boosting-Based Windows Malware Family Classification System Using Multi-Features Fusion - MDPI, truy cập vào tháng 11 29, 2025, https://www.mdpi.com/2076-3417/13/6/4060
autruonggiang/IE105-FL-Flower: Implementation of a Federated Learning Framework for Portable Executable (PE) Malware Classification. - GitHub, truy cập vào tháng 11 29, 2025, https://github.com/autruonggiang/IE105-FL-Flower
fabiocaiulo8/malware-detection: Quantum/classical machine learning models - GitHub, truy cập vào tháng 11 29, 2025, https://github.com/fabiocaiulo8/malware-detection
AMDDLmodel: Android smartphones malware detection using deep learning model - Research journals - PLOS, truy cập vào tháng 11 29, 2025, https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0296722
Machine learning models and dimensionality reduction for improving the Android malware detection - PMC - NIH, truy cập vào tháng 11 29, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11784760/
Malware Detection Dataset - Kaggle, truy cập vào tháng 11 29, 2025, https://www.kaggle.com/datasets/ankit1743/android-malware-detection-dataset
Drebin: Effective and Explainable Detection of Android Malware in Your Pocket, truy cập vào tháng 11 29, 2025, https://www.ndss-symposium.org/wp-content/uploads/2017/09/11_3_1.pdf
Using convolutional neural network for Android malware detection - Computer Modelling and New Technologies, truy cập vào tháng 11 29, 2025, http://www.cmnt.lv/upload-files/ns_96art03_CMNT2301_Karabey-Aksakalli.pdf
Detecting malware using the MLP algorithm - Warse, truy cập vào tháng 11 29, 2025, https://www.warse.org/IJATCSE/static/pdf/file/ijatcse214942020.pdf
(PDF) SecFedMDM-1: A Federated Learning-Based Malware Detection Model for Interconnected Cloud Infrastructures - ResearchGate, truy cập vào tháng 11 29, 2025, https://www.researchgate.net/publication/392544836_SecFedMDM-1_A_Federated_Learning-Based_Malware_Detection_Model_for_Interconnected_Cloud_Infrastructures
Deep Feature Extraction and Classification of Android Malware Images - MDPI, truy cập vào tháng 11 29, 2025, https://www.mdpi.com/1424-8220/20/24/7013
An Android Malware Detection Method Based on CNN Mixed-data Model, truy cập vào tháng 11 29, 2025, https://ceur-ws.org/Vol-2732/20200198.pdf
Deep Learning-Based Android Malware Detection with CNN-GRU Model - NORMA@NCI Library, truy cập vào tháng 11 29, 2025, https://norma.ncirl.ie/8327/1/harisankarkalathilsalim.pdf
Towards Quantum Machine Learning for Malicious Code Analysis - arXiv, truy cập vào tháng 11 29, 2025, https://arxiv.org/html/2508.19381v1
How to Create a Streamlit Dashboard with State Persistence in JSON? - Stack Overflow, truy cập vào tháng 11 29, 2025, https://stackoverflow.com/questions/77708961/how-to-create-a-streamlit-dashboard-with-state-persistence-in-json
Federated Learning with Flower: Practical Code and a Complete Beginner's Guide | by Sara younesi | Medium, truy cập vào tháng 11 29, 2025, https://medium.com/@sarayounesi/when-data-stays-local-the-rise-of-federated-learning-fe8c92465ace
Multi class flower classification using Python || Streamlit || FastAI - YouTube, truy cập vào tháng 11 29, 2025, https://www.youtube.com/watch?v=_e6p9SvPsiY
How to build a real-time live dashboard with Streamlit, truy cập vào tháng 11 29, 2025, https://discuss.streamlit.io/t/how-to-build-a-real-time-live-dashboard-with-streamlit/24437
Continuously updating dashboard - Using Streamlit, truy cập vào tháng 11 29, 2025, https://discuss.streamlit.io/t/continuously-updating-dashboard/532
Best practice for real-time app - Using Streamlit, truy cập vào tháng 11 29, 2025, https://discuss.streamlit.io/t/best-practice-for-real-time-app/2661
Efficiently Visualizing Multiple Live Data Streams in Streamlit, truy cập vào tháng 11 29, 2025, https://discuss.streamlit.io/t/efficiently-visualizing-multiple-live-data-streams-in-streamlit/88653
Leveraging Flower and Docker for Device Heterogeneity Management in Federated Learning - Flower Examples 1.24.0, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/examples/flower-via-docker-compose.html
Monitoring Celery: My Walk with Flower, Prometheus and Grafana | by Seun Runsewe, truy cập vào tháng 11 29, 2025, https://runsewe-seun.medium.com/monitoring-celery-my-walk-with-flower-prometheus-and-grafana-4dcab785561b
Monitor your infrastructure with Streamlit - Red Hat, truy cập vào tháng 11 29, 2025, https://www.redhat.com/en/blog/streamlit-monitor-infrastructure
Securing gRPC service-to-service communications with mTLS | by Charith Rajitha | Medium, truy cập vào tháng 11 29, 2025, https://medium.com/@rajithacharith/securing-grpc-service-to-service-communications-with-mtls-74c5a8583a4a
Deep Model Poisoning Attack on Federated Learning - MDPI, truy cập vào tháng 11 29, 2025, https://www.mdpi.com/1999-5903/13/3/73
[2303.16668] Protecting Federated Learning from Extreme Model Poisoning Attacks via Multidimensional Time Series Anomaly Detection - arXiv, truy cập vào tháng 11 29, 2025, https://arxiv.org/abs/2303.16668
FLANDERS: Protecting Federated Learning from Extreme Model Poisoning Attacks via Multidimensional Time Series Anomaly Detection - Flower Baselines 1.24.0, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/baselines/flanders.html
FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients - Zaixi Zhang, truy cập vào tháng 11 29, 2025, https://zaixizhang.github.io/ZaixiZhang_files/FLDetector.pdf
adap/flower - A Friendly Federated AI Framework - GitHub, truy cập vào tháng 11 29, 2025, https://github.com/adap/flower
Quickstart with Docker Compose - Flower Framework, truy cập vào tháng 11 29, 2025, https://flower.ai/docs/framework/main/ko/docker/tutorial-quickstart-docker-compose.html
